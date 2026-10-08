"""
VL and audio export pipelines (genai builder + liquidonnx graphs and precisions) on tiny models.

The VL checkpoint is built from a config (only the LFM2.5-VL-1.6B processor is downloaded); the
audio export is the synthetic one from tests/test_lfm2_audio/synthetic.py.

Run with:
    uv run pytest tests/test_genai_export_multimodal.py -v
    uv run pytest tests/test_genai_export_multimodal.py -v -k "vl and q4"
"""

import json
import pathlib
import shutil
import sys

import huggingface_hub
import numpy as np
import onnx
import onnxruntime_genai as og
import pytest
import torch
from helpers import Q8_FP32_HEAD, attributes, matmul_bits
from huggingface_hub.utils import filter_repo_objects
from PIL import Image
from test_lfm2_audio.synthetic import (
    HIDDEN,
    SmallDepthformer,
    build_model_dir,
    write_full_checkpoint,
)
from test_lfm2_audio.synthetic import write_wav as write_tone
from tokenizers import Regex, Tokenizer, models, pre_tokenizers
from transformers import AutoProcessor, Lfm2VlConfig, Lfm2VlForConditionalGeneration

from liquidonnx.compare import has_errors
from liquidonnx.compare import vl as compare_vl
from liquidonnx.compare.metrics import greedy
from liquidonnx.compare.text import eos_ids
from liquidonnx.embeddings import QUANT_TABLE, TABLE, TOKEN_ID, embed
from liquidonnx.genai_builder import Q4F32, export_decoder
from liquidonnx.genai_runtime import generate, load_model
from liquidonnx.lfm2_audio import export as audio_export
from liquidonnx.lfm2_audio.builder import depthformer_builder
from liquidonnx.lfm2_audio.infer import AudioChat, chat_prompt, single_turn
from liquidonnx.lfm2_vl import export as vl_export
from liquidonnx.lfm2_vl import infer as vl_infer
from liquidonnx.session import decoder_inputs, initialize_cache, load_onnx_session, update_cache
from liquidonnx.verify import compare_token_cosine

ASSETS = pathlib.Path(__file__).parent / "test_lfm2_vl/assets"
IMAGE = ASSETS / "cardinal.jpg"
MIXED = (ASSETS / "wide.jpg", ASSETS / "tall.jpg")  # patch grids 18x52 and 44x22
# Resized once, patch grids 26x36 and 44x22; split, the 1280x960 cardinal is 3x2 tiles of 32x32
# patches plus its 26x36 thumbnail, and the 480x960 tall image is under the splitting threshold.
TILED = (IMAGE, ASSETS / "tall.jpg")
TILED_SHAPES = [[32, 32]] * 6 + [[26, 36], [44, 22]]
PRECISIONS = ["fp32", "fp16", "q8", "q4"]
MIN_COSINE = {"fp32": 0.99999, "fp16": 0.9999, "q8": 0.999, "q4": 0.97}


def session(output_dir: pathlib.Path, filename: str):
    return load_onnx_session(output_dir / "onnx" / filename)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b)))


def initializers(path: pathlib.Path) -> dict[str, np.ndarray]:
    graph = onnx.load(str(path)).graph
    return {init.name: onnx.numpy_helper.to_array(init) for init in graph.initializer}


def signature(session) -> list[tuple]:
    return [(v.name, v.type, v.shape) for v in [*session.get_inputs(), *session.get_outputs()]]


def check_int8_embeddings(output_dir: pathlib.Path):
    """The q4 and q8 embedding model against the fp32 one: the same I/O, each looked-up value
    within half an int8 step (blocks of 32 along H) of the fp32 table, features passed through."""
    fp32 = initializers(output_dir / "onnx/embeddings.onnx")
    int8 = initializers(output_dir / "onnx/embeddings_q8.onnx")
    table, token_id = fp32[TABLE], int(fp32[TOKEN_ID])
    assert TABLE not in int8
    assert (int8[QUANT_TABLE[0]].dtype, int8[QUANT_TABLE[0]].shape) == (np.uint8, table.shape)

    quantized = session(output_dir, "embeddings_q8.onnx")
    assert signature(quantized) == signature(session(output_dir, "embeddings.onnx"))

    features = np.random.default_rng(0).standard_normal((1, table.shape[1])).astype(np.float32)
    actual = embed(quantized, np.arange(len(table))[None], features)[0]
    np.testing.assert_array_equal(actual[token_id], features[0])

    blocks = table.reshape(len(table), -1, 32)
    half_step = np.repeat((blocks.max(-1) - blocks.min(-1)) / 255 / 2, 32, axis=1)
    within = np.abs(actual - table) <= half_step + 1e-6
    assert np.delete(within, token_id, axis=0).all()


# (placeholders, feature rows): row i replaces placeholder i; the rest keep their table row.
# More rows than placeholders means the encoder and the prompt disagree, which must still fail.
SCATTER_CASES = [
    pytest.param(3, 3, id="one-row-each"),
    pytest.param(1, 0, id="sampled-placeholder"),
    pytest.param(3, 1, id="fewer-rows"),
    pytest.param(2, 4, id="more-rows"),
]


def check_scatter(output_dir: pathlib.Path, token_id: int, placeholders: int, rows: int):
    """The fp32 embedding model writes the feature rows in order into the first placeholders."""
    table = initializers(output_dir / "onnx/embeddings.onnx")[TABLE]
    input_ids = np.array([[1, 6, *[token_id] * placeholders, 7]], dtype=np.int64)
    features = np.random.default_rng(0).standard_normal((rows, table.shape[1])).astype(np.float32)
    embeddings = session(output_dir, "embeddings.onnx")
    if rows > placeholders:
        with pytest.raises(Exception, match="ScatterND"):
            embed(embeddings, input_ids, features)
        return

    expected = table[input_ids[0]].copy()
    expected[2 : 2 + rows] = features
    np.testing.assert_array_equal(embed(embeddings, input_ids, features)[0], expected)


def check_genai_continues_after(model: og.Model, inputs: og.NamedTensors, token_id: int):
    """onnxruntime-genai decodes on after a placeholder token that came with no features, as when
    the decoder samples one itself."""
    prompt_length = inputs["input_ids"].as_numpy().shape[-1]
    params = og.GeneratorParams(model)
    params.set_search_options(do_sample=False, max_length=prompt_length + 4)
    generator = og.Generator(model, params)
    generator.set_inputs(inputs)
    generator.generate_next_token()
    generator.append_tokens(np.array([token_id], dtype=np.int32))
    generator.generate_next_token()
    sequence = generator.get_sequence(0)
    assert len(sequence) == prompt_length + 3
    assert sequence[-2] == token_id


# === VL ===


@pytest.fixture(scope="module")
def vl(tmp_path_factory):
    """(PyTorch model, processor, export dir) with every precision derived."""
    root = tmp_path_factory.mktemp("vl")
    torch.manual_seed(0)
    config = Lfm2VlConfig(
        text_config={
            "vocab_size": 65536,
            "hidden_size": 64,
            "num_hidden_layers": 3,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "layer_types": ["conv", "full_attention", "conv"],
            "intermediate_size": 128,
            "max_position_embeddings": 8192,
        },
        vision_config={
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "patch_size": 16,
            "num_patches": 256,
        },
        projector_hidden_size=64,
    )
    model = Lfm2VlForConditionalGeneration(config).eval()
    model.save_pretrained(root / "checkpoint")
    AutoProcessor.from_pretrained("LiquidAI/LFM2.5-VL-1.6B").save_pretrained(root / "checkpoint")

    output_dir = root / "export"
    checkpoint = str(root / "checkpoint")
    vl_export.export_vl_model(checkpoint, output_dir)
    for precision in vl_export.PRECISIONS:
        vl_export.derive_precision_files(checkpoint, output_dir, precision)
    vl_export.write_genai_config(output_dir, "q4")
    processor = AutoProcessor.from_pretrained(output_dir)
    return model, processor, output_dir


def vl_prompt(processor, images: tuple[pathlib.Path, ...]) -> str:
    content = [*({"type": "image"} for _ in images), {"type": "text", "text": "Hi"}]
    messages = [{"role": "user", "content": content}]
    return processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def vl_inputs(
    processor, images: tuple[pathlib.Path, ...] = (IMAGE,), image_splitting: bool = False
) -> dict:
    """The Hugging Face processor's inputs; resized once by default, as genai's processor does."""
    pil = [Image.open(path).convert("RGB") for path in images]
    text = vl_prompt(processor, images)
    return processor(text=text, images=pil, return_tensors="pt", do_image_splitting=image_splitting)


def vl_features(output_dir: pathlib.Path, precision: str, inputs: dict) -> np.ndarray:
    vision = session(output_dir, vl_export.bundle(precision)["vision"])
    return vision.run(
        None,
        {
            "pixel_values": inputs["pixel_values"].numpy().astype(np.float32),
            "pixel_attention_mask": inputs["pixel_attention_mask"].numpy().astype(np.int64),
            "spatial_shapes": inputs["spatial_shapes"].numpy().astype(np.int64),
        },
    )[0]


def vl_embeds(output_dir: pathlib.Path, precision: str, inputs: dict) -> np.ndarray:
    features = vl_features(output_dir, precision, inputs)
    embeddings = session(output_dir, vl_export.bundle(precision)["embedding"])
    return embed(embeddings, inputs["input_ids"].numpy(), features)


@pytest.mark.parametrize("precision", PRECISIONS)
def test_vl_logits_match_pytorch(vl, precision: str):
    model, processor, output_dir = vl
    inputs = vl_inputs(processor)
    with torch.no_grad():
        expected = model(**inputs).logits[0].numpy()

    decoder = session(output_dir, vl_export.bundle(precision)["decoder"])
    feed = decoder_inputs(vl_embeds(output_dir, precision, inputs), initialize_cache(decoder), 0)
    actual = decoder.run(None, feed)[0][0].astype(np.float32)

    assert cosine(expected, actual) >= MIN_COSINE[precision]
    if precision == "fp32":
        np.testing.assert_allclose(actual, expected, atol=1e-4)


# Patch grids (h, w) of the images genai sends to the vision encoder in one call.
VISION_BATCHES = {
    "single": [(26, 36)],
    "mixed": [(18, 52), (44, 22)],  # neither image has the batch's max grid, 44x52
    "nested": [(26, 36), (14, 20)],  # the smaller image has an axis under 16 patches
    "strip": [(12, 80)],  # an axis under 16 patches downsamples the 16x16 position grid
}


def vision_batch(shapes: list[tuple[int, int]], num_patches: int = 1024) -> dict:
    """Random patches, padded to num_patches as genai pads them."""
    rng = np.random.default_rng(0)
    pixel_values = np.zeros((len(shapes), num_patches, 3 * 16 * 16), dtype=np.float32)
    mask = np.zeros((len(shapes), num_patches), dtype=np.int64)
    for i, (h, w) in enumerate(shapes):
        pixel_values[i, : h * w] = rng.uniform(-1, 1, (h * w, pixel_values.shape[-1]))
        mask[i, : h * w] = 1
    return {
        "pixel_values": pixel_values,
        "pixel_attention_mask": mask,
        "spatial_shapes": np.array(shapes, dtype=np.int64),
    }


@pytest.mark.parametrize("precision", ["fp32", "fp16"])
@pytest.mark.parametrize("shapes", list(VISION_BATCHES.values()), ids=list(VISION_BATCHES))
def test_vl_vision_matches_pytorch_per_image(vl, shapes: list, precision: str):
    """Every image gets the position grid resized to its own shape, antialiased as in Siglip2."""
    model, _, output_dir = vl
    feed = vision_batch(shapes)
    with torch.no_grad():
        features = model.model.get_image_features(
            **{name: torch.from_numpy(value) for name, value in feed.items()}
        ).pooler_output
    expected = torch.cat(features).numpy()

    actual = session(output_dir, vl_export.bundle(precision)["vision"]).run(None, feed)[0]

    assert actual.shape == expected.shape
    token_cosines = (expected * actual).sum(-1) / (
        np.linalg.norm(expected, axis=-1) * np.linalg.norm(actual, axis=-1)
    )
    assert token_cosines.min() >= MIN_COSINE[precision]
    if precision == "fp32":
        np.testing.assert_allclose(actual, expected, atol=1e-5)


@pytest.mark.parametrize("precision", ["fp32", "q4"])
def test_vl_cached_decode_matches_prefill(vl, precision: str):
    """Prefill the image prompt, then feed three text tokens one call at a time."""
    _, processor, output_dir = vl
    inputs = vl_inputs(processor)
    files = vl_export.bundle(precision)
    decoder = session(output_dir, files["decoder"])
    embeddings = session(output_dir, files["embedding"])

    prompt = vl_embeds(output_dir, precision, inputs)
    extra = np.array([[5, 77, 300]], dtype=np.int64)
    full = np.concatenate([prompt, embed(embeddings, extra)], axis=1)
    expected = decoder.run(None, decoder_inputs(full, initialize_cache(decoder), 0))[0]

    cache, outputs, n = initialize_cache(decoder), decoder.get_outputs(), prompt.shape[1]
    update_cache(cache, decoder.run(None, decoder_inputs(prompt, cache, 0)), outputs)
    for i in range(extra.shape[1]):
        result = decoder.run(
            None, decoder_inputs(embed(embeddings, extra[:, i : i + 1]), cache, n + i)
        )
        update_cache(cache, result, outputs)
        np.testing.assert_allclose(result[0][0, -1], expected[0, n + i], atol=5e-3)


@pytest.mark.parametrize("precision", ["q4", "q8"])
def test_vl_decoder_layout(vl, precision: str):
    """Every MatMul quantized, the LM head to int8; the embedding model holds the token table."""
    _, _, output_dir = vl
    path = output_dir / "onnx" / f"decoder_{precision}.onnx"
    graph = onnx.load(str(path), load_external_data=False).graph
    lm_head = next(node for node in graph.node if "logits" in node.output)
    assert lm_head.op_type == "MatMulNBits"
    assert attributes(lm_head)["bits"] == 8
    assert matmul_bits(graph)[int(precision[1])] > 1
    assert all(node.op_type not in ("MatMul", "GatherBlockQuantized") for node in graph.node)


def test_vl_genai_config(vl):
    _, _, output_dir = vl
    model = json.loads((output_dir / "genai_config.json").read_text())["model"]
    assert model["type"] == "lfm2_vl"
    assert model["decoder"]["filename"] == "onnx/decoder_q4.onnx"
    assert model["decoder"]["session_options"]["session.set_denormal_as_zero"] == "1"
    assert model["embedding"]["filename"] == "onnx/embeddings_q8.onnx"
    assert model["vision"]["filename"] == "onnx/vision_encoder_q8.onnx"
    assert model["vision"]["max_num_patches"] == 1024
    for section in ("decoder", "embedding", "vision"):
        assert (output_dir / model[section]["filename"]).exists()

    processor_config = json.loads((output_dir / model["vision"]["config_filename"]).read_text())
    resize = next(
        t["operation"]["attrs"]
        for t in processor_config["processor"]["transforms"]
        if t["operation"]["type"] == "Resize"
    )
    assert resize["interpolation"] == "LINEAR"  # LFM2.5-VL-1.6B resamples bilinearly
    assert (resize["min_pixels"], resize["max_pixels"]) == (64 * 32**2, 256 * 32**2)


def test_vl_int8_embeddings(vl):
    check_int8_embeddings(vl[2])


@pytest.mark.parametrize("placeholders,rows", SCATTER_CASES)
def test_vl_embeddings_scatter_features(vl, placeholders: int, rows: int):
    check_scatter(vl[2], vl[0].config.image_token_id, placeholders, rows)


def test_vl_int8_vision_encoder(vl):
    """q4 and q8 load one vision encoder: every weight MatMul int8, symmetric, in blocks of 128,
    with fp32 activations (accuracy_level 0), and each image token close to the fp32 encoder's."""
    _, processor, output_dir = vl
    assert vl_export.bundle("q4")["vision"] == vl_export.bundle("q8")["vision"]
    assert not (output_dir / "onnx/vision_encoder_q4.onnx").exists()

    def graph(precision):
        path = output_dir / "onnx" / vl_export.bundle(precision)["vision"]
        return onnx.load(str(path), load_external_data=False).graph

    fp32, int8 = graph("fp32"), graph("q4")
    weights = {init.name for init in fp32.initializer}
    weight_matmuls = [n for n in fp32.node if n.op_type == "MatMul" and n.input[1] in weights]
    quantized = [node for node in int8.node if node.op_type == "MatMulNBits"]
    assert len(quantized) == len(weight_matmuls)
    for node in quantized:
        attrs = attributes(node)
        assert (attrs["bits"], attrs["block_size"], attrs.get("accuracy_level", 0)) == (8, 128, 0)
        assert len(node.input) == 3  # no zero points

    inputs = vl_inputs(processor)
    expected = vl_features(output_dir, "fp32", inputs)
    actual = vl_features(output_dir, "q4", inputs)
    result = compare_token_cosine("vision", expected, actual, min_worst=0.999, min_mean=0.9995)
    assert result.passed, result.details


def vl_genai_inputs(model: og.Model, processor) -> og.NamedTensors:
    prompt = vl_prompt(processor, (IMAGE,))
    return model.create_multimodal_processor()(prompt, images=og.Images.open(str(IMAGE)))


def genai_answer(model, inputs: og.NamedTensors, max_new_tokens: int = 4):
    """The greedy sequence and the logits of the prompt's last position."""
    first = []

    def step(generator):
        if not first:
            first.append(np.asarray(generator.get_output("logits"))[0, -1])

    generator = generate(model, inputs, max_new_tokens, step)
    return generator.get_sequence(0).tolist(), first[0]


@pytest.mark.parametrize("precision", ["fp32", "q8", "q4"])
def test_vl_genai_runtime(vl, precision: str):
    """The lfm2_vl pipeline with genai's own processor, against the same files in plain
    onnxruntime."""
    _, processor, output_dir = vl
    model = load_model(output_dir, vl_export.genai_files(precision))

    inputs = vl_genai_inputs(model, processor)
    genai_ids = inputs["input_ids"].as_numpy()[0]
    reference_inputs = vl_inputs(processor)
    assert genai_ids.tolist() == reference_inputs["input_ids"][0].tolist()

    sequence, logits = genai_answer(model, inputs)
    assert len(sequence) > len(genai_ids)

    # genai resizes the image with onnxruntime-extensions, so its pixels differ slightly.
    decoder = session(output_dir, vl_export.bundle(precision)["decoder"])
    embeds = vl_embeds(output_dir, precision, reference_inputs)
    expected = decoder.run(None, decoder_inputs(embeds, initialize_cache(decoder), 0))[0]
    assert cosine(expected[0, -1], logits) >= 0.999


@pytest.mark.parametrize("count", [0, 1], ids=["text", "image"])
def test_vl_checkpoint_processor_matches_genai(vl, tmp_path, count: int):
    """lfm2-vl-infer's inputs without image splitting against genai's processor, which never
    tiles: the same names, dtypes, shapes and values, pixels within 2/255 (std 0.5), and the same
    greedy answer; with genai's pixels, the same logits."""
    _, processor, output_dir = vl
    png = tmp_path / "cardinal.png"
    Image.open(IMAGE).save(png)  # decoded once: the two JPEG decoders differ by a few steps
    paths = [str(png)] * count
    model = load_model(output_dir, vl_export.genai_files("fp32"))
    prompt = vl_prompt(processor, (png,) * count)
    genai_inputs = model.create_multimodal_processor()(
        prompt, images=og.Images.open(*paths) if paths else None
    )
    checkpoint_processor = vl_infer.CheckpointProcessor(output_dir, image_splitting=False)
    inputs = checkpoint_processor(prompt, paths)

    names = {"image_attention_mask": "pixel_attention_mask", "image_sizes": "spatial_shapes"}
    expected = {names.get(n, n): genai_inputs[n].as_numpy() for n in genai_inputs.keys()}
    actual = {name: inputs[name].as_numpy() for name in inputs.keys()}
    assert sorted(actual) == sorted(expected)
    for name, value in expected.items():
        assert (actual[name].dtype, actual[name].shape) == (value.dtype, value.shape), name
        if name == "pixel_values":
            np.testing.assert_allclose(actual[name], value, atol=2 / 255 / 0.5)
            assert np.abs(actual[name] - value).mean() < 1e-4
        else:
            np.testing.assert_array_equal(actual[name], value, err_msg=name)

    sequence, logits = genai_answer(model, genai_inputs, 8)
    assert genai_answer(model, inputs, 8)[0] == sequence

    inputs = checkpoint_processor(prompt, paths)
    if paths:
        inputs["pixel_values"] = expected["pixel_values"]
    np.testing.assert_array_equal(genai_answer(model, inputs)[1], logits)


@pytest.mark.parametrize("precision", ["fp32", "q4"])
def test_vl_checkpoint_processor_runs_mixed_aspect_images(vl, precision: str):
    """A wide and a tall image, which genai's own processor rejects, against the same files in
    plain onnxruntime given the Hugging Face features; at fp32 also against PyTorch."""
    model, processor, output_dir = vl
    genai_model = load_model(output_dir, vl_export.genai_files(precision))
    inputs = vl_infer.CheckpointProcessor(output_dir, image_splitting=False)(
        vl_prompt(processor, MIXED), [str(path) for path in MIXED]
    )
    assert inputs["spatial_shapes"].as_numpy().tolist() == [[18, 52], [44, 22]]
    sequence, logits = genai_answer(genai_model, inputs)

    reference_inputs = vl_inputs(processor, MIXED)
    prompt_ids = reference_inputs["input_ids"][0].tolist()
    assert sequence[: len(prompt_ids)] == prompt_ids
    decoder = session(output_dir, vl_export.bundle(precision)["decoder"])
    embeds = vl_embeds(output_dir, precision, reference_inputs)
    expected = decoder.run(None, decoder_inputs(embeds, initialize_cache(decoder), 0))[0]
    np.testing.assert_allclose(logits, expected[0, -1], atol=1e-5)
    if precision == "fp32":
        with torch.no_grad():
            pytorch = model(**reference_inputs).logits[0, -1].numpy()
        np.testing.assert_allclose(logits, pytorch, atol=1e-4)


@pytest.mark.parametrize("precision", ["fp32", "q4"])
def test_vl_genai_continues_after_an_image_token(vl, precision: str):
    model, processor, output_dir = vl
    genai_model = load_model(output_dir, vl_export.genai_files(precision))
    inputs = vl_genai_inputs(genai_model, processor)
    check_genai_continues_after(genai_model, inputs, model.config.image_token_id)


@pytest.mark.parametrize("precision", ["fp32", "q8", "q4"])
def test_vl_checkpoint_processor_splits_large_images(vl, precision: str):
    """With the checkpoint's do_image_splitting, an image over the splitting threshold reaches
    genai as tiles plus a thumbnail, next to an image under it: the Hugging Face processor's prompt
    with its tile tokens and its pixels, an encoder feature per image token (equal to PyTorch's at
    fp32), and the greedy answer of the same files in plain onnxruntime."""
    model, processor, output_dir = vl
    assert processor.image_processor.do_image_splitting
    inputs = vl_infer.CheckpointProcessor(output_dir)(
        vl_prompt(processor, TILED), [str(path) for path in TILED]
    )
    reference_inputs = vl_inputs(processor, TILED, image_splitting=True)
    for name in ("input_ids", *compare_vl.PIXEL_INPUTS):
        np.testing.assert_array_equal(
            inputs[name].as_numpy(), reference_inputs[name].numpy(), err_msg=name
        )
    assert inputs["spatial_shapes"].as_numpy().tolist() == TILED_SHAPES
    vision = json.loads((output_dir / "genai_config.json").read_text())["model"]["vision"]
    max_num_patches = vision["max_num_patches"]  # a 512x512 tile is 32x32 patches
    assert inputs["pixel_values"].as_numpy().shape[:2] == (len(TILED_SHAPES), max_num_patches)
    prompt = reference_inputs["input_ids"][0].numpy()
    text = processor.tokenizer.decode(prompt)
    assert (text.count("<|img_row_"), text.count("<|img_thumbnail|>")) == (6, 1)
    assert "<|img_row_2_col_3|>" in text

    features = vl_features(output_dir, precision, reference_inputs)
    image_tokens = 6 * 256 + 13 * 18 + 22 * 11
    assert inputs["num_image_tokens"].as_numpy().tolist() == [image_tokens]
    assert np.count_nonzero(prompt == processor.image_token_id) == image_tokens
    assert features.shape[0] == image_tokens
    if precision == "fp32":
        pixels = {name: reference_inputs[name] for name in compare_vl.PIXEL_INPUTS}
        with torch.no_grad():
            expected = torch.cat(model.model.get_image_features(**pixels).pooler_output)
        np.testing.assert_allclose(features, expected.numpy(), atol=1e-5)

    sequence, logits = genai_answer(
        load_model(output_dir, vl_export.genai_files(precision)), inputs, 8
    )

    files = vl_export.bundle(precision)
    decoder = session(output_dir, files["decoder"])
    embeddings = session(output_dir, files["embedding"])
    embeds = embed(embeddings, prompt[None], features)

    def lookup(ids):
        return embeds if ids.shape[1] == len(prompt) else embed(embeddings, ids)

    answer = greedy(decoder, prompt, 8, eos_ids(output_dir), lookup)
    assert sequence == [*prompt.tolist(), *answer]
    expected = decoder.run(None, decoder_inputs(embeds, initialize_cache(decoder), 0))[0]
    np.testing.assert_allclose(logits, expected[0, -1], atol=1e-5)
    if precision == "fp32":
        with torch.no_grad():
            pytorch = model(**reference_inputs).logits[0, -1].numpy()
        np.testing.assert_allclose(logits, pytorch, atol=1e-4)


@pytest.mark.parametrize(
    "image_splitting,sent_per_turn",
    [(None, (7, 6 * 256 + 234)), (False, (1, 234))],
    ids=["checkpoint", "resize-once"],
)
def test_vl_chat_keeps_images(vl, monkeypatch, image_splitting, sent_per_turn):
    """lfm2-vl-infer re-sends an image with every turn after the one it came with, as tiles plus a
    thumbnail unless image splitting is off."""
    chat = vl_infer.VLChat(vl[2], image_splitting=image_splitting)
    sent = []

    def record(model, inputs, *args):
        sent.append(
            (inputs["pixel_values"].as_numpy().shape[0], inputs["num_image_tokens"].as_numpy()[0])
        )
        return generate(model, inputs, *args)

    monkeypatch.setattr(vl_infer, "generate", record)
    chat.attach([str(IMAGE)])
    chat.send("Hi", 4, stream=False)
    chat.send("And now?", 4, stream=False)

    assert sent == [sent_per_turn] * 2
    assert chat.images() == [str(IMAGE)]


@pytest.mark.parametrize(
    "image_splitting,crops", [(None, 7), (False, 1)], ids=["checkpoint", "resize-once"]
)
def test_vl_compare_scores_every_case(vl, image_splitting, crops: int):
    """lfm2-compare vl on the tiny export: every case through plain onnxruntime and
    onnxruntime-genai, the mixed-aspect one included, with the reference's image splitting."""
    _, _, output_dir = vl
    refs = compare_vl.reference(output_dir.parent / "checkpoint", 4, "cpu", image_splitting)
    assert [path.name for path in MIXED] in [ref["images"] for ref in refs]
    assert {ref["image_splitting"] for ref in refs} == {crops > 1}
    cardinal = next(ref for ref in refs if ref["images"] == [IMAGE.name])
    assert len(cardinal["pixel_inputs"]["spatial_shapes"]) == crops
    row = compare_vl.score(output_dir, "fp32", refs, True, "cpu")
    assert not has_errors(row), row
    assert row["genai"]["prompt_ids_equal"] == f"{len(refs)}/{len(refs)}"
    assert row["genai"]["greedy"] == row["decoder"]["greedy"]
    assert row["vision"]["cosine_min"] >= MIN_COSINE["fp32"]


def pre_tokenize(pattern: str, text: str) -> list[str]:
    split = pre_tokenizers.Split(Regex(pattern), "isolated")
    return [piece for piece, _ in split.pre_tokenize_str(text)]


@pytest.mark.parametrize(
    "text",
    [
        "<|im_start|>user\nIt's 2026; THEY'LL say we'd 12345 things.<|im_end|>\n",
        "one\n\n  two\r\n\tthree   ",
        "  (a)b! 'x' 'Ve ll 3.14",
    ],
)
def test_vl_tokenizer_pattern_swap_keeps_splits(text: str):
    expected = pre_tokenize(vl_export.UNSUPPORTED_PATTERN, text)
    assert pre_tokenize(vl_export.SUPPORTED_PATTERN, text) == expected


def test_vl_fix_tokenizer_pattern(tmp_path):
    path = tmp_path / "tokenizer.json"
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Split(Regex(vl_export.UNSUPPORTED_PATTERN), "isolated")
    tokenizer.save(str(path))

    vl_export.fix_tokenizer_pattern(path)
    pattern = json.loads(path.read_text())["pre_tokenizer"]["pattern"]["Regex"]
    assert pattern == vl_export.SUPPORTED_PATTERN


# === Audio ===


@pytest.fixture(scope="module")
def audio(tmp_path_factory):
    """Synthetic export with every precision derived; genai_config.json at q4."""
    root = tmp_path_factory.mktemp("audio")
    output_dir = build_model_dir(root)
    for precision in audio_export.PRECISIONS:
        audio_export.derive_precision_files(
            str(root / "checkpoint"), output_dir, precision, block_size=32
        )
    audio_export.write_genai_config(output_dir, "q4")
    return output_dir


def test_audio_genai_config(audio):
    model = json.loads((audio / "genai_config.json").read_text())["model"]
    assert model["type"] == "lfm2_audio"
    assert model["audio_token_id"] == audio_export.AUDIO_TOKEN_ID
    assert not set(model["eos_token_id"]) & set(audio_export.MODALITY_SWITCH_TOKEN_IDS)
    assert model["decoder"]["session_options"]["session.set_denormal_as_zero"] == "1"
    assert model["audio_output"]["interleaved_n_text"] == 6
    assert model["audio_output"]["interleaved_n_audio"] == 9
    files = [
        model["decoder"]["filename"],
        model["embedding"]["filename"],
        model["speech"]["filename"],
        model["audio_output"]["depthformer"]["filename"],
        model["audio_output"]["embedding"]["filename"],
    ]
    assert files == [
        "onnx/decoder_q4.onnx",
        "onnx/embeddings_q8.onnx",
        "onnx/audio_encoder_q4.onnx",
        "onnx/vocoder_depthformer_q4.onnx",
        "onnx/audio_embedding_q4.onnx",
    ]
    for filename in files:
        assert (audio / filename).exists()


def matmuls(path: pathlib.Path) -> dict:
    """(op type, attributes, weight tensor bytes) of each MatMul(NBits), by output name."""
    graph = onnx.load(str(path)).graph
    tensors = {init.name: onnx.numpy_helper.to_array(init).tobytes() for init in graph.initializer}
    return {
        node.output[0]: (node.op_type, attributes(node), [tensors.get(i) for i in node.input[1:]])
        for node in graph.node
        if node.op_type in ("MatMul", "MatMulNBits")
    }


@pytest.mark.parametrize("precision,fp32_head", [("q4", Q4F32), ("q8", Q8_FP32_HEAD)])
def test_audio_decoder_adds_an_int8_lm_head(audio, tmp_path, precision: str, fp32_head):
    """The decoder the fp32-head preset builds, but with an int8 LM head."""
    checkpoint = str(audio.parent / "checkpoint")
    options = audio_export.DECODER_OPTIONS
    reference = export_decoder(
        checkpoint, tmp_path, "decoder.onnx", options, fp32_head, block_size=32
    )

    actual = matmuls(audio / "onnx" / audio_export.bundle(precision)["decoder"])
    expected = matmuls(reference)
    op_type, attrs, _ = actual.pop("logits")
    assert (op_type, attrs["bits"]) == ("MatMulNBits", 8)
    assert expected.pop("logits")[0] == "MatMul"
    assert actual == expected
    assert {attrs["bits"] for _, attrs, _ in actual.values()} == {int(precision[1])}


def test_audio_genai_config_checks_token_ids(audio, tmp_path):
    tokenizer = json.loads((audio / "tokenizer.json").read_text())
    for token in tokenizer["added_tokens"]:
        if token["content"] == "<|reserved_123|>":
            token["id"] = 134
    (tmp_path / "tokenizer.json").write_text(json.dumps(tokenizer))
    with pytest.raises(ValueError, match="reserved_123"):
        audio_export.write_genai_config(tmp_path, "q4")


def test_audio_genai_config_checks_codebooks(audio, tmp_path):
    shutil.copy(audio / "tokenizer.json", tmp_path)
    config = json.loads((audio / "config.json").read_text())
    (tmp_path / "config.json").write_text(json.dumps({**config, "codebooks": 4}))
    with pytest.raises(ValueError, match="4 codebooks"):
        audio_export.write_genai_config(tmp_path, "q4")


def test_audio_embedding_checks_codebook_size(tmp_path):
    weights = {"audio_embedding.embedding.weight": np.zeros((8 * 2048, 4), dtype=np.float32)}
    with pytest.raises(ValueError, match="16384 rows"):
        audio_export.export_audio_embedding_binary(weights, {}, tmp_path)


# What the LFM2.5-Audio Hub repos hold besides the files the export reads
UNUSED_HUB_FILES = (
    ".gitattributes",
    "LICENSE",
    "README.md",
    "demo.mp4",
    "tokenizer-e351c8d8-checkpoint125.safetensors",
)


def test_audio_exports_a_checkpoint_folder_without_liquid_audio(tmp_path, monkeypatch):
    """lfm2-audio-export of a local checkpoint folder, and of the same checkpoint as a Hub ID,
    with liquid-audio not installed: the same ONNX files, and no download of unused files."""
    checkpoint = write_full_checkpoint(tmp_path / "LFM2.5-Audio-synthetic")
    for name in UNUSED_HUB_FILES:
        (checkpoint / name).write_bytes(b"unused")
    hub_id, snapshot = "LiquidAI/LFM2.5-Audio-synthetic", tmp_path / "snapshot"
    downloaded = []

    def snapshot_download(repo_id: str, allow_patterns: list[str] | None = None) -> str:
        assert repo_id == hub_id
        files = [p.relative_to(checkpoint).as_posix() for p in checkpoint.rglob("*") if p.is_file()]
        for name in filter_repo_objects(files, allow_patterns=allow_patterns):
            (snapshot / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(checkpoint / name, snapshot / name)
            downloaded.append(name)
        return str(snapshot)

    def hf_hub_download(*args, **kwargs):
        raise AssertionError(f"hf_hub_download{args}")

    for name in [m for m in sys.modules if m.split(".")[0] == "liquid_audio"] + ["liquid_audio"]:
        monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", hf_hub_download)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")  # for the genai builder's process
    monkeypatch.setattr(depthformer_builder, "DepthformerUnifiedBuilder", SmallDepthformer)

    exports = {}
    for model, base in ((str(checkpoint), tmp_path / "local"), (hub_id, tmp_path / "hub")):
        argv = ["lfm2-audio-export", model, "--output-dir", str(base), "--precision", "q8"]
        monkeypatch.setattr(sys, "argv", argv)
        audio_export.main()
        onnx_dir = base / "exports" / "LFM2.5-Audio-synthetic-ONNX" / "onnx"
        exports[model] = {p.name: p.read_bytes() for p in sorted(onnx_dir.iterdir())}

    local, hub = exports.values()
    assert {*audio_export.bundle("fp32").values(), *audio_export.bundle("q8").values()} <= {*local}
    assert local.keys() == hub.keys()
    assert [name for name in local if local[name] != hub[name]] == []
    assert downloaded and not set(downloaded) & set(UNUSED_HUB_FILES)


@pytest.mark.parametrize("placeholders,rows", SCATTER_CASES)
def test_audio_embeddings_scatter_features(audio, placeholders: int, rows: int):
    check_scatter(audio, audio_export.AUDIO_TOKEN_ID, placeholders, rows)


@pytest.mark.parametrize("precision", ["fp32", "q4"])
def test_audio_genai_continues_after_an_audio_token(audio, tmp_path, precision: str):
    chat = AudioChat(audio, precision)
    turns, audios = single_turn(None, write_tone(tmp_path / "tone.wav"))
    inputs = chat.processor(chat_prompt(None, turns), audios=og.Audios.open(*audios))
    check_genai_continues_after(chat.model, inputs, audio_export.AUDIO_TOKEN_ID)


def test_audio_int8_embeddings(audio):
    check_int8_embeddings(audio)


def graph_nodes(graph: onnx.GraphProto) -> list[onnx.NodeProto]:
    """The nodes of graph and of its subgraphs."""
    nodes = []
    for node in graph.node:
        nodes.append(node)
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                nodes += graph_nodes(attr.g)
    return nodes


@pytest.mark.parametrize("precision", ["q8", "q4"])
def test_audio_output_graphs_are_quantized(audio, precision: str):
    """olive-recipes #639's choices: every depthformer weight matrix and the audio embedding table
    symmetric int4/int8, the depthformer's per-codebook tables fp32 under the node names #639
    excludes."""
    bits = int(precision[1])
    accuracy_level = audio_export.AUDIO_OUTPUT_QUANT[precision]["accuracy_level"]
    files = audio_export.bundle(precision)

    graph = onnx.load(str(audio / "onnx" / files["depthformer"])).graph
    dtypes = {init.name: init.data_type for init in graph.initializer}
    nodes = graph_nodes(graph)
    quantized = [
        (attributes(node)["bits"], attributes(node).get("accuracy_level", 0), len(node.input))
        for node in nodes
        if node.op_type == "MatMulNBits"
    ]
    # depth_linear (in the step-0 branch), then qkv, out, w1, w2 and w3 in each of 6 layers
    assert quantized == [(bits, accuracy_level, 3)] * (1 + 6 * 5)
    assert not [node.name for node in nodes if node.op_type == "MatMul" and node.input[1] in dtypes]
    tables = {
        node.name: node.input[0]
        for node in nodes
        if node.op_type == "Gather" and node.input[0] in dtypes
    }
    assert tables == {
        "/prev_embed/table": "stacked_embed_weights",
        "/logits/norm_w": "stacked_logits_norm_weights",
        "/logits/logits_w": "stacked_logits_weights",
    }
    assert {dtypes[table] for table in tables.values()} == {onnx.TensorProto.FLOAT}

    embedding = onnx.load(str(audio / "onnx" / files["audio_embedding"])).graph
    assert [
        (node.op_type, attributes(node)["bits"], len(node.input)) for node in embedding.node
    ] == [("GatherBlockQuantized", bits, 3)]


@pytest.mark.parametrize("precision", ["q8", "q4"])
def test_audio_embedding_precisions_follow_fp32(audio, precision: str):
    """Each looked-up value within half a quantization step (blocks of 32 along H) of the fp32
    table, or a whole one next to the block's largest magnitude: that maps to -2^(bits-1), so
    values opposite it clip at 2^(bits-1) - 1."""
    table = initializers(audio / "onnx/audio_embedding.onnx")["audio_embedding.weight"]
    embedding = session(audio, audio_export.bundle(precision)["audio_embedding"])
    actual = embedding.run(None, {"audio_codes": np.arange(len(table))[None]})[0][0]

    half = 2 ** (int(precision[1]) - 1)
    blocks = np.abs(table).reshape(len(table), -1, 32)
    step = np.repeat(blocks.max(-1) / half, 32, axis=1)
    near_peak = np.abs(table) > (half - 0.5) * step
    assert (np.abs(actual - table) <= np.where(near_peak, step, step / 2) + 1e-7).all()


def depthformer_frame(
    depthformer, hidden: np.ndarray, codes: list[int] | None = None
) -> tuple[list[int], np.ndarray]:
    """One frame's codes and the logits of its 8 steps; each step reads the previous code of
    codes, or the previous step's greedy one."""
    inputs = {i.name: i.shape for i in depthformer.get_inputs()}
    layers, _, kv_heads, _, head_dim = inputs["past_keys"]
    cache = np.zeros((layers, 1, kv_heads, 0, head_dim), np.float32)
    feed = {
        "hidden_states": hidden,
        "depth_slices_in": np.zeros((1, *inputs["depth_slices_in"][1:]), np.float32),
        "past_keys": cache,
        "past_values": cache,
    }
    chosen, logits = [], []
    for step in range(audio_export.NUM_CODEBOOKS):
        previous = (codes or chosen)[step - 1] if step else 0
        feed |= {
            "step_idx": np.array(step, np.int64),
            "prev_token": np.array([previous], np.int64),
            "seqlens_k": np.array([step], np.int32),
            "total_seq_len": np.array(step + 1, np.int32),
        }
        step_logits, slices, keys, values = depthformer.run(None, feed)
        feed |= {"depth_slices_in": slices, "past_keys": keys, "past_values": values}
        chosen.append(int(step_logits[0].argmax()))
        logits.append(step_logits[0])
    return chosen, np.stack(logits)


@pytest.mark.parametrize("precision", ["q8", "q4"])
def test_audio_depthformer_precisions_follow_fp32(audio, precision: str):
    """A frame's logits against the fp32 depthformer, both reading the fp32 one's greedy codes."""
    hidden = np.random.default_rng(2).standard_normal((1, HIDDEN)).astype(np.float32)
    codes, logits = depthformer_frame(session(audio, "vocoder_depthformer.onnx"), hidden)
    depthformer = session(audio, audio_export.bundle(precision)["depthformer"])
    _, precision_logits = depthformer_frame(depthformer, hidden, codes)
    for step in range(audio_export.NUM_CODEBOOKS):
        assert cosine(logits[step], precision_logits[step]) >= MIN_COSINE[precision]


def test_audio_fp32_embeddings_keep_a_plain_gather(audio):
    """olive-recipes' audio export renames this graph's first Gather and quantizes its table."""
    graph = onnx.load(str(audio / "onnx/embeddings.onnx"), load_external_data=False).graph
    assert next(node for node in graph.node if node.op_type == "Gather").input[0] == TABLE
    assert all(node.domain == "" for node in graph.node)


@pytest.mark.parametrize("precision", ["fp16", "q8", "q4"])
def test_audio_decoder_precisions_follow_fp32(audio, precision: str):
    """Logits and the hidden states the depthformer reads, against the fp32 decoder."""
    embeds = np.random.default_rng(1).standard_normal((1, 6, HIDDEN)).astype(np.float32)

    def run(filename):
        decoder = session(audio, filename)
        feed = decoder_inputs(embeds, initialize_cache(decoder), 0)
        names = [o.name for o in decoder.get_outputs()]
        result = dict(zip(names, decoder.run(None, feed), strict=True))
        return result["logits"].astype(np.float32), result["hidden_states"].astype(np.float32)

    logits, hidden = run("decoder.onnx")
    precision_logits, precision_hidden = run(audio_export.bundle(precision)["decoder"])
    assert cosine(logits, precision_logits) >= MIN_COSINE[precision]
    assert cosine(hidden, precision_hidden) >= MIN_COSINE[precision]


@pytest.mark.parametrize("precision", ["q8", "q4"])
def test_audio_genai_runtime(audio, precision: str):
    """The lfm2_audio pipeline, loaded as the CLI does, against the same files in plain onnxruntime."""
    chat = AudioChat(audio, precision)
    inputs = chat.processor(chat_prompt(None, [("user", "hello")]))
    first = []

    def step(generator):
        if not first:
            first.append(np.asarray(generator.get_output("logits"))[0, -1])

    generate(chat.model, inputs, 4, step)

    files = audio_export.bundle(precision)
    embeds = embed(session(audio, files["embedding"]), inputs["input_ids"].as_numpy())
    decoder = session(audio, files["decoder"])
    expected = decoder.run(["logits"], decoder_inputs(embeds, initialize_cache(decoder), 0))[0]
    np.testing.assert_allclose(first[0], expected[0, -1], atol=1e-5)


@pytest.mark.parametrize("precision", ["q8", "q4"])
def test_audio_genai_speech_output(audio, precision: str):
    """onnxruntime-genai runs the quantized depthformer and audio embedding: an interleaved answer
    turns to speech after 6 text tokens."""
    chat = AudioChat(audio, precision)
    answer = chat.answer("interleaved", *single_turn("hello"), max_new_tokens=6 + 12, random_seed=1)
    assert len(answer.codes) == 9
    assert answer.codes.min() >= 0 and answer.codes.max() <= 2048
