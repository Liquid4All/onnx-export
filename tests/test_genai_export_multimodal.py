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

import numpy as np
import onnx
import onnxruntime_genai as og
import pytest
import torch
from helpers import require_genai
from PIL import Image
from test_lfm2_audio.synthetic import HIDDEN, build_model_dir
from tokenizers import Regex, Tokenizer, models, pre_tokenizers
from transformers import AutoProcessor, Lfm2VlConfig, Lfm2VlForConditionalGeneration

from liquidonnx.embeddings import embed
from liquidonnx.genai_runtime import generate, load_model
from liquidonnx.lfm2_audio import export as audio_export
from liquidonnx.lfm2_vl import export as vl_export
from liquidonnx.lfm2_vl.infer import VLChat
from liquidonnx.session import decoder_inputs, initialize_cache, load_onnx_session, update_cache

IMAGE = pathlib.Path(__file__).parent / "test_lfm2_vl/assets/cardinal.jpg"
PRECISIONS = ["fp32", "fp16", "q8", "q4"]
MIN_COSINE = {"fp32": 0.99999, "fp16": 0.9999, "q8": 0.999, "q4": 0.97}
# int4 on the tiny 32-wide vision tower (one block per row) dominates the VL q4 error.
VL_MIN_COSINE = {**MIN_COSINE, "q4": 0.9}


def session(output_dir: pathlib.Path, filename: str):
    return load_onnx_session(output_dir / "onnx" / filename, ["CPUExecutionProvider"])


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b)))


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
    vl_export.export_vl_model(str(root / "checkpoint"), output_dir)
    for precision in vl_export.PRECISIONS:
        vl_export.derive_precision_files(output_dir / "onnx", precision)
    vl_export.write_genai_config(output_dir, "q4")
    processor = AutoProcessor.from_pretrained(output_dir)
    return model, processor, output_dir


def vl_inputs(processor) -> dict:
    """One image, resized once as onnxruntime-genai does (no tiling)."""
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Hi"}]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    image = Image.open(IMAGE).convert("RGB")
    return processor(text=text, images=[image], return_tensors="pt", do_image_splitting=False)


def vl_embeds(output_dir: pathlib.Path, precision: str, inputs: dict) -> np.ndarray:
    files = vl_export.bundle(precision)
    vision = session(output_dir, files["vision"])
    features = vision.run(
        None,
        {
            "pixel_values": inputs["pixel_values"].numpy().astype(np.float32),
            "pixel_attention_mask": inputs["pixel_attention_mask"].numpy().astype(np.int64),
            "spatial_shapes": inputs["spatial_shapes"].numpy().astype(np.int64),
        },
    )[0]
    return embed(session(output_dir, files["embedding"]), inputs["input_ids"].numpy(), features)


@pytest.mark.parametrize("precision", PRECISIONS)
def test_vl_logits_match_pytorch(vl, precision: str):
    model, processor, output_dir = vl
    inputs = vl_inputs(processor)
    with torch.no_grad():
        expected = model(**inputs).logits[0].numpy()

    decoder = session(output_dir, vl_export.bundle(precision)["decoder"])
    feed = decoder_inputs(vl_embeds(output_dir, precision, inputs), initialize_cache(decoder), 0)
    actual = decoder.run(None, feed)[0][0].astype(np.float32)

    assert cosine(expected, actual) >= VL_MIN_COSINE[precision]
    if precision == "fp32":
        np.testing.assert_allclose(actual, expected, atol=1e-4)


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


def test_vl_genai_config(vl):
    _, _, output_dir = vl
    model = json.loads((output_dir / "genai_config.json").read_text())["model"]
    assert model["type"] == "lfm2_vl"
    assert model["decoder"]["filename"] == "onnx/decoder_q4.onnx"
    assert model["embedding"]["filename"] == "onnx/embeddings_fp16.onnx"
    assert model["vision"]["filename"] == "onnx/vision_encoder_q4.onnx"
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


@pytest.mark.parametrize("precision", ["fp32", "q4"])
def test_vl_genai_runtime(vl, precision: str):
    """The lfm2_vl pipeline, loaded as the CLI does, against the same files in plain onnxruntime."""
    require_genai("lfm2_vl")
    _, processor, output_dir = vl
    model = load_model(output_dir, vl_export.genai_files(precision))

    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Hi"}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = model.create_multimodal_processor()(prompt, images=og.Images.open(str(IMAGE)))
    genai_ids = inputs["input_ids"].as_numpy()[0]
    reference_inputs = vl_inputs(processor)
    assert genai_ids.tolist() == reference_inputs["input_ids"][0].tolist()

    first = []

    def step(generator):
        if not first:
            first.append(np.asarray(generator.get_output("logits"))[0, -1])

    generator = generate(model, inputs, 4, step)
    assert len(generator.get_sequence(0)) > len(genai_ids)

    # genai resizes the image with onnxruntime-extensions, so its pixels differ slightly.
    decoder = session(output_dir, vl_export.bundle(precision)["decoder"])
    embeds = vl_embeds(output_dir, precision, reference_inputs)
    expected = decoder.run(None, decoder_inputs(embeds, initialize_cache(decoder), 0))[0]
    assert cosine(expected[0, -1], first[0]) >= 0.999


def test_vl_chat_keeps_images(vl, monkeypatch):
    """lfm2-vl-infer re-sends an image with every turn after the one it came with."""
    require_genai("lfm2_vl")
    chat = VLChat(vl[2])
    prompts, processor = [], chat.processor

    def record(prompt, images=None):
        prompts.append(prompt)
        return processor(prompt, images=images)

    monkeypatch.setattr(chat, "processor", record)
    chat.attach([str(IMAGE)])
    chat.send("Hi", 4, stream=False)
    chat.send("And now?", 4, stream=False)

    assert [prompt.count("<image>") for prompt in prompts] == [1, 1]
    assert chat.images() == [str(IMAGE)]


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
    output_dir = build_model_dir(tmp_path_factory.mktemp("audio"))
    for precision in audio_export.PRECISIONS:
        audio_export.derive_precision_files(output_dir / "onnx", precision, block_size=32)
    audio_export.write_genai_config(output_dir, "q4")
    return output_dir


def test_audio_genai_config(audio):
    model = json.loads((audio / "genai_config.json").read_text())["model"]
    assert model["type"] == "lfm2_audio"
    assert model["audio_token_id"] == audio_export.AUDIO_TOKEN_ID
    assert not set(model["eos_token_id"]) & set(audio_export.MODALITY_SWITCH_TOKEN_IDS)
    files = [
        model["decoder"]["filename"],
        model["embedding"]["filename"],
        model["speech"]["filename"],
        model["audio_output"]["depthformer"]["filename"],
        model["audio_output"]["embedding"]["filename"],
    ]
    assert files == [
        "onnx/decoder_q4.onnx",
        "onnx/embeddings_fp16.onnx",
        "onnx/audio_encoder_q4.onnx",
        "onnx/vocoder_depthformer_fp16.onnx",
        "onnx/audio_embedding_fp16.onnx",
    ]
    for filename in files:
        assert (audio / filename).exists()


def test_audio_genai_config_checks_token_ids(audio, tmp_path):
    tokenizer = json.loads((audio / "tokenizer.json").read_text())
    for token in tokenizer["added_tokens"]:
        if token["content"] == "<|reserved_123|>":
            token["id"] = 134
    (tmp_path / "tokenizer.json").write_text(json.dumps(tokenizer))
    with pytest.raises(ValueError, match="reserved_123"):
        audio_export.write_genai_config(tmp_path, "q4")


def test_audio_embeddings_scatter_features(audio):
    embeddings = session(audio, "embeddings.onnx")
    table = onnx.numpy_helper.to_array(
        next(
            i
            for i in onnx.load(str(audio / "onnx/embeddings.onnx")).graph.initializer
            if i.name == "embed_tokens.weight"
        )
    )
    token = audio_export.AUDIO_TOKEN_ID
    input_ids = np.array([[1, 6, token, token, token, 7]], dtype=np.int64)
    features = np.random.default_rng(0).standard_normal((3, HIDDEN)).astype(np.float32)

    expected = table[input_ids[0]].copy()
    expected[2:5] = features
    np.testing.assert_array_equal(embed(embeddings, input_ids, features)[0], expected)


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
