"""
The LFM3-VL export of a tiny random checkpoint against its reference, LFM3-VL-2B's vendored
remote code (no download).

Run with:
    uv run pytest tests/test_lfm3_vl/test_tiny_export.py -v
"""

import builtins
import json
import os
import pathlib
import re
import shutil
import stat
import sys
import types

import numpy as np
import onnx
import pytest
import torch

from liquidonnx.lfm3.export import convert_quantized_to_fp16
from liquidonnx.lfm3.quantize import (
    LM_HEAD,
    mixed_int8_nodes,
    quantize_decoder_q4,
    untie_lm_head,
)
from liquidonnx.lfm3.session import (
    PREFERRED,
    CachedDecoder,
    kv_feeds_attention,
    load_tokenizer,
    update_cache,
)
from liquidonnx.lfm3_vl.export import VISION_QUANT, derive
from liquidonnx.lfm3_vl.session import BUNDLES, ONNXVLModel, run_vision
from liquidonnx.lfm3_vl.weights import load_tensors
from liquidonnx.quantize import quantize_matmuls
from liquidonnx.session import initialize_cache, load_onnx_session

from .conftest import ASSETS
from .tiny import TINY_CONFIG, write_flat_checkpoint

IMAGE, VIDEO = TINY_CONFIG["image_token_id"], TINY_CONFIG["video_token_id"]
START, END, BOS = TINY_CONFIG["vision_start_token_id"], TINY_CONFIG["vision_end_token_id"], 124894
HIDDEN = TINY_CONFIG["hidden_size"]
NO_FEATURES = np.zeros((0, HIDDEN), dtype=np.float32)

# (t, h, w): square, wide, tall, and a two-frame video grid
GRIDS = [(1, 2, 2), (1, 4, 6), (1, 6, 4), (1, 10, 4), (2, 6, 8)]


def pixels(grid, seed=0) -> np.ndarray:
    t, h, w = grid
    rng = np.random.default_rng(seed)
    return rng.uniform(-1, 1, (t * h * w, 1536)).astype(np.float32)


def reference_features(reference, pixel_values, grids) -> np.ndarray:
    with torch.no_grad():
        features = reference.get_image_features(torch.from_numpy(pixel_values), torch.tensor(grids))
    return torch.cat(features.pooler_output).numpy()


def image_prompt(tokens: int, text: int = 20, seed: int = 0) -> list[int]:
    rng = np.random.default_rng(seed)
    return [BOS, START] + [IMAGE] * tokens + [END] + rng.integers(1000, 120000, text).tolist()


def decoder_logits(session, ids, image_features=NO_FEATURES, video_features=NO_FEATURES):
    n = len(ids)
    feed = {
        "input_ids": np.array([ids], dtype=np.int64),
        "image_features": image_features,
        "video_features": video_features,
        "attention_mask": np.ones((1, n), dtype=np.int64),
        "position_ids": np.arange(n, dtype=np.int64)[None, :],
        "logits_to_keep": np.array([0], dtype=np.int64),
        **initialize_cache(session),
    }
    return session.run(None, feed)[0][0]


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a * b).sum(-1) / np.linalg.norm(a, axis=-1) / np.linalg.norm(b, axis=-1)


@pytest.mark.parametrize("prefix", ["vision_", "language_model."])
def test_port_checkpoint_loads_the_flat_tensors(tiny_checkpoint, tmp_path, prefix):
    """LFM3-VL-2B's names, unfused, give the builders the flat layout's tensors."""
    flat = write_flat_checkpoint(tmp_path / "flat")
    expected, got = load_tensors(flat, prefix), load_tensors(tiny_checkpoint, prefix)
    assert sorted(got) == sorted(expected)
    for name, array in expected.items():
        np.testing.assert_array_equal(got[name], array, err_msg=name)


@pytest.fixture(scope="module")
def sessions(tiny_export):
    onnx_dir = tiny_export / "onnx"
    return {
        name: load_onnx_session(onnx_dir / name)
        for name in sorted({f for b in BUNDLES.values() for f in b})
    }


# === Vision encoder ===


@pytest.mark.parametrize("grid", GRIDS, ids=lambda g: "x".join(map(str, g)))
def test_vision_matches_reference(sessions, tiny_reference, grid):
    pv = pixels(grid)
    expected = reference_features(tiny_reference, pv, [grid])
    got = run_vision(sessions["vision_encoder.onnx"], pv, np.array([grid]), merge=2)
    assert got.shape == expected.shape
    assert np.abs(got - expected).max() < 1e-4
    assert cosine(got, expected).min() > 0.99999


def test_vision_two_images_in_prompt_order(sessions, tiny_reference):
    grids = [(1, 4, 6), (1, 6, 4)]
    pv = np.concatenate([pixels(grids[0], 1), pixels(grids[1], 2)])
    expected = reference_features(tiny_reference, pv, grids)
    got = run_vision(sessions["vision_encoder.onnx"], pv, np.array(grids), merge=2)
    assert np.abs(got - expected).max() < 1e-4


# === Decoder ===


def test_decoder_image_prompt_matches_reference(sessions, tiny_reference):
    grid = (1, 4, 6)
    pv = pixels(grid)
    # 29 tokens: past the tiny model's sliding window of 8
    ids = image_prompt(6)
    with torch.no_grad():
        expected = (
            tiny_reference(
                input_ids=torch.tensor([ids]),
                pixel_values=torch.from_numpy(pv),
                image_grid_thw=torch.tensor([grid]),
                use_cache=False,
            )
            .logits[0]
            .numpy()
        )
    features = run_vision(sessions["vision_encoder.onnx"], pv, np.array([grid]), merge=2)
    got = decoder_logits(sessions["decoder.onnx"], ids, features)
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=2e-3)


def test_decoder_two_image_spans_match_reference(sessions, tiny_reference):
    grids = [(1, 4, 6), (1, 6, 4)]
    pv = np.concatenate([pixels(grids[0], 1), pixels(grids[1], 2)])
    ids = image_prompt(6, text=5, seed=1) + [START] + [IMAGE] * 6 + [END] + [5000, 6000, 7000]
    with torch.no_grad():
        expected = (
            tiny_reference(
                input_ids=torch.tensor([ids]),
                pixel_values=torch.from_numpy(pv),
                image_grid_thw=torch.tensor(grids),
                use_cache=False,
            )
            .logits[0]
            .numpy()
        )
    features = run_vision(sessions["vision_encoder.onnx"], pv, np.array(grids), merge=2)
    got = decoder_logits(sessions["decoder.onnx"], ids, features)
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=2e-3)


def test_decoder_video_and_text_prompts_match_reference(sessions, tiny_reference):
    grid = (2, 4, 6)
    pv = pixels(grid, 3)
    ids = [BOS, START] + [VIDEO] * 12 + [END] + [5000, 6000, 7000]
    with torch.no_grad():
        expected = (
            tiny_reference(
                input_ids=torch.tensor([ids]),
                pixel_values_videos=torch.from_numpy(pv),
                video_grid_thw=torch.tensor([grid]),
                use_cache=False,
            )
            .logits[0]
            .numpy()
        )
        text_ids = image_prompt(0)
        expected_text = (
            tiny_reference(input_ids=torch.tensor([text_ids]), use_cache=False).logits[0].numpy()
        )
    features = run_vision(sessions["vision_encoder.onnx"], pv, np.array([grid]), merge=2)
    got = decoder_logits(sessions["decoder.onnx"], ids, video_features=features)
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=2e-3)
    got_text = decoder_logits(sessions["decoder.onnx"], text_ids)
    np.testing.assert_allclose(got_text, expected_text, rtol=1e-4, atol=2e-3)


def test_logits_to_keep(sessions):
    session = sessions["decoder.onnx"]
    ids = image_prompt(0)
    feed = {
        "input_ids": np.array([ids], dtype=np.int64),
        "image_features": NO_FEATURES,
        "video_features": NO_FEATURES,
        "attention_mask": np.ones((1, len(ids)), dtype=np.int64),
        "position_ids": np.arange(len(ids), dtype=np.int64)[None, :],
        **initialize_cache(session),
    }
    all_rows = session.run(None, {**feed, "logits_to_keep": np.array([0])})[0]
    last = session.run(None, {**feed, "logits_to_keep": np.array([1])})[0]
    assert last.shape[1] == 1
    np.testing.assert_array_equal(last[0, 0], all_rows[0, -1])


@pytest.mark.parametrize("prompt", ["short_text", "image"])
def test_cached_decode_matches_teacher_forced_reference(
    tiny_export, sessions, tiny_reference, prompt
):
    model = ONNXVLModel(tiny_export, precision="fp32", force_cpu=True)
    model.load_sessions()
    if prompt == "image":
        grid = (1, 4, 6)
        pv = pixels(grid)
        ids, features = image_prompt(6), run_vision(model.vision, pv, np.array([grid]), merge=2)
        visual = {"pixel_values": torch.from_numpy(pv), "image_grid_thw": torch.tensor([grid])}
    else:
        # shorter than the conv kernel: the first steps read a zero-padded conv state
        ids, features, visual = [BOS, 5000, 6000], None, {}
    tokens = model.generate_ids(ids, features, max_new_tokens=12)
    with torch.no_grad():
        logits = tiny_reference(
            input_ids=torch.tensor([ids + tokens[:-1]]), use_cache=False, **visual
        ).logits
    assert logits[0, len(ids) - 1 :].argmax(-1).tolist() == tokens


@pytest.mark.parametrize("ep", ["cpu", "cuda"])
def test_default_precision_follows_the_execution_provider(tiny_export, tmp_path, monkeypatch, ep):
    """Without a precision, ONNXVLModel loads the first bundle of PREFERRED for the sessions'
    execution provider that the folder has (here no fp16 or q4f16), as lfm3-vl-infer picks; an
    explicit precision stays."""
    from liquidonnx.lfm3_vl import session as vl_session

    shutil.copy2(tiny_export / "config.json", tmp_path)
    (tmp_path / "onnx").mkdir()
    exported = ("q4", "q8", "q8f16")
    for name in {name for precision in exported for name in BUNDLES[precision]}:
        (tmp_path / "onnx" / name).touch()
    loaded = []
    monkeypatch.setattr(
        vl_session, "load_session", lambda path, ep, tf32: loaded.append((path.name, ep))
    )
    monkeypatch.setattr(vl_session, "kv_feeds_attention", lambda path: False)

    expected = next(precision for precision in PREFERRED[ep] if precision in exported)
    for precision, picked in [(None, expected), ("q8", "q8")]:
        model = ONNXVLModel(tmp_path, precision)
        model.load_sessions(ep=ep)
        assert model.precision == picked
        assert loaded[-2:] == [(BUNDLES[picked][1], ep), (BUNDLES[picked][0], ep)]
    shutil.rmtree(tmp_path / "onnx")
    with pytest.raises(FileNotFoundError, match="no exported bundle"):
        ONNXVLModel(tmp_path).load_sessions(ep=ep)


@pytest.mark.parametrize("stdin", [False, True], ids=["prompt", "stdin"])
def test_infer_cli_answers_about_an_image(tiny_export, monkeypatch, capsys, stdin):
    """lfm3-vl-infer --cpu without --precision loads the first exported bundle the CPU prefers (q4:
    the q4 decoder and int8 vision), resizes the image to at most --max-image-tokens tokens,
    prompts with an empty think block (--no-think) and prints the greedy answer of --max-tokens
    tokens. With --prompt that is the only turn; without, it answers each line of input with the
    conversation so far, the images only in the first turn."""
    from liquidonnx.lfm3_vl import infer
    from liquidonnx.lfm3_vl import session as vl_session

    loaded, generations = [], []
    load_session, generate_ids = vl_session.load_session, ONNXVLModel.generate_ids
    lines = ["What bird is this?", "Again"] if stdin else ["Again"]

    def record_load(path, **kwargs):
        loaded.append(path.name)
        return load_session(path, **kwargs)

    def record_generation(model, ids, features, *args):
        tokens = generate_ids(model, ids, features, *args)
        generations.append((ids, features, tokens))
        return tokens

    def next_line(prompt=""):
        if not lines:
            raise EOFError
        return lines.pop(0)

    monkeypatch.setattr(vl_session, "load_session", record_load)
    monkeypatch.setattr(ONNXVLModel, "generate_ids", record_generation)
    monkeypatch.setattr(builtins, "input", next_line)
    argv = ["lfm3-vl-infer", "--model", str(tiny_export), "--images", str(ASSETS / "cardinal.jpg")]
    argv += ["--cpu", "--no-think", "--max-tokens", "3", "--max-image-tokens", "16"]
    if not stdin:
        argv += ["--prompt", "What bird is this?"]
    monkeypatch.setattr(sys, "argv", argv)
    infer.main()

    assert loaded == ["vision_encoder_q8.onnx", "decoder_q4.onnx"]
    (ids, features, tokens), *later = generations
    image_tokens = ids.count(IMAGE)
    assert 0 < image_tokens <= 16 and features.shape == (image_tokens, HIDDEN)
    tokenizer = load_tokenizer(tiny_export)
    first = (
        "<|startoftext|><|im_start|>user\n<|vision_start|>"
        + "<|image|>" * image_tokens
        + "<|vision_end|>What bird is this?<|im_end|>\n<|im_start|>assistant\n<think></think>"
    )
    assert tokenizer.decode(ids) == first
    assert len(tokens) == 3
    answer = tokenizer.decode(tokens, skip_special_tokens=True)
    assert answer.strip() and f"Assistant: {answer}" in capsys.readouterr().out
    if stdin:
        [(second, _, _)] = later
        assert tokenizer.decode(second) == (
            f"{first}{answer}<|im_end|>\n<|im_start|>user\nAgain<|im_end|>\n"
            "<|im_start|>assistant\n<think></think>"
        )
    else:
        assert later == []


@pytest.mark.parametrize(
    "decoder_file", ["decoder.onnx", "decoder_fp16.onnx", "decoder_q8f16.onnx"]
)
@pytest.mark.parametrize("capacity", [None, 64], ids=["own-buffers", "shared-kv"])
def test_cached_decoder_matches_numpy_loop(tiny_export, sessions, decoder_file, capacity):
    session = sessions[decoder_file]
    if capacity:
        assert kv_feeds_attention(tiny_export / "onnx" / decoder_file)
    grid = (1, 4, 6)
    features = run_vision(sessions["vision_encoder.onnx"], pixels(grid), np.array([grid]), merge=2)
    ids = image_prompt(6)
    decoder = CachedDecoder(session, capacity)
    cache, outputs = initialize_cache(session), session.get_outputs()
    step_ids, step_features, past = ids, features, 0
    for _ in range(8):
        extra = {"video_features": NO_FEATURES, "logits_to_keep": np.array([1], dtype=np.int64)}
        got = decoder.step(step_ids, image_features=step_features, **extra)
        feed = {
            "input_ids": np.array([step_ids], dtype=np.int64),
            "image_features": step_features,
            "attention_mask": np.ones((1, past + len(step_ids)), dtype=np.int64),
            "position_ids": np.arange(past, past + len(step_ids), dtype=np.int64)[None, :],
            **extra,
            **cache,
        }
        result = session.run(None, feed)
        update_cache(cache, result, outputs)
        np.testing.assert_array_equal(got, result[0])
        past += len(step_ids)
        step_ids, step_features = [int(np.argmax(got[0, -1]))], NO_FEATURES


# === Quantized and fp16 graphs ===
# Random weights 64 wide quantize far worse than trained ones; these floors catch broken wiring,
# the real checkpoint's tests hold the quality bars. The q4 logits measure a cosine of 0.68 at
# worst and 0.87 on average (0.51 and 0.82 with every MatMul in int4, the head included), q8 0.995
# and 0.998.


@pytest.mark.parametrize(
    "decoder_file,min_floor,mean_floor",
    [
        ("decoder_fp16.onnx", 0.9999, 0.9999),
        ("decoder_q8.onnx", 0.98, 0.99),
        ("decoder_q8f16.onnx", 0.98, 0.99),
        ("decoder_q4.onnx", 0.6, 0.85),
        ("decoder_q4f16.onnx", 0.6, 0.85),
    ],
)
def test_decoders_track_fp32(sessions, decoder_file, min_floor, mean_floor):
    grid = (1, 4, 6)
    features = run_vision(sessions["vision_encoder.onnx"], pixels(grid), np.array([grid]), merge=2)
    for ids, image_features in ((image_prompt(6), features), (image_prompt(0), NO_FEATURES)):
        got = decoder_logits(sessions[decoder_file], ids, image_features)
        expected = decoder_logits(sessions["decoder.onnx"], ids, image_features)
        similarity = cosine(got, expected)
        assert similarity.min() > min_floor and similarity.mean() > mean_floor


def attributes(node: onnx.NodeProto) -> dict:
    return {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}


def assert_one_int8_table(path: pathlib.Path) -> onnx.GraphProto:
    """Every MatMul is quantized with zero points, in blocks of 32 at accuracy_level 4, the LM head
    to int8, and the head's weights are the token table."""
    graph = onnx.load(str(path), load_external_data=False).graph
    producers = {output: node for node in graph.node for output in node.output}

    assert [n.name for n in graph.node if n.op_type == "MatMul"] == []
    for node in [n for n in graph.node if n.op_type == "MatMulNBits"]:
        assert len(node.input) == 4 and node.input[3], f"{node.name} has no zero points"
        assert attributes(node)["accuracy_level"] == 4
        assert attributes(node)["block_size"] == 32
    head = producers["logits"]
    if head.op_type == "Cast":  # fp16 activations: the logits are cast back to float32
        head = producers[head.input[0]]
    assert head.op_type == "MatMulNBits" and attributes(head)["bits"] == 8

    (gather,) = [n for n in graph.node if n.op_type == "GatherBlockQuantized"]
    assert attributes(gather)["bits"] == 8 and attributes(gather)["block_size"] == 32
    weights = producers[gather.input[0]]
    assert weights.op_type == "Reshape" and weights.input[0] == head.input[1]
    assert list(gather.input[2:]) == list(head.input[2:])

    vocab = TINY_CONFIG["vocab_size"]
    floats = (onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16)
    tables = [
        init.name
        for init in graph.initializer
        if init.data_type in floats and list(init.dims) in ([vocab, HIDDEN], [HIDDEN, vocab])
    ]
    assert tables == []
    return graph


def assert_q4_recipe(path: pathlib.Path, fp32_path: pathlib.Path, gate_up: bool = False):
    """k_quant MatMuls (assert_one_int8_table): LFM3's recipe (the LM head and the mixed nodes of
    the fp32 graph int8, the rest int4) or with gate_up d1's (only the MLP gate_proj and up_proj
    int4, one of each per layer)."""
    graph = assert_one_int8_table(path)
    fp32 = onnx.load(str(fp32_path), load_external_data=False)
    names = [
        LM_HEAD if "logits" in node.output else node.name
        for node in fp32.graph.node
        if node.op_type == "MatMul"
    ]
    int8 = {LM_HEAD, *mixed_int8_nodes(fp32)}
    if gate_up:
        int4 = {name for name in names if re.search(r"/mlp/(gate|up)_proj/", name)}
        layers = {name.split("/")[2] for name in names if name.startswith("/model/layers.")}
        assert len(int4) == 2 * len(layers)
        int8 = set(names) - int4
    bits = {name: 8 if name in int8 else 4 for name in names}
    matmuls = [n for n in graph.node if n.op_type == "MatMulNBits"]
    assert {n.name: attributes(n)["bits"] for n in matmuls} == {
        f"{name}_Q{b}": b for name, b in bits.items()
    }
    assert list(bits.values()).count(8) == len(int8)


@pytest.mark.parametrize("decoder_file", ["decoder_q8.onnx", "decoder_q8f16.onnx"])
def test_q8_decoders_store_one_int8_table(tiny_export, decoder_file):
    graph = assert_one_int8_table(tiny_export / "onnx" / decoder_file)
    assert {attributes(n)["bits"] for n in graph.node if n.op_type == "MatMulNBits"} == {8}


def test_accuracy_level_copy(tiny_export, sessions):
    """--matmulnbits-accuracy-level's copy: every MatMulNBits at the level, the export's files as
    they were, and the copy runs on the export's weights close to the original (int8 activations
    or not)."""
    import onnxruntime as ort
    from helpers import EXTERNAL_DATA_FOLDER, with_accuracy_level

    source = tiny_export / "onnx" / "decoder_q8.onnx"
    before = {path.name: path.read_bytes() for path in source.parent.glob("decoder_q8.onnx*")}
    copy = with_accuracy_level(source, 0)
    nodes = [n for n in onnx.load_from_string(copy).graph.node if n.op_type == "MatMulNBits"]
    assert {attributes(node)["accuracy_level"] for node in nodes} == {0}
    assert {p.name: p.read_bytes() for p in source.parent.glob("decoder_q8.onnx*")} == before
    assert with_accuracy_level(tiny_export / "onnx" / "decoder.onnx", 0) is None

    options = ort.SessionOptions()
    options.add_session_config_entry(EXTERNAL_DATA_FOLDER, str(source.parent))
    session = ort.InferenceSession(copy, options, providers=["CPUExecutionProvider"])
    ids = image_prompt(0)
    got, expected = decoder_logits(session, ids), decoder_logits(sessions["decoder_q8.onnx"], ids)
    assert cosine(got, expected).min() > 0.99


def test_untie_lm_head_rejects_logits_of_another_op(tiny_export):
    """decoder_fp16's logits come from a Cast, which has no weight input to read."""
    model = onnx.load(str(tiny_export / "onnx" / "decoder_fp16.onnx"), load_external_data=False)
    with pytest.raises(ValueError, match=r"\(Cast\) is not a MatMul"):
        untie_lm_head(model)


@pytest.mark.parametrize("decoder_file", ["decoder_q4.onnx", "decoder_q4f16.onnx"])
def test_q4_decoders_follow_the_lfm2_recipe(tiny_export, decoder_file):
    onnx_dir = tiny_export / "onnx"
    assert_q4_recipe(onnx_dir / decoder_file, onnx_dir / "decoder.onnx")


def named_matmuls(layer_types: list[str]) -> onnx.ModelProto:
    """A graph with the MatMul names of an LFM3 decoder, and nothing else."""
    names = []
    for i, layer_type in enumerate(layer_types):
        operator = ["conv/in_proj", "conv/out_proj"]
        if layer_type != "conv":
            operator = [f"attn/{p}_proj" for p in ("q", "k", "v", "gate", "o")]
        mlp = [f"mlp/{p}_proj" for p in ("gate", "up", "down")]
        names += [f"/model/layers.{i}/{name}/MatMul" for name in operator + mlp]
    nodes = [onnx.helper.make_node("MatMul", ["x", "w"], [name], name=name) for name in names]
    return onnx.helper.make_model(onnx.helper.make_graph(nodes, "names", [], []))


def test_mixed_int8_nodes_follow_genai_mixed_layers():
    """The 27 int8 MatMuls of the 24-layer LFM3 checkpoints (attention on the even layers from 2),
    and the two of the tiny one."""
    layer_types = ["conv" if i % 2 or i < 2 else "full_attention" for i in range(24)]
    layers = [0, 1, 2, 5, 8, 11, 14, 17, 20, 21, 22, 23]
    expected = []
    for i in layers:
        if layer_types[i] != "conv":
            expected += [f"/model/layers.{i}/attn/{p}_proj/MatMul" for p in "qkv"]
        expected.append(f"/model/layers.{i}/mlp/down_proj/MatMul")
    assert len(expected) == 27
    assert mixed_int8_nodes(named_matmuls(layer_types)) == expected

    assert mixed_int8_nodes(named_matmuls(TINY_CONFIG["layer_types"])) == [
        "/model/layers.0/mlp/down_proj/MatMul",
        "/model/layers.3/mlp/down_proj/MatMul",
    ]


def test_k_quant_takes_node_bits_on_a_plain_graph(tmp_path):
    """quantize_matmuls' k_quant on a graph without contrib ops: the bits of each node, zero
    points, and the com.microsoft domain, which onnxruntime's k_quant doesn't declare."""
    rng = np.random.default_rng(0)
    a, b = (rng.standard_normal(shape).astype(np.float32) for shape in ((64, 64), (64, 32)))
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("MatMul", ["x", "a"], ["h"], name="/a/MatMul"),
            onnx.helper.make_node("MatMul", ["h", "b"], ["y"], name="/b/MatMul"),
        ],
        "plain",
        [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [None, 64])],
        [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [None, 32])],
        [onnx.numpy_helper.from_array(a, "a"), onnx.numpy_helper.from_array(b, "b")],
    )
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 21)])
    model.ir_version = 10
    with pytest.raises(ValueError, match="names no MatMul"):
        quantize_matmuls(model, bits=4, algorithm="k_quant", node_bits={"/c/MatMul": 8})
    with pytest.raises(ValueError, match="needs algorithm"):
        quantize_matmuls(model, bits=4, node_bits={"/b/MatMul": 8})
    with pytest.raises(ValueError, match="zero points"):
        quantize_matmuls(model, bits=4, algorithm="k_quant", symmetric=True)

    quantized = quantize_matmuls(model, bits=4, algorithm="k_quant", node_bits={"/b/MatMul": 8})
    nodes = {n.name: n for n in quantized.graph.node}
    assert {name: attributes(n)["bits"] for name, n in nodes.items()} == {
        "/a/MatMul_Q4": 4,
        "/b/MatMul_Q8": 8,
    }
    assert all(n.op_type == "MatMulNBits" and len(n.input) == 4 for n in nodes.values())
    onnx.save(quantized, str(tmp_path / "plain.onnx"))
    x = rng.standard_normal((3, 64)).astype(np.float32)
    got = load_onnx_session(tmp_path / "plain.onnx").run(None, {"x": x})[0]
    assert cosine(got, x @ a @ b).min() > 0.98


def test_q4_rejects_a_block_size_of_odd_blocks(
    tiny_export, tiny_checkpoint, tmp_path, monkeypatch, capsys
):
    """At block size 64 the tiny decoder's 64-wide inputs are one block, whose int4 zero points
    onnxruntime's k_quant can't pack: quantize_decoder_q4 raises, and both CLIs stop with a usage
    error before they write a file."""
    from liquidonnx.lfm3 import export as text_export
    from liquidonnx.lfm3_vl import export as vl_export

    expected = (
        r"even number of blocks of 64 in the input size K of each int4 MatMul: K=64 is 1 block "
        r"in /model/layers\.0/conv/in_proj/MatMul, /model/layers\.0/conv/out_proj/MatMul and "
        r"\d+ more\. Block sizes that fit: 16, 32$"
    )
    decoder = tiny_export / "onnx" / "decoder.onnx"
    with pytest.raises(ValueError, match=expected):
        quantize_decoder_q4(decoder, tmp_path / "decoder_q4.onnx", block_size=64)

    # The graphs without their data: the check reads the weights' shapes
    onnx_dir = tmp_path / "exports" / "tiny-ONNX" / "onnx"
    onnx_dir.mkdir(parents=True)
    for name in ("model.onnx", "decoder.onnx"):
        shutil.copy2(decoder, onnx_dir / name)
    shutil.copy2(tiny_export / "onnx" / "vision_encoder.onnx", onnx_dir)
    before = sorted(tmp_path.rglob("*"))
    argv = ["--output-dir", str(tmp_path), "--output-name", "tiny-ONNX", "--skip-export"]
    for cli, module in (("lfm3-export", text_export), ("lfm3-vl-export", vl_export)):
        monkeypatch.setattr(
            sys, "argv", [cli, str(tiny_checkpoint), *argv, "--block-size", "64", "--precision"]
        )
        with pytest.raises(SystemExit) as stop:
            module.main()
        assert stop.value.code == 2
        assert re.search(
            rf"{cli}: error: --block-size 64 .*{expected}", capsys.readouterr().err, re.M
        )
    assert sorted(tmp_path.rglob("*")) == before


def test_derive_changes_only_the_requested_bundles(tiny_export, tmp_path):
    """q4f16 and q8f16 alone: the q4 and q8 decoders they are converted from are scratch files, so
    none is left behind and the decoder_q8.onnx of an earlier run stays as it was."""
    onnx_dir = tmp_path / "onnx"
    onnx_dir.mkdir()
    for name in ("decoder.onnx", "vision_encoder.onnx"):
        shutil.copy2(tiny_export / "onnx" / name, onnx_dir)
        shutil.copy2(tiny_export / "onnx" / f"{name}_data", onnx_dir)
    earlier = onnx_dir / "decoder_q8.onnx"
    earlier.write_bytes(b"an earlier run")
    before = {path.name for path in onnx_dir.iterdir()}

    derive(onnx_dir, ["q4f16", "q8f16"], block_size=32)
    written = {path.name for path in onnx_dir.iterdir()} - before
    assert written == {
        f"{name}{suffix}"
        for name in ("decoder_q4f16.onnx", "decoder_q8f16.onnx", "vision_encoder_fp16.onnx")
        for suffix in ("", "_data")
    }
    assert earlier.read_bytes() == b"an earlier run"
    for name in written:
        assert (onnx_dir / name).read_bytes() == (tiny_export / "onnx" / name).read_bytes()


def test_text_export_cli_rebuilds_the_requested_precisions(tiny_export, tmp_path, monkeypatch):
    """lfm3-export quantizes model.onnx as the VL export does its decoder, and in a folder with
    the files of an earlier export it rebuilds those of the requested precisions and no other,
    with the mode of the umask (onnx.save_model creates external data 0600). The tiny VL decoder
    stands in for a text model.onnx: the same LFM3Builder graph plus the two feature inputs."""
    from liquidonnx.lfm3 import export as text_export

    onnx_dir = tmp_path / "exports" / "tiny-ONNX" / "onnx"
    onnx_dir.mkdir(parents=True)
    shutil.copy2(tiny_export / "onnx" / "decoder.onnx", onnx_dir / "model.onnx")
    shutil.copy2(tiny_export / "onnx" / "decoder.onnx_data", onnx_dir)
    # The earlier export: copies of the fp32 graph, and the second data file of a split one
    requested = ["fp16", "q4", "q4f16", "q4f32", "q8f16"]
    earlier = (onnx_dir / "model.onnx").read_bytes()
    for precision in [*requested, "q8"]:
        (onnx_dir / f"model_{precision}.onnx").write_bytes(earlier)
    chunks = [onnx_dir / f"model_{precision}.onnx_data_1" for precision in ("fp16", "q8f16")]
    for chunk in chunks:
        chunk.write_bytes(b"an earlier export")

    argv = ["lfm3-export", "tiny", "--output-dir", str(tmp_path), "--skip-export"]
    monkeypatch.setattr(sys, "argv", [*argv, "--precision", *requested])
    previous = os.umask(0o022)
    try:
        text_export.main()
    finally:
        os.umask(previous)
    assert {oct(stat.S_IMODE(path.stat().st_mode)) for path in onnx_dir.iterdir()} == {oct(0o644)}
    assert (onnx_dir / "model_q8.onnx").read_bytes() == earlier
    for precision in requested:
        assert (onnx_dir / f"model_{precision}.onnx").read_bytes() != earlier
    assert not any(chunk.exists() for chunk in chunks)
    assert [path.name for path in onnx_dir.iterdir() if path.is_dir()] == []
    [record] = json.loads((onnx_dir.parent / "export_info.json").read_text())["exports"]
    assert record["checkpoint"]["model"] == "tiny" and record["precisions"] == requested
    assert record["packages"]["onnx"] == onnx.__version__
    assert_one_int8_table(onnx_dir / "model_q8f16.onnx")
    for name in ("model_q4.onnx", "model_q4f16.onnx"):
        assert_q4_recipe(onnx_dir / name, onnx_dir / "model.onnx")


def test_text_answer_cut_at_max_tokens_warns(tiny_export, monkeypatch, caplog):
    """lfm3-infer's greedy loop (ONNXTextModel.generate) with its KV cache sized to the prompt and
    max_new_tokens: one warning when the answer stops at the cap, none when it ends at the
    end-of-turn token. The tiny VL decoder stands in for model.onnx, fed zero-row features."""
    from liquidonnx.lfm3 import session as text_session

    class TextDecoder(CachedDecoder):
        def step(self, ids, **inputs):
            features = {"image_features": NO_FEATURES, "video_features": NO_FEATURES}
            return super().step(ids, **features, **inputs)

    # decode returns the ids, so generate returns the answer's tokens
    tokenizer = types.SimpleNamespace(
        eos_token_id=None,
        apply_chat_template=lambda messages, **kwargs: "",
        encode=lambda prompt, **kwargs: [BOS, 5000, 6000],
        decode=lambda ids, **kwargs: ids,
    )
    monkeypatch.setattr(text_session, "CachedDecoder", TextDecoder)
    monkeypatch.setattr(text_session, "load_tokenizer", lambda path: tokenizer)
    model = text_session.ONNXTextModel(tiny_export / "onnx" / "decoder.onnx", force_cpu=True)
    model.load()
    assert model.share_kv

    def warnings() -> list[str]:
        return [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]

    tokens = model.generate([], max_new_tokens=6, stream=False)
    assert len(tokens) == 6
    assert warnings() == ["The answer was cut at --max-tokens (6 tokens)"]

    caplog.clear()
    tokenizer.eos_token_id = tokens[3]
    answer = model.generate([], max_new_tokens=6, stream=False)
    assert answer == tokens[: tokens.index(tokens[3]) + 1]
    assert warnings() == []


def test_vision_encoder_q8_follows_its_recipe(tiny_export):
    """Every MatMul is quantized at VISION_QUANT's bits, block size and accuracy_level, with zero
    points only if it is asymmetric."""
    path = tiny_export / "onnx" / "vision_encoder_q8.onnx"
    graph = onnx.load(str(path), load_external_data=False).graph
    assert [n.name for n in graph.node if n.op_type == "MatMul"] == []
    matmuls = [n for n in graph.node if n.op_type == "MatMulNBits"]
    keys = ("bits", "block_size", "accuracy_level")
    assert matmuls
    for node in matmuls:
        # onnxruntime leaves out an accuracy_level of 0, the attribute's default
        found = {"accuracy_level": 0} | attributes(node)
        assert {k: found[k] for k in keys} == {k: VISION_QUANT[k] for k in keys}
        zero_points = len(node.input) == 4 and node.input[3] != ""
        assert zero_points != VISION_QUANT["symmetric"], node.name


@pytest.mark.parametrize("vision_file", ["vision_encoder_fp16.onnx", "vision_encoder_q8.onnx"])
def test_vision_encoders_track_fp32(sessions, vision_file):
    grid = (1, 6, 8)
    pv = pixels(grid)
    got = run_vision(sessions[vision_file], pv, np.array([grid]), merge=2)
    expected = run_vision(sessions["vision_encoder.onnx"], pv, np.array([grid]), merge=2)
    assert cosine(got, expected).min() > 0.999


@pytest.mark.parametrize(
    "decoder_file", ["decoder_fp16.onnx", "decoder_q4f16.onnx", "decoder_q8f16.onnx"]
)
def test_fp16_decoders_keep_float32_features_and_logits(sessions, decoder_file):
    types = {i.name: i.type for i in sessions[decoder_file].get_inputs()}
    types |= {o.name: o.type for o in sessions[decoder_file].get_outputs()}
    for name in ("image_features", "video_features", "logits"):
        assert types[name] == "tensor(float)"
    caches = [t for name, t in types.items() if name.startswith(("past", "present"))]
    assert caches and all(t == "tensor(float16)" for t in caches)


def test_fp16_activations_keep_float32_io_that_is_no_cache(tiny_export, sessions, tmp_path):
    """convert_quantized_to_fp16 on the int8 vision encoder: pixel_values (a name with "value",
    as the KV caches) and image_features stay float32 around an fp16 graph. Its worst cosine is
    0.9995 here, 0.9991 over 15 inputs (arm64 and x86)."""
    path = convert_quantized_to_fp16(
        tiny_export / "onnx" / "vision_encoder_q8.onnx", tmp_path / "vision_encoder_q8f16.onnx"
    )
    graph = onnx.load(str(path), load_external_data=False).graph
    assert onnx.TensorProto.FLOAT not in {init.data_type for init in graph.initializer}
    session = load_onnx_session(path)
    assert [(i.name, i.type) for i in session.get_inputs()] == [("pixel_values", "tensor(float)")]
    assert [(o.name, o.type) for o in session.get_outputs()] == [
        ("image_features", "tensor(float)")
    ]
    grid = (1, 6, 8)
    pv = pixels(grid)
    got = run_vision(session, pv, np.array([grid]), merge=2)
    expected = run_vision(sessions["vision_encoder.onnx"], pv, np.array([grid]), merge=2)
    assert cosine(got, expected).min() > 0.998


def test_graph_io(sessions):
    decoder = sessions["decoder.onnx"]
    names = [i.name for i in decoder.get_inputs()]
    assert names[:6] == [
        "input_ids",
        "attention_mask",
        "position_ids",
        "image_features",
        "video_features",
        "logits_to_keep",
    ]
    assert [o.name for o in decoder.get_outputs()][0] == "logits"
    vision = sessions["vision_encoder.onnx"]
    assert [i.name for i in vision.get_inputs()] == ["pixel_values"]
    assert [o.name for o in vision.get_outputs()] == ["image_features"]
