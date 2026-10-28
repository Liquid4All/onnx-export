"""
The d1 export of a tiny random checkpoint (tests/test_lfm3_vl/tiny.py's LFM3-VL with d1's keys)
against its LFM3-VL reference (LFM3-VL-2B's vendored remote code) with d1's set mask on its
full-attention layers, as the checkpoint's attention.Sees.rows builds it, and d1-export's q4
recipe on it and on a tiny d1-Omni (tests/test_lfm3_audio/tiny.py's LFM3-Omni with d1's keys)
(no download).

Run with:
    uv run pytest tests/test_d1/test_tiny_export.py -v
"""

import contextlib
import json
import sys

import numpy as np
import onnx
import pytest
import torch
import torch.nn.functional as F

from liquidonnx.d1.decoder import SET_BIAS
from liquidonnx.d1.export import copy_checkpoint_files, derive, export_fp32, main
from liquidonnx.lfm3_vl.session import run_vision
from liquidonnx.session import load_onnx_session

from ..test_lfm3_audio import tiny as audio_tiny
from ..test_lfm3_vl.reference.loader import load_reference
from ..test_lfm3_vl.test_tiny_export import (
    assert_one_int8_table,
    assert_q4_recipe,
    cosine,
    pixels,
)
from ..test_lfm3_vl.tiny import TINY_CONFIG, write_tiny_checkpoint
from .test_config import d1_config

IMAGE, BOS = TINY_CONFIG["image_token_id"], TINY_CONFIG["bos_token_id"]
START, END = TINY_CONFIG["vision_start_token_id"], TINY_CONFIG["vision_end_token_id"]
H = TINY_CONFIG["hidden_size"]
GRID = (1, 4, 6)
DECODERS = ["decoder.onnx", "decoder_fp16.onnx", "decoder_q8.onnx", "decoder_q8f16.onnx"]
DECODERS += ["decoder_q4.onnx", "decoder_q4f16.onnx"]


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory):
    directory = write_tiny_checkpoint(tmp_path_factory.mktemp("d1_tiny"))
    config = json.loads((directory / "config.json").read_text())
    (directory / "config.json").write_text(json.dumps(d1_config(config), indent=2))
    return directory


@pytest.fixture(scope="module")
def tiny_omni_checkpoint(tmp_path_factory):
    """d1 on LFM3-Omni (d1-Omni-2B's kind): the tiny LFM3-Omni checkpoint with d1's keys."""
    parts = tmp_path_factory.mktemp("d1_omni_tiny")
    audio = audio_tiny.write_tiny_checkpoint(parts / "audio")
    directory = audio_tiny.write_omni_checkpoint(audio, parts / "d1-Omni")
    config = json.loads((directory / "config.json").read_text())
    (directory / "config.json").write_text(json.dumps(d1_config(config), indent=2))
    return directory


@pytest.fixture(scope="module")
def tiny_reference(tiny_checkpoint):
    return load_reference(tiny_checkpoint)


@pytest.fixture(scope="module")
def tiny_export(tiny_checkpoint, tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("d1_tiny_export")
    onnx_dir = output_dir / "onnx"
    onnx_dir.mkdir()
    export_fp32(tiny_checkpoint, onnx_dir)
    copy_checkpoint_files(tiny_checkpoint, output_dir)
    derive(onnx_dir, ["fp16", "q4", "q8", "q4f16", "q8f16"], block_size=32)
    return output_dir


@pytest.fixture(scope="module")
def sessions(tiny_export):
    names = ["vision_encoder.onnx", *DECODERS]
    return {name: load_onnx_session(tiny_export / "onnx" / name) for name in names}


def packed_row(questions: list[int], seed: int = 0) -> tuple[list[int], list[int], list[int]]:
    """(ids, question ids, slots) of a state with an image and the given question lengths after
    it, each question's last token a slot and the first question's third one too."""
    rng = np.random.default_rng(seed)
    text = lambda n: rng.integers(1000, 120000, n).tolist()  # noqa: E731
    ids = [BOS, *text(5), START, *[IMAGE] * 6, END, *text(3)]
    question = [-1] * len(ids)
    slots = []
    for k, length in enumerate(questions, 1):
        ids += text(length)
        question += [k] * length
        slots.append(len(ids) - 1)
    if len(questions) > 1:
        slots.insert(0, question.index(1) + 2)
    return ids, question, slots


@contextlib.contextmanager
def set_attention(reference, question: list[int]):
    """The reference's full-attention layers under d1's set mask: their scaled_dot_product_attention
    takes it in place of the causal mask (Sees.rows: a query on the state sees every key, a key on
    the state every query, a question its own tokens)."""
    assert reference.config._attn_implementation == "sdpa"
    g = torch.tensor(question)
    seen = (g[:, None] == g[None, :]) | (g[:, None] == -1) | (g[None, :] == -1)
    active = []
    sdpa = F.scaled_dot_product_attention

    def set_sdpa(q, k, v, attn_mask=None, is_causal=False, **kwargs):
        if active:
            attn_mask = torch.zeros(seen.shape, dtype=q.dtype)
            attn_mask.masked_fill_(~seen, torch.finfo(q.dtype).min)
            is_causal = False
        return sdpa(q, k, v, attn_mask=attn_mask, is_causal=is_causal, **kwargs)

    language = reference.model.language_model
    hooks = []
    for layer, layer_type in zip(language.layers, language.config.layer_types, strict=True):
        if layer_type == "full_attention":
            hooks.append(layer.self_attn.register_forward_pre_hook(lambda *_: active.append(1)))
            hooks.append(layer.self_attn.register_forward_hook(lambda *_: active.clear()))
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(F, "scaled_dot_product_attention", set_sdpa)
        try:
            yield
        finally:
            for hook in hooks:
                hook.remove()


def reference_outputs(reference, ids, question, slots, set_mask=True):
    """(logits at the slots [N, V], final hidden states [S, H]) of the reference."""
    hidden = []
    language = reference.model.language_model
    hook = language.register_forward_hook(lambda m, i, out: hidden.append(out.last_hidden_state))
    mask = set_attention(reference, question) if set_mask else contextlib.nullcontext()
    try:
        with mask, torch.no_grad():
            logits = reference(
                input_ids=torch.tensor([ids]),
                pixel_values=torch.from_numpy(pixels(GRID)),
                image_grid_thw=torch.tensor([GRID]),
                use_cache=False,
            ).logits[0]
    finally:
        hook.remove()
    return logits[slots].numpy(), hidden[0][0].numpy()


def onnx_outputs(sessions, decoder, ids, question, slots):
    features = run_vision(sessions["vision_encoder.onnx"], pixels(GRID), np.array([GRID]), 2)
    n = len(ids)
    feed = {
        "input_ids": np.array([ids], dtype=np.int64),
        "attention_mask": np.ones((1, n), dtype=np.int64),
        "position_ids": np.arange(n, dtype=np.int64)[None],
        "image_features": features,
        "video_features": np.zeros((0, H), dtype=np.float32),
        "question_ids": np.array([question], dtype=np.int64),
        "slot_positions": np.array(slots, dtype=np.int64),
    }
    logits, hidden = sessions[decoder].run(["logits", "hidden_states"], feed)
    return logits, hidden[0]


# 3 questions: 32 tokens, past the sliding window of 8; 1 question: the whole row both ways
@pytest.mark.parametrize("questions", [[4, 6, 5], [7]], ids=["three", "one"])
def test_decoder_matches_reference(sessions, tiny_reference, questions):
    ids, question, slots = packed_row(questions)
    expected_logits, expected_hidden = reference_outputs(tiny_reference, ids, question, slots)
    logits, hidden = onnx_outputs(sessions, "decoder.onnx", ids, question, slots)
    np.testing.assert_allclose(logits, expected_logits, rtol=1e-4, atol=2e-3)
    np.testing.assert_allclose(hidden, expected_hidden, rtol=1e-4, atol=1e-4)
    # the mask matters: the causal reference reads otherwise
    causal, _ = reference_outputs(tiny_reference, ids, question, slots, set_mask=False)
    assert np.abs(causal - logits).max() > 0.1


# Slot logits' cosine to the fp32 graph's. Random weights 64 wide quantize far worse than trained
# ones (tests/test_lfm3_vl/test_tiny_export.py's floors): these catch broken wiring, the real
# checkpoints' tests hold the quality bars, hidden states included.
@pytest.mark.parametrize(
    "decoder,min_floor,mean_floor",
    [
        ("decoder_fp16.onnx", 0.9999, 0.9999),
        ("decoder_q8.onnx", 0.98, 0.99),
        ("decoder_q8f16.onnx", 0.98, 0.99),
        ("decoder_q4.onnx", 0.6, 0.85),
        ("decoder_q4f16.onnx", 0.6, 0.85),
    ],
)
def test_decoders_track_fp32(sessions, decoder, min_floor, mean_floor):
    ids, question, slots = packed_row([4, 6, 5], seed=1)
    got, _ = onnx_outputs(sessions, decoder, ids, question, slots)
    expected, _ = onnx_outputs(sessions, "decoder.onnx", ids, question, slots)
    similarity = cosine(got, expected)
    assert similarity.min() > min_floor and similarity.mean() > mean_floor


def test_graph_structure(tiny_export):
    """No cache I/O; the full layers' MultiHeadAttention all read the one set bias, the sliding
    layers keep GroupQueryAttention with their window."""
    graph = onnx.load(str(tiny_export / "onnx" / "decoder.onnx"), load_external_data=False).graph
    assert [i.name for i in graph.input] == [
        "input_ids",
        "attention_mask",
        "position_ids",
        "image_features",
        "video_features",
        "question_ids",
        "slot_positions",
    ]
    assert [o.name for o in graph.output] == ["logits", "hidden_states"]
    ops = [node.op_type for node in graph.node]
    layer_types = TINY_CONFIG["layer_types"]
    attention = [n for n in graph.node if n.op_type == "MultiHeadAttention"]
    assert len(attention) == layer_types.count("full_attention")
    assert all(node.input[5] == SET_BIAS for node in attention)
    assert ops.count("Where") == 1
    gqa = [n for n in graph.node if n.op_type == "GroupQueryAttention"]
    assert len(gqa) == layer_types.count("sliding_attention")
    for node in gqa:
        window = next(a.i for a in node.attribute if a.name == "local_window_size")
        assert window == TINY_CONFIG["sliding_window"]


@pytest.mark.parametrize("decoder", [d for d in DECODERS if "_q" in d])
def test_quantized_decoders_share_one_int8_table(tiny_export, decoder):
    assert_one_int8_table(tiny_export / "onnx" / decoder)


@pytest.mark.parametrize(
    "checkpoint", ["tiny_checkpoint", "tiny_omni_checkpoint"], ids=["d1", "d1-Omni"]
)
def test_q4_recipe(request, tmp_path, monkeypatch, checkpoint):
    """The q4 and q4f16 decoders of d1 on LFM3-VL (d1-2B) and on LFM3-Omni (d1-Omni-2B) keep only
    the MLP gate_proj and up_proj in int4: d1-export's main() end to end, so a recipe lost on the
    way from it to the quantizer fails here."""
    checkpoint = request.getfixturevalue(checkpoint)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["d1-export", str(checkpoint), "--precision", "q4", "q4f16"])
    main()
    onnx_dir = tmp_path / "exports" / f"{checkpoint.name}-ONNX" / "onnx"
    for decoder in ("decoder_q4.onnx", "decoder_q4f16.onnx"):
        assert_q4_recipe(onnx_dir / decoder, onnx_dir / "decoder.onnx", gate_up=True)


def test_checkpoint_code_is_copied(tiny_checkpoint, tiny_export):
    (tiny_checkpoint / "prompt.py").write_text("LEADS = {}\n")
    copy_checkpoint_files(tiny_checkpoint, tiny_export)
    assert (tiny_export / "prompt.py").read_text() == "LEADS = {}\n"
    assert json.loads((tiny_export / "config.json").read_text())["d1_attention"] == "set"
