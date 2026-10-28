"""
The speech encoder graph against the reference's get_audio_features (on the repo's
woodworks_question.wav, 62 tokens, past the encoder's window of 32), and GroupQueryAttention's
local_window_size against HF's sliding window.

Run with:
    uv run pytest tests/test_lfm3_audio/test_audio_encoder.py -v -k "tiny or window"
    uv run --no-sync pytest tests/test_lfm3_audio/test_audio_encoder.py -v \\
        --device cuda --lfm3-audio-model LiquidAI/LFM3-Omni-2B
"""

import logging
import pathlib

import numpy as np
import onnx
import pytest
import torch
from onnx import TensorProto, helper

from liquidonnx.lfm3_audio.config import load_config
from liquidonnx.lfm3_audio.encoder import INPUT, LFM3AudioEncoderBuilder
from liquidonnx.session import load_onnx_session

logger = logging.getLogger(__name__)

SAMPLES = pathlib.Path(__file__).parents[2] / "samples" / "audio"
CLIPS = {"woodworks": SAMPLES / "woodworks_question.wav"}
FILES = {"fp32": "audio_encoder.onnx", "fp16": "audio_encoder_fp16.onnx"}
# (rel L2 error, 1st-percentile token cosine) bounds against the float32 reference; bf16 PyTorch's
# own error is well above fp16's
BOUNDS = {"fp32": (2e-5, 0.99999), "fp16": (1e-2, 0.9999)}


def token_metrics(got: np.ndarray, expected: np.ndarray) -> tuple[float, np.ndarray]:
    """(rel L2 error, per-token cosine)"""
    rel = np.linalg.norm(got - expected) / np.linalg.norm(expected)
    norms = np.linalg.norm(got, axis=-1) * np.linalg.norm(expected, axis=-1)
    return float(rel), (got * expected).sum(-1) / norms


def test_tiny_encoder(tiny_checkpoint, tiny_reference, tmp_path):
    """Per-clip runs equal the reference's padded batch, past the window of 8 tokens."""
    model = LFM3AudioEncoderBuilder(load_config(tiny_checkpoint)).build(tiny_checkpoint)
    onnx.save(model, tmp_path / "audio_encoder.onnx")
    session = load_onnx_session(tmp_path / "audio_encoder.onnx")

    lengths = [150, 37, 1]  # 19, 5 and 1 tokens
    features = torch.randn(
        len(lengths), 128, max(lengths), generator=torch.Generator().manual_seed(0)
    )
    mask = torch.arange(max(lengths))[None] < torch.tensor(lengths)[:, None]
    with torch.no_grad():
        expected = tiny_reference.get_audio_features(features * mask[:, None], mask.long()).numpy()
    got = np.concatenate(
        [
            session.run(None, {INPUT: clip[None, :, :n].numpy()})[0]
            for clip, n in zip(features, lengths, strict=True)
        ]
    )
    assert got.shape == expected.shape == (25, 64)
    rel, cosine = token_metrics(got, expected)
    logger.info(f"  tiny: rel {rel:.2e}, worst cos {cosine.min():.8f}")
    assert rel < 1e-5


def gqa_model(window: int, heads: int, kv_heads: int, head_dim: int, dtype: int) -> onnx.ModelProto:
    """GroupQueryAttention without a past, as the encoder and codec graphs run it."""
    node = helper.make_node(
        "GroupQueryAttention",
        ["q", "k", "v", "", "", "seqlens_k", "total_sequence_length"],
        ["output", "present_key", "present_value"],
        domain="com.microsoft",
        num_heads=heads,
        kv_num_heads=kv_heads,
        local_window_size=window,
        scale=head_dim**-0.5,
    )
    inputs = [
        helper.make_tensor_value_info(name, dtype, [1, "S", n * head_dim])
        for name, n in (("q", heads), ("k", kv_heads), ("v", kv_heads))
    ]
    inputs += [
        helper.make_tensor_value_info("seqlens_k", TensorProto.INT32, [1]),
        helper.make_tensor_value_info("total_sequence_length", TensorProto.INT32, []),
    ]
    outputs = [helper.make_tensor_value_info(name, dtype, None) for name in node.output]
    graph = helper.make_graph([node], "gqa", inputs, outputs)
    opsets = [helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)]
    return helper.make_model(graph, opset_imports=opsets, ir_version=10)


def hf_window_attention(q, k, v, window: int, heads: int, kv_heads: int) -> np.ndarray:
    """float64 attention under transformers' sliding_window_causal_mask_function(window)."""
    from transformers.masking_utils import sliding_window_causal_mask_function

    S = q.shape[1]
    positions = torch.arange(S)
    allowed = sliding_window_causal_mask_function(window)(0, 0, positions[:, None], positions)
    q, k, v = (
        x.astype(np.float64).reshape(S, -1, x.shape[-1] // n).transpose(1, 0, 2)
        for x, n in ((q[0], heads), (k[0], kv_heads), (v[0], kv_heads))
    )
    k, v = (np.repeat(x, heads // kv_heads, axis=0) for x in (k, v))
    scores = q @ k.transpose(0, 2, 1) / np.sqrt(q.shape[-1])
    scores = np.where(allowed.numpy(), scores, -np.inf)
    weights = np.exp(scores - scores.max(-1, keepdims=True))
    weights /= weights.sum(-1, keepdims=True)
    return (weights @ v).transpose(1, 0, 2).reshape(1, S, -1)


@pytest.mark.parametrize("window", [30, 32, 256])
def test_gqa_window(ep, window, tmp_path):
    """local_window_size=W attends to the same W keys, the query's own included, as HF's
    sliding_window=W, and the windows one key wider or narrower measure far off. CUDA runs the
    fp16 kernel (it has no fp32 one)."""
    heads, kv_heads, head_dim, S = 8, 4, 96, 2 * window + 40
    dtype = np.float16 if ep == "cuda" else np.float32
    elem = TensorProto.FLOAT16 if ep == "cuda" else TensorProto.FLOAT
    rng = np.random.default_rng(0)
    q, k, v = (
        rng.standard_normal((1, S, n * head_dim)).astype(dtype) for n in (heads, kv_heads, kv_heads)
    )
    onnx.save(gqa_model(window, heads, kv_heads, head_dim, elem), tmp_path / "gqa.onnx")
    session = load_onnx_session(tmp_path / "gqa.onnx", ep=ep)
    feeds = {
        "q": q,
        "k": k,
        "v": v,
        "seqlens_k": np.array([S - 1], np.int32),
        "total_sequence_length": np.array(S, np.int32),
    }
    got = session.run(["output"], feeds)[0].astype(np.float64)
    errors = {
        w: float(np.abs(got - hf_window_attention(q, k, v, w, heads, kv_heads)).max())
        for w in (window - 1, window, window + 1)
    }
    logger.info(
        f"  {ep} window {window}: max|diff| against HF window "
        + ", ".join(f"{w}: {e:.2e}" for w, e in errors.items())
    )
    assert errors[window] < (5e-3 if ep == "cuda" else 1e-5)
    assert 10 * errors[window] < min(errors[window - 1], errors[window + 1])


# === Real checkpoint ===


@pytest.mark.parametrize("precision", list(FILES))
@pytest.mark.parametrize("clip", list(CLIPS))
def test_audio_features(export_dir, reference, reference_processor, device, ep, precision, clip):
    import librosa

    wave, _ = librosa.load(CLIPS[clip], sr=16000)
    inputs = reference_processor.feature_extractor(wave, sampling_rate=16000, return_tensors="pt")
    features = inputs["input_features"]
    with torch.no_grad():
        expected = reference.get_audio_features(
            features.to(device), inputs["feature_attention_mask"].to(device)
        )
    expected = expected.float().cpu().numpy()

    session = load_onnx_session(export_dir / "onnx" / FILES[precision], ep=ep, tf32=False)
    got = session.run(None, {INPUT: features.numpy()})[0]
    assert got.shape == expected.shape
    rel, cosine = token_metrics(got, expected)
    logger.info(
        f"  {clip} {precision} ({len(got)} tokens): rel {rel:.2e}, worst cos {cosine.min():.6f}, "
        f"p1 {np.percentile(cosine, 1):.6f}, mean {cosine.mean():.6f}, "
        f"max|diff| {np.abs(got - expected).max():.2e}"
    )
    max_rel, p1_floor = BOUNDS[precision]
    assert rel < max_rel
    assert np.percentile(cosine, 1) > p1_floor
