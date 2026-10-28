"""
The depth decoder graph against the reference's Lfm3AudioDepthDecoder.

Tiny checkpoint (no download): random hidden states, codes and masks, fp32 on the CPU.

Real checkpoint: every depth call of a greedy TTS run and a greedy S2S run of the fp32 reference
on --device, with injected MaskGIT orders, replayed through fp32, fp16 and q8 graphs built from the
checkpoint on --ep. Scored per codebook row ([calls x 12, 2017], at least 1332 rows) by logits rel
L2, argmax agreement and KL against the bf16 reference's (the floor). test_exported_depth_logits
scores the export's depth_decoder.onnx (--exports-dir; every bundle runs it) the same way, to fp32's
bounds.

Run with:
    uv run pytest tests/test_lfm3_audio/test_depth.py -v -k tiny
    uv run pytest tests/test_lfm3_audio/test_depth.py -v --device cuda --log-cli-level=INFO
    uv run pytest tests/test_lfm3_audio/test_depth.py -v \
        --lfm3-audio-model LiquidAI/LFM3-Omni-2B --log-cli-level=INFO
"""

import copy
import logging
import pathlib

import numpy as np
import pytest
import torch
from helpers import processed_probs, total_variation

from liquidonnx.compare.metrics import logit_metrics
from liquidonnx.lfm3.session import load_session
from liquidonnx.lfm3_audio import depth
from liquidonnx.lfm3_audio.config import load_config
from liquidonnx.lfm3_audio.export import DEPTH_QUANT
from liquidonnx.quantize import convert_to_fp16, load_model, quantize_matmuls, save_model

from .remote import inject_orders, maskgit_orders

logger = logging.getLogger(__name__)

SAMPLE = pathlib.Path(__file__).parents[2] / "samples" / "audio" / "woodworks_question.wav"
MIN_ROWS = 1332  # 37 frames x 3 passes x 12 codebooks
# precision -> (max logits rel L2, min argmax agreement, max kl_mean over the bf16 floor's). fp16's
# and q8's argmax misses sit at near-ties of the reference, which the bf16 reference flips too;
# fp16's come from fp16 activations in the trunk, q8's also from int8 activations on the CPU EP.
# Only fp32 keeps every greedy code, so every bundle runs the fp32 depth decoder.
TOLERANCES = {
    "fp32": (1e-5, 1.0, 1e-6),
    "fp16": (2e-2, 0.985, 0.1),
    "q8": (5e-2, 0.95, 2.0),
}
# Max mean total variation of the codebook distributions at LFM3-Omni-2B's speech sampling
# (temperature SAMPLED_TEMPERATURE; the repetition penalty needs a history the replayed rows lack),
# over the bf16 floor's: q8 may reach bf16's, fp16 half of it.
MAX_SAMPLED_TV = {"fp32": 1e-3, "fp16": 0.5, "q8": 1.0}
SAMPLED_TEMPERATURE = 0.4


def rel(reference: np.ndarray, actual: np.ndarray) -> float:
    reference, actual = reference.astype(np.float64), actual.astype(np.float64)
    return float(np.linalg.norm(actual - reference) / np.linalg.norm(reference))


def run_depth(session, calls: dict) -> np.ndarray:
    inputs = {name: calls[name] for name in depth.INPUTS}
    return session.run([depth.OUTPUT], inputs)[0]


def sampled_tv(reference: np.ndarray, actual: np.ndarray) -> float:
    """Mean total variation of rows [N, V] after SAMPLED_TEMPERATURE."""
    p = processed_probs(reference, SAMPLED_TEMPERATURE)
    return float(total_variation(p, processed_probs(actual, SAMPLED_TEMPERATURE)).mean())


def report(name: str, reference: np.ndarray, actual: np.ndarray) -> dict:
    """rel L2 and logit_metrics over the codebook rows [N, 12, V] → [N*12, V]."""
    vocab = reference.shape[-1]
    metrics = logit_metrics(reference.reshape(-1, vocab), actual.reshape(-1, vocab))
    metrics["rel"] = rel(reference, actual)
    metrics["sampled_tv"] = sampled_tv(reference.reshape(-1, vocab), actual.reshape(-1, vocab))
    logger.info(
        f"{name}: rel {metrics['rel']:.2e} kl_mean {metrics['kl_mean']:.2e} "
        f"kl_max {metrics['kl_max']:.2e} top1 {metrics['top1']:.4f} "
        f"({round(metrics['top1'] * metrics['n'])}/{metrics['n']}) max_abs {metrics['max_abs']:.2e} "
        f"sampled TV {metrics['sampled_tv']:.2e}"
    )
    return metrics


def build_graphs(checkpoint: pathlib.Path, out: pathlib.Path) -> dict[str, pathlib.Path]:
    """fp32 depth_decoder.onnx and its fp16 and q8 derivatives (export.derive's recipe)."""
    paths = {p: out / f"depth_decoder{'' if p == 'fp32' else f'_{p}'}.onnx" for p in TOLERANCES}
    builder = depth.LFM3AudioDepthBuilder(load_config(checkpoint))
    save_model(builder.build(checkpoint), paths["fp32"])
    convert_to_fp16(paths["fp32"], paths["fp16"], keep_io=True)
    quantized = quantize_matmuls(
        load_model(paths["fp32"]), **DEPTH_QUANT, exclude=[depth.HEAD_NODE]
    )
    save_model(quantized, paths["q8"])
    return paths


# === Tiny checkpoint ===


def test_tiny_depth(tiny_checkpoint, tiny_reference, tmp_path):
    config = load_config(tiny_checkpoint)
    path = tmp_path / "depth_decoder.onnx"
    save_model(depth.LFM3AudioDepthBuilder(config).build(tiny_checkpoint), path)

    rng = np.random.default_rng(0)
    batch = 5
    calls = {
        "hidden_states": rng.standard_normal((batch, config.language.hidden_size), np.float32),
        "codes": rng.integers(0, config.codebook_size, (batch, config.num_codebooks)),
        "masked": rng.random((batch, config.num_codebooks)) < 0.5,
    }
    calls["masked"][0] = True  # a frame's first pass
    with torch.no_grad():
        expected = tiny_reference.model.depth_decoder(
            *(torch.as_tensor(calls[name]) for name in depth.INPUTS)
        ).numpy()
    actual = run_depth(load_session(path, "cpu"), calls)
    assert actual.shape == (batch, config.num_codebooks, config.codebook_size)
    assert rel(expected, actual) < 1e-5
    assert np.array_equal(expected.argmax(-1), actual.argmax(-1))


# === Real checkpoint ===


@pytest.fixture(scope="module")
def depth_graphs(checkpoint, tmp_path_factory) -> dict[str, pathlib.Path]:
    return build_graphs(checkpoint, tmp_path_factory.mktemp("lfm3_audio_depth"))


def generate_frames(reference, processor, conversation: list[dict], mode: str, seed: int, **kw):
    """Greedy reference generation with injected MaskGIT orders: the inputs and logits of
    every depth decoder call."""
    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(reference.device)
    calls = []

    def record(module, args, output):
        calls.append([t.detach().cpu().numpy() for t in (*args, output)])

    hook = reference.model.depth_decoder.register_forward_hook(record)
    try:
        with torch.no_grad(), inject_orders(maskgit_orders(seed, 1000)):
            reference.generate(**inputs, generation_mode=mode, return_audio_eos=True, **kw)
    finally:
        hook.remove()
    return calls


def capture_depth_calls(reference, processor) -> dict[str, np.ndarray]:
    """hidden_states, codes, masked and the reference logits of every depth call of a TTS and an
    S2S run, stacked over the calls."""
    import scipy.io.wavfile

    rate, wave = scipy.io.wavfile.read(SAMPLE)
    assert rate == 16000
    wave = (wave / 32768.0).astype(np.float32)
    tts = [
        {"role": "system", "content": "Perform TTS in American English using a female voice."},
        {"role": "user", "content": "The quick brown fox jumps over the lazy dog"},
    ]
    s2s = [
        {"role": "system", "content": "Respond with interleaved text and audio."},
        {"role": "user", "content": [{"type": "audio", "audio": wave}]},
    ]
    calls = generate_frames(reference, processor, tts, "audio", 20261024, max_new_tokens=200)
    calls += generate_frames(reference, processor, s2s, "interleaved", 3, max_new_tokens=80)
    hidden, codes, masked, logits = (np.concatenate(parts) for parts in zip(*calls, strict=True))
    logger.info(f"{len(calls)} depth calls, {logits.shape[0] * logits.shape[1]} codebook rows")
    assert logits.shape[0] * logits.shape[1] >= MIN_ROWS
    return {"hidden_states": hidden, "codes": codes, "masked": masked, "logits": logits}


def bf16_logits(reference, calls: dict[str, np.ndarray]) -> np.ndarray:
    """The bf16 reference depth decoder's logits of the captured calls."""
    decoder = copy.deepcopy(reference.model.depth_decoder).to(torch.bfloat16)
    device = reference.device
    with torch.no_grad():
        logits = decoder(
            torch.as_tensor(calls["hidden_states"], device=device, dtype=torch.bfloat16),
            torch.as_tensor(calls["codes"], device=device),
            torch.as_tensor(calls["masked"], device=device),
        )
    return logits.float().cpu().numpy()


@pytest.fixture(scope="module")
def depth_calls(reference, reference_processor) -> dict[str, np.ndarray]:
    return capture_depth_calls(reference, reference_processor)


@pytest.fixture(scope="module")
def bf16_floor(reference, depth_calls) -> dict:
    """The bf16 reference scored against fp32 on the same inputs."""
    return report("bf16 floor", depth_calls["logits"], bf16_logits(reference, depth_calls))


def assert_depth_logits(path: pathlib.Path, precision: str, depth_calls, bf16_floor, ep):
    session = load_session(path, ep, tf32=False)
    metrics = report(f"{path.name} on {ep}", depth_calls["logits"], run_depth(session, depth_calls))
    max_rel, min_top1, max_kl = TOLERANCES[precision]
    assert metrics["rel"] <= max_rel, f"{precision} rel {metrics['rel']:.2e} > {max_rel:.0e}"
    assert metrics["top1"] >= min_top1, (
        f"{precision} argmax agreement {metrics['top1']:.4f} < {min_top1:.4f} "
        f"(bf16 floor {bf16_floor['top1']:.4f})"
    )
    kl = metrics["kl_mean"] / bf16_floor["kl_mean"]
    assert kl <= max_kl, f"{precision} kl_mean {kl:.2e} x the bf16 floor's > {max_kl:.0e}"
    tv = metrics["sampled_tv"] / bf16_floor["sampled_tv"]
    logger.info(f"{precision} sampled TV {tv:.2e} x the bf16 floor's")
    assert tv <= MAX_SAMPLED_TV[precision]


@pytest.mark.parametrize("precision", list(TOLERANCES))
def test_depth_logits(precision, depth_graphs, depth_calls, bf16_floor, ep):
    assert_depth_logits(depth_graphs[precision], precision, depth_calls, bf16_floor, ep)


def test_exported_depth_logits(export_dir, depth_calls, bf16_floor, ep):
    """The export's depth_decoder.onnx, the one every bundle runs, held to fp32's bounds."""
    path = export_dir / "onnx" / "depth_decoder.onnx"
    assert_depth_logits(path, "fp32", depth_calls, bf16_floor, ep)
