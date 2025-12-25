"""Shared test utilities for LFM2 tests."""

import logging
import pathlib
from dataclasses import dataclass

import numpy as np
import onnxruntime as ort
import pytest

logger = logging.getLogger(__name__)

ATOL = 1e-3
RTOL = 1e-2
ATOL_QUANT = 0.5
RTOL_QUANT = 0.5


@dataclass
class VerificationResult:
    name: str
    passed: bool
    max_diff: float
    mean_diff: float
    correlation: float
    details: str = ""


def bits_to_str(bits: int | None) -> str:
    return f"q{bits}" if bits else "fp32"


def get_tolerances(bits: int | None) -> tuple[float, float]:
    if bits:
        return ATOL_QUANT, RTOL_QUANT
    return ATOL, RTOL


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    a_flat = a.flatten()
    b_flat = b.flatten()
    dot = np.dot(a_flat, b_flat)
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def get_onnx_dir(exports_dir: pathlib.Path, size: str) -> pathlib.Path:
    """Get ONNX directory for a model size."""
    return exports_dir / f"LFM2-{size}-ONNX" / "onnx"


def get_onnx_file(onnx_dir: pathlib.Path, bits: int | None) -> pathlib.Path:
    """Get ONNX model file for given quantization.

    Args:
        onnx_dir: Directory containing ONNX files
        bits: None for fp32, 4 for q4, 8 for q8

    Returns:
        Path to ONNX file
    """
    if bits is None:
        return onnx_dir / "model.onnx"
    return onnx_dir / f"model_q{bits}.onnx"


def skip_if_missing(path: pathlib.Path, reason: str = "File not found"):
    """Skip test if path doesn't exist."""
    if not path.exists():
        pytest.skip(f"{reason}: {path}")


def load_onnx_session(onnx_path: pathlib.Path) -> ort.InferenceSession:
    """Load ONNX model as inference session."""
    return ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])


def compare_arrays(
    name: str, expected: np.ndarray, actual: np.ndarray, atol: float, rtol: float
) -> VerificationResult:
    if expected.shape != actual.shape:
        return VerificationResult(
            name=name,
            passed=False,
            max_diff=float("inf"),
            mean_diff=float("inf"),
            correlation=0.0,
            details=f"Shape mismatch: {expected.shape} vs {actual.shape}",
        )

    diff = np.abs(expected - actual)
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())
    correlation = float(np.corrcoef(expected.flatten(), actual.flatten())[0, 1])
    passed = np.allclose(expected, actual, atol=atol, rtol=rtol)

    return VerificationResult(
        name=name,
        passed=passed,
        max_diff=max_diff,
        mean_diff=mean_diff,
        correlation=correlation,
    )


def compare_top_k(
    name: str, expected: np.ndarray, actual: np.ndarray, k: int = 5, min_overlap: int = 5
) -> VerificationResult:
    """Compare top-k predictions between expected and actual logits.

    Args:
        name: Test name
        expected: Expected logits
        actual: Actual logits
        k: Number of top predictions to compare
        min_overlap: Minimum overlap required to pass (default: k for exact match)
    """
    exp_logits = expected[0, -1]
    act_logits = actual[0, -1]

    exp_top_k = np.argsort(exp_logits)[-k:][::-1]
    act_top_k = np.argsort(act_logits)[-k:][::-1]

    top1_match = exp_top_k[0] == act_top_k[0]
    top_k_overlap = len(set(exp_top_k) & set(act_top_k))
    passed = top_k_overlap >= min_overlap

    return VerificationResult(
        name=name,
        passed=passed,
        max_diff=0.0 if top1_match else 1.0,
        mean_diff=1.0 - (top_k_overlap / k),
        correlation=top_k_overlap / k,
        details=f"Top-1 match: {top1_match}, Top-{k} overlap: {top_k_overlap}/{k}, "
        f"Expected: {exp_top_k.tolist()}, Actual: {act_top_k.tolist()}",
    )


def assert_results(results: list[VerificationResult], log=None):
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        if log:
            log.info(f"  {r.name}: {status} max_diff={r.max_diff:.6f} corr={r.correlation:.4f}")
            if r.details:
                log.info(f"    {r.details}")
        assert r.passed, f"{r.name}: max_diff={r.max_diff:.6f}, corr={r.correlation:.4f}, {r.details}"
