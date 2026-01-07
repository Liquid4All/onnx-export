"""
Model output verification utilities.

Provides functions for comparing model outputs between reference (PyTorch)
and exported (ONNX) models, with tolerance handling for quantized models.
"""

from dataclasses import dataclass

import numpy as np

ATOL = 5e-2  # 0.05 - our fp32 exports have ~0.03 max_diff
RTOL = 5e-2
ATOL_FP16 = 1e-1  # 0.1 - fp16 has less precision than fp32
RTOL_FP16 = 1e-1
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


def get_tolerances(precision: str | None) -> tuple[float, float]:
    """Get (atol, rtol) based on precision."""
    if precision == "fp16":
        return ATOL_FP16, RTOL_FP16
    if precision and precision not in ("fp32", None):
        return ATOL_QUANT, RTOL_QUANT
    return ATOL, RTOL


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two arrays."""
    a_flat = a.flatten()
    b_flat = b.flatten()
    dot = np.dot(a_flat, b_flat)
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def compare_logits_similarity(expected: np.ndarray, actual: np.ndarray) -> float:
    """Compare sequences of logits, returning mean cosine similarity."""
    if len(expected) == 0 or len(actual) == 0:
        return 1.0
    min_steps = min(len(expected), len(actual))
    similarities = [cosine_similarity(expected[i], actual[i]) for i in range(min_steps)]
    return float(np.mean(similarities))


def compare_arrays(
    name: str, expected: np.ndarray, actual: np.ndarray, atol: float, rtol: float
) -> VerificationResult:
    """Compare two arrays with tolerance checking."""
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
    name: str,
    expected: np.ndarray,
    actual: np.ndarray,
    k: int = 5,
    min_overlap: int | None = None,
) -> VerificationResult:
    """Compare top-k predictions between expected and actual logits.

    Args:
        name: Test name
        expected: Expected logits
        actual: Actual logits
        k: Number of top predictions to compare
        min_overlap: Minimum overlap required to pass (default: requires top-1 match)
    """
    exp_logits = expected[0, -1]
    act_logits = actual[0, -1]

    exp_top_k = np.argsort(exp_logits)[-k:][::-1]
    act_top_k = np.argsort(act_logits)[-k:][::-1]

    top1_match = exp_top_k[0] == act_top_k[0]
    top_k_overlap = len(set(exp_top_k) & set(act_top_k))

    if min_overlap is not None:
        passed = top_k_overlap >= min_overlap
    else:
        passed = top1_match

    return VerificationResult(
        name=name,
        passed=passed,
        max_diff=0.0 if top1_match else 1.0,
        mean_diff=1.0 - (top_k_overlap / k),
        correlation=top_k_overlap / k,
        details=f"Top-1 match: {top1_match}, Top-{k} overlap: {top_k_overlap}/{k}, "
        f"Expected: {exp_top_k.tolist()}, Actual: {act_top_k.tolist()}",
    )


def compare_correlation(
    name: str, expected: np.ndarray, actual: np.ndarray, threshold: float
) -> VerificationResult:
    """Check correlation between arrays (for quantized models)."""
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
    passed = correlation >= threshold

    return VerificationResult(
        name=name,
        passed=passed,
        max_diff=max_diff,
        mean_diff=mean_diff,
        correlation=correlation,
        details=f"threshold={threshold}",
    )


class VerificationError(AssertionError):
    """Raised when verification fails."""

    pass


def check_results(results: list[VerificationResult], log=None):
    """Check all verification results, raise VerificationError if any failed."""
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        if log:
            log.info(f"  {r.name}: {status} max_diff={r.max_diff:.6f} corr={r.correlation:.4f}")
            if r.details:
                log.info(f"    {r.details}")
        if not r.passed:
            raise VerificationError(
                f"{r.name}: max_diff={r.max_diff:.6f}, corr={r.correlation:.4f}, {r.details}"
            )
