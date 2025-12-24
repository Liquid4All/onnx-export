"""LFM2-VL test helpers."""

import logging
import pathlib
from dataclasses import dataclass

import numpy as np
import onnxruntime as ort
import pytest
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

logger = logging.getLogger(__name__)

ATOL = 1e-3
RTOL = 1e-2
ATOL_QUANT = 0.5
RTOL_QUANT = 0.5


def bits_to_str(bits: int | None) -> str:
    return f"q{bits}" if bits else "fp32"


def pad_to_square(image: Image.Image) -> Image.Image:
    """Pad image to square with black borders, centered."""
    w, h = image.size
    if w == h:
        return image
    max_dim = max(w, h)
    square_img = Image.new('RGB', (max_dim, max_dim), (0, 0, 0))
    paste_x = (max_dim - w) // 2
    paste_y = (max_dim - h) // 2
    square_img.paste(image, (paste_x, paste_y))
    return square_img


@dataclass
class VerificationResult:
    name: str
    passed: bool
    max_diff: float
    mean_diff: float
    correlation: float
    details: str = ""


def skip_if_missing(path: pathlib.Path, reason: str):
    if not path.exists():
        pytest.skip(f"{reason}: {path}")


def get_onnx_file(onnx_dir: pathlib.Path, name: str, bits: int | None) -> pathlib.Path:
    if bits:
        return onnx_dir / "onnx" / f"{name}_q{bits}.onnx"
    return onnx_dir / "onnx" / f"{name}_fp32.onnx"


def get_vl_onnx_dir(exports_dir: pathlib.Path, size: str, vision_mode: str) -> pathlib.Path:
    return exports_dir / f"LFM2-VL-{size}-ONNX-{vision_mode}"


def get_tolerances(bits: int | None) -> tuple[float, float]:
    if bits:
        return ATOL_QUANT, RTOL_QUANT
    return ATOL, RTOL


def load_pytorch_model(model_path: str) -> tuple:
    logger.info(f"Loading PyTorch model from {model_path}...")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    model.eval()
    return model, processor


def load_onnx_session(onnx_dir: pathlib.Path, filename: str) -> ort.InferenceSession:
    path = onnx_dir / "onnx" / filename
    if not path.exists():
        raise FileNotFoundError(f"{filename} not found in {onnx_dir / 'onnx'}")
    logger.info(f"Loading {filename}...")
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def compare_arrays(name: str, expected: np.ndarray, actual: np.ndarray,
                   atol: float, rtol: float) -> VerificationResult:
    if expected.shape != actual.shape:
        return VerificationResult(
            name=name, passed=False,
            max_diff=float('inf'), mean_diff=float('inf'), correlation=0.0,
            details=f"Shape mismatch: {expected.shape} vs {actual.shape}"
        )

    diff = np.abs(expected - actual)
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())
    correlation = float(np.corrcoef(expected.flatten(), actual.flatten())[0, 1])
    passed = np.allclose(expected, actual, atol=atol, rtol=rtol)

    return VerificationResult(
        name=name, passed=passed,
        max_diff=max_diff, mean_diff=mean_diff, correlation=correlation,
    )


def assert_results(results: list[VerificationResult], logger=None):
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        if logger:
            logger.info(f"  {r.name}: {status} max_diff={r.max_diff:.6f} corr={r.correlation:.4f}")
            if r.details:
                logger.info(f"    {r.details}")
        assert r.passed, f"{r.name}: max_diff={r.max_diff:.6f}, corr={r.correlation:.4f}, {r.details}"


def compare_top_k(name: str, expected: np.ndarray, actual: np.ndarray,
                  k: int = 5) -> VerificationResult:
    """Compare top-k predictions between expected and actual logits."""
    exp_logits = expected[0, -1]
    act_logits = actual[0, -1]

    exp_top_k = np.argsort(exp_logits)[-k:][::-1]
    act_top_k = np.argsort(act_logits)[-k:][::-1]

    top1_match = exp_top_k[0] == act_top_k[0]
    top_k_overlap = len(set(exp_top_k) & set(act_top_k))

    return VerificationResult(
        name=name, passed=top1_match,
        max_diff=0.0 if top1_match else 1.0,
        mean_diff=1.0 - (top_k_overlap / k),
        correlation=top_k_overlap / k,
        details=f"Top-1 match: {top1_match}, Top-{k} overlap: {top_k_overlap}/{k}, "
                f"Expected: {exp_top_k.tolist()}, Actual: {act_top_k.tolist()}"
    )


def compare_correlation(name: str, expected: np.ndarray, actual: np.ndarray,
                        threshold: float) -> VerificationResult:
    """Check correlation between arrays (for quantized models where exact match isn't expected)."""
    if expected.shape != actual.shape:
        return VerificationResult(
            name=name, passed=False,
            max_diff=float('inf'), mean_diff=float('inf'), correlation=0.0,
            details=f"Shape mismatch: {expected.shape} vs {actual.shape}"
        )

    diff = np.abs(expected - actual)
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())
    correlation = float(np.corrcoef(expected.flatten(), actual.flatten())[0, 1])
    passed = correlation >= threshold

    return VerificationResult(
        name=name, passed=passed,
        max_diff=max_diff, mean_diff=mean_diff, correlation=correlation,
        details=f"threshold={threshold}"
    )
