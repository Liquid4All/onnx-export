"""Shared test utilities for LFM2 tests."""

import pathlib

import numpy as np
import onnxruntime as ort
import pytest


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
        # Try model.onnx first, then decoder_fp32.onnx for backwards compat
        fp32_path = onnx_dir / "model.onnx"
        if fp32_path.exists():
            return fp32_path
        return onnx_dir / "decoder_fp32.onnx"
    return onnx_dir / f"model_q{bits}.onnx"


def skip_if_missing(path: pathlib.Path, reason: str = "File not found"):
    """Skip test if path doesn't exist."""
    if not path.exists():
        pytest.skip(f"{reason}: {path}")


def load_onnx_session(onnx_path: pathlib.Path) -> ort.InferenceSession:
    """Load ONNX model as inference session."""
    return ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
