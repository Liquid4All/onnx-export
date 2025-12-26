"""Shared test utilities."""

import pathlib

import pytest


def skip_if_missing(path: pathlib.Path, reason: str = "File not found"):
    """Skip test if path doesn't exist."""
    if not path.exists():
        pytest.skip(f"{reason}: {path}")


def get_community_onnx_dir(community_dir: pathlib.Path, size: str) -> pathlib.Path:
    """Get onnx-community model directory."""
    return community_dir / f"LFM2-{size}-ONNX" / "onnx"


def get_community_onnx_file(onnx_dir: pathlib.Path, bits: int | None) -> pathlib.Path:
    """Get onnx-community model file."""
    if bits is None:
        return onnx_dir / "model.onnx"
    return onnx_dir / f"model_q{bits}.onnx"
