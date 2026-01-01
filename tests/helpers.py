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


def get_community_vl_onnx_dir(community_dir: pathlib.Path, size: str) -> pathlib.Path:
    """Get onnx-community VL model directory."""
    return community_dir / f"LFM2-VL-{size}-ONNX" / "onnx"


def get_community_vl_files(
    onnx_dir: pathlib.Path, use_fp16: bool = False
) -> dict[str, pathlib.Path]:
    """Get onnx-community VL model files.

    Community VL models use different naming:
    - embed_tokens.onnx / embed_tokens_fp16.onnx
    - vision_encoder.onnx / vision_encoder_fp16.onnx
    - decoder_model_merged.onnx / decoder_model_merged_fp16.onnx
    """
    suffix = "_fp16" if use_fp16 else ""
    return {
        "embed_tokens": onnx_dir / f"embed_tokens{suffix}.onnx",
        "vision_encoder": onnx_dir / f"vision_encoder{suffix}.onnx",
        "decoder": onnx_dir / f"decoder_model_merged{suffix}.onnx",
    }


def get_local_vl_files(onnx_dir: pathlib.Path, use_fp16: bool = False) -> dict[str, pathlib.Path]:
    """Get local VL model files.

    Local VL models use:
    - embed_tokens.onnx / embed_tokens_fp16.onnx
    - embed_images.onnx / embed_images_fp16.onnx
    - decoder.onnx / decoder_fp16.onnx
    """
    suffix = "_fp16" if use_fp16 else ""
    return {
        "embed_tokens": onnx_dir / f"embed_tokens{suffix}.onnx",
        "embed_images": onnx_dir / f"embed_images{suffix}.onnx",
        "decoder": onnx_dir / f"decoder{suffix}.onnx",
    }


def get_community_moe_onnx_dir(community_dir: pathlib.Path, size: str) -> pathlib.Path:
    """Get onnx-community MoE model directory."""
    return community_dir / f"LFM2-{size}-ONNX" / "onnx"


def get_community_moe_onnx_file(onnx_dir: pathlib.Path, precision: str | None) -> pathlib.Path:
    """Get onnx-community MoE model file."""
    if precision is None:
        return onnx_dir / "model.onnx"
    if precision == "fp16":
        return onnx_dir / "model_fp16.onnx"
    return onnx_dir / f"model_{precision}.onnx"
