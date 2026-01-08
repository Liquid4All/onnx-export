"""Shared test utilities."""

import pathlib

import pytest

# === LFM2 Model Mappings ===

# HuggingFace model ID -> onnx-community repo
COMMUNITY_MODELS = {
    "LiquidAI/LFM2-350M": "onnx-community/LFM2-350M-ONNX",
    "LiquidAI/LFM2-700M": "onnx-community/LFM2-700M-ONNX",
    "LiquidAI/LFM2-1.2B": "onnx-community/LFM2-1.2B-ONNX",
    "LiquidAI/LFM2-2.6B": "onnx-community/LFM2-2.6B-ONNX",
}

# === LFM2-MoE Model Mappings ===

COMMUNITY_MOE_MODELS = {
    "LiquidAI/LFM2-8B-A1B": "onnx-community/LFM2-8B-A1B-ONNX",
}


def get_model_name(model_id: str) -> str:
    """Extract model name from HF slug (e.g., 'LiquidAI/LFM2-350M' -> 'LFM2-350M')."""
    return model_id.split("/")[-1]


def get_community_model_id(model_id: str) -> str | None:
    """Get onnx-community HF repo for a model, or None if not available."""
    return COMMUNITY_MODELS.get(model_id)


def get_onnx_dir(exports_dir: pathlib.Path, model_id: str) -> pathlib.Path:
    """Get ONNX directory for a model."""
    model_name = get_model_name(model_id)
    return exports_dir / f"{model_name}-ONNX" / "onnx"


def skip_if_missing(path: pathlib.Path, reason: str = "File not found"):
    """Skip test if path doesn't exist."""
    if not path.exists():
        pytest.skip(f"{reason}: {path}")


def get_community_onnx_dir(community_dir: pathlib.Path, model_id: str) -> pathlib.Path:
    """Get onnx-community model directory for a HF model ID."""
    model_name = get_model_name(model_id)
    return community_dir / f"{model_name}-ONNX" / "onnx"


def get_community_onnx_file(onnx_dir: pathlib.Path, precision: str | None) -> pathlib.Path:
    """Get onnx-community model file."""
    if precision is None:
        return onnx_dir / "model.onnx"
    return onnx_dir / f"model_{precision}.onnx"


def download_community_onnx(model_id: str, precision: str | None) -> pathlib.Path | None:
    """Download community ONNX file from HuggingFace if available.

    Returns path to downloaded file, or None if not found.
    """
    from huggingface_hub import hf_hub_download, list_repo_files
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

    community_id = get_community_model_id(model_id)
    if not community_id:
        return None

    filename = "model.onnx" if precision is None else f"model_{precision}.onnx"
    onnx_path = f"onnx/{filename}"

    try:
        local_path = hf_hub_download(repo_id=community_id, filename=onnx_path)
    except (EntryNotFoundError, RepositoryNotFoundError):
        return None

    # Download all associated data files (model.onnx_data, model.onnx_data_1, etc.)
    repo_files = list_repo_files(repo_id=community_id)
    data_files = [f for f in repo_files if f.startswith(f"{onnx_path}_data")]
    for data_file in data_files:
        hf_hub_download(repo_id=community_id, filename=data_file)

    return pathlib.Path(local_path)


def get_community_moe_model_id(model_id: str) -> str | None:
    """Get onnx-community HF repo for a MoE model, or None if not available."""
    return COMMUNITY_MOE_MODELS.get(model_id)


def download_community_moe_onnx(model_id: str, precision: str | None) -> pathlib.Path | None:
    """Download community MoE ONNX file from HuggingFace if available.

    Returns path to downloaded file, or None if not found.
    """
    from huggingface_hub import hf_hub_download, list_repo_files
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

    community_id = get_community_moe_model_id(model_id)
    if not community_id:
        return None

    filename = "model.onnx" if precision is None else f"model_{precision}.onnx"
    onnx_path = f"onnx/{filename}"

    try:
        local_path = hf_hub_download(repo_id=community_id, filename=onnx_path)
    except (EntryNotFoundError, RepositoryNotFoundError):
        return None

    # Download all associated data files (model.onnx_data, model.onnx_data_1, etc.)
    repo_files = list_repo_files(repo_id=community_id)
    data_files = [f for f in repo_files if f.startswith(f"{onnx_path}_data")]
    for data_file in data_files:
        hf_hub_download(repo_id=community_id, filename=data_file)

    return pathlib.Path(local_path)


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
    """Get onnx-community MoE model file.

    Args:
        onnx_dir: ONNX directory
        precision: None for fp32, "fp16", "q4", "q4f16"
    """
    if precision is None:
        return onnx_dir / "model.onnx"
    return onnx_dir / f"model_{precision}.onnx"
