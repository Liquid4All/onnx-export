"""Shared ONNX inference utilities."""

import logging
import pathlib

import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)


def get_onnx_file(
    onnx_dir: pathlib.Path, precision: str | None, name: str = "model"
) -> pathlib.Path:
    """Get ONNX file path for given precision.

    Args:
        onnx_dir: Directory containing ONNX files
        precision: None for fp32, "fp16", "q4", "q8"
        name: Base name of the model file (default: "model")

    Returns:
        Path to the ONNX file (e.g., model.onnx, model_q4.onnx, decoder_fp16.onnx)
    """
    if precision:
        return onnx_dir / f"{name}_{precision}.onnx"
    return onnx_dir / f"{name}.onnx"


def load_onnx_session(path: pathlib.Path) -> ort.InferenceSession:
    """Load ONNX model as inference session."""
    if not path.exists():
        raise FileNotFoundError(f"ONNX file not found: {path}")
    logger.info(f"Loading {path.name}...")
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


ONNX_TYPE_TO_NUMPY = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
}


def initialize_cache(session: ort.InferenceSession) -> dict:
    """Initialize KV cache tensors for an ONNX inference session.

    Automatically detects cache inputs (past_*) and initializes them with zeros.
    Infers dtype from the ONNX model input specification.
    """
    skip_inputs = {"input_ids", "inputs_embeds", "attention_mask", "position_ids"}
    cache = {}

    for inp in session.get_inputs():
        if inp.name in skip_inputs:
            continue
        shape = [d if isinstance(d, int) else 1 for d in inp.shape]
        for i, d in enumerate(inp.shape):
            if isinstance(d, str) and "sequence" in d.lower():
                shape[i] = 0
        dtype = ONNX_TYPE_TO_NUMPY.get(inp.type, np.float32)
        cache[inp.name] = np.zeros(shape, dtype=dtype)

    return cache


def update_cache(cache: dict, outputs: list, output_infos: list) -> None:
    """Update cache from model outputs.

    Handles present_conv -> past_conv and present. -> past_key_values. mappings.
    """
    for i, out_info in enumerate(output_infos[1:], 1):  # Skip logits
        name = out_info.name
        if "present_conv" in name:
            cache_name = name.replace("present_conv", "past_conv")
        elif "present." in name:
            cache_name = name.replace("present.", "past_key_values.")
        else:
            continue
        if cache_name in cache:
            cache[cache_name] = outputs[i]
