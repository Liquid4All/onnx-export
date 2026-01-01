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


_cuda_works = None  # Cache CUDA availability check


def get_providers() -> list[str]:
    """Get available execution providers, preferring CUDA if it works."""
    global _cuda_works

    available = ort.get_available_providers()
    if "CUDAExecutionProvider" not in available:
        return ["CPUExecutionProvider"]

    # Check if CUDA actually works (cuDNN available, etc.)
    if _cuda_works is None:
        try:
            # Create a minimal session to test CUDA
            import tempfile

            from onnx import TensorProto, helper

            # Minimal valid ONNX model
            X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1])
            Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1])
            node = helper.make_node("Identity", ["X"], ["Y"])
            graph = helper.make_graph([node], "test", [X], [Y])
            model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

            with tempfile.NamedTemporaryFile(suffix=".onnx", delete=True) as f:
                import onnx
                onnx.save(model, f.name)
                ort.InferenceSession(f.name, providers=["CUDAExecutionProvider"])
            _cuda_works = True
            logger.info("CUDA execution provider available")
        except Exception as e:
            _cuda_works = False
            logger.info(f"CUDA not available, using CPU: {e}")

    if _cuda_works:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def load_onnx_session(path: pathlib.Path, providers: list[str] | None = None) -> ort.InferenceSession:
    """Load ONNX model as inference session.

    Args:
        path: Path to ONNX model
        providers: Execution providers to use. If None, auto-detects (CUDA if available, else CPU).

    Returns:
        ONNX Runtime InferenceSession
    """
    if not path.exists():
        raise FileNotFoundError(f"ONNX file not found: {path}")
    if providers is None:
        providers = get_providers()

    # Try with preferred providers, fallback to CPU if CUDA fails
    try:
        logger.info(f"Loading {path.name} with {providers[0]}...")
        return ort.InferenceSession(str(path), providers=providers)
    except Exception as e:
        if "CUDAExecutionProvider" in providers and "CPUExecutionProvider" in providers:
            logger.warning(f"CUDA failed ({e}), falling back to CPU...")
            return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        raise


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
