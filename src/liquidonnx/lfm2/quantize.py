"""
INT4/INT8 Quantization for LFM2 ONNX models.

Provides quantize_int4() and quantize_int8() functions for converting
FP32 ONNX models to INT4/INT8 using MatMulNBits quantization.

By default, lm_head is kept in FP32 (matches community approach).
Use exclude_lm_head=False to quantize it as well.
"""

import logging
import pathlib

import onnx
from onnxruntime.quantization.matmul_nbits_quantizer import (
    DefaultWeightOnlyQuantConfig,
    MatMulNBitsQuantizer,
)

logger = logging.getLogger(__name__)


def find_lm_head_node(model) -> str | None:
    """Find the lm_head MatMul node name."""
    for node in model.graph.node:
        if node.op_type == "MatMul":
            # Check if any input contains lm_head weight
            for inp in node.input:
                if "lm_head" in inp.lower():
                    return node.name
    return None


def quantize_int4(
    model_path: pathlib.Path,
    output_path: pathlib.Path,
    block_size: int = 32,
    exclude_lm_head: bool = True,
):
    """Quantize model to INT4 using MatMulNBits.

    By default, lm_head is kept in FP32 (matches community approach).
    Use exclude_lm_head=False to quantize it as well.
    """
    logger.info(f"Loading {model_path}...")
    model = onnx.load(str(model_path))

    # Load external data if present
    external_data = model_path.with_suffix(".onnx_data")
    if external_data.exists():
        onnx.load_external_data_for_model(model, str(model_path.parent))

    # Find nodes to exclude (by default exclude lm_head)
    nodes_to_exclude = None
    if exclude_lm_head:
        lm_head_node = find_lm_head_node(model)
        if lm_head_node:
            nodes_to_exclude = [lm_head_node]
            logger.info(f"Keeping lm_head in FP32 (excluding: {lm_head_node})")
        else:
            logger.warning("Could not find lm_head node")
    else:
        logger.info("Quantizing all layers including lm_head")

    logger.info(f"Quantizing to INT4 (block_size={block_size})...")
    quantizer = MatMulNBitsQuantizer(
        model,
        block_size=block_size,
        is_symmetric=True,
        accuracy_level=4,
        nodes_to_exclude=nodes_to_exclude,
    )
    quantizer.process()

    logger.info(f"Saving to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get the quantized model and save with onnx.save_model for compact external data
    # The quantizer's save_model_to_file can create bloated external data files
    quantized_model = quantizer.model.model

    # Remove any existing external data file to avoid appending
    external_data_path = output_path.parent / (output_path.stem + ".onnx_data")
    if external_data_path.exists():
        external_data_path.unlink()

    onnx.save_model(
        quantized_model,
        str(output_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=output_path.stem + ".onnx_data",
        size_threshold=1024,  # Keep small tensors inline for ONNX Runtime compatibility
        convert_attribute=False,
    )

    return output_path


def quantize_int8(
    model_path: pathlib.Path,
    output_path: pathlib.Path,
    block_size: int = 32,
    exclude_lm_head: bool = True,
):
    """Quantize model to INT8 using MatMulNBits (same approach as INT4).

    By default, lm_head is kept in FP32 (matches INT4 approach).
    Use exclude_lm_head=False to quantize it as well.
    """
    logger.info(f"Loading {model_path}...")
    model = onnx.load(str(model_path))

    # Load external data if present
    external_data = model_path.with_suffix(".onnx_data")
    if external_data.exists():
        onnx.load_external_data_for_model(model, str(model_path.parent))

    # Find nodes to exclude (by default exclude lm_head)
    nodes_to_exclude = None
    if exclude_lm_head:
        lm_head_node = find_lm_head_node(model)
        if lm_head_node:
            nodes_to_exclude = [lm_head_node]
            logger.info(f"Keeping lm_head in FP32 (excluding: {lm_head_node})")
        else:
            logger.warning("Could not find lm_head node")
    else:
        logger.info("Quantizing all layers including lm_head")

    logger.info(f"Quantizing to INT8 (block_size={block_size})...")
    algo_config = DefaultWeightOnlyQuantConfig(
        block_size=block_size,
        is_symmetric=True,
        accuracy_level=4,
        bits=8,  # INT8 instead of INT4
    )
    quantizer = MatMulNBitsQuantizer(
        model,
        block_size=block_size,
        is_symmetric=True,
        accuracy_level=4,
        nodes_to_exclude=nodes_to_exclude,
        algo_config=algo_config,
    )
    quantizer.process()

    logger.info(f"Saving to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get the quantized model and save with onnx.save_model for compact external data
    quantized_model = quantizer.model.model

    # Remove any existing external data file to avoid appending
    external_data_path = output_path.parent / (output_path.stem + ".onnx_data")
    if external_data_path.exists():
        external_data_path.unlink()

    onnx.save_model(
        quantized_model,
        str(output_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=output_path.stem + ".onnx_data",
        size_threshold=1024,  # Keep small tensors inline for ONNX Runtime compatibility
        convert_attribute=False,
    )

    return output_path


def get_model_size(path: pathlib.Path) -> tuple[float, float]:
    """Return (model_mb, data_gb)."""
    model_size = path.stat().st_size / 1e6 if path.exists() else 0
    data_path = path.with_suffix(".onnx_data")
    data_size = data_path.stat().st_size / 1e9 if data_path.exists() else 0
    return model_size, data_size
