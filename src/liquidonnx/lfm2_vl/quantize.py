"""INT4/INT8 Quantization for LFM2-VL ONNX models."""

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


def quantize_model(
    model_path: pathlib.Path,
    output_path: pathlib.Path,
    bits: int = 4,
    block_size: int = 32,
    exclude_lm_head: bool = False,
) -> pathlib.Path:
    """Quantize a single ONNX model to INT4 or INT8."""
    logger.info(f"Loading {model_path}...")
    model = onnx.load(str(model_path))

    # Load external data if present
    external_data = model_path.with_suffix(".onnx_data")
    if external_data.exists():
        onnx.load_external_data_for_model(model, str(model_path.parent))

    # Find nodes to exclude (lm_head for decoder)
    nodes_to_exclude = None
    if exclude_lm_head:
        lm_head_node = find_lm_head_node(model)
        if lm_head_node:
            nodes_to_exclude = [lm_head_node]
            logger.info(f"Keeping lm_head in FP32 (excluding: {lm_head_node})")
        else:
            logger.warning("Could not find lm_head node")

    logger.info(f"Quantizing to INT{bits} (block_size={block_size})...")

    if bits == 4:
        quantizer = MatMulNBitsQuantizer(
            model,
            block_size=block_size,
            is_symmetric=True,
            accuracy_level=4,
            nodes_to_exclude=nodes_to_exclude,
        )
    else:
        algo_config = DefaultWeightOnlyQuantConfig(
            block_size=block_size,
            is_symmetric=True,
            accuracy_level=4,
            bits=8,
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

    quantized_model = quantizer.model.model

    # Remove any existing external data file
    external_data_path = output_path.parent / (output_path.stem + ".onnx_data")
    if external_data_path.exists():
        external_data_path.unlink()

    onnx.save_model(
        quantized_model,
        str(output_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=output_path.stem + ".onnx_data",
        size_threshold=1024,
    )

    return output_path


def get_model_size(path: pathlib.Path) -> tuple[float, float]:
    """Return (model_mb, data_mb)."""
    model_size = path.stat().st_size / 1e6 if path.exists() else 0
    data_path = path.with_suffix(".onnx_data")
    data_size = data_path.stat().st_size / 1e6 if data_path.exists() else 0
    return model_size, data_size
