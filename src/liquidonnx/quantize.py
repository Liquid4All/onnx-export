"""
INT4/INT8 Quantization for ONNX models.

Provides quantize_model() for converting FP32 ONNX models to INT4/INT8
using MatMulNBits quantization.
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
            for inp in node.input:
                if "lm_head" in inp.lower():
                    return node.name
    return None


def get_model_size(path: pathlib.Path) -> tuple[float, float]:
    """Return (model_mb, data_mb)."""
    model_size = path.stat().st_size / 1e6 if path.exists() else 0
    data_path = path.with_suffix(".onnx_data")
    data_size = data_path.stat().st_size / 1e6 if data_path.exists() else 0
    return model_size, data_size


def get_total_model_size_mb(path: pathlib.Path) -> float:
    """Return total model size in MB (model + all external data files).

    Handles split external data files (e.g., model.onnx_data, model.onnx_data_1).
    """
    total = path.stat().st_size / 1e6 if path.exists() else 0

    # Check for external data files (model.onnx_data, model.onnx_data_1, etc.)
    base_data = path.with_suffix(".onnx_data")
    if base_data.exists():
        total += base_data.stat().st_size / 1e6

    # Check for split data files
    i = 1
    while True:
        split_data = path.parent / f"{path.stem}.onnx_data_{i}"
        if not split_data.exists():
            break
        total += split_data.stat().st_size / 1e6
        i += 1

    return total


def _rename_quantized_weights(model: onnx.ModelProto):
    """Rename quantized weight initializers to match community convention.

    Transforms:
      - model.layers.X.Y.MatMul.weight_Q4 -> model_layers_X_Y_MatMul_weight_quant
      - model.layers.X.Y.MatMul.weight_scales -> model_layers_X_Y_MatMul_weight_scales
      - model.layers.X.Y.MatMul.weight_zero_points -> model_layers_X_Y_MatMul_weight_zp

    The onnxruntime quantizer uses dots and _Q4/_zero_points suffixes,
    community uses underscores and _quant/_zp suffixes.
    """
    graph = model.graph
    renames = {}

    for init in graph.initializer:
        old_name = init.name
        # Only rename quantized weight tensors (contain MatMul and are quantized)
        if "MatMul" not in old_name:
            continue
        if not any(suffix in old_name for suffix in ["_Q4", "_scales", "_zero_points"]):
            continue

        # Convert dots to underscores for quantized weights
        new_name = old_name.replace(".", "_")
        # Rename suffixes to match community
        new_name = new_name.replace("_Q4", "_quant")
        new_name = new_name.replace("_zero_points", "_zp")

        if new_name != old_name:
            renames[old_name] = new_name
            init.name = new_name

    # Update node inputs that reference renamed initializers
    for node in graph.node:
        for i, inp in enumerate(node.input):
            if inp in renames:
                node.input[i] = renames[inp]


def quantize_model(
    model_path: pathlib.Path,
    output_path: pathlib.Path,
    *,
    bits: int = 4,
    block_size: int = 32,
    exclude_lm_head: bool = True,
    symmetric: bool = False,
) -> pathlib.Path:
    """Quantize ONNX model to INT4 or INT8 using MatMulNBits.

    By default, lm_head is kept in FP32 (matches community approach).
    Use exclude_lm_head=False to quantize it as well.

    Args:
        model_path: Input ONNX model path
        output_path: Output quantized model path
        bits: Quantization bits (4 or 8)
        block_size: Block size for quantization
        exclude_lm_head: Keep lm_head in FP32
        symmetric: Use symmetric quantization (no zero points). Default False matches community.
    """
    logger.info(f"Loading {model_path}...")
    model = onnx.load(str(model_path))

    external_data = model_path.with_suffix(".onnx_data")
    if external_data.exists():
        onnx.load_external_data_for_model(model, str(model_path.parent))

    nodes_to_exclude = None
    if exclude_lm_head:
        lm_head_node = find_lm_head_node(model)
        if lm_head_node:
            nodes_to_exclude = [lm_head_node]
            logger.info(f"Keeping lm_head in FP32 (excluding: {lm_head_node})")
        else:
            logger.warning("Could not find lm_head node")

    quant_type = "symmetric" if symmetric else "asymmetric"
    logger.info(f"Quantizing to INT{bits} (block_size={block_size}, {quant_type})...")

    if bits == 4:
        quantizer = MatMulNBitsQuantizer(
            model,
            block_size=block_size,
            is_symmetric=symmetric,
            accuracy_level=4,
            nodes_to_exclude=nodes_to_exclude,
        )
    else:
        algo_config = DefaultWeightOnlyQuantConfig(
            block_size=block_size,
            is_symmetric=symmetric,
            accuracy_level=4,
            bits=bits,
        )
        quantizer = MatMulNBitsQuantizer(
            model,
            block_size=block_size,
            is_symmetric=symmetric,
            accuracy_level=4,
            nodes_to_exclude=nodes_to_exclude,
            algo_config=algo_config,
        )

    quantizer.process()

    logger.info(f"Saving to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    quantized_model = quantizer.model.model

    # Rename quantized weights to match community convention
    _rename_quantized_weights(quantized_model)

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
        convert_attribute=False,
    )

    return output_path
