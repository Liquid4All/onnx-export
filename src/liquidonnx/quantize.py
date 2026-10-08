"""
Quantization and precision conversion for ONNX models.

Weight-only MatMulNBits quantization (quantize_model / quantize_matmuls) for the vision encoder and
the audio graphs, and convert_to_fp16 for their fp16 versions and the fp16 and q4f16 decoders. The
genai builder quantizes the decoders (liquidonnx.genai_builder.DECODER_PRESETS).
"""

import logging
import pathlib

import numpy as np
import onnx
from onnxruntime.quantization.matmul_nbits_quantizer import (
    DefaultWeightOnlyQuantConfig,
    MatMulNBitsQuantizer,
)

logger = logging.getLogger(__name__)

DEFAULT_BLOCK_SIZE = 32
SCALE_EPS = 1e-10


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


def load_model(path: pathlib.Path) -> onnx.ModelProto:
    return onnx.load(str(path), load_external_data=True)


def save_model(model: onnx.ModelProto, path: pathlib.Path) -> pathlib.Path:
    """Save with all tensors in one `{stem}.onnx_data` file next to the graph."""
    path.parent.mkdir(parents=True, exist_ok=True)
    for old in path.parent.glob(f"{path.stem}.onnx_data*"):
        old.unlink()
    onnx.save_model(
        model,
        str(path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=f"{path.stem}.onnx_data",
        size_threshold=1024,
        convert_attribute=False,
    )
    return path


# === Block quantization ===


def _blocks(weight: np.ndarray, block_size: int) -> np.ndarray:
    """[..., K] -> [..., n_blocks, block_size], zero-padding K to a whole number of blocks."""
    *batch_dims, k = weight.shape
    n_blocks = (k + block_size - 1) // block_size
    if n_blocks * block_size != k:
        pad = np.zeros((*batch_dims, n_blocks * block_size - k), dtype=weight.dtype)
        weight = np.concatenate([weight, pad], axis=-1)
    return weight.reshape(*batch_dims, n_blocks, block_size)


def quantize_int8_block(
    weight: np.ndarray, block_size: int = DEFAULT_BLOCK_SIZE
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Asymmetric uint8 quantization along the last axis: (quant [..., K], scales, zero points)."""
    blocked = _blocks(weight, block_size)
    batch_dims = blocked.shape[:-2]
    w_min = blocked.min(axis=-1, keepdims=True)
    w_max = blocked.max(axis=-1, keepdims=True)

    scale = (w_max - w_min) / 255.0
    scale = np.where(scale < SCALE_EPS, 1.0, scale)
    zero_point = np.round(-w_min / scale).clip(0, 255)
    quant = np.round(blocked / scale + zero_point).clip(0, 255).astype(np.uint8)

    return (
        quant.reshape(*batch_dims, -1),
        scale.squeeze(-1).astype(np.float32),
        zero_point.squeeze(-1).astype(np.uint8),
    )


def convert_to_fp16(
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    keep_io: tuple[str, ...] | bool = ("logits", "hidden_states", "inputs_embeds"),
) -> pathlib.Path:
    """fp16 weights, activations and cache I/O; the named graph I/O (all of it if True) stays fp32.

    Keeping inputs_embeds fp32 lets one embedding model feed a decoder of any precision.
    """
    from onnxruntime.transformers.float16 import convert_float_to_float16

    logger.info(f"Converting {input_path.name} to FP16...")
    model_fp16 = convert_float_to_float16(
        load_model(input_path),
        keep_io_types=keep_io if isinstance(keep_io, bool) else list(keep_io),
        force_fp16_initializers=True,
        disable_shape_infer=True,
    )
    save_model(model_fp16, output_path)

    orig_mb = get_total_model_size_mb(input_path)
    fp16_mb = get_total_model_size_mb(output_path)
    logger.info(f"  {input_path.name}: {orig_mb:.1f} -> {fp16_mb:.1f} MB")
    return output_path


def rename_quantized_weights(model: onnx.ModelProto):
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


def quantize_matmuls(
    model: onnx.ModelProto,
    *,
    bits: int,
    block_size: int = DEFAULT_BLOCK_SIZE,
    symmetric: bool = False,
    accuracy_level: int = 4,
    exclude: list[str] | None = None,
) -> onnx.ModelProto:
    """MatMul -> MatMulNBits (int4 or int8) for every MatMul not named in exclude.

    accuracy_level is MatMulNBits' compute type: 4 quantizes the activations to int8, 0 keeps them
    in the input type.
    """
    quant_type = "symmetric" if symmetric else "asymmetric"
    logger.info(
        f"Quantizing to INT{bits} (block_size={block_size}, {quant_type}, "
        f"accuracy_level={accuracy_level})..."
    )
    # The quantizer logs every node it leaves alone, hundreds of lines for an MoE graph.
    logging.getLogger("onnxruntime.quantization.matmul_nbits_quantizer").setLevel(logging.WARNING)

    kwargs = {}
    if bits != 4:
        kwargs["algo_config"] = DefaultWeightOnlyQuantConfig(
            block_size=block_size,
            is_symmetric=symmetric,
            accuracy_level=accuracy_level,
            bits=bits,
        )
    quantizer = MatMulNBitsQuantizer(
        model,
        block_size=block_size,
        is_symmetric=symmetric,
        accuracy_level=accuracy_level,
        nodes_to_exclude=exclude or None,
        **kwargs,
    )
    quantizer.process()

    quantized = quantizer.model.model
    rename_quantized_weights(quantized)
    return quantized


def quantize_model(
    model_path: pathlib.Path,
    output_path: pathlib.Path,
    *,
    bits: int = 4,
    block_size: int = DEFAULT_BLOCK_SIZE,
    exclude_lm_head: bool = True,
    symmetric: bool = False,
    accuracy_level: int = 4,
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
        accuracy_level: MatMulNBits compute type (see quantize_matmuls)
    """
    logger.info(f"Loading {model_path}...")
    model = load_model(model_path)

    exclude = []
    if exclude_lm_head:
        lm_head_node = find_lm_head_node(model)
        if lm_head_node:
            exclude.append(lm_head_node)
            logger.info(f"Keeping lm_head in FP32 (excluding: {lm_head_node})")
        else:
            logger.warning("Could not find lm_head node")

    quantized = quantize_matmuls(
        model,
        bits=bits,
        block_size=block_size,
        symmetric=symmetric,
        accuracy_level=accuracy_level,
        exclude=exclude,
    )
    logger.info(f"Saving to {output_path}...")
    return save_model(quantized, output_path)
