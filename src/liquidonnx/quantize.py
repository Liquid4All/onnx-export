"""
Quantization and precision conversion for ONNX models.

Weight-only MatMulNBits quantization (quantize_model / quantize_matmuls), plus the passes that
turn an fp32 onnxruntime-genai LFM2 decoder into the published precisions:

    tie_embedding_int4  embedding Gather + tied lm_head -> GatherBlockQuantized + MatMulNBits
                        sharing one int4 table
    moe_to_qmoe         com.microsoft MoE -> QMoE with int4/int8 block-quantized experts
    convert_to_fp16     fp16 weights, activations and caches; logits kept fp32
"""

import logging
import pathlib

import numpy as np
import onnx
from onnx import helper, numpy_helper
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


def find_router_nodes(model: onnx.ModelProto) -> list[str]:
    """MoE router MatMuls; they stay fp32 so rounding cannot flip an expert choice."""
    return [n.name for n in model.graph.node if n.op_type == "MatMul" and "/moe/router/" in n.name]


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


def quantize_int4_block(
    weight: np.ndarray, block_size: int = DEFAULT_BLOCK_SIZE
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Asymmetric uint4 quantization along the last axis.

    Returns (quant [..., K/2] two values per byte low nibble first, scales [..., n_blocks] fp32,
    zero points [..., ceil(n_blocks/2)] packed the same way).
    """
    blocked = _blocks(weight, block_size)
    batch_dims = blocked.shape[:-2]
    w_min = blocked.min(axis=-1, keepdims=True)
    w_max = blocked.max(axis=-1, keepdims=True)

    scale = (w_max - w_min) / 15.0
    scale = np.where(scale < SCALE_EPS, 1.0, scale)
    zero_point = np.round(-w_min / scale).clip(0, 15).astype(np.uint8)
    quant = np.round(blocked / scale + zero_point).clip(0, 15).astype(np.uint8)
    quant_packed = (quant[..., 0::2] | (quant[..., 1::2] << 4)).reshape(*batch_dims, -1)

    zero_point = zero_point.squeeze(-1)
    if zero_point.shape[-1] % 2:
        zero_point = np.concatenate([zero_point, np.zeros_like(zero_point[..., :1])], axis=-1)
    zp_packed = zero_point[..., 0::2] | (zero_point[..., 1::2] << 4)

    return quant_packed, scale.squeeze(-1).astype(np.float32), zp_packed


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


# === Graph passes ===


def _replace_nodes(model: onnx.ModelProto, replacements: dict[int, onnx.NodeProto | None]):
    """Swap nodes by index in graph.node; None drops the node. Unused initializers are removed."""
    nodes = []
    for index, node in enumerate(model.graph.node):
        if index in replacements:
            if replacements[index] is not None:
                nodes.append(replacements[index])
        else:
            nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(nodes)

    used = {name for node in model.graph.node for name in node.input}
    used |= {o.name for o in model.graph.output}
    kept = [init for init in model.graph.initializer if init.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)


def tie_embedding_int4(model: onnx.ModelProto, block_size: int = DEFAULT_BLOCK_SIZE) -> bool:
    """Quantize the tied embedding table once for both the embedding lookup and lm_head.

    The genai builder stores a tied table as lm_head.MatMul.weight [H, V] and feeds the embedding
    through Transpose -> Gather. Returns False when there is no tied embedding in the graph.
    """
    graph = model.graph
    initializers = {init.name: init for init in graph.initializer}
    nodes = list(graph.node)
    producers = {out: i for i, node in enumerate(nodes) for out in node.output}
    lm_head = next(
        (i for i, n in enumerate(nodes) if n.op_type == "MatMul" and n.name == "/lm_head/MatMul"),
        None,
    )
    gather = next(
        (i for i, n in enumerate(nodes) if n.op_type == "Gather" and n.input[1] == "input_ids"),
        None,
    )
    if lm_head is None or gather is None or nodes[lm_head].input[1] not in initializers:
        return False
    transpose = producers.get(nodes[gather].input[0])
    if (
        transpose is None
        or nodes[transpose].op_type != "Transpose"
        or nodes[transpose].input[0] != nodes[lm_head].input[1]
    ):
        return False
    lm_head_input, logits = nodes[lm_head].input[0], nodes[lm_head].output[0]
    embeddings = nodes[gather].output[0]

    table = numpy_helper.to_array(initializers[nodes[lm_head].input[1]]).T  # [H, V] -> [V, H]
    vocab_size, hidden_size = table.shape
    quant, scales, zero_points = quantize_int4_block(table, block_size)
    n_blocks = (hidden_size + block_size - 1) // block_size

    # One uint8 table; lm_head sees it as [V, n_blocks, block_size / 2] through a constant Reshape.
    graph.initializer.extend(
        [
            numpy_helper.from_array(quant, "model_embed_tokens_weight_quant"),
            numpy_helper.from_array(scales, "model_embed_tokens_weight_scales"),
            numpy_helper.from_array(zero_points, "model_embed_tokens_weight_zp"),
            numpy_helper.from_array(
                np.array([vocab_size, n_blocks, block_size // 2], np.int64), "/lm_head/quant_shape"
            ),
        ]
    )
    gather_quantized = helper.make_node(
        "GatherBlockQuantized",
        [
            "model_embed_tokens_weight_quant",
            "input_ids",
            "model_embed_tokens_weight_scales",
            "model_embed_tokens_weight_zp",
        ],
        [embeddings],
        name="/model/embed_tokens/GatherBlockQuantized",
        domain="com.microsoft",
        bits=4,
        block_size=block_size,
        gather_axis=0,
        quantize_axis=1,
    )
    reshape = helper.make_node(
        "Reshape",
        ["model_embed_tokens_weight_quant", "/lm_head/quant_shape"],
        ["/lm_head/quant_reshaped"],
        name="/lm_head/Reshape",
    )
    matmul_nbits = helper.make_node(
        "MatMulNBits",
        [
            lm_head_input,
            "/lm_head/quant_reshaped",
            "model_embed_tokens_weight_scales",
            "model_embed_tokens_weight_zp",
        ],
        [logits],
        name="/lm_head/MatMulNBits",
        domain="com.microsoft",
        K=hidden_size,
        N=vocab_size,
        bits=4,
        block_size=block_size,
    )
    _replace_nodes(model, {gather: gather_quantized, transpose: reshape, lm_head: matmul_nbits})
    return True


def _expert_bias(node: onnx.NodeProto, index: int, initializers: dict) -> str:
    """The MoE bias input at index, or "" when absent or all zero (as genai writes it)."""
    name = node.input[index] if len(node.input) > index else ""
    if not name or not np.any(numpy_helper.to_array(initializers[name])):
        return ""
    return name


def moe_to_qmoe(model: onnx.ModelProto, bits: int, block_size: int = DEFAULT_BLOCK_SIZE) -> int:
    """Rewrite each fp32 com.microsoft MoE node as QMoE with block-quantized experts.

    MoE inputs: [x, router_probs, fc1 [E, 2I, H], fc1_bias, fc2 [E, H, I], fc2_bias]. QMoE takes
    uint8 weights [E, N, K * bits / 8], scales [E, N, K / block] and asymmetric zero points at
    inputs 11 / 12. Returns the number of nodes rewritten.
    """
    quantize = {4: quantize_int4_block, 8: quantize_int8_block}[bits]
    initializers = {init.name: init for init in model.graph.initializer}
    replacements = {}
    for index, node in enumerate(model.graph.node):
        if node.op_type != "MoE":
            continue
        packed = []
        for weight_name in (node.input[2], node.input[4]):
            weight = numpy_helper.to_array(initializers[weight_name]).astype(np.float32)
            base = weight_name.removesuffix(".weight").replace(".", "_")
            names = (f"{base}_weight_quant", f"{base}_weight_scales", f"{base}_weight_zp")
            for array, name in zip(quantize(weight, block_size), names, strict=True):
                model.graph.initializer.append(numpy_helper.from_array(array, name))
            packed.append(names)

        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        (fc1, fc1_scales, fc1_zp), (fc2, fc2_scales, fc2_zp) = packed
        replacements[index] = helper.make_node(
            "QMoE",
            [
                node.input[0],
                node.input[1],
                fc1,
                fc1_scales,
                _expert_bias(node, 3, initializers),
                fc2,
                fc2_scales,
                _expert_bias(node, 5, initializers),
                "",
                "",
                "",
                fc1_zp,
                fc2_zp,
                "",
            ],
            list(node.output),
            name=node.name.replace("/MoE", "/QMoE"),
            domain="com.microsoft",
            **attrs,
            expert_weight_bits=bits,
            block_size=block_size,
        )
    if replacements:
        _replace_nodes(model, replacements)
    return len(replacements)


def convert_to_fp16(
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    keep_io: tuple[str, ...] = ("logits", "hidden_states"),
) -> pathlib.Path:
    """fp16 weights, activations and cache I/O; the named graph I/O stays fp32."""
    from onnxruntime.transformers.float16 import convert_float_to_float16

    logger.info(f"Converting {input_path.name} to FP16...")
    model_fp16 = convert_float_to_float16(
        load_model(input_path),
        keep_io_types=list(keep_io),
        force_fp16_initializers=True,
        disable_shape_infer=True,
    )
    save_model(model_fp16, output_path)

    orig_mb = get_total_model_size_mb(input_path)
    fp16_mb = get_total_model_size_mb(output_path)
    logger.info(f"  {input_path.name}: {orig_mb:.1f} -> {fp16_mb:.1f} MB")
    return output_path


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


def quantize_matmuls(
    model: onnx.ModelProto,
    *,
    bits: int,
    block_size: int = DEFAULT_BLOCK_SIZE,
    symmetric: bool = False,
    exclude: list[str] | None = None,
) -> onnx.ModelProto:
    """MatMul -> MatMulNBits (int4 or int8) for every MatMul not named in exclude."""
    quant_type = "symmetric" if symmetric else "asymmetric"
    logger.info(f"Quantizing to INT{bits} (block_size={block_size}, {quant_type})...")
    # The quantizer logs every node it leaves alone, hundreds of lines for an MoE graph.
    logging.getLogger("onnxruntime.quantization.matmul_nbits_quantizer").setLevel(logging.WARNING)

    kwargs = {}
    if bits != 4:
        kwargs["algo_config"] = DefaultWeightOnlyQuantConfig(
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
        nodes_to_exclude=exclude or None,
        **kwargs,
    )
    quantizer.process()

    quantized = quantizer.model.model
    _rename_quantized_weights(quantized)
    return quantized


def quantize_model(
    model_path: pathlib.Path,
    output_path: pathlib.Path,
    *,
    bits: int = 4,
    block_size: int = DEFAULT_BLOCK_SIZE,
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
        model, bits=bits, block_size=block_size, symmetric=symmetric, exclude=exclude
    )
    logger.info(f"Saving to {output_path}...")
    return save_model(quantized, output_path)
