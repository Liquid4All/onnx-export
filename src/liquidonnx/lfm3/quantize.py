"""
Quantization of the LFM3 decoders, shared by the text and LFM3-VL exports.

The builder ties the LM head to the token table through a Transpose:

    input_ids → Gather(model.embed_tokens.weight [V, H]) → ... → normed
    logits = MatMul(normed, Transpose(model.embed_tokens.weight))

onnxruntime's MatMulNBitsQuantizer leaves a MatMul alone unless its weight is an initializer, so
that head stays fp32 next to a quantized body, and onnxruntime folds the Transpose into a second
dense table when it loads the graph. The q8 and q4 decoders, and their q8f16 and q4f16 versions,
instead store the table once, as the head's int8 weights (the layout of the LFM2 decoders):

    logits = MatMulNBits(normed, weights [V, n_blocks, block_size], scales, zero points)
    input_ids → GatherBlockQuantized(Reshape(weights) [V, n_blocks * block_size], scales,
                                     zero points) → ...

q8 has every MatMul in int8. q4 is the recipe of the LFM2 q4 decoders (liquidonnx.genai_builder.Q4):
k_quant int4, with the head and the MatMuls most sensitive to int4 (mixed_int8_nodes) in int8. An
export can name another int8 set (int8_nodes): d1's keeps only the MLP gate_proj and up_proj in
int4 (gate_up_int4_int8_nodes, README 3.1f).
"""

import logging
import pathlib
import re
from collections.abc import Callable

import numpy as np
import onnx
from onnx import numpy_helper

from liquidonnx.genai_builder import gather_embeddings_from_lm_head
from liquidonnx.quantize import (
    DEFAULT_BLOCK_SIZE,
    get_total_model_size_mb,
    load_model,
    quantize_matmuls,
    save_model,
)

logger = logging.getLogger(__name__)

LM_HEAD = "/lm_head/MatMul"
LM_HEAD_WEIGHT = "lm_head.MatMul.weight"
MIXED_INT8 = re.compile(r"/model/layers\.(\d+)/(?:attn/[qkv]_proj|mlp/down_proj)/MatMul")
MLP_GATE_UP = re.compile(r"/model/layers\.\d+/mlp/(?:gate|up)_proj/MatMul")


def untie_lm_head(model: onnx.ModelProto) -> onnx.ModelProto:
    """MatMul(normed, Transpose(table)) → MatMul(normed, LM_HEAD_WEIGHT [H, V]), named LM_HEAD.

    The fp32 table stays for the embedding Gather until gather_embeddings_from_lm_head replaces
    it with the quantized head's weights.
    """
    graph = model.graph
    head = next(node for node in graph.node if "logits" in node.output)
    transpose = None
    if head.op_type == "MatMul":
        transpose = next((node for node in graph.node if head.input[1] in node.output), None)
    if transpose is None or transpose.op_type != "Transpose":
        raise ValueError(
            f"{head.name} ({head.op_type}) is not a MatMul over the transposed embedding table"
        )
    table = next(init for init in graph.initializer if init.name == transpose.input[0])

    # [V, H] → [H, V]
    weight = np.ascontiguousarray(numpy_helper.to_array(table).T)
    graph.initializer.append(numpy_helper.from_array(weight, LM_HEAD_WEIGHT))
    head.input[1] = LM_HEAD_WEIGHT
    head.name = LM_HEAD
    graph.node.remove(transpose)
    for value in [v for v in graph.value_info if v.name == transpose.output[0]]:
        graph.value_info.remove(value)
    return model


def mixed_int8_nodes(model: onnx.ModelProto) -> list[str]:
    """The MatMuls of the q4 decoder that stay int8: onnxruntime-genai's mixed_layers, for LFM3.

    genai (make_matmul_mixed_precision, after llama.cpp's Q4_K_M) takes the first and the last
    eighth of the layers and every third layer between, and upgrades their attn/qkv_proj,
    attn/v_proj and mlp/down_proj. LFM3 has a q_proj, k_proj and v_proj where genai names a packed
    qkv_proj, so all three are upgraded: the mapping the recipe was measured with on LFM3-VL. A
    conv layer has only the mlp/down_proj. The 24-layer checkpoints get 27 int8 MatMuls: the
    down_proj of layers 0-2, 5, 8, 11, 14, 17, 20 and 21-23, and the q/k/v_proj of the attention
    layers among them (2, 8, 14, 20, 22).
    """
    candidates = [
        (int(match[1]), node.name)
        for node in model.graph.node
        if node.op_type == "MatMul" and (match := MIXED_INT8.fullmatch(node.name))
    ]
    n = 1 + max(layer for layer, _ in candidates)
    return [
        name for i, name in candidates if i < n / 8 or i >= 7 * n / 8 or (i - round(n / 8)) % 3 == 2
    ]


def q4_int8_nodes(model: onnx.ModelProto) -> list[str]:
    """The MatMuls the q4 decoder keeps in int8: the LM head and mixed_int8_nodes."""
    return [LM_HEAD, *mixed_int8_nodes(model)]


def gate_up_int4_int8_nodes(model: onnx.ModelProto) -> list[str]:
    """The int8 MatMuls of d1's q4 decoders: every one but the MLP gate_proj and up_proj, the LM
    head included.

    d1's outputs are the probabilities themselves, and under q4_int8_nodes d1-Omni-2B's
    spoken-question answer flips: the error builds up over all layers, with no single group to keep
    in int8 (README 3.1f). The MLP gate/up are 48 of the 24-layer checkpoints' 154 MatMuls, with
    44% of their weights (the tied head's included).
    """
    return [LM_HEAD] + [
        node.name
        for node in model.graph.node
        if node.op_type == "MatMul"
        and node.name != LM_HEAD
        and not MLP_GATE_UP.fullmatch(node.name)
    ]


def q4_block_size_error(
    model: onnx.ModelProto,
    block_size: int,
    int8_nodes: Callable[[onnx.ModelProto], list[str]] = q4_int8_nodes,
) -> str | None:
    """Why block_size can't quantize this fp32 decoder to q4 (int8_nodes in int8), or None if it
    can.

    onnxruntime's k_quant stores the int4 zero points two to a byte, running on from one row of
    blocks into the next, and fails with a numpy reshape error when a row has an odd number of
    blocks: when an int4 MatMul's input size K is (or pads to) an odd multiple of block_size. int8
    has no such limit. Only the weights' shapes are read, so the graph needs no external data.
    """
    int8 = set(int8_nodes(model))
    shapes = {init.name: list(init.dims) for init in model.graph.initializer}
    sizes = {}  # int4 MatMul → K
    for node in model.graph.node:
        dims = shapes.get(node.input[1]) if node.op_type == "MatMul" else None
        if dims and len(dims) == 2 and node.name not in int8:
            sizes[node.name] = dims[0]

    def odd(size: int) -> dict[int, list[str]]:
        by_k = {}
        for name, k in sizes.items():
            if -(-k // size) % 2:
                by_k.setdefault(k, []).append(name)
        return by_k

    if not (by_k := odd(block_size)):
        return None
    found = []
    for k, names in by_k.items():
        blocks = -(-k // block_size)
        more = f" and {len(names) - 2} more" if len(names) > 2 else ""
        unit = "block" if blocks == 1 else "blocks"
        found.append(f"K={k} is {blocks} {unit} in {', '.join(names[:2])}{more}")
    fit = [str(size) for size in (16, 32, 64, 128, 256) if not odd(size)]
    return (
        f"onnxruntime's k_quant int4 needs an even number of blocks of {block_size} in the input "
        f"size K of each int4 MatMul: {'; '.join(found)}"
        + (f". Block sizes that fit: {', '.join(fit)}" if fit else "")
    )


def save_tied(
    model: onnx.ModelProto, input_path: pathlib.Path, output_path: pathlib.Path
) -> pathlib.Path:
    """Save a decoder with a quantized head, its token embeddings gathered from that head."""
    gather_embeddings_from_lm_head(model)
    logger.info(f"Saving to {output_path}...")
    save_model(model, output_path)

    orig_mb = get_total_model_size_mb(input_path)
    quant_mb = get_total_model_size_mb(output_path)
    logger.info(f"  {input_path.name}: {orig_mb:.1f} -> {quant_mb:.1f} MB")
    return output_path


def quantize_decoder_q8(
    input_path: pathlib.Path, output_path: pathlib.Path, block_size: int = DEFAULT_BLOCK_SIZE
) -> pathlib.Path:
    """fp32 decoder → q8: every MatMul int8 (asymmetric), the LM head included, and the token
    embeddings gathered from the head's weights."""
    logger.info(f"Loading {input_path}...")
    model = untie_lm_head(load_model(input_path))
    # accuracy_level 4 (int8 activations) on the CPU EP costs some KL against level 0, but level 0
    # has no fast int8 kernel there and decodes many times slower (README 3.1b). CUDA ignores it.
    model = quantize_matmuls(
        model, bits=8, block_size=block_size, symmetric=False, accuracy_level=4
    )
    return save_tied(model, input_path, output_path)


def quantize_decoder_q4(
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    block_size: int = DEFAULT_BLOCK_SIZE,
    int8_nodes: Callable[[onnx.ModelProto], list[str]] = q4_int8_nodes,
) -> pathlib.Path:
    """fp32 decoder → q4: k_quant int4 MatMuls, with the LM head and the other int8_nodes in int8
    (by default mixed_int8_nodes), and the token embeddings gathered from the head's weights."""
    logger.info(f"Loading {input_path}...")
    model = untie_lm_head(load_model(input_path))
    if error := q4_block_size_error(model, block_size, int8_nodes):
        raise ValueError(error)
    int8 = int8_nodes(model)
    logger.info(f"  int8 ({int8_nodes.__name__}): the LM head and {len(int8) - 1} more MatMuls")
    model = quantize_matmuls(
        model, bits=4, block_size=block_size, algorithm="k_quant", node_bits=dict.fromkeys(int8, 8)
    )
    return save_tied(model, input_path, output_path)
