"""
Embedding models for the onnxruntime-genai multimodal pipelines (lfm2_vl, lfm2_audio).

    input_ids [B, S] ──Gather──► token embeddings [B, S, H]
                                        │ ScatterND: row i of features replaces the i-th
    features [N, H] ────────────────────┘ position whose id is token_id
                                        ▼
                                 inputs_embeds [B, S, H] (fp32)

The decoders are built with exclude_embeds, so this graph owns the token table. The runtime
calls it every step; decode steps pass an empty features tensor.

The fp32 and fp16 bundles store the table in their own precision. The quantized bundles share an
int8 one (GatherBlockQuantized), a little over half the size of the fp16 table.
"""

import pathlib

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from liquidonnx.quantize import DEFAULT_BLOCK_SIZE, load_model, quantize_int8_block, save_model

TABLE = "embed_tokens.weight"
QUANT_TABLE = ("embed_tokens_weight_quant", "embed_tokens_weight_scales", "embed_tokens_weight_zp")
TOKEN_ID = "token_id"
TABLE_DTYPES = {"fp32": np.float32, "fp16": np.float16, "q4": np.uint8, "q8": np.uint8}


def embeddings_file(precision: str) -> str:
    """The embedding model (in onnx/) a precision loads."""
    suffix = {"fp32": "", "fp16": "_fp16", "q4": "_q8", "q8": "_q8"}[precision]
    return f"embeddings{suffix}.onnx"


def build_embeddings(
    table: np.ndarray,
    token_id: int,
    feature: str,
    path: pathlib.Path,
    table_dtype: type = np.float32,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> pathlib.Path:
    """Write the embedding model with the [V, H] table stored as table_dtype.

    np.uint8 stores it quantized to 8 bits in blocks of block_size along H, with zero points.
    """
    hidden = table.shape[1]
    lookup = "token_embeds"
    opsets = [helper.make_opsetid("", 17)]
    if table_dtype == np.uint8:
        quant, scales, zero_points = quantize_int8_block(table, block_size)
        # quantize_int8_block pads H to whole blocks; the op reads the row width from the table.
        tables = dict(zip(QUANT_TABLE, (quant[:, :hidden], scales, zero_points), strict=True))
        opsets.append(helper.make_opsetid("com.microsoft", 1))
        nodes = [
            helper.make_node(
                "GatherBlockQuantized",
                [QUANT_TABLE[0], "input_ids", *QUANT_TABLE[1:]],
                [lookup],
                domain="com.microsoft",
                bits=8,
                block_size=block_size,
                gather_axis=0,
                quantize_axis=1,
            )
        ]
    else:
        tables = {TABLE: table.astype(table_dtype)}
        nodes = [helper.make_node("Gather", [TABLE, "input_ids"], [lookup], axis=0)]
    if table_dtype == np.float16:
        nodes.append(
            helper.make_node("Cast", [lookup], ["token_embeds_fp32"], to=TensorProto.FLOAT)
        )
        lookup = "token_embeds_fp32"
    nodes += [
        helper.make_node("Shape", [lookup], ["embeds_shape"]),
        helper.make_node("Reshape", [lookup, "flat_rows"], ["flat_embeds"]),
        helper.make_node("Reshape", ["input_ids", "flat"], ["flat_ids"]),
        helper.make_node("Equal", ["flat_ids", TOKEN_ID], ["is_feature"]),
        helper.make_node("NonZero", ["is_feature"], ["positions_t"]),
        helper.make_node("Transpose", ["positions_t"], ["positions"], perm=[1, 0]),
        helper.make_node("ScatterND", ["flat_embeds", "positions", feature], ["merged"]),
        helper.make_node("Reshape", ["merged", "embeds_shape"], ["inputs_embeds"]),
    ]
    graph = helper.make_graph(
        nodes,
        "embedding",
        [
            helper.make_tensor_value_info(
                "input_ids", TensorProto.INT64, ["batch_size", "sequence_length"]
            ),
            helper.make_tensor_value_info(feature, TensorProto.FLOAT, [f"num_{feature}", hidden]),
        ],
        [
            helper.make_tensor_value_info(
                "inputs_embeds", TensorProto.FLOAT, ["batch_size", "sequence_length", hidden]
            )
        ],
        initializer=[
            *(numpy_helper.from_array(array, name) for name, array in tables.items()),
            numpy_helper.from_array(np.array([-1, hidden], np.int64), "flat_rows"),
            numpy_helper.from_array(np.array([-1], np.int64), "flat"),
            numpy_helper.from_array(np.array(token_id, np.int64), TOKEN_ID),
        ],
    )
    model = helper.make_model(graph, opset_imports=opsets, ir_version=8)
    model.producer_name = "liquidonnx"
    onnx.checker.check_model(model)
    return save_model(model, path)


def derive_embeddings(
    onnx_dir: pathlib.Path, precision: str, block_size: int = DEFAULT_BLOCK_SIZE
) -> pathlib.Path:
    """Write the embedding model precision loads from the fp32 one in onnx_dir; the output stays
    fp32."""
    model = load_model(onnx_dir / embeddings_file("fp32"))
    initializers = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}
    return build_embeddings(
        initializers[TABLE],
        int(initializers[TOKEN_ID]),
        model.graph.input[1].name,
        onnx_dir / embeddings_file(precision),
        TABLE_DTYPES[precision],
        block_size,
    )


def embed(session, input_ids: np.ndarray, features: np.ndarray | None = None) -> np.ndarray:
    """inputs_embeds of input_ids, with features (None: plain text) at the placeholder ids."""
    feature = session.get_inputs()[1]
    if features is None:
        features = np.zeros((0, feature.shape[1]), np.float32)
    feed = {"input_ids": input_ids.astype(np.int64), feature.name: features.astype(np.float32)}
    return session.run(None, feed)[0]
