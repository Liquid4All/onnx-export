"""
Embedding models for the onnxruntime-genai multimodal pipelines (lfm2_vl, lfm2_audio).

    input_ids [B, S] ──Gather──► token embeddings [B, S, H]
                                        │ ScatterND: row i of features replaces the i-th
    features [N, H] ────────────────────┘ position whose id is token_id
                                        ▼
                                 inputs_embeds [B, S, H] (fp32)

The decoders are built with exclude_embeds, so this graph owns the token table. The runtime
calls it every step; decode steps pass an empty features tensor.
"""

import pathlib

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from liquidonnx.quantize import load_model, save_model

TABLE = "embed_tokens.weight"
TOKEN_ID = "token_id"


def build_embeddings(
    table: np.ndarray,
    token_id: int,
    feature: str,
    path: pathlib.Path,
    table_dtype: type = np.float32,
) -> pathlib.Path:
    """Write the embedding model with the [V, H] table stored as table_dtype."""
    hidden = table.shape[1]
    lookup = "token_embeds"
    nodes = [helper.make_node("Gather", [TABLE, "input_ids"], [lookup], axis=0)]
    if table_dtype != np.float32:
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
            numpy_helper.from_array(table.astype(table_dtype), TABLE),
            numpy_helper.from_array(np.array([-1, hidden], np.int64), "flat_rows"),
            numpy_helper.from_array(np.array([-1], np.int64), "flat"),
            numpy_helper.from_array(np.array(token_id, np.int64), TOKEN_ID),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
    model.producer_name = "liquidonnx"
    onnx.checker.check_model(model)
    return save_model(model, path)


def embeddings_to_fp16(source: pathlib.Path, path: pathlib.Path) -> pathlib.Path:
    """Rebuild the embedding model at source with an fp16 table; the output stays fp32."""
    model = load_model(source)
    initializers = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}
    return build_embeddings(
        initializers[TABLE],
        int(initializers[TOKEN_ID]),
        model.graph.input[1].name,
        path,
        np.float16,
    )


def embedding_feed(session, input_ids: np.ndarray, features: np.ndarray | None = None) -> dict:
    """Feed for an embedding model; features=None means no feature rows (plain text).

    An onnx-community embed_tokens.onnx (input_ids only) is a plain lookup and takes no features.
    """
    inputs = session.get_inputs()
    feed = {"input_ids": input_ids.astype(np.int64)}
    if len(inputs) > 1:
        if features is None:
            features = np.zeros((0, inputs[1].shape[1]), np.float32)
        feed[inputs[1].name] = features.astype(np.float32)
    elif features is not None:
        raise ValueError("this embedding model has no feature input")
    return feed


def embed(session, input_ids: np.ndarray, features: np.ndarray | None = None) -> np.ndarray:
    return session.run(None, embedding_feed(session, input_ids, features))[0]
