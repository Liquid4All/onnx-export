"""
GroupQueryAttention(local_window_size=W) attends to the keys of HF's sliding_window=W: the W
positions up to and including the query, j in (i - W, i]. The LFM3 decoder (W=256), the LFM3-Audio
encoder (W=32) and its codec (W=30) take local_window_size = sliding_window. Checked on the --ep
execution provider with its attention kernel: float32 on the CPU, float16 on CUDA (it has no
float32 GroupQueryAttention), with lfm3.session's CUDA options.

The keys' scores fall by 16 per position, so each query attends (to e^-16) to the oldest key it
may see: output i is v[max(0, i - W + 1)], and a window one key wider or narrower picks a
neighbour.

Run with:
    uv run pytest tests/test_lfm3_audio/test_gqa_window.py -v
    uv run pytest tests/test_lfm3_audio/test_gqa_window.py -v --device cuda
"""

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from liquidonnx.lfm3.session import load_session

HEAD_DIM = 64
INPUTS = ("query", "key", "value", "past_key", "past_value", "seqlens_k", "total_sequence_length")


def gqa_model(window: int, elem_type: int) -> onnx.ModelProto:
    node = helper.make_node(
        "GroupQueryAttention",
        list(INPUTS),
        ["output", "present_key", "present_value"],
        domain="com.microsoft",
        num_heads=1,
        kv_num_heads=1,
        local_window_size=window,
        scale=1.0,
    )
    qkv = [helper.make_tensor_value_info(n, elem_type, [1, "S", HEAD_DIM]) for n in INPUTS[:3]]
    past = [helper.make_tensor_value_info(n, elem_type, [1, 1, 0, HEAD_DIM]) for n in INPUTS[3:5]]
    lengths = [
        helper.make_tensor_value_info("seqlens_k", TensorProto.INT32, [1]),
        helper.make_tensor_value_info("total_sequence_length", TensorProto.INT32, []),
    ]
    outputs = [
        helper.make_tensor_value_info(n, elem_type, None)
        for n in ("output", "present_key", "present_value")
    ]
    graph = helper.make_graph([node], "gqa", qkv + past + lengths, outputs)
    opsets = [helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)]
    return helper.make_model(graph, opset_imports=opsets, ir_version=10)


@pytest.mark.parametrize("window", [256, 32, 30])
def test_local_window_matches_hf(window, ep, tmp_path):
    dtype, elem_type = {
        "cpu": (np.float32, TensorProto.FLOAT),
        "cuda": (np.float16, TensorProto.FLOAT16),
    }[ep]
    path = tmp_path / "gqa.onnx"
    onnx.save(gqa_model(window, elem_type), str(path))
    session = load_session(path, ep, tf32=False)

    S = 2 * window + 7
    query = np.zeros((1, S, HEAD_DIM), dtype)
    query[..., 0] = 1
    key = np.zeros((1, S, HEAD_DIM), dtype)
    key[0, :, 0] = -16 * np.arange(S)  # exact in float16 up to S = 1024
    value = np.random.default_rng(0).standard_normal((1, S, HEAD_DIM)).astype(dtype)
    empty = np.zeros((1, 1, 0, HEAD_DIM), dtype)
    output = session.run(
        ["output"],
        {
            "query": query,
            "key": key,
            "value": value,
            "past_key": empty,
            "past_value": empty,
            "seqlens_k": np.array([S - 1], np.int32),
            "total_sequence_length": np.array(S, np.int32),
        },
    )[0][0].astype(np.float32)

    oldest = np.maximum(0, np.arange(S) - window + 1)
    np.testing.assert_allclose(output, value[0, oldest].astype(np.float32), atol=1e-2, rtol=0)
