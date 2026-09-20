"""Structural checks for the LFM2.5-Audio ONNX graphs using synthetic weights.

Builds each graph with small random weights (no checkpoint download, no reference
model) and checks that it loads under ORT CPU, produces finite output, handles
batched input, and quantizes completely.

    uv run pytest tests/test_lfm2_audio/test_graph_structure.py -v
"""

import pathlib

import numpy as np
import onnx
import onnxruntime as ort
import pytest

from liquidonnx.quantize import quantize_model

from .synthetic import DETOK_CONFIG, HIDDEN, build_depthformer, build_detokenizer, build_encoder

OPSET = 21
DEPTHFORMER_LAYERS = 2


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(0)


@pytest.fixture(scope="module")
def encoder_path(tmp_path_factory, rng) -> pathlib.Path:
    path = tmp_path_factory.mktemp("audio") / "audio_encoder.onnx"
    onnx.save(build_encoder(rng), str(path))
    return path


@pytest.fixture(scope="module")
def detokenizer_path(tmp_path_factory, rng) -> pathlib.Path:
    path = tmp_path_factory.mktemp("audio") / "audio_detokenizer.onnx"
    onnx.save(build_detokenizer(rng), str(path))
    return path


@pytest.fixture(scope="module")
def depthformer_path(tmp_path_factory, rng) -> pathlib.Path:
    path = tmp_path_factory.mktemp("audio") / "vocoder_depthformer.onnx"
    onnx.save(build_depthformer(rng, num_layers=DEPTHFORMER_LAYERS), str(path))
    return path


def _session(path: pathlib.Path) -> ort.InferenceSession:
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def _depthformer_feed(
    rng, step: int, num_layers: int = DEPTHFORMER_LAYERS
) -> dict[str, np.ndarray]:
    B, nkv, hd = 1, 8, 32
    return {
        "hidden_states": rng.standard_normal((B, HIDDEN)).astype(np.float32),
        "depth_slices_in": rng.standard_normal((B, 8, 1024)).astype(np.float32),
        "step_idx": np.array(step, dtype=np.int64),
        "prev_token": np.array([5], dtype=np.int64),
        "past_keys": rng.standard_normal((num_layers, B, nkv, step, hd)).astype(np.float32),
        "past_values": rng.standard_normal((num_layers, B, nkv, step, hd)).astype(np.float32),
        "seqlens_k": np.array([step], dtype=np.int32),
        "total_seq_len": np.array(step + 1, dtype=np.int32),
    }


# === Tests ===


@pytest.mark.parametrize("graph", ["encoder_path", "detokenizer_path", "depthformer_path"])
def test_graph_targets_opset(graph, request):
    model = onnx.load(str(request.getfixturevalue(graph)), load_external_data=False)
    opsets = {o.domain: o.version for o in model.opset_import}
    assert opsets[""] == OPSET
    assert model.ir_version == 10


def test_encoder_batched_matches_per_item(encoder_path, rng):
    """Batched, ragged-length encoding must equal encoding each item alone.

    Regression: the subsampling length mask used to broadcast [T] against [B],
    which failed outright for batch > 1.
    """
    sess = _session(encoder_path)
    mel = rng.standard_normal((2, 200, 128)).astype(np.float32)
    lens = np.array([200, 96], dtype=np.int64)

    emb_batched, len_batched = sess.run(None, {"mel_spectrogram": mel, "mel_lengths": lens})
    assert emb_batched.shape[0] == 2
    assert np.isfinite(emb_batched).all()

    for i in range(2):
        emb, length = sess.run(
            None, {"mel_spectrogram": mel[i : i + 1], "mel_lengths": lens[i : i + 1]}
        )
        n = int(length[0])
        assert len_batched[i] == n
        np.testing.assert_allclose(emb_batched[i, :n], emb[0, :n], rtol=1e-5, atol=1e-6)


def test_detokenizer_runs(detokenizer_path, rng):
    sess = _session(detokenizer_path)
    codes = rng.integers(0, 2048, (1, 8, 12)).astype(np.int64)
    (stft,) = sess.run(None, {"audio_codes": codes})
    assert stft.shape == (1, 12 * 6, DETOK_CONFIG["output_size"])
    assert np.isfinite(stft).all()


@pytest.mark.parametrize("step", [0, 1, 7])
def test_depthformer_step_runs(depthformer_path, rng, step):
    sess = _session(depthformer_path)
    outs = dict(
        zip(
            [o.name for o in sess.get_outputs()],
            sess.run(None, _depthformer_feed(rng, step)),
            strict=True,
        )
    )
    assert outs["logits"].shape == (1, 2049)
    assert outs["new_keys"].shape == (DEPTHFORMER_LAYERS, 1, 8, step + 1, 32)
    assert all(np.isfinite(v).all() for v in outs.values())


@pytest.mark.parametrize("graph", ["encoder_path", "detokenizer_path", "depthformer_path"])
def test_q4_quantizes_every_constant_matmul(graph, request, tmp_path):
    """Every MatMul with a constant weight must become MatMulNBits, including inside subgraphs."""
    src = request.getfixturevalue(graph)
    dst = tmp_path / f"{src.stem}_q4.onnx"
    quantize_model(src, dst, bits=4, block_size=32, exclude_lm_head=False, symmetric=True)

    model = onnx.load(str(dst), load_external_data=False)

    def constant_matmuls(g: onnx.GraphProto, initializers: set[str]) -> list[str]:
        initializers = initializers | {i.name for i in g.initializer}
        found = []
        for n in g.node:
            if n.op_type == "MatMul" and n.input[1] in initializers:
                found.append(n.name)
            for a in n.attribute:
                if a.type == onnx.AttributeProto.GRAPH:
                    found += constant_matmuls(a.g, initializers)
        return found

    assert constant_matmuls(model.graph, set()) == []
    assert any(n.op_type == "MatMulNBits" for n in model.graph.node)
    assert _session(dst) is not None


def test_depthformer_gqa_present_has_its_own_seq_dim(depthformer_path):
    """GQA present_k/v must not share a symbolic seq dim with past_k/v.

    ORT's GQA shape inference copies the past seq dim onto present; the WebGPU kernel marks
    present as MayInplace(past), so identical symbolic shapes make the planner alias the two
    buffers and the run fails as soon as the cache grows.
    """
    model = onnx.load(str(depthformer_path), load_external_data=False)
    value_info = {vi.name: vi for vi in model.graph.value_info}
    gqa_nodes = [n for n in model.graph.node if n.op_type == "GroupQueryAttention"]
    assert gqa_nodes

    for node in gqa_nodes:
        past_dim = value_info.get(node.input[3])
        past_param = past_dim.type.tensor_type.shape.dim[2].dim_param if past_dim else "past_len"
        for present in node.output[1:3]:
            assert present in value_info, f"{present} has no declared shape"
            dims = value_info[present].type.tensor_type.shape.dim
            assert dims[2].dim_param and dims[2].dim_param != past_param
