"""
Export pipeline (genai builder + liquidonnx precisions) on tiny random LFM2 / LFM2-MoE checkpoints.

The checkpoints are built from a config, so only the LFM2 tokenizer is downloaded.

Run with:
    uv run pytest tests/test_genai_export.py -v
    uv run pytest tests/test_genai_export.py -v -k "moe and q4"
"""

import collections
import json
import pathlib
import shutil

import numpy as np
import onnx
import onnxruntime as ort
import onnxruntime_genai as og
import pytest
import torch
from transformers import (
    AutoTokenizer,
    Lfm2Config,
    Lfm2ForCausalLM,
    Lfm2MoeConfig,
    Lfm2MoeForCausalLM,
)

from liquidonnx.compare.metrics import greedy
from liquidonnx.genai_builder import export_decoder
from liquidonnx.genai_runtime import (
    EXECUTION_PROVIDERS,
    check_genai_version,
    generate,
    load_model,
)
from liquidonnx.lfm2.export import ALL_PRECISIONS, genai_files, model_file, set_default_decoder
from liquidonnx.quantize import derive_precision
from liquidonnx.session import (
    cached_outputs,
    decoder_inputs,
    initialize_cache,
    load_onnx_session,
    preload_cuda_libraries,
)

CPU_EP, CUDA_EP = "CPUExecutionProvider", "CUDAExecutionProvider"
TOKENS = np.array([[1, 5, 77, 300, 42, 9, 128, 64, 3, 250]], dtype=np.int64)
HEAD_SIZE = 16
COMMON = {
    "vocab_size": 512,
    "hidden_size": 64,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "max_position_embeddings": 256,
    "tie_word_embeddings": True,
}
MIN_COSINE = {
    "fp32": 0.999999,
    "fp16": 0.9999,
    "q8": 0.999,
    "q4f32": 0.98,
    "q4": 0.97,
    "q4f16": 0.97,
}


def make_checkpoint(kind: str, path: pathlib.Path) -> torch.nn.Module:
    torch.manual_seed(0)
    if kind == "dense":
        config = Lfm2Config(
            num_hidden_layers=4,
            intermediate_size=128,
            layer_types=["conv", "conv", "full_attention", "conv"],
            **COMMON,
        )
        model = Lfm2ForCausalLM(config)
    else:
        config = Lfm2MoeConfig(
            num_hidden_layers=4,
            intermediate_size=128,
            moe_intermediate_size=64,
            num_experts=8,
            num_experts_per_tok=2,
            num_dense_layers=1,
            layer_types=["conv", "full_attention", "conv", "full_attention"],
            **COMMON,
        )
        model = Lfm2MoeForCausalLM(config)
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name.endswith("expert_bias"):
                    param.normal_(0, 0.1)
    model.eval().save_pretrained(path)
    AutoTokenizer.from_pretrained("LiquidAI/LFM2-350M").save_pretrained(path)
    return model


@pytest.fixture(scope="module", params=["dense", "moe"])
def export(request, tmp_path_factory):
    """(kind, PyTorch model, export dir) with every precision derived."""
    root = tmp_path_factory.mktemp(request.param)
    model = make_checkpoint(request.param, root / "checkpoint")
    output_dir = root / "export"
    output_dir.mkdir()
    export_decoder(str(root / "checkpoint"), output_dir)
    for precision in ALL_PRECISIONS:
        derive_precision(output_dir / "onnx", precision, reuse_q4=True)
    set_default_decoder(output_dir, list(ALL_PRECISIONS))
    return request.param, model, output_dir


def decoder(output_dir: pathlib.Path, precision: str):
    return load_onnx_session(output_dir / "onnx" / model_file(precision))


@pytest.mark.parametrize("precision", ["fp32", *ALL_PRECISIONS])
def test_logits_match_pytorch(export, precision: str):
    _, model, output_dir = export
    with torch.no_grad():
        expected = model(torch.from_numpy(TOKENS)).logits[0].numpy()

    session = decoder(output_dir, precision)
    actual = session.run(None, decoder_inputs(TOKENS, initialize_cache(session), 0))[0][0]
    actual = actual.astype(np.float32)

    cosine = (expected * actual).sum() / (np.linalg.norm(expected) * np.linalg.norm(actual))
    assert cosine >= MIN_COSINE[precision]
    if precision == "fp32":
        np.testing.assert_allclose(actual, expected, atol=1e-5)


@pytest.mark.parametrize("precision", ["fp32", "fp16", "q8", "q4"])
def test_cached_decode_matches_prefill(export, precision: str):
    _, _, output_dir = export
    session = decoder(output_dir, precision)
    prefill = session.run(None, decoder_inputs(TOKENS, initialize_cache(session), 0))[0][0, 3:]
    # MatMulNBits takes different kernels for one token and for a whole prompt.
    atol = 5e-3 if precision == "q4" else 1e-3
    np.testing.assert_allclose(cached_outputs(session, TOKENS, 4)["logits"], prefill, atol=atol)


def test_graph_layout(export):
    kind, _, output_dir = export

    def ops(precision):
        path = output_dir / "onnx" / model_file(precision)
        graph = onnx.load(str(path), load_external_data=False).graph
        return graph, collections.Counter(node.op_type for node in graph.node)

    graph, fp32_ops = ops("fp32")
    for value in graph.input:
        if value.name.startswith("past_key_values."):
            assert value.type.tensor_type.shape.dim[-1].dim_value == HEAD_SIZE

    routers = 3 if kind == "moe" else 0
    assert fp32_ops["MoE"] == routers

    graph, q4_ops = ops("q4")
    assert q4_ops["GatherBlockQuantized"] == 1
    assert q4_ops["MatMul"] == routers
    assert not [i.name for i in graph.initializer if i.name.endswith("_quant_matmul")]
    lm_head = next(n for n in graph.node if n.name == "/lm_head/MatMulNBits")
    assert lm_head.input[1] == "/lm_head/quant_reshaped"

    _, q4f32_ops = ops("q4f32")
    assert q4f32_ops["GatherBlockQuantized"] == 0
    assert q4f32_ops["MatMul"] == routers + 1  # lm_head stays fp32

    for precision, bits in (("q4", 4), ("q8", 8)):
        graph, precision_ops = ops(precision)
        assert precision_ops["MoE"] == 0
        assert precision_ops["QMoE"] == routers
        for node in graph.node:
            if node.op_type == "QMoE":
                attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
                assert attrs["expert_weight_bits"] == bits


def test_genai_config(export):
    _, _, output_dir = export
    config = json.loads((output_dir / "genai_config.json").read_text())
    assert config["model"]["decoder"]["filename"] == "onnx/model_q4.onnx"
    for name in ("tokenizer.json", "tokenizer_config.json", "config.json"):
        assert (output_dir / name).exists()


def test_q4f16_ignores_a_stale_q4(export, tmp_path):
    """Without reuse_q4, q4f16 quantizes the fp32 graph instead of converting model_q4.onnx."""
    _, _, output_dir = export
    for path in (output_dir / "onnx").glob("model.onnx*"):
        shutil.copy(path, tmp_path)
    (tmp_path / "model_q4.onnx").write_bytes(b"left over from another checkpoint")

    derive_precision(tmp_path, "q4f16")

    def logits(path: pathlib.Path) -> np.ndarray:
        session = load_onnx_session(path)
        return session.run(None, decoder_inputs(TOKENS, initialize_cache(session), 0))[0]

    expected = logits(output_dir / "onnx" / "model_q4f16.onnx")
    np.testing.assert_array_equal(logits(tmp_path / "model_q4f16.onnx"), expected)


@pytest.mark.parametrize("precision", ["fp32", "q4"])
def test_genai_runtime_matches_onnxruntime(export, precision: str):
    """Greedy answers through liquidonnx.genai_runtime (the CLIs' path) and plain onnxruntime."""
    _, _, output_dir = export

    model = load_model(output_dir, genai_files(precision))
    genai_tokens = generate(model, TOKENS[0], 8).get_sequence(0)[TOKENS.shape[1] :].tolist()

    session = decoder(output_dir, precision)
    assert genai_tokens == greedy(session, TOKENS[0], len(genai_tokens), eos=set())


def test_genai_version_floor():
    """load_model refuses onnxruntime-genai releases before 0.17.1 and accepts builds of main."""
    for version in ("0.16.0", "0.17.0"):
        with pytest.raises(RuntimeError, match="0.17.1"):
            check_genai_version(version)
    for version in ("0.17.1", "0.18.0-dev"):
        check_genai_version(version)


@pytest.mark.parametrize("ep", EXECUTION_PROVIDERS)
def test_load_model_preloads_cuda_libraries(export, ep: str, monkeypatch):
    """--ep cuda loads the CUDA libraries before the model, --ep cpu does not."""
    calls = []
    monkeypatch.setattr(ort, "preload_dlls", lambda: calls.append("preload"))
    monkeypatch.setattr(og, "Model", lambda config: calls.append("model"))
    load_model(export[2], ep=ep)
    assert calls == {"cpu": ["model"], "cuda": ["preload", "model"]}[ep]


@pytest.mark.parametrize("tf32", [True, False])
@pytest.mark.parametrize("ep", EXECUTION_PROVIDERS)
def test_load_model_turns_tf32_off_on_cuda(export, ep: str, tf32: bool, monkeypatch):
    """tf32=False sets the CUDA EP's use_tf32 to 0; otherwise the providers keep their defaults."""
    options = []
    set_provider_option = og.Config.set_provider_option

    def record(config, *args):
        options.append(args)
        set_provider_option(config, *args)

    monkeypatch.setattr(ort, "preload_dlls", lambda: None)
    monkeypatch.setattr(og.Config, "set_provider_option", record)
    monkeypatch.setattr(og, "Model", lambda config: None)
    load_model(export[2], ep=ep, tf32=tf32)
    assert options == ([("cuda", "use_tf32", "0")] if ep == "cuda" and not tf32 else [])


def fake_sessions(monkeypatch, loaded: list[str]) -> list:
    """Sessions that load only the providers in loaded; returns the log of CUDA library preloads
    and the providers each session asked for."""
    calls = []

    class Session:
        def __init__(self, path: str, sess_options=None, providers: list | None = None):
            calls.append(providers)
            names = [p[0] if isinstance(p, tuple) else p for p in providers]
            self.providers = [p for p in names if p in loaded]

        def get_providers(self) -> list[str]:
            return self.providers

    monkeypatch.setattr(ort, "preload_dlls", lambda: calls.append("preload"))
    monkeypatch.setattr(ort, "get_available_providers", lambda: loaded)
    monkeypatch.setattr(ort, "InferenceSession", Session)
    return calls


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ((), [[CPU_EP]]),
        (("cpu",), [[CPU_EP]]),
        (("cuda",), ["preload", [CUDA_EP, CPU_EP]]),
        (("cpu", False), [[CPU_EP]]),
        (("cuda", False), ["preload", [(CUDA_EP, {"use_tf32": "0"}), CPU_EP]]),
    ],
    ids=["default", "cpu", "cuda", "cpu-no-tf32", "cuda-no-tf32"],
)
def test_sessions_run_on_the_requested_ep(args: tuple, expected: list, monkeypatch, tmp_path):
    """Single-graph sessions run on CPU unless asked for CUDA, even where CUDA works, and load the
    CUDA libraries only for CUDA. tf32=False turns TF32 off on CUDA."""
    calls = fake_sessions(monkeypatch, [CUDA_EP, CPU_EP])
    path = tmp_path / "model.onnx"
    path.touch()
    load_onnx_session(path, *args)
    assert calls == expected


@pytest.mark.parametrize("tf32", [True, False])
def test_cuda_session_fails_on_cpu_fallback(tf32: bool, monkeypatch, tmp_path):
    """onnxruntime runs on CPU when the CUDA EP does not load; a CUDA session then fails."""
    fake_sessions(monkeypatch, [CPU_EP])
    path = tmp_path / "model.onnx"
    path.touch()
    with pytest.raises(RuntimeError, match=f"without {CUDA_EP}"):
        load_onnx_session(path, "cuda", tf32)


def test_preload_cuda_libraries_without_preload_dlls(monkeypatch):
    """onnxruntime-gpu releases without preload_dlls still load models."""
    monkeypatch.delattr(ort, "preload_dlls")
    preload_cuda_libraries()
