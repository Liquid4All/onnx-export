"""
Export pipeline (genai builder + liquidonnx precisions) on tiny random LFM2 / LFM2-MoE checkpoints.

The checkpoints are built from a config, so only the LFM2 tokenizer is downloaded.

Run with:
    uv run pytest tests/test_genai_export.py -v
    uv run pytest tests/test_genai_export.py -v -k "moe and q4"
"""

import argparse
import collections
import json
import logging
import os
import pathlib
import shutil
import stat
import subprocess
import sys

import numpy as np
import onnx
import onnxruntime as ort
import onnxruntime_genai as og
import pytest
import torch
from helpers import Q8_FP32_HEAD, attributes, matmul_bits
from onnx import numpy_helper
from transformers import (
    AutoTokenizer,
    Lfm2Config,
    Lfm2ForCausalLM,
    Lfm2MoeConfig,
    Lfm2MoeForCausalLM,
)

from liquidonnx import genai_builder
from liquidonnx.compare.metrics import greedy
from liquidonnx.genai_builder import EMBED_GATHER, Q8, export_decoder, export_precision
from liquidonnx.genai_runtime import (
    EXECUTION_PROVIDERS,
    check_genai_version,
    generate,
    load_model,
)
from liquidonnx.lfm2 import benchmark
from liquidonnx.lfm2 import infer as text_infer
from liquidonnx.lfm2.export import (
    ALL_PRECISIONS,
    genai_files,
    model_file,
    reusable_q4,
    set_default_decoder,
)
from liquidonnx.lfm2.export import main as export_main
from liquidonnx.lfm2_audio import infer as audio_infer
from liquidonnx.lfm2_vl import infer as vl_infer
from liquidonnx.quantize import save_model
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
FAMILIES = {"dense": "lfm2", "moe": "lfm2_moe"}
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


def make_checkpoint(kind: str, path: pathlib.Path, **overrides) -> torch.nn.Module:
    torch.manual_seed(0)
    common = COMMON | overrides
    if kind == "dense":
        config = Lfm2Config(
            num_hidden_layers=4,
            intermediate_size=128,
            layer_types=["conv", "conv", "full_attention", "conv"],
            **common,
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
            **common,
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
    checkpoint = str(root / "checkpoint")
    export_decoder(checkpoint, output_dir)
    for precision in ALL_PRECISIONS:
        export_precision(checkpoint, output_dir, FAMILIES[request.param], precision, reuse_q4=True)
    set_default_decoder(output_dir, list(ALL_PRECISIONS))
    return request.param, model, output_dir


def decoder(output_dir: pathlib.Path, precision: str):
    return load_onnx_session(output_dir / "onnx" / model_file(precision))


def prefill_logits(path: pathlib.Path) -> np.ndarray:
    session = load_onnx_session(path)
    return session.run(None, decoder_inputs(TOKENS, initialize_cache(session), 0))[0][0]


def pytorch_logits(model: torch.nn.Module) -> np.ndarray:
    with torch.no_grad():
        return model(torch.from_numpy(TOKENS)).logits[0].numpy()


def cosine(expected: np.ndarray, actual: np.ndarray) -> float:
    return (expected * actual).sum() / (np.linalg.norm(expected) * np.linalg.norm(actual))


@pytest.mark.parametrize("precision", ["fp32", *ALL_PRECISIONS])
def test_logits_match_pytorch(export, precision: str):
    _, model, output_dir = export
    expected = pytorch_logits(model)
    actual = prefill_logits(output_dir / "onnx" / model_file(precision)).astype(np.float32)

    assert cosine(expected, actual) >= MIN_COSINE[precision]
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

    # q4: int4 body, with the LM head, the embedding table it shares and the sensitive layers int8
    graph, q4_ops = ops("q4")
    assert q4_ops["MatMul"] == routers
    producers = {output: node for node in graph.node for output in node.output}
    lm_head = producers["logits"]
    gather = next(n for n in graph.node if n.op_type == "GatherBlockQuantized")
    assert lm_head.op_type == "MatMulNBits"
    assert producers[gather.input[0]].input[0] == lm_head.input[1]
    assert attributes(lm_head)["bits"] == attributes(gather)["bits"] == 8
    bits = matmul_bits(graph)
    assert bits[4] > 0
    assert bits[8] > 1

    # q4f32: int4 MatMuls, fp32 embedding and LM head
    graph, q4f32_ops = ops("q4f32")
    assert q4f32_ops["GatherBlockQuantized"] == 0
    assert q4f32_ops["MatMul"] == routers + 1
    assert set(matmul_bits(graph)) == {4}

    # q8: int8 MatMuls, the LM head included
    graph, q8_ops = ops("q8")
    assert q8_ops["MatMul"] == routers
    assert set(matmul_bits(graph)) == {8}

    for precision, expert_bits in (("q4", 4), ("q4f32", 4), ("q8", 8)):
        graph, precision_ops = ops(precision)
        assert precision_ops["MoE"] == 0
        assert precision_ops["QMoE"] == routers
        for node in graph.node:
            if node.op_type == "QMoE":
                assert attributes(node)["expert_weight_bits"] == expert_bits
                assert attributes(node)["weights_prepacked"] == 0


def test_q8_shares_one_int8_table(export):
    """The q8 embeddings gather from the int8 weights, scales and zero points of the LM head, and
    no other [V, H] table is left."""
    _, _, output_dir = export
    graph = onnx.load(str(output_dir / "onnx" / model_file("q8")), load_external_data=False).graph
    producers = {output: node for node in graph.node for output in node.output}
    lm_head = producers["logits"]
    (gather,) = [node for node in graph.node if node.op_type == "GatherBlockQuantized"]
    assert lm_head.op_type == "MatMulNBits"
    assert attributes(lm_head)["bits"] == attributes(gather)["bits"] == 8
    assert producers[gather.input[0]].input[0] == lm_head.input[1]
    assert list(gather.input[1:]) == ["input_ids", *lm_head.input[2:]]

    vocab, hidden = COMMON["vocab_size"], COMMON["hidden_size"]
    tables = [
        init.name
        for init in graph.initializer
        if vocab in init.dims and np.prod(init.dims) == vocab * hidden
    ]
    assert tables == [lm_head.input[1]]


def test_q8_logits_match_the_fp32_head_q8(export, tmp_path):
    """The shared int8 table keeps the logits of the q8 decoder with an fp32 LM head and table."""
    _, _, output_dir = export
    checkpoint = str(output_dir.parent / "checkpoint")
    reference = export_decoder(checkpoint, tmp_path, "model_q8.onnx", preset=Q8_FP32_HEAD)
    actual = prefill_logits(output_dir / "onnx" / model_file("q8"))
    assert cosine(prefill_logits(reference), actual) >= MIN_COSINE["q8"]


def test_q8_slices_the_block_padding_off_the_embeddings(export, tmp_path):
    """With blocks longer than the hidden size, the shared table's rows are padded and the
    embeddings sliced back to the hidden size."""
    _, model, output_dir = export
    checkpoint = str(output_dir.parent / "checkpoint")
    path = export_decoder(checkpoint, tmp_path, "model_q8.onnx", preset=Q8, block_size=128)

    graph = onnx.load(str(path), load_external_data=False).graph
    producers = {output: node for node in graph.node for output in node.output}
    embeddings = producers[f"{EMBED_GATHER}/output_0"]
    assert embeddings.op_type == "Slice"
    assert producers[embeddings.input[0]].op_type == "GatherBlockQuantized"
    assert cosine(pytorch_logits(model), prefill_logits(path)) >= MIN_COSINE["q8"]


def test_q8_keeps_the_table_of_an_untied_checkpoint(tmp_path):
    """A checkpoint whose LM head is not its token embedding keeps an fp32 table at q8."""
    model = make_checkpoint("dense", tmp_path / "checkpoint", tie_word_embeddings=False)
    path = export_decoder(str(tmp_path / "checkpoint"), tmp_path, "model_q8.onnx", preset=Q8)

    graph = onnx.load(str(path), load_external_data=False).graph
    gather = next(node for node in graph.node if node.name == EMBED_GATHER)
    assert gather.op_type == "Gather"
    assert cosine(pytorch_logits(model), prefill_logits(path)) >= MIN_COSINE["q8"]


def test_genai_config(export):
    _, _, output_dir = export
    config = json.loads((output_dir / "genai_config.json").read_text())
    assert config["model"]["decoder"]["filename"] == "onnx/model_q4.onnx"
    assert config["model"]["decoder"]["session_options"]["session.set_denormal_as_zero"] == "1"
    for name in ("tokenizer.json", "tokenizer_config.json", "config.json"):
        assert (output_dir / name).exists()


@pytest.mark.parametrize("umask", [0o022, 0o077], ids=oct)
def test_export_files_get_the_umask_mode(umask: int, tmp_path, monkeypatch):
    """onnx.save_model creates external data 0600 and copy2 keeps the checkpoint's modes."""
    make_checkpoint("dense", tmp_path / "checkpoint")
    argv = ["lfm2-export", str(tmp_path / "checkpoint"), "--output-dir", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", argv)
    previous = os.umask(umask)
    try:
        export_main()
    finally:
        os.umask(previous)

    files = [f for f in (tmp_path / "exports").rglob("*") if f.is_file()]
    assert "model.onnx_data" in {f.name for f in files}
    modes = {f.name: oct(stat.S_IMODE(f.stat().st_mode)) for f in files}
    assert modes == {f.name: oct(0o666 & ~umask) for f in files}


@pytest.mark.parametrize("precision", ["fp32", "q4"])
def test_denormal_flush_keeps_logits(export, precision: str):
    _, _, output_dir = export
    flushed = decoder(output_dir, precision)
    entry = flushed.get_session_options().get_session_config_entry("session.set_denormal_as_zero")
    assert entry == "1"

    path = output_dir / "onnx" / model_file(precision)
    plain = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    feed = decoder_inputs(TOKENS, initialize_cache(plain), 0)
    np.testing.assert_array_equal(flushed.run(None, feed)[0], plain.run(None, feed)[0])


def test_q4f16_ignores_a_stale_q4(export, tmp_path):
    """Without reuse_q4, q4f16 converts a fresh q4 build instead of onnx/model_q4.onnx."""
    kind, _, output_dir = export
    (tmp_path / "onnx").mkdir()
    (tmp_path / "onnx" / "model_q4.onnx").write_bytes(b"left over from another checkpoint")

    checkpoint = str(output_dir.parent / "checkpoint")
    export_precision(checkpoint, tmp_path, FAMILIES[kind], "q4f16")

    expected = prefill_logits(output_dir / "onnx" / "model_q4f16.onnx")
    np.testing.assert_array_equal(prefill_logits(tmp_path / "onnx" / "model_q4f16.onnx"), expected)


def add_q4f16(
    export, tmp_path, monkeypatch, args: list[str], other_checkpoint: bool = False
) -> tuple[pathlib.Path, list[str]]:
    """Run `lfm2-export --precision q4f16 <args>` on a copy of the export without its q4f16.

    Returns the copy and the precisions the genai builder built. other_checkpoint doubles the
    q4's final norm, as a q4 of another checkpoint differs from model.onnx.
    """
    kind, _, output_dir = export
    folder = tmp_path / "exports" / "export"
    shutil.copytree(output_dir, folder, ignore=shutil.ignore_patterns("model_q4f16.onnx*"))
    if other_checkpoint:
        q4 = onnx.load(folder / "onnx" / "model_q4.onnx")
        norm = next(t for t in q4.graph.initializer if "final_norm" in t.name)
        norm.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(norm) * 2, norm.name))
        save_model(q4, folder / "onnx" / "model_q4.onnx")

    built = []
    build_decoder = genai_builder.build_decoder

    def spy(model, build_dir, precision, *rest):
        built.append(precision)
        return build_decoder(model, build_dir, precision, *rest)

    monkeypatch.setattr(genai_builder, "build_decoder", spy)
    checkpoint = str(output_dir.parent / "checkpoint")
    argv = ["lfm2-export", checkpoint, "--output-dir", str(tmp_path), "--output-name", "export"]
    monkeypatch.setattr(sys, "argv", [*argv, "--precision", "q4f16", *args])
    export_main(family=FAMILIES[kind])
    return folder, built


@pytest.mark.parametrize(
    ("args", "other_checkpoint", "builds"),
    [
        (["--skip-export"], False, []),
        (["--skip-export"], True, ["int4"]),
        ([], False, ["fp32", "int4"]),
    ],
    ids=["skip-export", "q4-of-another-checkpoint", "fp32-rebuilt"],
)
def test_q4f16_add_on_converts_a_matching_q4(
    export, args: list[str], other_checkpoint: bool, builds: list[str], tmp_path, monkeypatch
):
    """--skip-export converts the folder's q4 when it matches model.onnx and builds a fresh one
    otherwise, as every export that rebuilds fp32 does; the q4f16 is the one of a fresh export."""
    _, _, output_dir = export
    folder, built = add_q4f16(export, tmp_path, monkeypatch, args, other_checkpoint)

    assert built == builds
    for name in ("model_q4f16.onnx", "model_q4f16.onnx_data"):
        assert (folder / "onnx" / name).read_bytes() == (output_dir / "onnx" / name).read_bytes()


def test_reusable_q4_needs_the_block_size(export):
    """A q4 in blocks of 32 is rebuilt for --block-size 64 (the tiny models cannot build that)."""
    _, _, output_dir = export
    assert reusable_q4(output_dir / "onnx", 32)
    assert not reusable_q4(output_dir / "onnx", 64)


def test_reusable_q4_skips_hard_linked_data(export, tmp_path):
    """onnx loads no external data file with several hard links, so such a q4 is rebuilt."""
    _, _, output_dir = export
    onnx_dir = tmp_path / "onnx"
    shutil.copytree(output_dir / "onnx", onnx_dir)
    os.link(onnx_dir / "model_q4.onnx_data", tmp_path / "model_q4.onnx_data")
    assert not reusable_q4(onnx_dir, 32)


@pytest.mark.parametrize("precision", ["fp32", "q4", "q4f16", "q8"])
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


@pytest.mark.parametrize(("value", "expected"), [(None, "1"), ("0", "0")])
def test_onnxruntime_telemetry_is_off_unless_set(value: str | None, expected: str):
    """onnxruntime's telemetry thread can crash a process holding torch at exit."""
    env = {k: v for k, v in os.environ.items() if k != "ORT_DISABLE_TELEMETRY"}
    if value is not None:
        env["ORT_DISABLE_TELEMETRY"] = value
    code = "import os, liquidonnx; print(os.environ['ORT_DISABLE_TELEMETRY'])"
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == expected


@pytest.mark.parametrize(
    ("version", "ep", "warns"),
    [
        ("1.30.0", "cpu", True),
        ("1.31.0.dev20261007001", "cpu", False),
        ("1.31.0", "cpu", False),
        ("1.30.0", "cuda", False),
    ],
)
def test_load_model_warns_on_slow_moe_onnxruntime(
    export, version: str, ep: str, warns: bool, monkeypatch, caplog
):
    """LFM2-MoE on CPU warns once with onnxruntime before 1.31; 1.31 nightlies, CUDA and dense
    models do not."""
    kind, _, output_dir = export
    monkeypatch.setattr(ort, "__version__", version)
    monkeypatch.setattr(ort, "preload_dlls", lambda: None)
    monkeypatch.setattr(og, "Model", lambda config: None)
    with caplog.at_level(logging.WARNING, logger="liquidonnx.genai_runtime"):
        load_model(output_dir, ep=ep)
    messages = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    if warns and kind == "moe":
        assert len(messages) == 1
        assert f"onnxruntime {version}" in messages[0]
        assert "2. Installation" in messages[0]
        assert "uv sync" not in messages[0]  # it would swap onnxruntime-gpu for the CPU nightly
    else:
        assert messages == []


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


class Loaded(Exception):
    """Stops a CLI once it has asked for its model."""


# CLI -> (module, arguments that get it to loading the model)
INFERENCE_CLIS = {
    "lfm2-infer": (text_infer, ["--model", "export", "--prompt", "Hi"]),
    "lfm2-bench": (benchmark, ["--model", "export", "--precision", "q4"]),
    "lfm2-vl-infer": (vl_infer, ["--model", "export"]),
    "lfm2-audio-infer": (audio_infer, ["export", "--prompt", "Hi"]),
}


@pytest.mark.parametrize(
    ("flags", "tf32"), [([], True), (["--no-tf32"], False)], ids=["default", "no-tf32"]
)
@pytest.mark.parametrize("cli", INFERENCE_CLIS)
def test_inference_clis_turn_tf32_off_on_request(cli: str, flags: list, tf32: bool, monkeypatch):
    """--no-tf32 reaches load_model; without it the CLIs keep the CUDA EP's TF32 default."""
    module, arguments = INFERENCE_CLIS[cli]
    loads = []

    def load(model_dir, files=None, ep="cpu", tf32=True):
        loads.append((ep, tf32))
        raise Loaded

    monkeypatch.setattr(module, "load_model", load)
    monkeypatch.setattr(sys, "argv", [cli, *arguments, "--ep", "cuda", *flags])
    with pytest.raises(Loaded):
        module.main()
    assert loads == [("cuda", tf32)]


class Parsed(Exception):
    """Stops a CLI once it has parsed its arguments."""


@pytest.mark.parametrize(
    ("cli", "max_tokens"),
    [("lfm2-infer", 4096), ("lfm2-vl-infer", 4096), ("lfm2-bench", 20), ("lfm2-audio-infer", None)],
)
def test_inference_clis_max_tokens_defaults(cli: str, max_tokens: int | None, monkeypatch):
    """lfm2-infer (and lfm2-moe-infer, the same CLI) and lfm2-vl-infer leave room for
    LFM2.5-2.6B and LFM2.5-8B-A1B to think and then answer. lfm2-bench keeps its short run and
    lfm2-audio-infer its per-mode defaults."""
    module, arguments = INFERENCE_CLIS[cli]
    parsed = []
    parse_args = argparse.ArgumentParser.parse_args

    def parse(parser, *args, **kwargs):
        parsed.append(parse_args(parser, *args, **kwargs))
        raise Parsed

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", parse)
    monkeypatch.setattr(sys, "argv", [cli, *arguments])
    with pytest.raises(Parsed):
        module.main()
    assert parsed[0].max_tokens == max_tokens


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
