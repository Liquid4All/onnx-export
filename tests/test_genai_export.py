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

import numpy as np
import onnx
import pytest
import torch
from transformers import (
    AutoTokenizer,
    Lfm2Config,
    Lfm2ForCausalLM,
    Lfm2MoeConfig,
    Lfm2MoeForCausalLM,
)

from liquidonnx.lfm2.export import (
    ALL_PRECISIONS,
    derive_precision,
    export_model,
    set_default_decoder,
)
from liquidonnx.session import decoder_inputs, initialize_cache, load_onnx_session, update_cache

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
    export_model(str(root / "checkpoint"), output_dir)
    for precision in ALL_PRECISIONS:
        derive_precision(output_dir / "onnx", precision)
    set_default_decoder(output_dir, list(ALL_PRECISIONS))
    return request.param, model, output_dir


def model_file(output_dir: pathlib.Path, precision: str) -> pathlib.Path:
    name = "model.onnx" if precision == "fp32" else f"model_{precision}.onnx"
    return output_dir / "onnx" / name


def run_cached(session, tokens: np.ndarray, prefill: int) -> np.ndarray:
    """Logits at positions prefill-1 .. end: prefill in one call, then one token per call."""
    cache, outputs = initialize_cache(session), session.get_outputs()
    result = session.run(None, decoder_inputs(session, tokens[:, :prefill], cache, 0))
    update_cache(cache, result, outputs)
    rows = [result[0][0, -1]]
    for pos in range(prefill, tokens.shape[1]):
        result = session.run(None, decoder_inputs(session, tokens[:, pos : pos + 1], cache, pos))
        update_cache(cache, result, outputs)
        rows.append(result[0][0, -1])
    return np.stack(rows).astype(np.float32)


@pytest.mark.parametrize("precision", ["fp32", *ALL_PRECISIONS])
def test_logits_match_pytorch(export, precision: str):
    _, model, output_dir = export
    with torch.no_grad():
        expected = model(torch.from_numpy(TOKENS)).logits[0].numpy()

    session = load_onnx_session(model_file(output_dir, precision), ["CPUExecutionProvider"])
    feed = decoder_inputs(session, TOKENS, initialize_cache(session), 0)
    actual = session.run(None, feed)[0][0].astype(np.float32)

    cosine = (expected * actual).sum() / (np.linalg.norm(expected) * np.linalg.norm(actual))
    assert cosine >= MIN_COSINE[precision]
    if precision == "fp32":
        np.testing.assert_allclose(actual, expected, atol=1e-5)


@pytest.mark.parametrize("precision", ["fp32", "fp16", "q8", "q4"])
def test_cached_decode_matches_prefill(export, precision: str):
    _, _, output_dir = export
    session = load_onnx_session(model_file(output_dir, precision), ["CPUExecutionProvider"])
    feed = decoder_inputs(session, TOKENS, initialize_cache(session), 0)
    prefill = session.run(None, feed)[0][0, 3:].astype(np.float32)
    # MatMulNBits takes different kernels for one token and for a whole prompt.
    atol = 5e-3 if precision == "q4" else 1e-3
    np.testing.assert_allclose(run_cached(session, TOKENS, prefill=4), prefill, atol=atol)


def test_graph_layout(export):
    kind, _, output_dir = export

    def ops(precision):
        graph = onnx.load(str(model_file(output_dir, precision)), load_external_data=False).graph
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


def test_genai_runtime_matches_onnxruntime(export):
    og = pytest.importorskip("onnxruntime_genai")
    kind, _, output_dir = export
    try:
        model = og.Model(str(output_dir))
    except RuntimeError as e:
        pytest.skip(f"onnxruntime-genai {og.__version__} cannot load {kind}: {e}")

    max_new = 8
    params = og.GeneratorParams(model)
    params.set_search_options(do_sample=False, max_length=TOKENS.shape[1] + max_new)
    generator = og.Generator(model, params)
    generator.append_tokens(TOKENS.astype(np.int32))
    while not generator.is_done():
        generator.generate_next_token()
    genai_tokens = list(generator.get_sequence(0)[TOKENS.shape[1] :])

    session = load_onnx_session(output_dir / "onnx/model_q4.onnx", ["CPUExecutionProvider"])
    cache, outputs = initialize_cache(session), session.get_outputs()
    tokens, past, ort_tokens = TOKENS, 0, []
    for _ in range(len(genai_tokens)):
        result = session.run(None, decoder_inputs(session, tokens, cache, past))
        update_cache(cache, result, outputs)
        past += tokens.shape[1]
        ort_tokens.append(int(result[0][0, -1].argmax()))
        tokens = np.array([[ort_tokens[-1]]], dtype=np.int64)
    assert genai_tokens == ort_tokens
