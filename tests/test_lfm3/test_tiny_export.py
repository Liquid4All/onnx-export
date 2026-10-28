"""
lfm3-export's main() on a tiny random checkpoint in LFM3-1.7B's format (tiny.py): fp32 against its
reference, LFM3-1.7B's modeling code, the other precisions against fp32, the recipes of the
quantized files, the checkpoint's config.json; and lfm3-infer's main() on the export (no download).

Run with:
    uv run pytest tests/test_lfm3/test_tiny_export.py -v
"""

import builtins
import json
import shutil
import sys

import numpy as np
import onnx
import pytest
import torch

from liquidonnx.lfm3 import infer
from liquidonnx.lfm3 import session as text_session
from liquidonnx.lfm3.export import main as export_main
from liquidonnx.lfm3.session import CachedDecoder, load_tokenizer
from liquidonnx.session import load_onnx_session

from ..test_lfm3_vl.test_tiny_export import (
    assert_one_int8_table,
    assert_q4_recipe,
    attributes,
    cosine,
)
from .tiny import TINY_CONFIG

# A prefill of 20 tokens, past the sliding window of 8 and the conv kernel of 7, then 8 cached steps
IDS = [TINY_CONFIG["bos_token_id"], *np.random.default_rng(0).integers(1000, 120000, 27).tolist()]
PREFILL = 20


@pytest.fixture(scope="module")
def sessions(tiny_export):
    return {path.name: load_onnx_session(path) for path in (tiny_export / "onnx").glob("*.onnx")}


def logits(session) -> np.ndarray:
    """[len(IDS), vocab]: one prefill of IDS[:PREFILL] with every position's logits, then a cached
    step per token."""
    decoder = CachedDecoder(session)
    rows = [decoder.step(IDS[:PREFILL], logits_to_keep=np.array([0], dtype=np.int64))[0]]
    for token in IDS[PREFILL:]:
        rows.append(decoder.step([token], logits_to_keep=np.array([1], dtype=np.int64))[0])
    return np.concatenate(rows)


def test_fp32_prefill_and_cached_steps_match_reference(sessions, tiny_reference):
    with torch.no_grad():
        expected = tiny_reference(input_ids=torch.tensor([IDS]), use_cache=False).logits[0].numpy()
    np.testing.assert_allclose(logits(sessions["model.onnx"]), expected, rtol=1e-4, atol=2e-3)


# Random weights 64 wide quantize far worse than trained ones; these floors catch broken wiring,
# the real checkpoint's tests hold the quality bars. The int4 logits measure a cosine of 0.73 at
# worst and 0.85 on average, q8 0.994 and 0.998.
@pytest.mark.parametrize(
    "precision,min_floor,mean_floor",
    [
        ("fp16", 0.9999, 0.9999),
        ("q8", 0.98, 0.99),
        ("q8f16", 0.98, 0.99),
        ("q4", 0.6, 0.8),
        ("q4f16", 0.6, 0.8),
        ("q4f32", 0.6, 0.8),
    ],
)
def test_precisions_track_fp32(sessions, precision, min_floor, mean_floor):
    similarity = cosine(logits(sessions[f"model_{precision}.onnx"]), logits(sessions["model.onnx"]))
    assert similarity.min() > min_floor and similarity.mean() > mean_floor


@pytest.mark.parametrize("precision", ["q8", "q8f16"])
def test_q8_files_store_one_int8_table(tiny_export, precision):
    graph = assert_one_int8_table(tiny_export / "onnx" / f"model_{precision}.onnx")
    assert {attributes(n)["bits"] for n in graph.node if n.op_type == "MatMulNBits"} == {8}


@pytest.mark.parametrize("precision", ["q4", "q4f16"])
def test_q4_files_follow_the_lfm2_recipe(tiny_export, precision):
    onnx_dir = tiny_export / "onnx"
    assert_q4_recipe(onnx_dir / f"model_{precision}.onnx", onnx_dir / "model.onnx")


def assert_q4f32_recipe(path, fp32_path, zero_points: bool):
    """Every MatMul but the LM head int4 (round to nearest) in blocks of 32 at accuracy_level 4,
    with zero points only if asymmetric; the head an fp32 MatMul over the transposed token table,
    which the embedding gathers from as it is."""
    graph = onnx.load(str(path), load_external_data=False).graph
    producers = {output: node for node in graph.node for output in node.output}
    head = producers["logits"]
    transpose = producers[head.input[1]]
    assert head.op_type == "MatMul" and transpose.op_type == "Transpose"
    table = next(init for init in graph.initializer if init.name == transpose.input[0])
    assert table.data_type == onnx.TensorProto.FLOAT
    assert table.name in {node.input[0] for node in graph.node if node.op_type == "Gather"}
    assert "GatherBlockQuantized" not in {node.op_type for node in graph.node}

    fp32 = onnx.load(str(fp32_path), load_external_data=False).graph
    matmuls = [node for node in graph.node if node.op_type == "MatMulNBits"]
    assert len(matmuls) == [node.op_type for node in fp32.node].count("MatMul") - 1
    for node in matmuls:
        attrs = attributes(node)
        assert (attrs["bits"], attrs["block_size"], attrs["accuracy_level"]) == (4, 32, 4)
        assert (len(node.input) == 4 and node.input[3] != "") == zero_points, node.name


def test_q4f32_file_follows_its_recipe(tiny_export):
    onnx_dir = tiny_export / "onnx"
    assert_q4f32_recipe(onnx_dir / "model_q4f32.onnx", onnx_dir / "model.onnx", zero_points=False)


def test_q4_asymmetric_writes_q4f32_zero_points(
    tiny_checkpoint, tiny_export, tmp_path, monkeypatch
):
    """--q4-asymmetric with --skip-export: q4f32 from the folder's model.onnx, with zero points."""
    onnx_dir = tmp_path / "exports" / "tiny-ONNX" / "onnx"
    onnx_dir.mkdir(parents=True)
    for name in ("model.onnx", "model.onnx_data"):
        shutil.copy2(tiny_export / "onnx" / name, onnx_dir)
    argv = ["lfm3-export", str(tiny_checkpoint), "--output-dir", str(tmp_path)]
    argv += ["--output-name", "tiny-ONNX", "--skip-export", "--precision", "q4f32"]
    monkeypatch.setattr(sys, "argv", [*argv, "--q4-asymmetric"])
    export_main()
    assert_q4f32_recipe(onnx_dir / "model_q4f32.onnx", onnx_dir / "model.onnx", zero_points=True)


def test_export_writes_the_checkpoint_config(tiny_export):
    config = json.loads((tiny_export / "config.json").read_text())
    assert config.pop("transformers.js_config") and config == TINY_CONFIG


@pytest.mark.parametrize("think", [True, False], ids=["think", "no-think"])
def test_infer_cli_answers_a_prompt(tiny_export, monkeypatch, capsys, think):
    """lfm3-infer --cpu without --precision loads the first exported file the CPU prefers (q4),
    prompts with the chat template (an empty think block with --no-think), prints the greedy
    answer of --max-tokens tokens and answers a second line with the conversation so far."""
    loaded, steps = [], []
    load_session, step = text_session.load_session, CachedDecoder.step
    lines = ["Again"]

    def record_load(path, **kwargs):
        loaded.append(path.name)
        return load_session(path, **kwargs)

    def record_step(decoder, ids, **inputs):
        step_logits = step(decoder, ids, **inputs)
        steps.append((ids, int(np.argmax(step_logits[0, -1]))))
        return step_logits

    def next_line(prompt=""):
        if not lines:
            raise EOFError
        return lines.pop(0)

    monkeypatch.setattr(text_session, "load_session", record_load)
    monkeypatch.setattr(CachedDecoder, "step", record_step)
    monkeypatch.setattr(builtins, "input", next_line)
    argv = ["lfm3-infer", "--model", str(tiny_export), "--prompt", "Hi", "--cpu"]
    argv += ["--max-tokens", "3", *([] if think else ["--no-think"])]
    monkeypatch.setattr(sys, "argv", argv)
    infer.main()

    assert loaded == ["model_q4.onnx"]
    tokenizer = load_tokenizer(tiny_export)
    prompt, tokens = steps[0][0], [token for _, token in steps]
    opening = "<think>" if think else "<think></think>"
    assert tokenizer.decode(prompt) == (
        f"<|startoftext|><|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n{opening}"
    )
    assert len(tokens) == 6
    answer = "".join(tokenizer.decode([token]) for token in tokens[:3])
    assert answer.strip() and f"User: Hi\nAssistant: {answer}" in capsys.readouterr().out
    # The template gives an answer before the last user turn an empty think block
    assert tokenizer.decode(steps[3][0]) == (
        "<|startoftext|><|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think></think>"
        f"{answer}<|im_end|>\n<|im_start|>user\nAgain<|im_end|>\n<|im_start|>assistant\n{opening}"
    )


def test_infer_cli_answers_up_to_4096_tokens():
    assert infer.build_parser().parse_args(["--model", "export"]).max_tokens == 4096
