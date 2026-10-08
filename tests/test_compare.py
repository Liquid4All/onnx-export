"""
lfm2-compare's reference cache and devices, mostly with stubs in place of the models.

Run with:
    uv run pytest tests/test_compare.py -v
"""

import inspect
import json
import logging
import pathlib
import re
import sys

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer, Lfm2Config, Lfm2ForCausalLM

from liquidonnx import compare
from liquidonnx.compare import text
from liquidonnx.compare import vl as compare_vl
from liquidonnx.genai_builder import export_decoder
from liquidonnx.genai_runtime import EXECUTION_PROVIDERS, load_model
from liquidonnx.session import load_onnx_session

ITEMS = [
    {"ids": np.arange(6), "logits": np.linspace(-1, 1, 48, dtype=np.float32).reshape(6, 8)},
    {"ids": np.arange(3), "logits": np.ones((3, 8), np.float32)},
]


@pytest.fixture
def checkpoint(tmp_path, monkeypatch) -> pathlib.Path:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    return checkpoint


@pytest.fixture
def runs(monkeypatch) -> list:
    """The reference runs (device, and whether TF32 matmuls and convolutions were allowed: both
    are at first, and are restored afterwards) and the CUDA cache releases."""
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", True)
    runs = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: runs.append("empty_cache"))

    def reference(checkpoint: pathlib.Path, prompts: int, max_new: int, device: str) -> list[dict]:
        tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        runs.append((device, *tf32))
        return ITEMS

    monkeypatch.setattr(text, "reference", reference)
    return runs


def check_reference(checkpoint: pathlib.Path, device: str = "cpu"):
    items = compare.cached_reference(str(checkpoint), checkpoint, "text", 2, 4, device)
    assert len(items) == len(ITEMS)
    for got, want in zip(items, ITEMS, strict=True):
        assert got.keys() == want.keys()
        for key in want:
            np.testing.assert_array_equal(got[key], want[key])


def cache_files(checkpoint: pathlib.Path) -> list[pathlib.Path]:
    return sorted((checkpoint.parent / "cache" / "liquidonnx" / "compare").iterdir())


def test_reference_is_cached(checkpoint, runs):
    check_reference(checkpoint)
    check_reference(checkpoint)
    assert len(runs) == 1
    [path] = cache_files(checkpoint)
    assert path.suffix == ".npz"


# A reference copied to the Mac had 917,504 trailing bytes, and np.load failed with "Bad magic
# number for file header", as it does when the trailing bytes end with a zip directory.
@pytest.mark.parametrize(
    "corrupt",
    [
        lambda raw: raw + bytes(917_504),
        lambda raw: raw + raw[len(raw) // 2 :],
        lambda raw: raw[: len(raw) // 2],
        lambda raw: b"",
    ],
    ids=["trailing-zeros", "trailing-directory", "truncated", "empty"],
)
def test_corrupt_reference_is_recomputed(checkpoint, runs, corrupt, caplog):
    check_reference(checkpoint)
    [path] = cache_files(checkpoint)
    path.write_bytes(corrupt(path.read_bytes()))

    with caplog.at_level(logging.WARNING):
        check_reference(checkpoint)
    assert len(runs) == 2
    assert f"Recomputing the unreadable reference {path}" in caplog.text

    check_reference(checkpoint)
    assert len(runs) == 2
    assert cache_files(checkpoint) == [path]


def test_failed_write_keeps_the_previous_reference(tmp_path, monkeypatch):
    path = tmp_path / "reference.npz"
    compare.save_reference(path, ITEMS)
    raw = path.read_bytes()

    def savez(file, **arrays):
        file.write(b"partial")
        raise OSError("No space left on device")

    monkeypatch.setattr(np, "savez", savez)
    with pytest.raises(OSError, match="No space left"):
        compare.save_reference(path, ITEMS[:1])
    assert path.read_bytes() == raw
    assert list(tmp_path.iterdir()) == [path]


def test_cpu_and_cuda_references_never_mix(checkpoint, runs):
    """Each device computes its reference once; CUDA's without TF32 and then hands the GPU memory
    back, CPU's with the flags as found."""
    for device in ("cpu", "cuda", "cpu", "cuda"):
        check_reference(checkpoint, device)
    assert runs == [("cpu", True, True), ("cuda", False, False), "empty_cache"]
    assert len(cache_files(checkpoint)) == 2


def test_cpu_references_keep_their_names(tmp_path, monkeypatch, runs):
    """CPU references cached before --device are still found, like the verified 350M one."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    snapshot = tmp_path / "snapshots" / "9e6c6ccf47cd318696e137d381a7ded8fe4df09f"
    for device in EXECUTION_PROVIDERS:
        compare.cached_reference("LiquidAI/LFM2.5-350M", snapshot, "text", 8, 48, device)
    assert sorted(p.name for p in (tmp_path / "liquidonnx" / "compare").iterdir()) == [
        "LFM2.5-350M-text-0b7c2174540b.npz",
        "LFM2.5-350M-text-cuda-0b7c2174540b.npz",
    ]


def test_vl_references_per_image_splitting(checkpoint, monkeypatch):
    """A VL reference is computed once per --image-splitting setting; the checkpoint's own setting
    (None) gets a key of its own, so references cached when VL always resized once are not
    reused."""
    runs = []

    def reference(checkpoint, max_new, device, image_splitting):
        runs.append(image_splitting)
        return ITEMS

    monkeypatch.setattr(compare_vl, "reference", reference)
    for image_splitting in (None, True, False, None, False, True):
        compare.cached_reference(
            str(checkpoint), checkpoint, "vl", 8, 4, image_splitting=image_splitting
        )
    assert runs == [None, True, False]
    assert len(cache_files(checkpoint)) == 3


@pytest.mark.parametrize(
    "flag,image_splitting",
    [([], None), (["--image-splitting"], True), (["--no-image-splitting"], False)],
    ids=["checkpoint", "on", "off"],
)
def test_vl_image_splitting_reaches_the_reference(flag, image_splitting, monkeypatch, tmp_path):
    calls = []
    decoder = {
        "teacher_forced": {"kl_mean": 0.0, "kl_max": 0.0, "top1": 1.0, "max_abs": 0.0},
        "greedy": {"exact": 1, "of": 1, "prefix_frac": 1.0},
        "size_mb": 1.0,
    }
    monkeypatch.setattr(compare, "resolve_checkpoint", lambda model: tmp_path / "snapshot")
    monkeypatch.setattr(
        compare, "cached_reference", lambda *args, **kwargs: calls.append(kwargs) or []
    )
    monkeypatch.setattr(compare_vl, "precisions", lambda export: ["q4"])
    monkeypatch.setattr(compare_vl, "score", lambda *args: {"decoder": decoder})
    argv = ["vl", "--model", "m", "--export", str(tmp_path), "--no-genai", *flag]
    argv += ["--output", str(tmp_path / "c.json")]
    monkeypatch.setattr(sys, "argv", ["lfm2-compare", *argv])

    compare.main()

    assert calls == [{"image_splitting": image_splitting}]


@pytest.mark.parametrize("device", EXECUTION_PROVIDERS)
@pytest.mark.parametrize("family", ["text", "vl", "audio"])
def test_scores_run_on_the_device(family: str, device: str, monkeypatch, tmp_path):
    """Every ONNX session and onnxruntime-genai model of a score runs on the device, without TF32."""
    module = compare.load_family(family)
    loads = []

    def stub(load):
        def record(*args, **kwargs):
            call = inspect.signature(load).bind(*args, **kwargs)
            call.apply_defaults()
            loads.append(call.arguments)
            raise RuntimeError("stub")

        return record

    monkeypatch.setattr(module, "load_onnx_session", stub(load_onnx_session))
    monkeypatch.setattr(module, "load_model", stub(load_model))
    (tmp_path / "genai_config.json").write_text(json.dumps({"model": {"eos_token_id": 7}}))
    max_new = (16,) if family == "audio" else ()
    row = module.score(tmp_path, "fp32", [], True, *max_new, device)

    assert all(part == {"error": "stub"} for part in row.values())
    assert len(loads) == len(row)
    assert all(load["ep"] == device and load["tf32"] is False for load in loads)


@pytest.mark.parametrize("device", [None, *EXECUTION_PROVIDERS])
def test_device_reaches_the_reference_and_the_scores(device, monkeypatch, tmp_path):
    """--device (default cpu) picks the reference and the sessions, and is in the results."""
    calls = []
    row = {
        "decoder": {
            "teacher_forced": {"kl_mean": 0.0, "kl_max": 0.0, "top1": 1.0, "max_abs": 0.0},
            "greedy": {"exact": 1, "of": 1, "prefix_frac": 1.0},
            "size_mb": 1.0,
        }
    }
    monkeypatch.setattr(compare, "resolve_checkpoint", lambda model: tmp_path / "snapshot")
    monkeypatch.setattr(compare, "cached_reference", lambda *args: calls.append(args[-1]) or [])
    monkeypatch.setattr(text, "precisions", lambda export: ["q4"])
    monkeypatch.setattr(text, "score", lambda *args: calls.append(args[-1]) or dict(row))
    monkeypatch.setattr(text, "eos_ids", lambda export: {7})
    genai = {"greedy": row["decoder"]["greedy"], "prompt_ids_equal": "1/1"}
    monkeypatch.setattr(text, "score_genai", lambda *args: calls.append(args[-1]) or genai)
    output = tmp_path / "compare.json"
    argv = ["text", "--model", "m", "--export", str(tmp_path), "--output", str(output)]
    if device:
        argv += ["--device", device]
    monkeypatch.setattr(sys, "argv", ["lfm2-compare", *argv])

    compare.main()

    expected = device or "cpu"
    assert calls == [expected, expected, expected]
    assert json.loads(output.read_text())["device"] == expected
    header = output.with_suffix(".md").read_text().splitlines()[0]
    assert header.startswith("### m (text) on ") and header.endswith(f" ({expected})")


def test_cpu_run_of_a_tiny_export(tmp_path, monkeypatch):
    """lfm2-compare text on CPU, end to end, on a tiny random LFM2 with the LFM2 vocabulary (the
    prompts need it): the fp32 decoder and onnxruntime-genai reproduce the PyTorch reference."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    checkpoint, export = tmp_path / "checkpoint", tmp_path / "export"
    tokenizer = AutoTokenizer.from_pretrained("LiquidAI/LFM2-350M")
    torch.manual_seed(0)
    config = Lfm2Config(
        vocab_size=len(tokenizer),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        layer_types=["conv", "full_attention"],
        max_position_embeddings=256,
    )
    Lfm2ForCausalLM(config).eval().save_pretrained(checkpoint)
    tokenizer.save_pretrained(checkpoint)
    export_decoder(str(checkpoint), export)
    output = tmp_path / "compare.json"
    argv = ["text", "--model", str(checkpoint), "--export", str(export), "--output", str(output)]
    monkeypatch.setattr(sys, "argv", ["lfm2-compare", *argv, "--prompts", "2", "--max-new", "8"])

    compare.main()

    results = json.loads(output.read_text())
    assert results["device"] == "cpu"
    assert list(results["rows"]) == ["fp32"]
    row = results["rows"]["fp32"]
    assert row["decoder"]["teacher_forced"]["kl_max"] < 1e-6
    assert row["decoder"]["teacher_forced"]["top1"] == 1.0
    assert row["decoder"]["greedy"]["exact"] == 2
    assert row["genai"] == {"greedy": row["decoder"]["greedy"], "prompt_ids_equal": "2/2"}
    [cache] = (tmp_path / "cache" / "liquidonnx" / "compare").iterdir()
    assert re.fullmatch(r"checkpoint-text-[0-9a-f]{12}\.npz", cache.name)


def test_onnxruntime_sessions_come_before_genai(monkeypatch, tmp_path):
    """Every precision's onnxruntime sessions run before the first onnxruntime-genai model."""
    events = []
    decoder = {
        "teacher_forced": {"kl_mean": 0.0, "kl_max": 0.0, "top1": 1.0, "max_abs": 0.0},
        "greedy": {"exact": 1, "of": 1, "prefix_frac": 1.0},
        "size_mb": 1.0,
    }
    genai = {"greedy": decoder["greedy"], "prompt_ids_equal": "1/1"}
    monkeypatch.setattr(compare, "resolve_checkpoint", lambda model: tmp_path / "snapshot")
    monkeypatch.setattr(compare, "cached_reference", lambda *args: [])
    monkeypatch.setattr(text, "precisions", lambda export: ["q4", "q4f16"])
    monkeypatch.setattr(text, "eos_ids", lambda export: {7})
    monkeypatch.setattr(
        text,
        "score",
        lambda e, p, r, g, d: events.append(("onnxruntime", p, g)) or {"decoder": decoder},
    )
    monkeypatch.setattr(
        text, "score_genai", lambda e, p, *rest: events.append(("genai", p)) or genai
    )
    argv = ["text", "--model", "m", "--export", str(tmp_path), "--output", str(tmp_path / "c.json")]
    monkeypatch.setattr(sys, "argv", ["lfm2-compare", *argv])

    compare.main()

    assert events == [
        ("onnxruntime", "q4", False),
        ("onnxruntime", "q4f16", False),
        ("genai", "q4"),
        ("genai", "q4f16"),
    ]
