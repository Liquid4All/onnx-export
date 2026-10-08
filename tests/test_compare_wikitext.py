"""
lfm2-compare wikitext on tiny random models against synthetic references, and its llama.cpp
KL-divergence base files.

The text and VL checkpoints are built from a config with the LFM2 vocabulary, so only the LFM2
tokenizer and the LFM2.5-VL-1.6B processor are downloaded; the audio export is the synthetic one
from tests/test_lfm2_audio/synthetic.py.

Run with:
    uv run pytest tests/test_compare_wikitext.py -v
"""

import json
import pathlib
import sys

import numpy as np
import pytest
import torch
from test_lfm2_audio.synthetic import build_model_dir
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Lfm2Config,
    Lfm2ForCausalLM,
    Lfm2VlConfig,
    Lfm2VlForConditionalGeneration,
)

from liquidonnx import compare
from liquidonnx.compare import kld_base, wikitext
from liquidonnx.embeddings import embed
from liquidonnx.genai_builder import export_decoder, export_precision
from liquidonnx.lfm2_audio import export as audio_export
from liquidonnx.lfm2_vl import export as vl_export
from liquidonnx.quantize import get_total_model_size_mb
from liquidonnx.session import decoder_inputs, initialize_cache, load_onnx_session

N_CTX, N_CHUNK, BOS = 32, 4, 1
SCORED = N_CHUNK * (N_CTX // 2 - 1)
TEXT = "The quick brown fox jumps over the lazy dog.\r\nPack my box with five dozen liquor jugs.\n"


def random_tokens(vocab: int) -> np.ndarray:
    """A stream [N_CHUNK, N_CTX] of int32 ids that starts with BOS, as llama.cpp's does."""
    tokens = np.random.default_rng(0).integers(0, vocab, (N_CHUNK, N_CTX), dtype=np.int32)
    tokens[0, 0] = BOS
    return tokens


def write_reference(path: pathlib.Path, logits_of, tokens: np.ndarray, model: str) -> pathlib.Path:
    """The file and .json sidecar make_ref.py writes, from logits_of(ids [B, S]) -> [B, S, V]:
    BOS in place of the first token of each chunk, the second half of each chunk kept."""
    ids = tokens.astype(np.int64)
    ids[:, 0] = BOS
    logits = logits_of(ids)[:, N_CTX // 2 : N_CTX - 1]
    n_vocab = logits.shape[-1]
    with path.open("wb") as f:
        kld_base.write_header(f, N_CTX, n_vocab, tokens)
        for chunk in logits:
            f.write(kld_base.encode_rows(chunk, n_vocab).tobytes())
    meta = {"model": model, "add_bos": True, "bos": BOS}
    path.with_name(f"{path.name}.json").write_text(json.dumps(meta))
    return path


def torch_logits(model):
    def logits_of(ids: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return model(torch.from_numpy(ids)).logits.numpy()

    return logits_of


def run(monkeypatch, *argv) -> dict:
    """lfm2-compare wikitext with argv (which names --output); the results JSON."""
    output = pathlib.Path(argv[argv.index("--output") + 1])
    monkeypatch.setattr(sys, "argv", ["lfm2-compare", "wikitext", *map(str, argv)])
    compare.main()
    return json.loads(output.read_text())


# === llama.cpp KL-divergence base files ===


@pytest.mark.parametrize("n_vocab", [7, 8])
def test_base_file_layout(tmp_path, n_vocab: int):
    """perplexity.cpp's layout: rows of 2 * ceil(n_vocab / 2) + 4 uint16 holding the
    log-probabilities over the top 16 nats, and the rest at max - 16."""
    rng = np.random.default_rng(0)
    tokens = rng.integers(0, n_vocab, (3, 10), dtype=np.int32)
    logits = (rng.standard_normal((3, 4, n_vocab)) * 8).astype(np.float32)
    logits[:, :, 0] = -100
    path = tmp_path / "base.kld"
    with path.open("wb") as f:
        kld_base.write_header(f, 10, n_vocab, tokens)
        for chunk in logits:
            f.write(kld_base.encode_rows(chunk, n_vocab).tobytes())

    width = 2 * ((n_vocab + 1) // 2) + 4
    assert path.stat().st_size == 8 + 12 + 4 * 30 + 2 * 3 * 4 * width
    n_ctx, vocab, stream, rows = kld_base.open_rows(path)
    assert (n_ctx, vocab, rows.shape) == (10, n_vocab, (3, 4, width))
    np.testing.assert_array_equal(stream, tokens)
    for chunk, block in zip(logits, rows, strict=True):
        decoded = kld_base.decode_rows(np.asarray(block), n_vocab)
        exact = kld_base.log_softmax(chunk)
        floor = exact.max(axis=1, keepdims=True) - 16
        step = 16 / 65535
        kept = exact > floor
        np.testing.assert_allclose(decoded[kept], exact[kept], atol=step)
        np.testing.assert_allclose(
            decoded[~kept], np.broadcast_to(floor, exact.shape)[~kept], atol=1e-5
        )


@pytest.mark.parametrize(
    "corrupt",
    [lambda raw: raw[:-2], lambda raw: raw + b"\0\0", lambda raw: raw[:30], lambda raw: b"x" + raw],
    ids=["truncated", "trailing", "inside-tokens", "magic"],
)
def test_damaged_base_files_are_rejected(tmp_path, corrupt):
    path = tmp_path / "base.kld"
    with path.open("wb") as f:
        kld_base.write_header(f, 4, 3, np.zeros((2, 4), np.int32))
        f.write(kld_base.encode_rows(np.zeros((2, 3), np.float32), 3).tobytes())
    kld_base.open_rows(path)
    path.write_bytes(corrupt(path.read_bytes()))
    with pytest.raises(ValueError):
        kld_base.open_rows(path)


def test_kld_statistics():
    """KLD over the reference's top 16 nats, and its SE over the chunk means."""
    base = np.log(np.array([[0.5, 0.5, 1e-9]], np.float32))  # the third is below max - 16
    stats = kld_base.KldStats()
    stats.add(np.log(np.array([[0.75, 0.25, 1e-9]], np.float32)), base, np.array([0]))
    stats.add(base.copy(), base, np.array([1]))
    summary = stats.summary()
    first = 0.5 * np.log(0.5 / 0.75) + 0.5 * np.log(0.5 / 0.25)
    assert summary["kld_mean"] == pytest.approx(first / 2, rel=1e-5)
    assert summary["kld_mean_err_chunks"] == pytest.approx(first / 2, rel=1e-5)
    assert summary["same_top_pct"] == 100.0
    assert stats.arrays()["kld"].shape == (2, 1)


# === Text ===


@pytest.fixture(scope="module")
def text(tmp_path_factory):
    """(checkpoint, PyTorch model, export with fp32, q8 and q4) of a tiny LFM2."""
    root = tmp_path_factory.mktemp("wikitext")
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
    model = Lfm2ForCausalLM(config).eval()
    model.save_pretrained(root / "checkpoint")
    tokenizer.save_pretrained(root / "checkpoint")
    export = root / "LFM2.5-tiny-ONNX"
    export_decoder(str(root / "checkpoint"), export)
    for precision in ("q4", "q8"):
        export_precision(str(root / "checkpoint"), export, "lfm2", precision)
    return root / "checkpoint", model, export


def test_scores_and_the_gate(text, tmp_path, monkeypatch):
    """fp32 scores ~0 against the PyTorch reference and q8 and q4 more; a precision above
    --max-kld fails the command, which still writes its results."""
    _, model, export = text
    reference = tmp_path / "ref.kld"
    write_reference(reference, torch_logits(model), random_tokens(model.config.vocab_size), "tiny")
    output, dump = tmp_path / "wikitext.json", tmp_path / "scores"
    argv = ["--export", export, "--reference", reference, "--output", output]

    results = run(monkeypatch, *argv, "--precision", "fp32", "q8", "q4", "--dump", dump)
    rows = results["rows"]
    assert rows["fp32"]["kld_mean"] < 1e-6
    assert rows["fp32"]["same_top_pct"] == 100.0
    assert rows["fp32"]["kld_mean"] < rows["q8"]["kld_mean"] < rows["q4"]["kld_mean"]
    assert all(row["tokens"] == SCORED for row in rows.values())
    assert all(row["ceiling"] is None and row["pass"] is None for row in rows.values())
    assert {k: results["reference"][k] for k in ("model", "bos", "n_ctx", "chunks")} == {
        "model": "tiny",
        "bos": BOS,
        "n_ctx": N_CTX,
        "chunks": N_CHUNK,
    }
    for precision, row in rows.items():
        with np.load(dump / f"{export.name}.{precision}.cpu.npz") as arrays:
            assert arrays["kld"].shape == (N_CHUNK, N_CTX // 2 - 1)
            assert arrays["kld"].mean() == pytest.approx(row["kld_mean"])

    limit = (rows["q8"]["kld_mean"] + rows["q4"]["kld_mean"]) / 2
    with pytest.raises(SystemExit) as exit_info:
        run(monkeypatch, *argv, "--max-kld", limit, "--chunks", 2)
    assert exit_info.value.code == 1
    results = json.loads(output.read_text())
    assert results["reference"]["chunks"] == 2
    assert {p: row["pass"] for p, row in results["rows"].items()} == {"q4": False, "q8": True}
    assert "| q4 |" in output.with_suffix(".md").read_text()


@pytest.mark.parametrize("starts_with_bos", [True, False])
def test_reference_without_a_sidecar(text, tmp_path, monkeypatch, starts_with_bos: bool):
    """Without make_ref.py's .json, the export's bos_token_id replaces each chunk's first token
    if the stream starts with it, as llama-perplexity does."""
    _, model, export = text
    tokens = random_tokens(model.config.vocab_size)
    tokens[0, 0] = BOS if starts_with_bos else BOS + 1
    reference = write_reference(tmp_path / "ref.kld", torch_logits(model), tokens, "tiny")
    reference.with_name(f"{reference.name}.json").unlink()
    output = tmp_path / "wikitext.json"
    argv = ["--export", export, "--reference", reference, "--output", output]

    results = run(monkeypatch, *argv, "--precision", "fp32")
    assert results["reference"]["bos"] == (BOS if starts_with_bos else None)
    # write_reference always puts BOS first: only then does the decoder see the same tokens
    assert (results["rows"]["fp32"]["kld_mean"] < 1e-6) is starts_with_bos


def test_known_models_have_ceilings(text, tmp_path, monkeypatch):
    """The reference names the model, whose baselines give each precision KLD + 2 SE."""
    _, model, export = text
    reference = tmp_path / "ref.kld"
    tokens = random_tokens(model.config.vocab_size)
    write_reference(reference, torch_logits(model), tokens, "LiquidAI/LFM2.5-350M")
    output = tmp_path / "wikitext.json"
    argv = ["--export", export, "--reference", reference, "--output", output]

    rows = run(monkeypatch, *argv)["rows"]
    assert list(rows) == ["q4", "q8"]
    for precision, row in rows.items():
        kld, se = wikitext.BASELINES["LFM2.5-350M"][precision]
        assert row["ceiling"] == kld + 2 * se
        assert row["pass"] is True
    assert wikitext.ceiling(["LFM2.5-tiny", "LFM2.5-2.6B"], "q8", None) == 0.002723 + 2 * 0.000099
    assert wikitext.ceiling(["LFM2.5-tiny"], "q4", None) is None
    assert wikitext.ceiling(["LFM2.5-350M"], "q4", 0.5) == 0.5


def test_reference_from_the_checkpoint(text, tmp_path, monkeypatch):
    """--model on --tokens writes make_ref.py's file once, and scores as that file does."""
    checkpoint, model, export = text
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    tokens = random_tokens(model.config.vocab_size)
    given = write_reference(tmp_path / "given.kld", torch_logits(model), tokens, "tiny")
    devices = []
    compute = wikitext.compute_reference
    monkeypatch.setattr(
        wikitext, "compute_reference", lambda *args: devices.append(args[4]) or compute(*args)
    )
    output = tmp_path / "wikitext.json"
    argv = ["--export", export, "--output", output, "--precision", "fp32", "q4"]

    expected = run(monkeypatch, *argv, "--reference", given)
    first, second = (
        run(monkeypatch, *argv, "--model", checkpoint, "--tokens", given) for _ in range(2)
    )

    assert devices == ["cpu"]
    [path] = (tmp_path / "cache" / "liquidonnx" / "compare").glob("*.kld")
    assert path.name.startswith("checkpoint-wikitext-")
    assert first["reference"]["path"] == second["reference"]["path"] == str(path)
    meta = json.loads(path.with_name(f"{path.name}.json").read_text())
    assert {k: meta[k] for k in ("add_bos", "bos", "device", "dtype")} == {
        "add_bos": True,
        "bos": BOS,
        "device": "cpu",
        "dtype": "float32",
    }
    _, n_vocab, stream, rows = kld_base.open_rows(path)
    _, _, _, given_rows = kld_base.open_rows(given)
    np.testing.assert_array_equal(stream, tokens)
    for got, want in zip(rows, given_rows, strict=True):
        np.testing.assert_allclose(
            kld_base.decode_rows(np.asarray(got), n_vocab),
            kld_base.decode_rows(np.asarray(want), n_vocab),
            atol=1e-5,
        )
    for precision, row in expected["rows"].items():
        assert first["rows"][precision]["kld_mean"] == pytest.approx(
            row["kld_mean"], rel=1e-3, abs=1e-8
        )


def test_text_is_tokenized_as_llama_perplexity_does(text, tmp_path, monkeypatch):
    """--text: BOS, then the tokens of the raw text (no newline translation), in chunks."""
    checkpoint, _, export = text
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    path = tmp_path / "wiki.test.raw"
    path.write_bytes((TEXT * 20).encode())
    expected = [BOS, *tokenizer.encode(TEXT * 20, add_special_tokens=False)]

    tokens = wikitext.text_tokens(tokenizer, path, N_CTX, N_CHUNK)
    np.testing.assert_array_equal(tokens.ravel(), expected[: N_CTX * N_CHUNK])
    with pytest.raises(ValueError, match="fewer than"):
        wikitext.text_tokens(tokenizer, path, N_CTX, len(expected))

    output = tmp_path / "wikitext.json"
    argv = ["--export", export, "--model", checkpoint, "--text", path, "--output", output]
    results = run(monkeypatch, *argv, "--ctx", N_CTX, "--chunks", N_CHUNK, "--precision", "fp32")
    assert results["rows"]["fp32"]["kld_mean"] < 1e-6
    _, _, stream, _ = kld_base.open_rows(pathlib.Path(results["reference"]["path"]))
    np.testing.assert_array_equal(stream, tokens)


@pytest.mark.parametrize("reference_device", [None, "cpu", "cuda"])
def test_devices_and_threads(text, tmp_path, monkeypatch, reference_device: str | None):
    """Sessions run on --device without TF32 and with --threads; the reference model on
    --reference-device, or on --device without it."""
    checkpoint, model, export = text
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    tokens = random_tokens(model.config.vocab_size)
    given = write_reference(tmp_path / "given.kld", torch_logits(model), tokens, "tiny")
    calls = []

    def compute_reference(checkpoint, tokens, n_vocab, vl, device, path):
        calls.append(("reference", device))
        path.parent.mkdir(parents=True)
        path.write_bytes(given.read_bytes())
        return {"add_bos": True, "bos": BOS}

    def load_onnx_session(path, ep, tf32, threads):
        calls.append(("session", path.name, ep, tf32, threads))
        raise RuntimeError("stub")

    monkeypatch.setattr(wikitext, "compute_reference", compute_reference)
    monkeypatch.setattr(wikitext, "load_onnx_session", load_onnx_session)
    output = tmp_path / "wikitext.json"
    argv = ["--export", export, "--model", checkpoint, "--tokens", given, "--output", output]
    argv += ["--device", "cpu", "--threads", 3, "--precision", "q4"]
    if reference_device:
        argv += ["--reference-device", reference_device]

    with pytest.raises(SystemExit) as exit_info:
        run(monkeypatch, *argv)
    assert exit_info.value.code == 1
    assert calls == [
        ("reference", reference_device or "cpu"),
        ("session", "model_q4.onnx", "cpu", False, 3),
    ]
    assert json.loads(output.read_text())["rows"]["q4"] == {"error": "stub"}


@pytest.mark.parametrize("threads", [0, 2])
def test_session_threads(text, threads: int):
    _, _, export = text
    session = load_onnx_session(export / "onnx" / "model_q4.onnx", threads=threads)
    assert session.get_session_options().intra_op_num_threads == threads


@pytest.mark.parametrize(
    "family, argv, threads",
    [("text", [], 13), ("vl", ["--threads", 2], 2)],
    ids=["text-default", "vl-given"],
)
def test_gate_pins_the_thread_count(
    request, tmp_path, monkeypatch, family: str, argv: list, threads: int
):
    """Every session runs 13 intra-op threads on any machine, the count of the 8B-A1B ceilings,
    unless --threads says otherwise; the results and the report record the count."""
    _, model, export = request.getfixturevalue(family)
    tokens = random_tokens(model.config.get_text_config().vocab_size)
    reference = write_reference(tmp_path / "ref.kld", torch_logits(model), tokens, family)
    sessions = []
    load = wikitext.load_onnx_session

    def recording_load(*args, **kwargs):
        sessions.append(load(*args, **kwargs))
        return sessions[-1]

    monkeypatch.setattr(wikitext, "load_onnx_session", recording_load)
    output = tmp_path / "wikitext.json"
    argv = [*argv, "--export", export, "--reference", reference, "--output", output]

    results = run(monkeypatch, *argv, "--precision", "q4", "--chunks", 2)
    assert "error" not in results["rows"]["q4"]
    expected = [threads] * (2 if family == "vl" else 1)
    assert [s.get_session_options().intra_op_num_threads for s in sessions] == expected
    assert results["threads"] == threads
    assert f"(cpu, {threads} threads)" in output.with_suffix(".md").read_text()


def test_arguments_are_checked(text, tmp_path, monkeypatch, capsys):
    _, _, export = text
    output = tmp_path / "wikitext.json"
    output.write_bytes(b"")
    for argv, message in [
        (["--model", "m"], "--model needs --tokens or --text"),
        (["--reference", output, "--tokens", output], "--tokens and --text go with --model"),
        (["--reference", tmp_path / "missing.kld"], "does not exist"),
        (["--reference", output, "--precision", "q4f16"], "has no q4f16"),
        (["--reference", output, "--device", "cuda"], "has none of fp16, q4f16"),
    ]:
        with pytest.raises(SystemExit) as exit_info:
            run(monkeypatch, "--export", export, "--output", output, *argv)
        assert exit_info.value.code == 2
        assert message in capsys.readouterr().err


# === VL and audio: the decoder fed by the embedding model ===


def bundle_mb(export: pathlib.Path, files: dict[str, str]) -> float:
    parts = ("decoder", "embedding")
    return sum(get_total_model_size_mb(export / "onnx" / files[part]) for part in parts)


@pytest.fixture(scope="module")
def vl(tmp_path_factory):
    """(checkpoint, PyTorch model, export with fp32 and q4) of a tiny LFM2-VL."""
    root = tmp_path_factory.mktemp("vl")
    torch.manual_seed(0)
    config = Lfm2VlConfig(
        text_config={
            "vocab_size": 65536,
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "layer_types": ["conv", "full_attention"],
            "intermediate_size": 128,
            "max_position_embeddings": 4096,
        },
        vision_config={
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "patch_size": 16,
            "num_patches": 256,
        },
        projector_hidden_size=64,
    )
    model = Lfm2VlForConditionalGeneration(config).eval()
    model.save_pretrained(root / "checkpoint")
    AutoProcessor.from_pretrained("LiquidAI/LFM2.5-VL-1.6B").save_pretrained(root / "checkpoint")
    export = root / "export"
    vl_export.export_vl_model(str(root / "checkpoint"), export)
    vl_export.derive_precision_files(str(root / "checkpoint"), export, "q4")
    vl_export.write_genai_config(export, "q4")
    return root / "checkpoint", model, export


def test_vl_text_from_the_checkpoint(vl, tmp_path, monkeypatch):
    """VL: the reference is the text-only model, and the decoder takes the embedding model's
    output; both graphs count towards the bytes."""
    checkpoint, model, export = vl
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    given = write_reference(tmp_path / "given.kld", torch_logits(model), random_tokens(65536), "vl")
    output = tmp_path / "wikitext.json"
    argv = ["--export", export, "--model", checkpoint, "--tokens", given, "--output", output]

    rows = run(monkeypatch, *argv, "--precision", "fp32", "q4")["rows"]
    assert rows["fp32"]["kld_mean"] < 1e-6
    assert rows["q4"]["kld_mean"] > 10 * rows["fp32"]["kld_mean"]
    for precision, row in rows.items():
        assert row["size_mb"] == pytest.approx(bundle_mb(export, vl_export.bundle(precision)))


def test_audio_against_a_reference(tmp_path, monkeypatch, capsys):
    """Audio exports score against --reference, through the embedding model's audio_features;
    computing a reference needs liquid-audio."""
    export = build_model_dir(tmp_path)
    audio_export.derive_precision_files(str(tmp_path / "checkpoint"), export, "q4", block_size=32)
    files = audio_export.bundle("fp32")
    decoder = load_onnx_session(export / "onnx" / files["decoder"])
    embedding = load_onnx_session(export / "onnx" / files["embedding"])

    def logits_of(ids: np.ndarray) -> np.ndarray:
        feeds = [
            decoder_inputs(embed(embedding, row[None]), initialize_cache(decoder), 0) for row in ids
        ]
        return np.concatenate([decoder.run(["logits"], feed)[0] for feed in feeds])

    reference = write_reference(tmp_path / "ref.kld", logits_of, random_tokens(256), "audio")
    output = tmp_path / "wikitext.json"
    argv = ["--export", export, "--output", output]

    rows = run(monkeypatch, *argv, "--reference", reference, "--precision", "fp32", "q4")["rows"]
    # the synthetic decoder's distribution is nearly flat: q4 scores ~0 too
    assert rows["fp32"]["kld_mean"] < 1e-6
    for precision, row in rows.items():
        assert row["size_mb"] == pytest.approx(bundle_mb(export, audio_export.bundle(precision)))
    with pytest.raises(SystemExit) as exit_info:
        run(monkeypatch, *argv, "--model", "m", "--tokens", reference)
    assert exit_info.value.code == 2
    assert "audio references need liquid-audio" in capsys.readouterr().err
