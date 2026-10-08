"""
lfm2-compare's reference cache, with a stub in place of the reference model.

Run with:
    uv run pytest tests/test_compare.py -v
"""

import logging
import pathlib

import numpy as np
import pytest

from liquidonnx import compare
from liquidonnx.compare import text

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
def runs(monkeypatch) -> list[pathlib.Path]:
    runs = []

    def reference(checkpoint: pathlib.Path, prompts: int, max_new: int) -> list[dict]:
        runs.append(checkpoint)
        return ITEMS

    monkeypatch.setattr(text, "reference", reference)
    return runs


def check_reference(checkpoint: pathlib.Path):
    items = compare.cached_reference(str(checkpoint), checkpoint, "text", 2, 4)
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
