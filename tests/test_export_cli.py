"""
Arguments shared by the lfm2*-export CLIs.

Run with:
    uv run pytest tests/test_export_cli.py -v
"""

import argparse
import json
import pathlib
import platform

import pytest

from liquidonnx import export_cli
from liquidonnx.export_cli import add_export_arguments, output_dir, parse_precisions
from liquidonnx.lfm2.export import ALL_PRECISIONS, TEXT_PRECISIONS


def parse(*argv: str) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser()
    add_export_arguments(parser, ALL_PRECISIONS, TEXT_PRECISIONS, "LiquidAI/LFM2.5-350M")
    args = parser.parse_args(argv)
    return args, parse_precisions(parser, args, ALL_PRECISIONS, TEXT_PRECISIONS)


def test_precisions():
    assert parse("LiquidAI/LFM2.5-350M")[1] == []
    assert parse("LiquidAI/LFM2.5-350M", "--precision")[1] == list(TEXT_PRECISIONS)
    # q4f16 converts q4 when both are exported, so q4 comes first.
    assert parse("m", "--precision", "Q4F16", "q8", "q4", "q4")[1] == ["q4", "q4f16", "q8"]
    with pytest.raises(SystemExit):
        parse("m", "--precision", "int3")


def test_output_dir():
    args, _ = parse("LiquidAI/LFM2.5-350M", "--output-dir", "out")
    assert output_dir(args) == pathlib.Path("out/exports/LFM2.5-350M-ONNX")
    args, _ = parse("/models/checkpoint/", "--output-name", "custom")
    assert output_dir(args) == pathlib.Path("exports/custom")


def test_export_info_records_each_run(tmp_path, monkeypatch):
    """export_info.json: one record per run, the checkpoint's revision read from the cache."""
    sha = "0123456789abcdef0123456789abcdef01234567"
    repo = tmp_path / "hub" / "models--LiquidAI--tiny"
    (repo / "refs").mkdir(parents=True)
    (repo / "refs" / "main").write_text(sha)
    (repo / "snapshots" / sha).mkdir(parents=True)
    (repo / "snapshots" / sha / "config.json").write_text("{}")
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path / "hub"))

    export_cli.write_export_info(tmp_path, "LiquidAI/tiny", ["fp32", "q8"])
    export_cli.write_export_info(tmp_path, "LiquidAI/tiny", ["q4"], revision=sha)
    export_cli.write_export_info(tmp_path, "LiquidAI/other", ["q4"])
    export_cli.write_export_info(tmp_path, str(tmp_path / "hub"), ["fp16"])
    first, pinned, uncached, local = json.loads((tmp_path / "export_info.json").read_text())[
        "exports"
    ]

    assert first["checkpoint"] == {"model": "LiquidAI/tiny", "revision": sha, "pinned": None}
    assert first["precisions"] == ["fp32", "q8"]
    assert pinned["checkpoint"]["revision"] == pinned["checkpoint"]["pinned"] == sha
    assert uncached["checkpoint"]["revision"] is None
    assert local["checkpoint"]["revision"] is None
    versions = first["packages"]
    assert {"onnx", "torch", "transformers"} <= set(versions)
    assert {"onnxruntime", "onnxruntime-gpu"} & set(versions)
    assert first["host"]["machine"] == platform.machine()
    commit, dirty = first["liquidonnx"]["commit"], first["liquidonnx"]["dirty"]
    # None outside a checkout of the repository (an installed package)
    assert (commit is None and dirty is None) or (len(commit) == 40 and isinstance(dirty, bool))
