"""
Arguments shared by the lfm2*-export CLIs.

Run with:
    uv run pytest tests/test_export_cli.py -v
"""

import argparse
import pathlib

import pytest

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
