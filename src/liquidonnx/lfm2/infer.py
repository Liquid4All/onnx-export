#!/usr/bin/env python3
"""
ONNX inference script for LFM2 text models.

Usage:
    uv run lfm2-infer --model exports/LFM2-1.2B-ONNX
    uv run lfm2-infer --model exports/LFM2-1.2B-ONNX --prompt "Hello"
    uv run lfm2-infer --model exports/LFM2-1.2B-ONNX/onnx/model_q4.onnx
"""

import argparse
import logging
import pathlib

from liquidonnx.session import ONNXTextModel, run_chat_loop


def get_onnx_dir(exports_dir: pathlib.Path, size: str) -> pathlib.Path:
    """Get ONNX directory for a model size."""
    return exports_dir / f"LFM2-{size}-ONNX" / "onnx"


def main():
    parser = argparse.ArgumentParser(description="ONNX inference for LFM2 text models")
    parser.add_argument("--model", required=True, help="Path to ONNX model directory or file")
    parser.add_argument("--prompt", default=None, help="Initial prompt (optional)")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming output")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    model = ONNXTextModel(args.model)
    model.load()

    run_chat_loop(model, args)


if __name__ == "__main__":
    main()
