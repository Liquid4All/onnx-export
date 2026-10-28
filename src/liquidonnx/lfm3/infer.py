#!/usr/bin/env python3
"""
ONNX inference script for LFM3 text models.

Usage:
    uv run lfm3-infer --model exports/LFM3-1.7B-ONNX
    uv run lfm3-infer --model exports/LFM3-1.7B-ONNX --prompt "Hello"
    uv run lfm3-infer --model exports/LFM3-1.7B-ONNX --precision q4f16
    uv run lfm3-infer --model exports/LFM3-1.7B-ONNX/onnx/model_q4.onnx
"""

import argparse
import logging

from liquidonnx.lfm3.session import ONNXTextModel, run_chat_loop


def main():
    parser = argparse.ArgumentParser(description="ONNX inference for LFM3 text models")
    parser.add_argument("--model", required=True, help="Path to ONNX model directory or file")
    parser.add_argument("--prompt", default=None, help="Initial prompt (optional)")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--no-think", action="store_true", help="enable_thinking=False")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming output")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution (skip CUDA)")
    parser.add_argument(
        "--precision",
        choices=["fp32", "fp16", "q4", "q4f16", "q4f32", "q8", "q8f16"],
        help="Precision in an export folder (default: q4f16, fp16 or q8f16 on CUDA, q4 or q8 on CPU)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    model = ONNXTextModel(args.model, force_cpu=args.cpu, precision=args.precision)
    model.load()

    run_chat_loop(model, args)


if __name__ == "__main__":
    main()
