#!/usr/bin/env python3
"""
ONNX inference script for LFM2-MoE models.

Usage:
    uv run lfm2-moe-infer --model exports/LFM2.5-8B-A1B-ONNX
    uv run lfm2-moe-infer --model exports/LFM2.5-8B-A1B-ONNX --prompt "Hello"
    uv run lfm2-moe-infer --model exports/LFM2.5-8B-A1B-ONNX/onnx/model_q4.onnx
"""

import argparse
import logging

from liquidonnx.session import ONNXTextModel, run_chat_loop


def main():
    parser = argparse.ArgumentParser(description="ONNX inference for LFM2-MoE models")
    parser.add_argument("--model", required=True, help="Path to ONNX model directory or file")
    parser.add_argument("--prompt", default=None, help="Initial prompt (optional)")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming output")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution (skip CUDA)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    model = ONNXTextModel(args.model, force_cpu=args.cpu)
    model.load()

    run_chat_loop(model, args)


if __name__ == "__main__":
    main()
