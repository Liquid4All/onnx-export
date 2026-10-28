#!/usr/bin/env python3
"""
Chat with an LFM3-VL ONNX export on onnxruntime.

Usage:
    uv run lfm3-vl-infer --model exports/LFM3-VL-2B-ONNX \\
        --images tests/test_lfm2_vl/assets/cardinal.jpg --prompt "What bird is this?"
    uv run lfm3-vl-infer --model exports/LFM3-VL-2B-ONNX --precision fp16
"""

import argparse
import logging

from PIL import Image

from liquidonnx.lfm3.session import default_ep
from liquidonnx.lfm3_vl.session import BUNDLES, ONNXVLModel, available_precisions


def user_message(text: str, image_count: int) -> dict:
    return {
        "role": "user",
        "content": [{"type": "image"}] * image_count + [{"type": "text", "text": text}],
    }


def main():
    parser = argparse.ArgumentParser(description="ONNX inference for LFM3-VL")
    parser.add_argument("--model", required=True, help="Export folder")
    parser.add_argument(
        "--precision",
        choices=sorted(BUNDLES),
        help="default: q4f16, fp16 or q8f16 on CUDA, q4 or q8 on CPU (the first one exported)",
    )
    parser.add_argument("--images", nargs="*", default=[], help="Images for the first turn")
    parser.add_argument("--prompt", help="First turn (default: read turns from stdin)")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument(
        "--max-image-tokens", type=int, help="Cap tokens per image (smaller resize)"
    )
    parser.add_argument("--no-think", action="store_true", help="enable_thinking=False")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming output")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution (skip CUDA)")
    parser.add_argument("--no-tf32", action="store_true", help="Keep CUDA fp32 math in fp32")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    exported = available_precisions(args.model, default_ep(args.cpu))
    precision = args.precision or next(iter(exported), None)
    if precision is None:
        parser.error(f"no exported bundle in {args.model}/onnx")
    model = ONNXVLModel(args.model, precision, force_cpu=args.cpu, tf32=not args.no_tf32)
    model.load()
    images = [Image.open(path).convert("RGB") for path in args.images]

    messages = []
    prompt = args.prompt
    while True:
        if prompt is None:
            try:
                prompt = input("\nUser: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if prompt.lower() in ("", "quit", "exit"):
                break
        messages.append(user_message(prompt, len(images) if not messages else 0))
        print("Assistant: ", end="" if not args.no_stream else "\n")
        answer = model.generate(
            messages,
            images,
            max_new_tokens=args.max_tokens,
            stream=not args.no_stream,
            enable_thinking=not args.no_think,
            max_image_tokens=args.max_image_tokens,
        )
        if args.no_stream:
            print(answer)
        messages.append({"role": "assistant", "content": answer})
        if args.prompt is not None:
            break
        prompt = None


if __name__ == "__main__":
    main()
