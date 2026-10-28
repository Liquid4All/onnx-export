#!/usr/bin/env python3
"""
Answer named questions over a state with a d1 ONNX export (d1.session: the checkpoint's own
system_one on onnxruntime). Prints {"answers": ..., "usage": ...} as JSON, and logs the time
of the first call and the mean of --repeat more.

A request file is the API's request as JSON: {"state": ..., "questions": {name: question},
"media": [{"type": "image" | "audio" | "video", "path": ...}]}, the questions in the checkpoint
README's schema.

Usage:
    uv run d1-infer --model exports/d1-2B-ONNX --request samples/d1/ticket.json
    curl -LO http://images.cocodataset.org/val2017/000000039769.jpg  # the model cards' picture
    echo "dea9e7ef97386345f7cff32f9055da4982da5471c48d575146c796ab4563b04e  000000039769.jpg" \\
        | sha256sum -c -
    uv run d1-infer --model exports/d1-2B-ONNX --request samples/d1/picture.json \\
        --media 000000039769.jpg --precision fp16 --repeat 10
    uv run d1-infer --model exports/d1-Omni-2B-ONNX --request samples/d1/clip.json \\
        --media samples/audio/woodworks_question.wav
"""

import argparse
import json
import logging
import pathlib
import time

from liquidonnx.d1.session import BUNDLES, load
from liquidonnx.lfm3.session import default_ep

logger = logging.getLogger(__name__)

AUDIO = {".wav", ".npy", ".mp3", ".flac", ".ogg"}
VIDEO = {".mp4", ".mov", ".webm", ".mkv", ".avi"}


def media_item(path: str) -> dict:
    suffix = pathlib.Path(path).suffix.lower()
    kind = "audio" if suffix in AUDIO else "video" if suffix in VIDEO else "image"
    return {"type": kind, "path": path}


def main():
    parser = argparse.ArgumentParser(description="ONNX inference for d1")
    parser.add_argument("--model", required=True, help="Export folder")
    parser.add_argument("--request", required=True, help="Request JSON: state, questions, media")
    parser.add_argument("--state", help="State text in place of the request's")
    parser.add_argument("--media", nargs="*", default=[], help="Pictures, clips, videos to add")
    parser.add_argument(
        "--precision",
        choices=sorted(BUNDLES),
        help="default: q4f16, fp16 or q8f16 on CUDA, q4 or q8 on CPU (the first one exported)",
    )
    parser.add_argument("--set-size", type=int, help="Questions read together (d1_set_size)")
    parser.add_argument("--repeat", type=int, default=0, help="Timed calls after the first")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution (skip CUDA)")
    parser.add_argument("--no-tf32", action="store_true", help="Keep CUDA fp32 math in fp32")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    request = json.loads(pathlib.Path(args.request).read_text())
    state = request.get("state") if args.state is None else args.state
    media = request.get("media", []) + [media_item(path) for path in args.media]
    ep = default_ep(args.cpu)
    model = load(args.model, args.precision, ep, tf32=not args.no_tf32, set_size=args.set_size)
    logger.info(f"{args.model}: {model.precision} on {ep}")

    start = time.perf_counter()
    out = model.system_one(state, request["questions"], media)
    logger.info(f"first call: {(time.perf_counter() - start) * 1000:.1f} ms")
    if args.repeat:
        start = time.perf_counter()
        for _ in range(args.repeat):
            model.system_one(state, request["questions"], media)
        mean = (time.perf_counter() - start) * 1000 / args.repeat
        logger.info(f"{args.repeat} more calls: {mean:.1f} ms each")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
