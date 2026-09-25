#!/usr/bin/env python3
"""
Chat with an LFM2-VL export through onnxruntime-genai, with images on any turn.

Images stay in the conversation once attached, and every turn re-sends them. onnxruntime-genai
resizes each image once instead of splitting it into tiles.

Usage:
    uv run lfm2-vl-infer --model exports/LFM2.5-VL-1.6B-ONNX
    uv run lfm2-vl-infer --model exports/LFM2.5-VL-1.6B-ONNX --images photo.jpg
    uv run lfm2-vl-infer --model exports/LFM2.5-VL-1.6B-ONNX --images a.jpg b.jpg --prompt "Compare"
    uv run lfm2-vl-infer --model exports/LFM2.5-VL-1.6B-ONNX --precision fp16
"""

import argparse
import logging
import pathlib

import onnxruntime_genai as og

from liquidonnx.genai_runtime import TokenPrinter, add_runtime_arguments, generate, load_model
from liquidonnx.lfm2.infer import run_chat_loop
from liquidonnx.lfm2_vl.export import PRECISIONS, genai_files


class VLChat:
    """A conversation with an LFM2-VL export; images join the next message sent."""

    def __init__(self, model_dir: pathlib.Path, precision: str | None = None, ep: str = "cpu"):
        from transformers import AutoTokenizer

        self.model = load_model(model_dir, precision and genai_files(precision), ep)
        self.tokenizer = og.Tokenizer(self.model)
        self.processor = self.model.create_multimodal_processor()
        # The chat template writes one <image> per image; the genai processor expands each one
        # into the image's feature placeholders.
        self.hf_tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.messages: list[dict] = []
        self.pending: list[str] = []

    def attach(self, paths: list[str]):
        """Images for the next message; missing files are left out."""
        self.pending = []
        for path in paths:
            if pathlib.Path(path).exists():
                self.pending.append(path)
                print(f"Loaded image: {path}")
            else:
                print(f"Image not found: {path}")

    def images(self) -> list[str]:
        """The conversation's images, in the order of their <image> placeholders."""
        return [
            item["path"]
            for message in self.messages
            if isinstance(message["content"], list)
            for item in message["content"]
            if item["type"] == "image"
        ]

    def send(self, text: str, max_new_tokens: int = 100, stream: bool = True) -> str:
        content = text
        if self.pending:
            content = [{"type": "image", "path": path} for path in self.pending]
            content.append({"type": "text", "text": text})
            self.pending = []
        self.messages.append({"role": "user", "content": content})
        prompt = self.hf_tokenizer.apply_chat_template(
            self.messages, tokenize=False, add_generation_prompt=True
        )
        images = self.images()
        inputs = self.processor(prompt, images=og.Images.open(*images) if images else None)
        prompt_length = inputs["input_ids"].as_numpy().shape[-1]

        printer = TokenPrinter(self.tokenizer) if stream else None
        generator = generate(self.model, inputs, max_new_tokens, printer)
        if stream:
            print()
        response = self.tokenizer.decode(generator.get_sequence(0)[prompt_length:])
        self.messages.append({"role": "assistant", "content": response})
        return response

    def clear(self):
        self.messages.clear()
        self.pending = []

    def command(self, line: str) -> bool:
        command, _, paths = line.partition(" ")
        if command.lower() not in ("image", "images"):
            return False
        self.attach(paths.split())
        return True


def main():
    parser = argparse.ArgumentParser(description="Chat with an LFM2-VL export (images optional)")
    parser.add_argument("--model", required=True, type=pathlib.Path, help="Export folder")
    add_runtime_arguments(parser, PRECISIONS)
    parser.add_argument("--images", nargs="*", default=[], help="Images for the first turn")
    parser.add_argument("--prompt", default=None, help="Initial prompt (optional)")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming output")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    chat = VLChat(args.model, args.precision, args.ep)
    chat.attach(args.images)
    run_chat_loop(
        chat, args, "LFM2-VL", "'images <path> [<path> ...]' attaches images to the next message"
    )


if __name__ == "__main__":
    main()
