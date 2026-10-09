#!/usr/bin/env python3
"""
Chat with an LFM2-VL export through onnxruntime-genai, with images on any turn.

Images stay in the conversation once attached, and every turn re-sends them. The checkpoint's own
processor prepares them (see CheckpointProcessor). Every LFM2-VL and LFM2.5-VL checkpoint sets
do_image_splitting, so an image of more than twice max_image_tokens worth of pixels (524,288,
about 724x724) becomes 512x512 tiles plus a thumbnail; --no-image-splitting resizes each image
once instead.

Usage:
    uv run lfm2-vl-infer --model exports/LFM2.5-VL-1.6B-ONNX
    uv run lfm2-vl-infer --model exports/LFM2.5-VL-1.6B-ONNX \\
        --images tests/test_lfm2_vl/assets/cardinal.jpg
    uv run lfm2-vl-infer --model exports/LFM2.5-VL-1.6B-ONNX --prompt "Compare" \\
        --images tests/test_lfm2_vl/assets/cardinal.jpg tests/test_lfm2_vl/assets/bluejay.jpg
    uv run lfm2-vl-infer --model exports/LFM2.5-VL-1.6B-ONNX --precision fp16
    uv run lfm2-vl-infer --model exports/LFM2.5-VL-1.6B-ONNX --no-image-splitting \\
        --images tests/test_lfm2_vl/assets/wide.jpg
"""

import argparse
import json
import logging
import pathlib

import numpy as np
import onnxruntime_genai as og

from liquidonnx.genai_runtime import TokenPrinter, add_runtime_arguments, generate, load_model
from liquidonnx.lfm2.infer import MAX_NEW_TOKENS, run_chat_loop
from liquidonnx.lfm2_vl.export import PRECISIONS, genai_files

# Hugging Face processor output -> its entry in genai_config.json model.vision.inputs
VISION_INPUTS = {
    "pixel_values": "pixel_values",
    "pixel_attention_mask": "attention_mask",
    "spatial_shapes": "image_sizes",
}


class CheckpointProcessor:
    """onnxruntime-genai inputs for a prompt and its images, made by the export's Hugging Face
    processor (processor_config.json) instead of genai's own (genai_processor_config.json).

    genai's processor rejects every prompt whose tallest image is not also its widest, and it
    never tiles. The Hugging Face processor pads each image's patches on their own, to the
    max_num_patches genai pads to. With image_splitting (default: the checkpoint's
    do_image_splitting) it splits a large image into tiles plus a thumbnail, which the vision
    encoder takes as separate images, and puts the tile tokens in the prompt. Without, both
    processors resize each image once; from the same decoded image, their pixels are a 1/255 step
    or two apart.
    """

    def __init__(self, model_dir: pathlib.Path, image_splitting: bool | None = None):
        from transformers import AutoProcessor

        self.hf = AutoProcessor.from_pretrained(model_dir)
        if image_splitting is None:
            image_splitting = self.hf.image_processor.do_image_splitting
        self.image_splitting = image_splitting
        config = json.loads((model_dir / "genai_config.json").read_text())
        inputs = config["model"]["vision"]["inputs"]
        self.names = {key: inputs[entry] for key, entry in VISION_INPUTS.items()}

    def __call__(self, prompt: str, images: list[str]) -> og.NamedTensors:
        """prompt holds one <image> per image, which the processor expands into its placeholders."""
        from PIL import Image

        pil = [Image.open(path).convert("RGB") for path in images]
        features = self.hf(
            text=prompt,
            images=pil or None,
            return_tensors="np",
            do_image_splitting=self.image_splitting,
        )
        input_ids = features["input_ids"]
        inputs = og.NamedTensors()
        inputs["input_ids"] = input_ids.astype(np.int32)
        # genai sizes the image features by it, and skips the vision model at 0
        image_tokens = np.count_nonzero(input_ids == self.hf.image_token_id)
        inputs["num_image_tokens"] = np.array([image_tokens], dtype=np.int64)
        if pil:
            for key, name in self.names.items():
                dtype = np.float32 if key == "pixel_values" else np.int64
                inputs[name] = features[key].astype(dtype)
        return inputs


class VLChat:
    """A conversation with an LFM2-VL export; images join the next message sent."""

    def __init__(
        self,
        model_dir: pathlib.Path,
        precision: str | None = None,
        ep: str = "cpu",
        image_splitting: bool | None = None,
        tf32: bool = True,
    ):
        self.model = load_model(model_dir, precision and genai_files(precision), ep, tf32)
        self.tokenizer = og.Tokenizer(self.model)
        self.processor = CheckpointProcessor(model_dir, image_splitting)
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

    def send(self, text: str, max_new_tokens: int = MAX_NEW_TOKENS, stream: bool = True) -> str:
        content = text
        if self.pending:
            content = [{"type": "image", "path": path} for path in self.pending]
            content.append({"type": "text", "text": text})
            self.pending = []
        self.messages.append({"role": "user", "content": content})
        prompt = self.processor.hf.apply_chat_template(
            self.messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(prompt, self.images())
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
    parser.add_argument(
        "--image-splitting",
        action=argparse.BooleanOptionalAction,
        help="Split large images into 512x512 tiles plus a thumbnail, as the reference processor "
        "does (default: the checkpoint's do_image_splitting, on for LFM2-VL and LFM2.5-VL)",
    )
    parser.add_argument("--prompt", default=None, help="Initial prompt (optional)")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=MAX_NEW_TOKENS,
        help=f"Max tokens per answer (default: {MAX_NEW_TOKENS})",
    )
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming output")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    chat = VLChat(args.model, args.precision, args.ep, args.image_splitting, args.tf32)
    chat.attach(args.images)
    run_chat_loop(
        chat, args, "LFM2-VL", "'images <path> [<path> ...]' attaches images to the next message"
    )


if __name__ == "__main__":
    main()
