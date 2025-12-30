#!/usr/bin/env python3
"""
ONNX inference script for LFM2-VL vision-language models.

Usage:
    uv run lfm2-vl-infer --model exports/LFM2-VL-450M-ONNX
    uv run lfm2-vl-infer --model exports/LFM2-VL-450M-ONNX --images photo.jpg
    uv run lfm2-vl-infer --model exports/LFM2-VL-450M-ONNX --images a.jpg b.jpg --prompt "Compare"
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from transformers import AutoProcessor

from liquidonnx.lfm2_vl import VISION_MODE_CONV2D, VISION_MODE_TILED
from liquidonnx.lfm2_vl.preprocessing import (
    build_inputs_embeds,
    detect_vision_format,
    preprocess_conv2d,
    preprocess_tiled,
)
from liquidonnx.session import initialize_cache, update_cache

logger = logging.getLogger(__name__)


def get_onnx_dir(exports_dir: Path, size: str) -> Path:
    """Get ONNX directory for a VL model size."""
    return exports_dir / f"LFM2-VL-{size}-ONNX" / "onnx"


class VLModelInference:
    """ONNX inference for LFM2-VL models."""

    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.processor = None
        self.tokenizer = None
        self.embed_tokens_sess = None
        self.embed_images_sess = None
        self.decoder_sess = None
        self.image_token_id = None
        self.vision_format = VISION_MODE_TILED

    def load(self):
        """Load processor and ONNX models."""
        logger.info(f"Loading VL model from {self.model_path}...")

        # Try loading processor from local path first if it has tokenizer files
        if (self.model_path / "tokenizer.json").exists():
            hf_model = str(self.model_path)
        else:
            dir_name = self.model_path.name
            if "450M" in dir_name:
                hf_model = "LiquidAI/LFM2-VL-450M"
            elif "1.6B" in dir_name:
                hf_model = "LiquidAI/LFM2-VL-1.6B"
            elif "3B" in dir_name:
                hf_model = "LiquidAI/LFM2-VL-3B"
            else:
                hf_model = str(self.model_path)

        logger.info(f"Loading processor from {hf_model}...")
        self.processor = AutoProcessor.from_pretrained(hf_model, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer
        self.image_token_id = self.tokenizer.convert_tokens_to_ids("<image>")

        onnx_dir = self.model_path / "onnx"

        logger.info("Loading embed_tokens.onnx...")
        self.embed_tokens_sess = ort.InferenceSession(
            str(onnx_dir / "embed_tokens.onnx"), providers=["CPUExecutionProvider"]
        )

        logger.info("Loading embed_images.onnx...")
        self.embed_images_sess = ort.InferenceSession(
            str(onnx_dir / "embed_images.onnx"), providers=["CPUExecutionProvider"]
        )

        logger.info("Loading decoder.onnx...")
        self.decoder_sess = ort.InferenceSession(
            str(onnx_dir / "decoder.onnx"), providers=["CPUExecutionProvider"]
        )

        self.vision_format = detect_vision_format(self.embed_images_sess)
        logger.info(f"Vision format: {self.vision_format}")

    def _get_image_embeddings(self, images: list[Image.Image]) -> list[np.ndarray]:
        """Get embeddings for a list of images."""
        embeddings = []

        for image in images:
            if self.vision_format == VISION_MODE_CONV2D:
                pixel_values, spatial_h, spatial_w = preprocess_conv2d(image)
                outputs = self.embed_images_sess.run(
                    None,
                    {
                        "pixel_values": pixel_values,
                        "spatial_h": np.array(spatial_h, dtype=np.int64),
                        "spatial_w": np.array(spatial_w, dtype=np.int64),
                    },
                )
            else:
                pixel_values, patch_attention_mask, spatial_shapes = preprocess_tiled(
                    image, self.processor, do_pad_to_square=True
                )
                outputs = self.embed_images_sess.run(
                    None,
                    {
                        "pixel_values": pixel_values,
                        "patch_attention_mask": patch_attention_mask,
                        "spatial_shapes": spatial_shapes,
                    },
                )
                # Tiled output is [num_tiles, tokens_per_tile, hidden], flatten to [total_patches, hidden]
                onnx_embeds = outputs[0]
                num_tiles, tokens_per_tile, hidden = onnx_embeds.shape
                embeddings.append(onnx_embeds.reshape(-1, hidden))
                continue
            embeddings.append(outputs[0][0])

        return embeddings

    def _get_text_embeddings(self, input_ids: np.ndarray) -> np.ndarray:
        """Get text embeddings from token IDs."""
        outputs = self.embed_tokens_sess.run(None, {"input_ids": input_ids.astype(np.int64)})
        return outputs[0]  # [1, seq_len, hidden_dim]

    def _build_inputs_embeds_expanded(
        self, input_ids: np.ndarray, image_embeds_list: list[np.ndarray]
    ) -> np.ndarray:
        """Build inputs_embeds for expanded token sequence using liquidonnx utility."""
        text_embeds = self._get_text_embeddings(input_ids)[0]  # [seq_len, hidden]
        return build_inputs_embeds(text_embeds, image_embeds_list, self.image_token_id, input_ids)

    def generate(
        self,
        messages: list,
        images: list[Image.Image] | None = None,
        max_new_tokens: int = 100,
        stream: bool = True,
    ) -> str:
        """Generate response for chat messages with optional images."""
        images = images or []

        if images:
            # Images are added to the LAST user message only (current turn)
            messages_with_images = []
            last_user_idx = max(
                (i for i, msg in enumerate(messages) if msg["role"] == "user"), default=-1
            )
            for i, msg in enumerate(messages):
                if msg["role"] == "user" and i == last_user_idx:
                    content = [{"type": "image", "image": img} for img in images]
                    content.append({"type": "text", "text": msg["content"]})
                    messages_with_images.append({"role": "user", "content": content})
                else:
                    messages_with_images.append(msg)

            prompt = self.processor.apply_chat_template(
                messages_with_images, add_generation_prompt=True
            )

            # Processor expands each <image> into N tokens (one per patch)
            inputs = self.processor(
                images=images,
                text=prompt,
                return_tensors="pt",
            )
            input_ids = inputs["input_ids"].numpy()

            image_embeds_list = self._get_image_embeddings(images)
            inputs_embeds = self._build_inputs_embeds_expanded(input_ids, image_embeds_list)
        else:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            input_ids = np.array(
                [self.tokenizer.encode(prompt, add_special_tokens=False)], dtype=np.int64
            )
            inputs_embeds = self._get_text_embeddings(input_ids)

        cache = initialize_cache(self.decoder_sess)
        has_position_ids = "position_ids" in {inp.name for inp in self.decoder_sess.get_inputs()}

        seq_len = inputs_embeds.shape[1]
        generated_tokens = []
        cur_len = seq_len

        for step in range(max_new_tokens):
            if step == 0:
                embeds = inputs_embeds
                pos = np.arange(seq_len, dtype=np.int64).reshape(1, -1)
            else:
                last_token = np.array([[generated_tokens[-1]]], dtype=np.int64)
                embeds = self._get_text_embeddings(last_token)
                pos = np.array([[cur_len - 1]], dtype=np.int64)

            attn_mask = np.ones((1, cur_len), dtype=np.int64)

            feed = {
                "inputs_embeds": embeds.astype(np.float32),
                "attention_mask": attn_mask,
            }
            if has_position_ids:
                feed["position_ids"] = pos
            feed.update(cache)

            outputs = self.decoder_sess.run(None, feed)
            logits = outputs[0][0, -1]

            next_token = int(np.argmax(logits))
            generated_tokens.append(next_token)

            update_cache(cache, outputs, self.decoder_sess.get_outputs())
            cur_len += 1

            if stream:
                token_str = self.tokenizer.decode([next_token])
                print(token_str, end="", flush=True)

            if next_token == self.tokenizer.eos_token_id:
                break

        if stream:
            print()

        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(
        description="ONNX inference for LFM2-VL models (0-2 images per turn)"
    )
    parser.add_argument("--model", required=True, help="Path to ONNX model directory")
    parser.add_argument("--images", nargs="*", default=[], help="Image paths (0-2 images)")
    parser.add_argument("--prompt", default=None, help="Initial prompt (optional)")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming output")
    args = parser.parse_args()

    if len(args.images) > 2:
        print("Warning: Only 0-2 images supported per turn. Using first 2.")
        args.images = args.images[:2]

    model = VLModelInference(args.model)
    model.load()

    print("\n" + "=" * 50)
    print("LFM2-VL Model - ONNX Inference")
    print("Supports 0-2 images per message")
    print("Commands: 'quit', 'exit', 'clear', 'image <path>' or 'images <path1> <path2>'")
    print("=" * 50 + "\n")

    messages = []
    current_images = []

    for img_path in args.images:
        if Path(img_path).exists():
            current_images.append(Image.open(img_path).convert("RGB"))
            print(f"Loaded image: {img_path}")

    if args.prompt:
        messages.append({"role": "user", "content": args.prompt})
        print(f"User: {args.prompt}")
        if current_images:
            print(f"  [with {len(current_images)} image(s)]")
        print("Assistant: ", end="")
        response = model.generate(
            messages.copy(),
            images=current_images if current_images else None,
            max_new_tokens=args.max_tokens,
            stream=not args.no_stream,
        )
        messages.append({"role": "assistant", "content": response})
        if args.no_stream:
            print(response)
        current_images = []

    while True:
        try:
            user_input = input("\nUser: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break

        if user_input.lower() == "clear":
            messages = []
            current_images = []
            print("Chat history and images cleared.")
            continue

        if user_input.lower().startswith("image "):
            img_path = user_input[6:].strip()
            if Path(img_path).exists():
                current_images = [Image.open(img_path).convert("RGB")]
                print(f"Loaded 1 image: {img_path}")
            else:
                print(f"Image not found: {img_path}")
            continue

        if user_input.lower().startswith("images "):
            paths = user_input[7:].strip().split()
            current_images = []
            for p in paths[:2]:
                if Path(p).exists():
                    current_images.append(Image.open(p).convert("RGB"))
                    print(f"Loaded image: {p}")
                else:
                    print(f"Image not found: {p}")
            continue

        messages.append({"role": "user", "content": user_input})
        if current_images:
            print(f"  [with {len(current_images)} image(s)]")
        print("Assistant: ", end="")

        response = model.generate(
            messages.copy(),
            images=current_images if current_images else None,
            max_new_tokens=args.max_tokens,
            stream=not args.no_stream,
        )
        messages.append({"role": "assistant", "content": response})
        if args.no_stream:
            print(response)

        current_images = []


if __name__ == "__main__":
    main()
