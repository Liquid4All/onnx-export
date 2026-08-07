#!/usr/bin/env python3
"""
ONNX inference script for LFM2-VL vision-language models.

Usage:
    uv run lfm2-vl-infer --model exports/LFM2-VL-450M-ONNX
    uv run lfm2-vl-infer --model exports/LFM2-VL-450M-ONNX --images photo.jpg
    uv run lfm2-vl-infer --model exports/LFM2-VL-450M-ONNX --images a.jpg b.jpg --prompt "Compare"

    # Run with quantized models
    uv run lfm2-vl-infer --model exports/LFM2-VL-450M-ONNX --precision q4
    uv run lfm2-vl-infer --model exports/LFM2-VL-450M-ONNX --precision fp16

    # Specify individual component files
    uv run lfm2-vl-infer --model exports/LFM2-VL-450M-ONNX \
        --embed-tokens embed_tokens_fp16.onnx \
        --embed-images vision_encoder_q4.onnx \
        --decoder decoder_model_merged_q4.onnx
"""

import argparse
import logging
from pathlib import Path

import numpy as np
from PIL import Image
from transformers import AutoProcessor

from liquidonnx.lfm2_vl import VISION_MODE_CONV2D, VISION_MODE_TILED
from liquidonnx.lfm2_vl.preprocessing import (
    build_inputs_embeds,
    detect_vision_format,
    pad_to_square,
    preprocess_conv2d,
    preprocess_tiled,
)
from liquidonnx.session import initialize_cache, load_onnx_session, update_cache

logger = logging.getLogger(__name__)


def get_onnx_dir(exports_dir: Path, size: str) -> Path:
    """Get ONNX directory for a VL model size."""
    return exports_dir / f"LFM2-VL-{size}-ONNX" / "onnx"


class VLModelInference:
    """ONNX inference for LFM2-VL models.

    Public API (re-exported from `liquidonnx.lfm2_vl`): embedding servers and
    smoke harnesses load a repo directory and drive `generate` directly — the
    CLI below is one such consumer. `generate` also handles text-only
    conversations (no vision encoder run), so a VL export can serve plain chat.
    """

    def __init__(
        self,
        model_path: str,
        embed_tokens_file: str | None = None,
        embed_images_file: str | None = None,
        decoder_file: str | None = None,
        force_cpu: bool = False,
    ):
        self.model_path = Path(model_path)
        self.embed_tokens_file = embed_tokens_file
        self.embed_images_file = embed_images_file
        self.decoder_file = decoder_file
        self.force_cpu = force_cpu
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

        # Resolve file paths (use provided or default)
        embed_tokens_path = onnx_dir / (self.embed_tokens_file or "embed_tokens.onnx")
        embed_images_path = onnx_dir / (self.embed_images_file or "vision_encoder.onnx")
        decoder_path = onnx_dir / (self.decoder_file or "decoder_model_merged.onnx")

        logger.info(f"  embed_tokens: {embed_tokens_path.name}")
        logger.info(f"  embed_images: {embed_images_path.name}")
        logger.info(f"  decoder: {decoder_path.name}")

        providers = ["CPUExecutionProvider"] if self.force_cpu else None
        self.embed_tokens_sess = load_onnx_session(embed_tokens_path, providers=providers)
        self.embed_images_sess = load_onnx_session(embed_images_path, providers=providers)
        self.decoder_sess = load_onnx_session(decoder_path, providers=providers)

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
                pixel_values, pixel_attention_mask, spatial_shapes = preprocess_tiled(
                    image, self.processor, do_pad_to_square=True
                )
                outputs = self.embed_images_sess.run(
                    None,
                    {
                        "pixel_values": pixel_values,
                        "pixel_attention_mask": pixel_attention_mask,
                        "spatial_shapes": spatial_shapes,
                    },
                )
                # Tiled output is 2D [total_tokens, hidden] after Compress
                embeddings.append(outputs[0])
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
        skip_special_tokens: bool = True,
    ) -> str:
        """Generate response for chat messages with optional images.

        `skip_special_tokens=False` keeps in-band markers (tool-call
        delimiters, reasoning boundaries) in the returned text — serving
        consumers parse them; the interactive CLI wants them stripped.
        """
        images = images or []

        if images:
            # Pad images to square for consistent tiling between processor and ONNX model
            # This ensures the number of <image> tokens matches the number of patch embeddings
            if self.vision_format == VISION_MODE_TILED:
                images = [pad_to_square(img) for img in images]

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

        return self.tokenizer.decode(generated_tokens, skip_special_tokens=skip_special_tokens)


def resolve_precision_files(
    precision: str | None,
) -> tuple[str | None, str | None, str | None]:
    """Resolve file names from precision shorthand.

    Args:
        precision: One of "fp16", "q4", "q8", or None for default (fp32)

    Returns:
        Tuple of (embed_tokens_file, embed_images_file, decoder_file)
    """
    if precision is None:
        return None, None, None

    precision = precision.lower()
    if precision == "fp16":
        return (
            "embed_tokens_fp16.onnx",
            "vision_encoder_fp16.onnx",
            "decoder_model_merged_fp16.onnx",
        )
    elif precision in ("q4", "q8"):
        # embed_tokens has no quantized version, use fp32
        return (
            "embed_tokens.onnx",
            f"vision_encoder_{precision}.onnx",
            f"decoder_model_merged_{precision}.onnx",
        )
    else:
        raise ValueError(f"Invalid precision: {precision}. Use fp16, q4, or q8.")


def main():
    parser = argparse.ArgumentParser(
        description="ONNX inference for LFM2-VL models (0-2 images per turn)"
    )
    parser.add_argument("--model", required=True, help="Path to ONNX model directory")
    parser.add_argument("--images", nargs="*", default=[], help="Image paths (0-2 images)")
    parser.add_argument("--prompt", default=None, help="Initial prompt (optional)")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming output")
    parser.add_argument(
        "--precision",
        choices=["fp16", "q4", "q8"],
        help="Model precision: fp16, q4, or q8 (default: fp32)",
    )
    parser.add_argument(
        "--embed-tokens",
        metavar="FILE",
        help="Custom embed_tokens ONNX file (relative to onnx/ dir)",
    )
    parser.add_argument(
        "--embed-images",
        metavar="FILE",
        help="Custom embed_images ONNX file (relative to onnx/ dir)",
    )
    parser.add_argument(
        "--decoder",
        metavar="FILE",
        help="Custom decoder ONNX file (relative to onnx/ dir)",
    )
    args = parser.parse_args()

    if len(args.images) > 2:
        print("Warning: Only 0-2 images supported per turn. Using first 2.")
        args.images = args.images[:2]

    # Resolve component files from --precision or explicit file args
    embed_tokens_file, embed_images_file, decoder_file = resolve_precision_files(args.precision)

    # Explicit file args override --precision
    if args.embed_tokens:
        embed_tokens_file = args.embed_tokens
    if args.embed_images:
        embed_images_file = args.embed_images
    if args.decoder:
        decoder_file = args.decoder

    model = VLModelInference(
        args.model,
        embed_tokens_file=embed_tokens_file,
        embed_images_file=embed_images_file,
        decoder_file=decoder_file,
    )
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
