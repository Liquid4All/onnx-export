#!/usr/bin/env python3
"""
ONNX inference script for LFM2-VL vision-language models.

Supports 0-2 images per turn.

Usage:
    # Text-only
    uv run inference_vl.py --model LFM2-VL-450M-ONNX-B4V4

    # With single image
    uv run inference_vl.py --model LFM2-VL-450M-ONNX-B4V4 --images cardinal.jpg

    # With two images
    uv run inference_vl.py --model LFM2-VL-450M-ONNX-B4V4 --images cardinal.jpg bluejay.jpg

    # With prompt
    uv run inference_vl.py --model LFM2-VL-450M-ONNX-B4V4 --images cardinal.jpg --prompt "What is this?"
"""

import argparse
import math
from pathlib import Path
from typing import List, Optional

import numpy as np
import onnxruntime as ort
from PIL import Image
from transformers import AutoProcessor


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
        self.vision_format = "tiled"  # "tiled" or "conv2d"

    def load(self):
        """Load processor and ONNX models."""
        print(f"Loading VL model from {self.model_path}...")

        # Find the HuggingFace model ID based on directory name
        dir_name = self.model_path.name
        if "450M" in dir_name:
            hf_model = "LiquidAI/LFM2-VL-450M"
        elif "1.6B" in dir_name:
            hf_model = "LiquidAI/LFM2-VL-1.6B"
        elif "3B" in dir_name:
            hf_model = "LiquidAI/LFM2-VL-3B"
        else:
            hf_model = str(self.model_path)

        # Load processor from HuggingFace (for image processing)
        print(f"Loading processor from {hf_model}...")
        self.processor = AutoProcessor.from_pretrained(hf_model, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer

        # Get image token ID
        self.image_token_id = self.tokenizer.convert_tokens_to_ids("<image>")
        print(f"Image token ID: {self.image_token_id}")

        # Load ONNX models
        onnx_dir = self.model_path / "onnx"

        print("Loading embed_tokens.onnx...")
        self.embed_tokens_sess = ort.InferenceSession(
            str(onnx_dir / "embed_tokens.onnx"), providers=["CPUExecutionProvider"]
        )

        print("Loading embed_images.onnx...")
        self.embed_images_sess = ort.InferenceSession(
            str(onnx_dir / "embed_images.onnx"), providers=["CPUExecutionProvider"]
        )

        print("Loading decoder.onnx...")
        self.decoder_sess = ort.InferenceSession(
            str(onnx_dir / "decoder.onnx"), providers=["CPUExecutionProvider"]
        )

        # Detect vision format from embed_images input shape
        self.vision_format = self._detect_vision_format()
        print(f"Vision format: {self.vision_format}")
        print("Model loaded successfully!")

    def _detect_vision_format(self) -> str:
        """Detect vision input format from ONNX model inputs."""
        input_names = {inp.name for inp in self.embed_images_sess.get_inputs()}
        # conv2d format has spatial_h and spatial_w inputs
        if "spatial_h" in input_names:
            return "conv2d"
        return "tiled"

    def _preprocess_conv2d(self, image: Image.Image) -> tuple[np.ndarray, int, int]:
        """Preprocess image for conv2d format.

        Matches llama.cpp preprocessing: preserves aspect ratio while keeping
        total pixels within min/max bounds, aligned to patch_size * n_merge.

        Returns (pixel_values, spatial_h, spatial_w) where spatial dimensions
        are AFTER n_merge (i.e., the final projector output dimensions).
        """
        patch_size, n_merge = 16, 2
        align_size = patch_size * n_merge  # 32

        # Token limits from model config (after n_merge downsampling)
        # image_min_tokens: 64, image_max_tokens: 256
        min_tokens, max_tokens = 64, 256
        # patch_area = patch_size^2 * n_merge^2 = 16*16*2*2 = 1024 pixels per output token
        patch_area = patch_size * patch_size * n_merge * n_merge
        min_pixels = min_tokens * patch_area  # 64 * 1024 = 65536 (256x256)
        max_pixels = max_tokens * patch_area  # 256 * 1024 = 262144 (512x512)

        w, h = image.size
        current_pixels = h * w

        # Scale to fit within min/max pixels while preserving aspect ratio
        if current_pixels > max_pixels:
            # Scale down
            beta = math.sqrt(current_pixels / max_pixels)
            h_new = max(align_size, int(h / beta) // align_size * align_size)
            w_new = max(align_size, int(w / beta) // align_size * align_size)
        elif current_pixels < min_pixels:
            # Scale up
            beta = math.sqrt(min_pixels / current_pixels)
            h_new = math.ceil(h * beta / align_size) * align_size
            w_new = math.ceil(w * beta / align_size) * align_size
        else:
            # Within bounds, just align to grid
            h_new = max(align_size, round(h / align_size) * align_size)
            w_new = max(align_size, round(w / align_size) * align_size)

        # Resize with bilinear interpolation (matches llama.cpp)
        image_resized = image.resize((w_new, h_new), Image.BILINEAR)

        # Compute spatial dimensions AFTER n_merge (for projector)
        spatial_h = h_new // patch_size // n_merge
        spatial_w = w_new // patch_size // n_merge

        # Convert to float and normalize
        # SigLIP2 normalization: mean=0.5, std=0.5 -> range [-1, 1]
        pixels = np.array(image_resized).astype(np.float32) / 255.0
        pixels = (pixels - 0.5) / 0.5
        pixels = pixels.transpose(2, 0, 1)[np.newaxis, ...]

        return pixels.astype(np.float32), spatial_h, spatial_w

    def _get_image_embeddings(self, images: List[Image.Image]) -> List[np.ndarray]:
        """Get embeddings for a list of images."""
        embeddings = []

        for image in images:
            if self.vision_format == "conv2d":
                # Conv2d format: [B, 3, H, W] raw image input with spatial dims
                pixel_values, spatial_h, spatial_w = self._preprocess_conv2d(image)
                outputs = self.embed_images_sess.run(
                    None,
                    {
                        "pixel_values": pixel_values,
                        "spatial_h": np.array(spatial_h, dtype=np.int64),
                        "spatial_w": np.array(spatial_w, dtype=np.int64),
                    },
                )
                # Output: [1, num_tokens, hidden_dim] where num_tokens = spatial_h * spatial_w
                img_embeds = outputs[0][0]  # [num_tokens, hidden_dim]
            else:
                # Tiled format: use HuggingFace processor for patch extraction
                # The ONNX model's position embedding interpolation assumes square layout
                # so we need to pad non-square images to square first
                w, h = image.size
                if w != h:
                    # Pad to square by adding black bars
                    max_dim = max(w, h)
                    square_img = Image.new('RGB', (max_dim, max_dim), (0, 0, 0))
                    # Center the original image
                    paste_x = (max_dim - w) // 2
                    paste_y = (max_dim - h) // 2
                    square_img.paste(image, (paste_x, paste_y))
                    image = square_img

                # Process with image splitting disabled for single square tile
                processed = self.processor(
                    images=[image],
                    text="<image>",  # Minimal text with image token (required by processor)
                    return_tensors='pt',
                    do_image_splitting=False  # Single tile for simpler position embeddings
                )

                # Get pixel values - with square input, all patches should be valid
                # pixel_values: [1, num_patches, 768] where num_patches should be square
                pixel_values_pt = processed['pixel_values']
                attention_mask_pt = processed['pixel_attention_mask']

                # Convert to numpy
                pixel_values = pixel_values_pt.numpy().astype(np.float32)
                patch_attention_mask = attention_mask_pt.numpy().astype(np.int64)

                outputs = self.embed_images_sess.run(
                    None,
                    {
                        "pixel_values": pixel_values,
                        "patch_attention_mask": patch_attention_mask,
                    },
                )

                # Output: [1, num_tokens, hidden_dim]
                img_embeds = outputs[0][0]  # [num_tokens, hidden_dim]

            embeddings.append(img_embeds)

        return embeddings

    def _get_text_embeddings(self, input_ids: np.ndarray) -> np.ndarray:
        """Get text embeddings from token IDs."""
        outputs = self.embed_tokens_sess.run(
            None, {"input_ids": input_ids.astype(np.int64)}
        )
        return outputs[0]  # [1, seq_len, hidden_dim]

    def _build_inputs_embeds(
        self, input_ids: np.ndarray, image_embeds_list: List[np.ndarray]
    ) -> np.ndarray:
        """Build inputs_embeds by replacing image tokens with image embeddings."""
        text_embeds = self._get_text_embeddings(input_ids)[0]  # [seq_len, hidden]

        # Find image token positions
        image_positions = np.where(input_ids[0] == self.image_token_id)[0].tolist()

        if len(image_positions) == 0 or len(image_embeds_list) == 0:
            return text_embeds[np.newaxis, ...]  # [1, seq_len, hidden]

        # Build combined embeddings
        result_parts = []
        prev_end = 0

        for img_idx, image_embeds in enumerate(image_embeds_list):
            if img_idx >= len(image_positions):
                break
            pos = image_positions[img_idx]
            # Add text before this image token
            result_parts.append(text_embeds[prev_end:pos])
            # Add image embeddings
            result_parts.append(image_embeds)
            prev_end = pos + 1  # Skip the image token

        # Add remaining text after last image token
        result_parts.append(text_embeds[prev_end:])

        combined = np.concatenate(result_parts, axis=0)
        return combined[np.newaxis, ...].astype(np.float32)

    def _initialize_cache(self) -> dict:
        """Initialize cache tensors for decoder."""
        cache = {}
        for inp in self.decoder_sess.get_inputs():
            name = inp.name
            if name in ["inputs_embeds", "attention_mask", "position_ids"]:
                continue
            shape = [d if isinstance(d, int) else 1 for d in inp.shape]
            # Set sequence dimension to 0 for KV cache
            for i, d in enumerate(inp.shape):
                if isinstance(d, str) and "sequence" in d.lower():
                    shape[i] = 0
            cache[name] = np.zeros(shape, dtype=np.float32)
        return cache

    def _update_cache(self, cache: dict, outputs: list):
        """Update cache from decoder outputs."""
        output_names = [out.name for out in self.decoder_sess.get_outputs()]
        for i, name in enumerate(output_names[1:], 1):  # Skip logits
            if "present_conv" in name:
                cache_name = name.replace("present_conv", "past_conv")
            elif "present." in name:
                cache_name = name.replace("present.", "past_key_values.")
            else:
                continue
            if cache_name in cache:
                cache[cache_name] = outputs[i]

    def generate(
        self,
        messages: list,
        images: Optional[List[Image.Image]] = None,
        max_new_tokens: int = 100,
        stream: bool = True,
    ) -> str:
        """Generate response for chat messages with optional images."""
        images = images or []

        # Build conversation with image placeholders
        if images:
            # Insert image tokens into first user message
            for i, msg in enumerate(messages):
                if msg["role"] == "user":
                    image_tokens = "<image>" * len(images)
                    messages[i] = {
                        "role": "user",
                        "content": image_tokens + msg["content"],
                    }
                    break

        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Tokenize
        input_ids = np.array(
            [self.tokenizer.encode(prompt, add_special_tokens=False)], dtype=np.int64
        )

        # Get image embeddings if images provided
        if images:
            image_embeds_list = self._get_image_embeddings(images)
            inputs_embeds = self._build_inputs_embeds(input_ids, image_embeds_list)
        else:
            inputs_embeds = self._get_text_embeddings(input_ids)

        # Initialize cache
        cache = self._initialize_cache()

        # Check for position_ids input
        has_position_ids = "position_ids" in {
            inp.name for inp in self.decoder_sess.get_inputs()
        }

        seq_len = inputs_embeds.shape[1]
        generated_tokens = []
        cur_len = seq_len

        for step in range(max_new_tokens):
            if step == 0:
                embeds = inputs_embeds
                pos = np.arange(seq_len, dtype=np.int64).reshape(1, -1)
            else:
                # Get embedding for last generated token
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

            # Run decoder
            outputs = self.decoder_sess.run(None, feed)
            logits = outputs[0][0, -1]

            # Greedy decoding
            next_token = int(np.argmax(logits))
            generated_tokens.append(next_token)

            # Update cache
            self._update_cache(cache, outputs)
            cur_len += 1

            # Stream output
            if stream:
                token_str = self.tokenizer.decode([next_token])
                print(token_str, end="", flush=True)

            # Check for EOS
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
    parser.add_argument(
        "--images", nargs="*", default=[], help="Image paths (0-2 images)"
    )
    parser.add_argument("--prompt", default=None, help="Initial prompt (optional)")
    parser.add_argument(
        "--max-tokens", type=int, default=100, help="Max tokens to generate"
    )
    parser.add_argument(
        "--no-stream", action="store_true", help="Disable streaming output"
    )
    args = parser.parse_args()

    if len(args.images) > 2:
        print("Warning: Only 0-2 images supported per turn. Using first 2.")
        args.images = args.images[:2]

    # Load model
    model = VLModelInference(args.model)
    model.load()

    print("\n" + "=" * 50)
    print("LFM2-VL Model - ONNX Inference")
    print("Supports 0-2 images per message")
    print("Commands: 'quit', 'exit', 'clear', 'image <path>' or 'images <path1> <path2>'")
    print("=" * 50 + "\n")

    messages = []
    current_images = []

    # Load initial images
    for img_path in args.images:
        if Path(img_path).exists():
            current_images.append(Image.open(img_path).convert("RGB"))
            print(f"Loaded image: {img_path}")

    # Initial prompt if provided
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
        # Clear images after first turn
        current_images = []

    # Interactive loop
    while True:
        try:
            user_input = input("\nUser: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        # Handle commands
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

        # Generate response
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

        # Clear images after use (images are per-turn)
        current_images = []


if __name__ == "__main__":
    main()
