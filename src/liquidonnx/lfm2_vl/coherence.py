#!/usr/bin/env python3
"""
Multi-turn coherence testing for LFM2-VL ONNX models.

Tests whether ONNX VL models maintain coherent multi-turn conversations
with image context compared to PyTorch ground truth.

This test uses the full e2e flow:
1. embed_images.onnx - process image to embeddings
2. embed_tokens.onnx - process text tokens to embeddings
3. Replace <image> placeholder with image embeddings
4. decoder.onnx - generate with combined inputs_embeds

Metrics:
- Token-level: exact match of generated tokens per turn
- Semantic: cosine similarity of logits per turn
- Accumulated error across turns

Available formats: -T (tiled), -C (conv2d)

Usage:
    # Test single model with specific variant
    uv run coherence_vl.py --model LiquidAI/LFM2-VL-450M --onnx LFM2-VL-450M-ONNX-B4V8-T --image cardinal.jpg

    # Test tiled format models
    uv run coherence_vl.py -T --models 450M --variants FP32 B4V4 B4V8 B8V8 --image cardinal.jpg

    # Test conv2d format models
    uv run coherence_vl.py -C --models 450M --variants B4V8 --image cardinal.jpg

    # Test both formats
    uv run coherence_vl.py -T -C --models 450M --variants B4V8 --image cardinal.jpg

    # More tokens per turn
    uv run coherence_vl.py -T --models 450M --variants B4V8 --image cardinal.jpg --max-tokens 50
"""

import argparse
import gc
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from liquidonnx.lfm2_vl import detect_vision_format, preprocess_conv2d, preprocess_tiled

logger = logging.getLogger(__name__)

# ============================================================================
# MODEL PATHS
# ============================================================================

PYTORCH_MODELS = {
    "450M": "LiquidAI/LFM2-VL-450M",
    "1.6B": "LiquidAI/LFM2-VL-1.6B",
    "3B": "LiquidAI/LFM2-VL-3B",
}

# Quantization variants: (backbone_bits, vision_bits) or None for FP32
VARIANTS = {
    "FP32": None,
    "B4V4": (4, 4),
    "B4V8": (4, 8),
    "B8V8": (8, 8),
}

# Vision input formats
FORMATS = {
    "T": "tiled",   # [B, N, 768] pre-extracted patches
    "C": "conv2d",  # [B, 3, H, W] raw image
}


def get_onnx_dir(size: str, variant: str, format_key: str = "T") -> str:
    """Get ONNX directory name for a model/variant/format combination."""
    suffix = f"-{format_key}"
    if VARIANTS[variant] is None:
        return f"LFM2-VL-{size}-ONNX{suffix}"
    else:
        backbone_bits, vision_bits = VARIANTS[variant]
        return f"LFM2-VL-{size}-ONNX-B{backbone_bits}V{vision_bits}{suffix}"


# Default conversation for VL coherence testing - tests context retention with image
DEFAULT_PROMPTS = [
    "What do you see in this image? Describe the main elements.",
    "What colors are present in the image?",
    "Can you identify any shapes or patterns?",
    "Based on what you described, what type of image is this?",
    "If I wanted to recreate this image, what would I need?",
]

# Prompts for multi-image testing
# Note: Some prompts like "I'm showing you multiple images..." trigger a refusal
# response on certain model sizes. Use simpler prompts that work reliably.
MULTI_IMAGE_PROMPTS = [
    "Which one most important thing do you see on each image? Be concise and exact.",
    "What are the similarities between these images?",
    "What are the differences between these images?",
    "Which image do you prefer and why?",
]


# ============================================================================

@dataclass
class TurnResult:
    """Result of a single conversation turn."""
    turn: int
    prompt: str
    pytorch_response: str
    onnx_response: str
    token_match_rate: float  # Percentage of matching tokens
    semantic_similarity: float  # Cosine similarity of logits
    max_logit_diff: float
    mean_logit_diff: float


@dataclass
class CoherenceResult:
    """Result of multi-turn coherence test."""
    model_size: str
    source: str
    quant_type: str
    turns: List[TurnResult] = field(default_factory=list)
    avg_token_match: float = 0.0
    avg_semantic_sim: float = 0.0
    accumulated_error: float = 0.0


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    a_flat = a.flatten()
    b_flat = b.flatten()
    dot = np.dot(a_flat, b_flat)
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


class VLMultiTurnTester:
    """Tests multi-turn coherence of VL models with proper e2e ONNX flow."""

    def __init__(self, pytorch_path: str, max_new_tokens: int = 30):
        self.pytorch_path = pytorch_path
        self.max_new_tokens = max_new_tokens
        self.processor = None
        self.tokenizer = None
        self.torch_model = None
        # ONNX sessions
        self.embed_tokens_sess = None
        self.embed_images_sess = None
        self.decoder_sess = None
        self.images: List = []  # Support multiple images
        # Cached image embeddings
        self.pytorch_image_embeds = None
        self.onnx_image_embeds = None

    def load_pytorch(self):
        """Load PyTorch VL model."""
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        logger.info(f"Loading PyTorch VL model: {self.pytorch_path}")
        self.processor = AutoProcessor.from_pretrained(
            self.pytorch_path, trust_remote_code=True
        )
        self.tokenizer = self.processor.tokenizer
        self.torch_model = AutoModelForImageTextToText.from_pretrained(
            self.pytorch_path,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        self.torch_model.eval()

    def load_onnx(self, onnx_path: str):
        """Load all ONNX VL models (embed_tokens, embed_images, decoder)."""
        import onnxruntime as ort

        # Clean up previous sessions to avoid resource leaks
        self.embed_tokens_sess = None
        self.embed_images_sess = None
        self.decoder_sess = None
        gc.collect()

        onnx_dir = os.path.join(onnx_path, "onnx")

        embed_tokens_file = os.path.join(onnx_dir, "embed_tokens.onnx")
        embed_images_file = os.path.join(onnx_dir, "embed_images.onnx")
        decoder_file = os.path.join(onnx_dir, "decoder.onnx")

        if not os.path.exists(embed_tokens_file):
            raise FileNotFoundError(f"embed_tokens.onnx not found in {onnx_dir}")
        if not os.path.exists(embed_images_file):
            raise FileNotFoundError(f"embed_images.onnx not found in {onnx_dir}")
        if not os.path.exists(decoder_file):
            raise FileNotFoundError(f"decoder.onnx not found in {onnx_dir}")

        logger.info(f"Loading embed_tokens from {embed_tokens_file}...")
        self.embed_tokens_sess = ort.InferenceSession(
            embed_tokens_file, providers=["CPUExecutionProvider"]
        )

        logger.info(f"Loading embed_images from {embed_images_file}...")
        self.embed_images_sess = ort.InferenceSession(
            embed_images_file, providers=["CPUExecutionProvider"]
        )

        logger.info(f"Loading decoder from {decoder_file}...")
        self.decoder_sess = ort.InferenceSession(
            decoder_file, providers=["CPUExecutionProvider"]
        )

        # Clear cached embeddings when loading new ONNX model
        self.onnx_image_embeds = None

    def _detect_vision_format(self) -> str:
        """Detect vision input format from ONNX model inputs."""
        return detect_vision_format(self.embed_images_sess)

    def load_images(self, image_paths: List[str]) -> List:
        """Load images from paths or create test image."""
        from PIL import Image

        self.images = []
        for image_path in image_paths:
            if image_path and os.path.exists(image_path):
                logger.info(f"Loading image from {image_path}")
                self.images.append(Image.open(image_path).convert("RGB"))
            else:
                logger.info(f"Image not found: {image_path}, creating test image...")
                self.images.append(self._create_test_image())

        # Clear cached embeddings when loading new images
        self.pytorch_image_embeds = None
        self.onnx_image_embeds = None

        logger.info(f"Loaded {len(self.images)} image(s)")
        return self.images

    def _create_test_image(self, size: int = 512) -> "Image":
        """Create a test image with identifiable content."""
        from PIL import Image

        img = Image.new('RGB', (size, size), color=(200, 200, 200))
        pixels = np.array(img)
        # Red square (top-left)
        pixels[50:150, 50:150] = [255, 0, 0]
        # Blue circle approximation (top-right)
        center = (100, 350)
        for y in range(50, 150):
            for x in range(300, 400):
                if (y - center[0])**2 + (x - center[1])**2 < 50**2:
                    pixels[y, x] = [0, 0, 255]
        # Green triangle approximation (bottom-center)
        for y in range(300, 400):
            width = int((y - 300) * 0.5)
            x_start = 256 - width
            x_end = 256 + width
            if x_start >= 0 and x_end < size:
                pixels[y, x_start:x_end] = [0, 255, 0]
        return Image.fromarray(pixels)

    def get_pytorch_image_embeddings(self) -> List[np.ndarray]:
        """Get image embeddings from PyTorch model (cached). Returns list of embeddings per image."""
        import torch

        if self.pytorch_image_embeds is not None:
            return self.pytorch_image_embeds

        self.pytorch_image_embeds = []

        for image in self.images:
            inputs = self.processor.image_processor(images=image, return_tensors="pt")
            pixel_values = inputs["pixel_values"]
            pixel_attention_mask = inputs["pixel_attention_mask"]
            spatial_shapes = inputs["spatial_shapes"]

            with torch.no_grad():
                vision_outputs = self.torch_model.model.vision_tower(
                    pixel_values=pixel_values,
                    pixel_attention_mask=pixel_attention_mask,
                    spatial_shapes=spatial_shapes,
                ).last_hidden_state

                # Process through projector per tile
                embeddings_list = []
                num_tiles = pixel_values.shape[0]
                for tile_idx in range(num_tiles):
                    feature = vision_outputs[tile_idx]
                    h, w = spatial_shapes[tile_idx].tolist()
                    feature = feature[:h * w].reshape(1, h, w, -1)
                    proj_out = self.torch_model.model.multi_modal_projector(feature)
                    proj_out = proj_out.reshape(-1, proj_out.shape[-1])
                    embeddings_list.append(proj_out)

                # Concatenate all tile embeddings for this image
                image_embeds = torch.cat(embeddings_list, dim=0).numpy()
                self.pytorch_image_embeds.append(image_embeds)

        return self.pytorch_image_embeds

    def get_onnx_image_embeddings(self) -> List[np.ndarray]:
        """Get image embeddings from ONNX model (cached). Returns list of embeddings per image."""
        if self.onnx_image_embeds is not None:
            return self.onnx_image_embeds

        self.onnx_image_embeds = []
        vision_format = self._detect_vision_format()

        for image in self.images:
            if vision_format == "conv2d":
                # Conv2d format: use liquidonnx preprocess_conv2d
                pixel_values, spatial_h, spatial_w = preprocess_conv2d(image)
                outputs = self.embed_images_sess.run(None, {
                    "pixel_values": pixel_values,
                    "spatial_h": np.array(spatial_h, dtype=np.int64),
                    "spatial_w": np.array(spatial_w, dtype=np.int64),
                })
                onnx_embeds = outputs[0][0]  # (num_tokens, hidden_dim)
            else:
                # Tiled format: use liquidonnx preprocess_tiled
                pixel_values, patch_attention_mask, _ = preprocess_tiled(
                    image, self.processor, do_image_splitting=False, pad_to_square=True
                )
                outputs = self.embed_images_sess.run(None, {
                    "pixel_values": pixel_values,
                    "patch_attention_mask": patch_attention_mask,
                })
                # Flatten tiles to (total_tokens, hidden_dim)
                onnx_embeds = outputs[0]
                num_tiles, tokens_per_tile, hidden = onnx_embeds.shape
                onnx_embeds = onnx_embeds.reshape(-1, hidden)

            self.onnx_image_embeds.append(onnx_embeds)

        return self.onnx_image_embeds

    def _get_image_token_id(self) -> int:
        """Get the image placeholder token ID."""
        # Try common image token names
        for token_name in ["<image>", "<|image|>", "[IMG]"]:
            token_id = self.tokenizer.convert_tokens_to_ids(token_name)
            if token_id != self.tokenizer.unk_token_id:
                return token_id
        # Fallback: check tokenizer config
        if hasattr(self.tokenizer, "image_token_id"):
            return self.tokenizer.image_token_id
        raise ValueError("Could not find image token ID")

    def _build_inputs_embeds_pytorch(
        self, input_ids: "torch.Tensor", image_embeds_list: List[np.ndarray]
    ) -> "torch.Tensor":
        """Build inputs_embeds by replacing image tokens with image embeddings (PyTorch)."""
        import torch

        image_token_id = self._get_image_token_id()

        # Get text embeddings
        with torch.no_grad():
            text_embeds = self.torch_model.model.language_model.embed_tokens(input_ids)

        # Find all image token positions
        image_positions = (input_ids[0] == image_token_id).nonzero(as_tuple=True)[0].tolist()

        if len(image_positions) == 0:
            return text_embeds

        # Build combined embeddings by replacing each image token with corresponding embeddings
        text_embeds_np = text_embeds[0].numpy()
        result_parts = []
        prev_end = 0

        for img_idx, image_embeds in enumerate(image_embeds_list):
            if img_idx >= len(image_positions):
                break
            pos = image_positions[img_idx]
            # Add text before this image token
            result_parts.append(text_embeds_np[prev_end:pos])
            # Add image embeddings
            result_parts.append(image_embeds)
            prev_end = pos + 1  # Skip the image token

        # Add remaining text after last image token
        result_parts.append(text_embeds_np[prev_end:])

        combined = np.concatenate(result_parts, axis=0)
        return torch.from_numpy(combined).unsqueeze(0).float()

    def _build_inputs_embeds_onnx(
        self, input_ids: np.ndarray, image_embeds_list: List[np.ndarray]
    ) -> np.ndarray:
        """Build inputs_embeds by replacing image tokens with image embeddings (ONNX)."""
        image_token_id = self._get_image_token_id()

        # Get text embeddings from embed_tokens
        text_embeds = self.embed_tokens_sess.run(None, {
            "input_ids": input_ids.astype(np.int64),
        })[0][0]  # Remove batch dim

        # Find all image token positions
        image_positions = np.where(input_ids[0] == image_token_id)[0].tolist()

        if len(image_positions) == 0:
            return text_embeds[np.newaxis, ...]  # Add batch dim back

        # Build combined embeddings by replacing each image token with corresponding embeddings
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

    def _conversation_has_images(self, messages: List[dict]) -> bool:
        """Check if any message in conversation has image content."""
        for msg in messages:
            content = msg.get("content", [])
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image":
                        return True
        return False

    def _normalize_messages_for_vl(self, messages: List[dict]) -> List[dict]:
        """
        Normalize messages for VL model processing:
        1. Convert string content to list format
        2. Inject actual image objects into image placeholders
        """
        result = []
        img_idx = 0
        for msg in messages:
            content = msg.get("content", "")

            # Convert string content to list format
            if isinstance(content, str):
                new_content = [{"type": "text", "text": content}]
            else:
                # Process list content - inject images
                new_content = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image":
                        # Inject actual image object if not already present
                        if "image" not in item and img_idx < len(self.images):
                            new_content.append({"type": "image", "image": self.images[img_idx]})
                            img_idx += 1
                        else:
                            new_content.append(item)
                    else:
                        new_content.append(item)

            result.append({**msg, "content": new_content})

        return result

    def generate_pytorch(
        self, messages: List[dict], max_new_tokens: int, has_image: bool = False, stream: bool = False
    ) -> Tuple[List[int], np.ndarray, str]:
        """Generate tokens with PyTorch VL model using model.generate() for correct image handling."""
        import torch

        # Normalize messages for VL processing (inject images, convert string to list format)
        conv_has_images = self._conversation_has_images(messages)
        if conv_has_images and self.images:
            messages_with_images = self._normalize_messages_for_vl(messages)
            # Use processor.apply_chat_template with images embedded in conversation
            inputs = self.processor.apply_chat_template(
                messages_with_images,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                tokenize=True,
            )
        else:
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(text=text, return_tensors="pt")

        all_logits = []
        generated_tokens = []

        with torch.no_grad():
            # Use model.generate for correct image embedding handling
            output_ids = self.torch_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_logits=True,
            )

            # Extract generated tokens (skip input tokens)
            input_len = inputs['input_ids'].shape[1]
            generated_tokens = output_ids.sequences[0, input_len:].tolist()

            # Get logits for each generated token
            # output_ids.logits is a tuple of tensors with shape (batch, vocab_size)
            if output_ids.logits:
                for logit in output_ids.logits:
                    all_logits.append(logit[0].numpy())

        if stream:
            print(self.tokenizer.decode(generated_tokens, skip_special_tokens=True))

        response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return generated_tokens, np.stack(all_logits) if all_logits else np.array([]), response_text

    def generate_onnx(
        self, messages: List[dict], max_new_tokens: int, has_image: bool = False, stream: bool = False
    ) -> Tuple[List[int], np.ndarray, str]:
        """Generate tokens with ONNX VL model using proper e2e flow with processor."""
        sess = self.decoder_sess

        # Apply chat template
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Pass images if conversation contains image content (not just first turn)
        conv_has_images = self._conversation_has_images(messages)
        if conv_has_images and self.images:
            inputs = self.processor(text=text, images=self.images, return_tensors="pt")
            input_ids = inputs['input_ids'].numpy().astype(np.int64)

            # Get image embeddings from ONNX
            image_embeds_list = self.get_onnx_image_embeddings()
            # Concatenate all image embeddings
            all_image_embeds = np.concatenate(image_embeds_list, axis=0)

            # Get text embeddings
            text_embeds = self.embed_tokens_sess.run(None, {"input_ids": input_ids})[0][0]

            # Find and replace image tokens with image embeddings
            image_token_id = self._get_image_token_id()
            image_mask = (input_ids[0] == image_token_id)
            num_image_tokens = image_mask.sum()

            if num_image_tokens > 0 and len(all_image_embeds) > 0:
                # Build combined embeddings
                result_embeds = []
                img_idx = 0
                for i, is_image in enumerate(image_mask):
                    if is_image and img_idx < len(all_image_embeds):
                        result_embeds.append(all_image_embeds[img_idx])
                        img_idx += 1
                    else:
                        result_embeds.append(text_embeds[i])
                inputs_embeds = np.stack(result_embeds, axis=0)[np.newaxis, ...].astype(np.float32)
            else:
                inputs_embeds = text_embeds[np.newaxis, ...].astype(np.float32)
        else:
            input_ids = np.array([self.tokenizer.encode(text, add_special_tokens=False)], dtype=np.int64)
            inputs_embeds = self.embed_tokens_sess.run(None, {"input_ids": input_ids})[0]

        seq_len = inputs_embeds.shape[1]
        attention_mask = np.ones((1, seq_len), dtype=np.int64)
        position_ids = np.arange(seq_len, dtype=np.int64).reshape(1, -1)

        # Check which inputs the model expects
        input_names = {inp.name for inp in sess.get_inputs()}
        has_position_ids = "position_ids" in input_names

        # Initialize caches
        cache = {}
        for inp in sess.get_inputs():
            if inp.name not in ["inputs_embeds", "attention_mask", "position_ids"]:
                shape = [d if isinstance(d, int) else 1 for d in inp.shape]
                cache[inp.name] = np.zeros(shape, dtype=np.float32)

        outputs_info = sess.get_outputs()
        all_logits = []
        generated_tokens = []
        cur_len = seq_len

        for step in range(max_new_tokens):
            if step == 0:
                embeds = inputs_embeds
                pos = position_ids
            else:
                # Get embedding for last generated token
                last_token = np.array([[generated_tokens[-1]]], dtype=np.int64)
                embeds = self.embed_tokens_sess.run(None, {"input_ids": last_token})[0]
                pos = np.array([[cur_len - 1]], dtype=np.int64)

            attn_mask = np.ones((1, cur_len), dtype=np.int64)

            feed = {"inputs_embeds": embeds.astype(np.float32), "attention_mask": attn_mask}
            if has_position_ids:
                feed["position_ids"] = pos
            feed.update(cache)

            result = sess.run(None, feed)
            logits = result[0][0, -1]
            all_logits.append(logits)

            # Update caches
            for i, out_info in enumerate(outputs_info[1:], 1):
                out_name = out_info.name
                if "present_conv" in out_name:
                    cache_name = out_name.replace("present_conv", "past_conv")
                elif "present." in out_name:
                    cache_name = out_name.replace("present.", "past_key_values.")
                else:
                    continue
                if cache_name in cache:
                    cache[cache_name] = result[i]

            next_token = int(np.argmax(logits))
            generated_tokens.append(next_token)
            cur_len += 1

            if stream:
                print(self.tokenizer.decode([next_token]), end="", flush=True)

            if next_token == self.tokenizer.eos_token_id:
                break

        if stream:
            print()

        response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return generated_tokens, np.stack(all_logits) if all_logits else np.array([]), response_text

    def compare_turn(
        self,
        turn: int,
        prompt: str,
        messages_pytorch: List[dict],
        messages_onnx: List[dict],
        is_first_turn: bool = False,
    ) -> TurnResult:
        """Compare a single turn between PyTorch and ONNX."""
        # Build message with image(s) on first turn
        if is_first_turn:
            content = []
            # Add image placeholders for each image
            for _ in self.images:
                content.append({"type": "image"})
            content.append({"type": "text", "text": prompt})
            user_message = {"role": "user", "content": content}
        else:
            user_message = {"role": "user", "content": prompt}

        messages_pytorch = messages_pytorch + [user_message]
        messages_onnx = messages_onnx + [user_message]

        print(f"  Turn {turn} prompt: {prompt}")
        print(f"  Turn {turn} PyTorch: ", end="", flush=True)
        pytorch_tokens, pytorch_logits, pytorch_response = self.generate_pytorch(
            messages_pytorch, self.max_new_tokens, has_image=is_first_turn, stream=True
        )
        print(f"  Turn {turn} ONNX:    ", end="", flush=True)
        onnx_tokens, onnx_logits, onnx_response = self.generate_onnx(
            messages_onnx, self.max_new_tokens, has_image=is_first_turn, stream=True
        )

        # Token match rate
        min_len = min(len(pytorch_tokens), len(onnx_tokens))
        if min_len > 0:
            matches = sum(
                1 for i in range(min_len) if pytorch_tokens[i] == onnx_tokens[i]
            )
            token_match_rate = matches / max(len(pytorch_tokens), len(onnx_tokens))
        else:
            token_match_rate = 1.0 if len(pytorch_tokens) == len(onnx_tokens) == 0 else 0.0

        # Semantic similarity
        if len(pytorch_logits) > 0 and len(onnx_logits) > 0:
            min_steps = min(len(pytorch_logits), len(onnx_logits))
            similarities = [
                cosine_similarity(pytorch_logits[i], onnx_logits[i])
                for i in range(min_steps)
            ]
            semantic_similarity = np.mean(similarities)

            diffs = [
                np.abs(pytorch_logits[i] - onnx_logits[i])
                for i in range(min_steps)
            ]
            max_logit_diff = float(np.max([d.max() for d in diffs]))
            mean_logit_diff = float(np.mean([d.mean() for d in diffs]))
        else:
            semantic_similarity = 1.0
            max_logit_diff = 0.0
            mean_logit_diff = 0.0

        return TurnResult(
            turn=turn,
            prompt=prompt,
            pytorch_response=pytorch_response,
            onnx_response=onnx_response,
            token_match_rate=token_match_rate,
            semantic_similarity=semantic_similarity,
            max_logit_diff=max_logit_diff,
            mean_logit_diff=mean_logit_diff,
        )

    def test_coherence(
        self,
        model_size: str,
        onnx_path: str,
        source: str,
        quant_type: str,
        prompts: List[str],
    ) -> CoherenceResult:
        """Run multi-turn coherence test."""
        # Clear cached ONNX embeddings for new model
        self.onnx_image_embeds = None

        self.load_onnx(onnx_path)

        result = CoherenceResult(
            model_size=model_size,
            source=source,
            quant_type=quant_type,
        )

        # Start with empty message history
        messages_pytorch: List[dict] = []
        messages_onnx: List[dict] = []
        accumulated_error = 0.0

        for turn, prompt in enumerate(prompts, 1):
            is_first = (turn == 1)
            turn_result = self.compare_turn(
                turn, prompt, messages_pytorch, messages_onnx, is_first_turn=is_first
            )
            result.turns.append(turn_result)
            accumulated_error += (1.0 - turn_result.semantic_similarity)

            # Update message histories
            if is_first:
                content = []
                for _ in self.images:
                    content.append({"type": "image"})
                content.append({"type": "text", "text": prompt})
                user_msg = {"role": "user", "content": content}
            else:
                user_msg = {"role": "user", "content": prompt}

            messages_pytorch = messages_pytorch + [
                user_msg,
                {"role": "assistant", "content": turn_result.pytorch_response},
            ]
            messages_onnx = messages_onnx + [
                user_msg,
                {"role": "assistant", "content": turn_result.onnx_response},
            ]

        if result.turns:
            result.avg_token_match = np.mean([t.token_match_rate for t in result.turns])
            result.avg_semantic_sim = np.mean([t.semantic_similarity for t in result.turns])
            result.accumulated_error = accumulated_error

        return result


def print_turn_results(result: CoherenceResult):
    """Print detailed turn-by-turn results."""
    print(f"\n{'='*80}")
    print(f"VL MULTI-TURN COHERENCE: {result.source.upper()} {result.quant_type.upper()}")
    print(f"Model: LFM2-VL-{result.model_size}")
    print(f"{'='*80}")

    for turn in result.turns:
        print(f"\n--- Turn {turn.turn} ---")
        print(f"Prompt: {turn.prompt}")
        print(f"PyTorch: {turn.pytorch_response[:100]}{'...' if len(turn.pytorch_response) > 100 else ''}")
        print(f"ONNX:    {turn.onnx_response[:100]}{'...' if len(turn.onnx_response) > 100 else ''}")
        print(f"Token Match: {turn.token_match_rate*100:.1f}%")
        print(f"Semantic Sim: {turn.semantic_similarity:.4f}")
        print(f"Max Logit Diff: {turn.max_logit_diff:.4f}")

    print(f"\n--- Summary ---")
    print(f"Avg Token Match: {result.avg_token_match*100:.1f}%")
    print(f"Avg Semantic Sim: {result.avg_semantic_sim:.4f}")
    print(f"Accumulated Error: {result.accumulated_error:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-turn coherence testing for LFM2-VL models (e2e flow)"
    )
    # Vision input format
    parser.add_argument(
        "-T", "--tiled",
        action="store_true",
        help="Test tiled format models [B, N, 768]",
    )
    parser.add_argument(
        "-C", "--conv2d",
        action="store_true",
        help="Test conv2d format models [B, 3, H, W]",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="PyTorch model path (e.g., LiquidAI/LFM2-VL-450M)",
    )
    parser.add_argument(
        "--onnx",
        type=str,
        help="ONNX model directory",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["450M", "1.6B", "3B"],
        default=None,
        help="Model sizes to test (for batch testing)",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=list(VARIANTS.keys()),
        default=["FP32", "B4V8"],
        help="Variants to test (default: FP32 B4V8)",
    )
    parser.add_argument(
        "--image",
        type=str,
        nargs="+",
        default=["cardinal.jpg", "bluejay.jpg"],
        help="Test image path(s) (default: cardinal.jpg bluejay.jpg)",
    )
    parser.add_argument(
        "--single-image-only",
        action="store_true",
        help="Only test with first image (skip multi-image test)",
    )
    parser.add_argument(
        "--multi-image-only",
        action="store_true",
        help="Only test with multiple images (skip single-image test)",
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=3,
        help="Number of conversation turns (default: 3)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=30,
        help="Max tokens to generate per turn",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed turn-by-turn results",
    )
    args = parser.parse_args()

    # Determine which formats to test
    format_keys = []
    if args.tiled:
        format_keys.append("T")
    if args.conv2d:
        format_keys.append("C")
    if not format_keys:
        format_keys = ["T"]  # Default to tiled

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    results = []

    # Determine which tests to run
    run_single = not args.multi_image_only
    run_multi = not args.single_image_only and len(args.image) > 1

    # Single model mode (--model and --onnx specified)
    if args.model and args.onnx:
        tester = VLMultiTurnTester(args.model, max_new_tokens=args.max_tokens)
        tester.load_pytorch()

        # Determine variant from path
        variant = "FP32"
        for v in VARIANTS.keys():
            if v in args.onnx:
                variant = v
                break

        size = "custom"
        for s in ["450M", "1.6B", "3B"]:
            if s in args.onnx:
                size = s
                break

        # Single image test
        if run_single:
            tester.load_images([args.image[0]])
            prompts = DEFAULT_PROMPTS[: args.turns]
            result = tester.test_coherence(size, args.onnx, variant, f"{variant}_1img", prompts)
            results.append(result)
            print_turn_results(result)

        # Multi-image test
        if run_multi:
            tester.load_images(args.image)
            prompts = MULTI_IMAGE_PROMPTS[: args.turns]
            result = tester.test_coherence(size, args.onnx, variant, f"{variant}_{len(args.image)}img", prompts)
            results.append(result)
            print_turn_results(result)

        return

    # Batch mode
    if args.models is None:
        args.models = ["450M"]

    for size in args.models:
        print(f"\n{'='*60}")
        print(f"TESTING LFM2-VL-{size}")
        print(f"{'='*60}")

        pytorch_path = PYTORCH_MODELS[size]
        tester = VLMultiTurnTester(pytorch_path, max_new_tokens=args.max_tokens)
        tester.load_pytorch()

        for format_key in format_keys:
            format_name = FORMATS[format_key]
            print(f"\n=== Format: {format_key} ({format_name}) ===")

            for variant in args.variants:
                onnx_path = get_onnx_dir(size, variant, format_key)

                # Single image test
                if run_single:
                    print(f"\n--- {variant} / 1 image ({onnx_path}) ---")
                    try:
                        tester.load_images([args.image[0]])
                        prompts = DEFAULT_PROMPTS[: args.turns]
                        result = tester.test_coherence(size, onnx_path, variant, f"{variant}_{format_key}_1img", prompts)
                        results.append(result)
                        if args.verbose:
                            print_turn_results(result)
                        else:
                            print(f"  Avg Token Match: {result.avg_token_match*100:.1f}%")
                            print(f"  Avg Semantic Sim: {result.avg_semantic_sim:.4f}")
                            print(f"  Accumulated Error: {result.accumulated_error:.4f}")
                    except Exception as e:
                        print(f"  ERROR: {e}")

                # Multi-image test
                if run_multi:
                    print(f"\n--- {variant} / {len(args.image)} images ({onnx_path}) ---")
                    try:
                        tester.load_images(args.image)
                        prompts = MULTI_IMAGE_PROMPTS[: args.turns]
                        result = tester.test_coherence(size, onnx_path, variant, f"{variant}_{format_key}_{len(args.image)}img", prompts)
                        results.append(result)
                        if args.verbose:
                            print_turn_results(result)
                        else:
                            print(f"  Avg Token Match: {result.avg_token_match*100:.1f}%")
                            print(f"  Avg Semantic Sim: {result.avg_semantic_sim:.4f}")
                            print(f"  Accumulated Error: {result.accumulated_error:.4f}")
                    except Exception as e:
                        print(f"  ERROR: {e}")

    # Print summary
    if results:
        print(f"\n{'='*110}")
        print("COHERENCE SUMMARY")
        print(f"{'='*110}")
        print(f"\n{'Model':<15} | {'Test':<15} | {'Avg Token%':<12} | {'Avg Semantic':<12} | {'Accum Error':<12}")
        print("-" * 110)
        for r in results:
            print(
                f"LFM2-VL-{r.model_size:<6} | {r.quant_type:<15} | "
                f"{r.avg_token_match*100:<12.1f} | {r.avg_semantic_sim:<12.4f} | "
                f"{r.accumulated_error:<12.4f}"
            )
        print("=" * 110)


if __name__ == "__main__":
    main()
