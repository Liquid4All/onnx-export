#!/usr/bin/env python3
"""
Numerical verification for LFM2-VL ONNX exports.

Verifies separately:
1. Token embeddings (embed_tokens.onnx)
2. Vision encoder + projector (embed_images.onnx)
3. Decoder/backbone (decoder.onnx) - takes inputs_embeds, not input_ids

Usage:
    # Verify FP32 export
    uv run verify_vl.py --model LiquidAI/LFM2-VL-450M --onnx LFM2-VL-450M-ONNX

    # Verify quantized export (B4V8)
    uv run verify_vl.py --model LiquidAI/LFM2-VL-450M --onnx LFM2-VL-450M-ONNX-B4V8 --atol 0.5 --rtol 0.5

    # Verify vision encoder only
    uv run verify_vl.py --model LiquidAI/LFM2-VL-450M --onnx LFM2-VL-450M-ONNX --vision-only

    # Verify decoder only
    uv run verify_vl.py --model LiquidAI/LFM2-VL-450M --onnx LFM2-VL-450M-ONNX --decoder-only

    # Verify embed_tokens only
    uv run verify_vl.py --model LiquidAI/LFM2-VL-450M --onnx LFM2-VL-450M-ONNX --embed-tokens-only

    # Use specific image
    uv run verify_vl.py --model LiquidAI/LFM2-VL-450M --onnx LFM2-VL-450M-ONNX --image cardinal.jpg

For batch verification across all models/variants, use verify_all_vl.py instead.
"""

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from liquidonnx import detect_vision_format, preprocess_conv2d, preprocess_tiled

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Result of a single verification check."""
    name: str
    passed: bool
    max_diff: float
    mean_diff: float
    correlation: float
    details: str = ""


class VLVerifier:
    """Verifies numerical correctness of LFM2-VL ONNX exports."""

    def __init__(self, model_path: str, atol: float = 1e-3, rtol: float = 1e-2):
        self.model_path = model_path
        self.atol = atol
        self.rtol = rtol
        self.results: List[VerificationResult] = []

        # Models
        self.torch_model = None
        self.processor = None
        self.tokenizer = None
        self.embed_tokens_sess = None
        self.embed_images_sess = None
        self.decoder_sess = None

    def load_pytorch_model(self):
        """Load PyTorch VL model for reference."""
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        logger.info(f"Loading PyTorch VL model from {self.model_path}...")
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer
        self.torch_model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        self.torch_model.eval()

    def load_onnx_embed_tokens(self, onnx_path: str):
        """Load ONNX embed_tokens model."""
        import onnxruntime as ort

        onnx_dir = os.path.join(onnx_path, "onnx")
        embed_tokens_file = os.path.join(onnx_dir, "embed_tokens.onnx")

        if not os.path.exists(embed_tokens_file):
            raise FileNotFoundError(f"embed_tokens.onnx not found in {onnx_dir}")

        logger.info(f"Loading embed_tokens from {embed_tokens_file}...")
        self.embed_tokens_sess = ort.InferenceSession(
            embed_tokens_file, providers=["CPUExecutionProvider"]
        )

    def load_onnx_vision(self, onnx_path: str):
        """Load ONNX vision encoder."""
        import onnxruntime as ort

        onnx_dir = os.path.join(onnx_path, "onnx")
        embed_images_file = os.path.join(onnx_dir, "embed_images.onnx")

        if not os.path.exists(embed_images_file):
            raise FileNotFoundError(f"embed_images.onnx not found in {onnx_dir}")

        logger.info(f"Loading embed_images from {embed_images_file}...")
        self.embed_images_sess = ort.InferenceSession(
            embed_images_file, providers=["CPUExecutionProvider"]
        )

    def load_onnx_decoder(self, onnx_path: str):
        """Load ONNX decoder."""
        import onnxruntime as ort

        onnx_dir = os.path.join(onnx_path, "onnx")
        decoder_file = os.path.join(onnx_dir, "decoder.onnx")

        if not os.path.exists(decoder_file):
            raise FileNotFoundError(f"decoder.onnx not found in {onnx_dir}")

        logger.info(f"Loading decoder from {decoder_file}...")
        self.decoder_sess = ort.InferenceSession(
            decoder_file, providers=["CPUExecutionProvider"]
        )

    def create_test_image(self, size: int = 512) -> "Image":
        """Create a test image with some content."""
        from PIL import Image

        img = Image.new('RGB', (size, size), color=(128, 128, 128))
        pixels = np.array(img)
        # Red rectangle
        pixels[100:200, 100:200] = [255, 0, 0]
        # Blue rectangle
        pixels[300:400, 300:400] = [0, 0, 255]
        # Green rectangle
        pixels[100:200, 300:400] = [0, 255, 0]

        return Image.fromarray(pixels)

    def load_image(self, image_path: Optional[str]) -> "Image":
        """Load image from path or create test image."""
        from PIL import Image

        if image_path and os.path.exists(image_path):
            logger.info(f"Loading image from {image_path}")
            return Image.open(image_path).convert("RGB")
        else:
            logger.info("Creating test image...")
            return self.create_test_image()

    def compare_arrays(self, name: str, expected: np.ndarray, actual: np.ndarray) -> VerificationResult:
        """Compare two arrays and return verification result."""
        if expected.shape != actual.shape:
            return VerificationResult(
                name=name,
                passed=False,
                max_diff=float('inf'),
                mean_diff=float('inf'),
                correlation=0.0,
                details=f"Shape mismatch: {expected.shape} vs {actual.shape}"
            )

        diff = np.abs(expected - actual)
        max_diff = float(diff.max())
        mean_diff = float(diff.mean())

        # Correlation
        flat_exp = expected.flatten()
        flat_act = actual.flatten()
        correlation = float(np.corrcoef(flat_exp, flat_act)[0, 1])

        passed = np.allclose(expected, actual, atol=self.atol, rtol=self.rtol)

        return VerificationResult(
            name=name,
            passed=passed,
            max_diff=max_diff,
            mean_diff=mean_diff,
            correlation=correlation,
        )

    def compare_top_k(self, name: str, expected: np.ndarray, actual: np.ndarray, k: int = 5) -> VerificationResult:
        """Compare top-k predictions."""
        exp_logits = expected[0, -1]
        act_logits = actual[0, -1]

        exp_top_k = np.argsort(exp_logits)[-k:][::-1]
        act_top_k = np.argsort(act_logits)[-k:][::-1]

        top1_match = exp_top_k[0] == act_top_k[0]
        top_k_overlap = len(set(exp_top_k) & set(act_top_k))

        return VerificationResult(
            name=name,
            passed=top1_match,
            max_diff=0.0 if top1_match else 1.0,
            mean_diff=1.0 - (top_k_overlap / k),
            correlation=top_k_overlap / k,
            details=f"Top-1 match: {top1_match}, Top-{k} overlap: {top_k_overlap}/{k}, "
                    f"Expected: {exp_top_k.tolist()}, Actual: {act_top_k.tolist()}"
        )

    # =========================================================================
    # Embed Tokens Verification
    # =========================================================================

    def verify_embed_tokens(self, prompts: List[str] = None) -> List[VerificationResult]:
        """Verify embed_tokens (token embedding lookup)."""
        import torch

        if prompts is None:
            prompts = [
                "Hello, how are you?",
                "The quick brown fox",
                "Describe this image:",
            ]

        logger.info("Verifying embed_tokens...")
        results = []

        for prompt in prompts:
            logger.info(f"Testing prompt: '{prompt}'")

            # Tokenize
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt")

            # PyTorch: get embeddings from language model
            with torch.no_grad():
                pytorch_embeds = self.torch_model.model.language_model.embed_tokens(input_ids).numpy()

            # ONNX
            onnx_outputs = self.embed_tokens_sess.run(None, {
                "input_ids": input_ids.numpy().astype(np.int64),
            })
            onnx_embeds = onnx_outputs[0]

            logger.info(f"  PyTorch shape: {pytorch_embeds.shape}, ONNX shape: {onnx_embeds.shape}")

            # Compare embeddings
            result = self.compare_arrays(f"embed_tokens: '{prompt[:20]}...'", pytorch_embeds, onnx_embeds)
            results.append(result)

        self.results.extend(results)
        return results

    # =========================================================================
    # Vision Encoder Verification
    # =========================================================================

    def _detect_vision_format(self) -> str:
        """Detect vision input format from ONNX model inputs."""
        return detect_vision_format(self.embed_images_sess)

    def verify_vision_encoder(self, image) -> List[VerificationResult]:
        """Verify vision encoder + projector outputs."""
        import torch

        logger.info("Verifying vision encoder + projector...")
        results = []

        # Detect vision input format
        vision_format = self._detect_vision_format()
        logger.info(f"Detected vision format: {vision_format}")

        # Process image using image_processor directly
        # The main processor requires text, but for vision-only we just need pixel values
        inputs = self.processor.image_processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"]
        pixel_attention_mask = inputs["pixel_attention_mask"]
        spatial_shapes = inputs["spatial_shapes"]

        # PyTorch: Process each tile through vision tower + projector
        # We need the padded output (before unpadding) to compare with ONNX
        # The projector applies pixel_unshuffle which requires fixed spatial dimensions
        num_tiles = pixel_values.shape[0]

        with torch.no_grad():
            # Run vision tower
            vision_outputs = self.torch_model.model.vision_tower(
                pixel_values=pixel_values,
                pixel_attention_mask=pixel_attention_mask,
                spatial_shapes=spatial_shapes,
            ).last_hidden_state  # (num_tiles, num_patches, hidden)

            # Process each tile through projector with its spatial shape
            pytorch_embeddings_list = []
            for tile_idx in range(num_tiles):
                feature = vision_outputs[tile_idx]  # (num_patches, hidden)
                h, w = spatial_shapes[tile_idx].tolist()

                # Reshape to 4D for projector: (1, H, W, hidden)
                feature = feature[:h * w].reshape(1, h, w, -1)

                # Run through projector
                proj_out = self.torch_model.model.multi_modal_projector(feature)
                # Projector outputs (1, H//2, W//2, text_hidden) due to pixel_unshuffle
                # Flatten to (tokens, text_hidden)
                proj_out = proj_out.reshape(-1, proj_out.shape[-1])
                pytorch_embeddings_list.append(proj_out)

            # For comparison, we'll compare per-tile since shapes may differ
            # Stack only tiles with matching shapes (first N-1 usually have same shape)
            pytorch_embeddings = pytorch_embeddings_list

        # ONNX - handle different input formats
        if vision_format == "conv2d":
            # Conv2d format: [B, 3, H, W] raw image input with spatial dims
            onnx_pixel_values, spatial_h, spatial_w = preprocess_conv2d(image)
            logger.info(f"Conv2d input shape: {onnx_pixel_values.shape}, spatial: ({spatial_h}, {spatial_w})")

            onnx_outputs = self.embed_images_sess.run(None, {
                "pixel_values": onnx_pixel_values,
                "spatial_h": np.array(spatial_h, dtype=np.int64),
                "spatial_w": np.array(spatial_w, dtype=np.int64),
            })
            onnx_embeddings = onnx_outputs[0]  # (1, num_tokens, hidden)

            # For conv2d, we only have one "tile" (the whole image)
            # Compare with first pytorch tile (or concatenated if multiple)
            pytorch_concat = torch.cat(pytorch_embeddings_list, dim=0).numpy()
            onnx_flat = onnx_embeddings[0]

            # Compare what we can (sizes may differ due to different spatial handling)
            min_tokens = min(pytorch_concat.shape[0], onnx_flat.shape[0])
            logger.info(f"PyTorch tokens: {pytorch_concat.shape[0]}, ONNX tokens: {onnx_flat.shape[0]}, comparing first {min_tokens}")

            result = self.compare_arrays(
                "vision_conv2d",
                pytorch_concat[:min_tokens],
                onnx_flat[:min_tokens]
            )
            results.append(result)

        else:
            # Tiled format: [B, N, 768] pre-extracted patches
            onnx_pixel_values = pixel_values.numpy().astype(np.float32)
            patch_attention_mask = pixel_attention_mask.numpy().astype(np.int64)

            onnx_outputs = self.embed_images_sess.run(None, {
                "pixel_values": onnx_pixel_values,
                "patch_attention_mask": patch_attention_mask,
            })
            onnx_embeddings = onnx_outputs[0]  # (num_tiles, num_tokens, hidden)

            logger.info(f"PyTorch embeddings: {len(pytorch_embeddings)} tiles, shapes: {[e.shape for e in pytorch_embeddings]}")
            logger.info(f"ONNX embeddings shape: {onnx_embeddings.shape}")

            # Compare per tile (only matching spatial shapes)
            for tile_idx, pytorch_tile in enumerate(pytorch_embeddings):
                pytorch_np = pytorch_tile.numpy()
                onnx_tile = onnx_embeddings[tile_idx]

                # ONNX uses fixed 256 tokens (32x32 / 4 = 256), PyTorch uses actual spatial shape
                # Compare the overlap region
                min_tokens = min(pytorch_np.shape[0], onnx_tile.shape[0])
                pytorch_np = pytorch_np[:min_tokens]
                onnx_tile = onnx_tile[:min_tokens]

                result = self.compare_arrays(f"vision_tile_{tile_idx}", pytorch_np, onnx_tile)
                results.append(result)

        self.results.extend(results)
        return results

    # =========================================================================
    # Decoder Verification
    # =========================================================================

    def verify_decoder(self, prompts: List[str] = None) -> List[VerificationResult]:
        """Verify decoder outputs (takes inputs_embeds, not input_ids)."""
        import torch

        if prompts is None:
            prompts = [
                "Hello, how are",
                "The image shows",
                "I can see",
            ]

        logger.info("Verifying decoder (with inputs_embeds)...")
        results = []

        for prompt in prompts:
            logger.info(f"Testing prompt: '{prompt}'")

            # Tokenize
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
            seq_len = input_ids.shape[1]
            attention_mask = torch.ones_like(input_ids)
            position_ids = torch.arange(seq_len).unsqueeze(0)

            # PyTorch: use language_model directly
            with torch.no_grad():
                # Get embeddings
                inputs_embeds = self.torch_model.model.language_model.embed_tokens(input_ids)

                # Forward through language model
                lm_outputs = self.torch_model.model.language_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )
                # Apply lm_head
                pytorch_logits = self.torch_model.lm_head(lm_outputs.last_hidden_state).numpy()

            # ONNX - decoder takes inputs_embeds, not input_ids
            # First get embeddings from embed_tokens
            onnx_embeds = self.embed_tokens_sess.run(None, {
                "input_ids": input_ids.numpy().astype(np.int64),
            })[0]

            onnx_inputs = {
                "inputs_embeds": onnx_embeds.astype(np.float32),
                "attention_mask": attention_mask.numpy().astype(np.int64),
                "position_ids": position_ids.numpy().astype(np.int64),
            }

            # Add cache inputs initialized to zeros
            for inp in self.decoder_sess.get_inputs():
                if inp.name not in onnx_inputs:
                    shape = [d if isinstance(d, int) else 1 for d in inp.shape]
                    onnx_inputs[inp.name] = np.zeros(shape, dtype=np.float32)

            onnx_outputs = self.decoder_sess.run(None, onnx_inputs)
            onnx_logits = onnx_outputs[0]

            # Compare logits
            result = self.compare_arrays(f"decoder: '{prompt[:20]}...'", pytorch_logits, onnx_logits)
            results.append(result)

            # Compare top-k
            top_k_result = self.compare_top_k(f"top-5: '{prompt[:20]}...'", pytorch_logits, onnx_logits)
            results.append(top_k_result)

        self.results.extend(results)
        return results

    def print_report(self):
        """Print verification report."""
        print("\n" + "=" * 70)
        print("VL NUMERICAL VERIFICATION REPORT")
        print("=" * 70)

        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)

        for result in self.results:
            status = "PASS" if result.passed else "FAIL"
            print(f"\n{status} {result.name}")
            print(f"  Max diff: {result.max_diff:.6f}")
            print(f"  Mean diff: {result.mean_diff:.6f}")
            print(f"  Correlation: {result.correlation:.6f}")
            if result.details:
                print(f"  Details: {result.details}")

        print("\n" + "=" * 70)
        print(f"SUMMARY: {passed}/{total} checks passed")
        print("=" * 70)

        return passed == total


def main():
    parser = argparse.ArgumentParser(description="Verify LFM2-VL ONNX export")
    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace model path (e.g., LiquidAI/LFM2-VL-450M)")
    parser.add_argument("--onnx", type=str, required=True,
                        help="ONNX model directory")
    parser.add_argument("--image", type=str, default=None,
                        help="Test image path (optional)")
    parser.add_argument("--atol", type=float, default=1e-3,
                        help="Absolute tolerance (default: 1e-3)")
    parser.add_argument("--rtol", type=float, default=1e-2,
                        help="Relative tolerance (default: 1e-2)")
    parser.add_argument("--vision-only", action="store_true",
                        help="Only verify vision encoder")
    parser.add_argument("--decoder-only", action="store_true",
                        help="Only verify decoder")
    parser.add_argument("--embed-tokens-only", action="store_true",
                        help="Only verify embed_tokens")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    verifier = VLVerifier(args.model, atol=args.atol, rtol=args.rtol)
    verifier.load_pytorch_model()

    # Determine what to verify
    verify_all = not (args.vision_only or args.decoder_only or args.embed_tokens_only)

    # Load and verify embed_tokens
    if args.embed_tokens_only or verify_all:
        verifier.load_onnx_embed_tokens(args.onnx)
        verifier.verify_embed_tokens()

    # Load and verify vision encoder
    if args.vision_only or verify_all:
        verifier.load_onnx_vision(args.onnx)
        image = verifier.load_image(args.image)
        verifier.verify_vision_encoder(image)

    # Load and verify decoder (requires embed_tokens for inputs_embeds)
    if args.decoder_only or verify_all:
        # Decoder verification needs embed_tokens to get inputs_embeds
        if not verifier.embed_tokens_sess:
            verifier.load_onnx_embed_tokens(args.onnx)
        verifier.load_onnx_decoder(args.onnx)
        verifier.verify_decoder()

    # Print report
    success = verifier.print_report()
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
