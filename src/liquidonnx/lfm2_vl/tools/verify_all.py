#!/usr/bin/env python3
"""
Batch verification for all LFM2-VL ONNX exports.

Verifies numerical correctness across all model sizes and quantization variants.

Available models: 450M, 1.6B, 3B
Available variants: FP32, B4V4, B4V8, B8V8
Available formats: -T (tiled), -C (conv2d)

Usage:
    # Verify all models with tiled format (default)
    lfm2-vl-verify-all -T

    # Verify all models with conv2d format
    lfm2-vl-verify-all -C

    # Verify both formats
    lfm2-vl-verify-all -T -C

    # Verify specific models
    lfm2-vl-verify-all -T --models 450M 1.6B

    # Verify specific variants
    lfm2-vl-verify-all -T --variants FP32 B4V8

    # Verify single combination
    lfm2-vl-verify-all -T --models 450M --variants B4V8

    # Custom tolerances for quantized models
    lfm2-vl-verify-all -T --quant-atol 0.1 --quant-rtol 0.1

    # Use specific image for vision tests
    lfm2-vl-verify-all -T --image cardinal.jpg
"""

import argparse
import gc
import logging
from pathlib import Path

from liquidonnx.lfm2_vl.verify import VLVerifier

logger = logging.getLogger(__name__)

MODELS = {
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


def get_onnx_dir(size: str, variant: str, format_key: str) -> str:
    """Get ONNX directory name for a model/variant/format combination."""
    suffix = f"-{format_key}"
    if VARIANTS[variant] is None:
        return f"LFM2-VL-{size}-ONNX{suffix}"
    else:
        backbone_bits, vision_bits = VARIANTS[variant]
        return f"LFM2-VL-{size}-ONNX-B{backbone_bits}V{vision_bits}{suffix}"


def verify_model(
    verifier: VLVerifier,
    onnx_dir: Path,
    image_path: str | None,
) -> tuple[bool, str]:
    """Verify a single model/variant/format combination."""
    if not onnx_dir.exists():
        return False, f"Not found: {onnx_dir}"

    # Clear previous results
    verifier.results.clear()

    # Load ONNX models
    verifier.load_onnx_embed_tokens(str(onnx_dir))
    verifier.load_onnx_vision(str(onnx_dir))
    verifier.load_onnx_decoder(str(onnx_dir))

    # Run verifications
    verifier.verify_embed_tokens()
    image = verifier.load_image(image_path)
    verifier.verify_vision_encoder(image)
    verifier.verify_decoder()

    # Print report and get result
    success = verifier.print_report()

    passed = sum(1 for r in verifier.results if r.passed)
    total = len(verifier.results)

    # Build summary with failed checks
    failed = [r for r in verifier.results if not r.passed]
    if failed:
        failed_names = ", ".join(r.name for r in failed)
        summary = f"{passed}/{total} passed, FAILED: {failed_names}"
    else:
        summary = f"{passed}/{total} passed"

    # Clean up ONNX sessions to avoid resource leaks
    verifier.embed_tokens_sess = None
    verifier.embed_images_sess = None
    verifier.decoder_sess = None
    gc.collect()

    return success, summary


def main():
    parser = argparse.ArgumentParser(
        description="Batch verification for LFM2-VL ONNX exports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    # Vision input format
    parser.add_argument(
        "-T", "--tiled",
        action="store_true",
        help="Verify tiled format models [B, N, 768]",
    )
    parser.add_argument(
        "-C", "--conv2d",
        action="store_true",
        help="Verify conv2d format models [B, 3, H, W]",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS.keys()),
        default=list(MODELS.keys()),
        help="Model sizes to verify (default: all)",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=list(VARIANTS.keys()),
        default=list(VARIANTS.keys()),
        help="Variants to verify (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory containing exported models (default: current directory)",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Test image path for vision verification",
    )
    parser.add_argument(
        "--fp32-atol",
        type=float,
        default=1e-3,
        help="Absolute tolerance for FP32 models (default: 1e-3)",
    )
    parser.add_argument(
        "--fp32-rtol",
        type=float,
        default=1e-2,
        help="Relative tolerance for FP32 models (default: 1e-2)",
    )
    parser.add_argument(
        "--quant-atol",
        type=float,
        default=10.0,
        help="Absolute tolerance for quantized models (default: 10.0)",
    )
    parser.add_argument(
        "--quant-rtol",
        type=float,
        default=1.0,
        help="Relative tolerance for quantized models (default: 1.0)",
    )
    args = parser.parse_args()

    # Determine which formats to verify
    format_keys = []
    if args.tiled:
        format_keys.append("T")
    if args.conv2d:
        format_keys.append("C")
    if not format_keys:
        format_keys = ["T"]  # Default to tiled

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(message)s"
    )

    results = {}

    logger.info("=" * 70)
    logger.info("LFM2-VL BATCH VERIFICATION")
    logger.info("=" * 70)
    logger.info(f"Formats: {format_keys}")
    logger.info(f"Models: {args.models}")
    logger.info(f"Variants: {args.variants}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info("")

    # Cache verifiers per model to avoid reloading PyTorch model
    verifiers: dict[str, VLVerifier] = {}

    for format_key in format_keys:
        format_name = FORMATS[format_key]
        logger.info(f"=== Format: {format_key} ({format_name}) ===")

        for size in args.models:
            model_path = MODELS[size]
            logger.info(f"\n--- Model: {size} ({model_path}) ---")

            for variant in args.variants:
                # Use different tolerances for FP32 vs quantized
                if variant == "FP32":
                    atol, rtol = args.fp32_atol, args.fp32_rtol
                else:
                    atol, rtol = args.quant_atol, args.quant_rtol

                key = f"{size}_{variant}_{format_key}"
                onnx_dir = args.output_dir / get_onnx_dir(size, variant, format_key)

                logger.info(f"  Verifying {variant} ({onnx_dir.name})...")

                # Get or create verifier for this model size
                verifier_key = f"{size}_{atol}_{rtol}"
                if verifier_key not in verifiers:
                    verifiers[verifier_key] = VLVerifier(model_path, atol=atol, rtol=rtol)
                    verifiers[verifier_key].load_pytorch_model()

                verifier = verifiers[verifier_key]
                # Update tolerances if different
                verifier.atol = atol
                verifier.rtol = rtol

                success, message = verify_model(verifier, onnx_dir, args.image)
                results[key] = (success, message)

                status = "OK" if success else "FAIL"
                logger.info(f"    {status}: {message}")

        logger.info("")

    # Summary
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)

    passed = sum(1 for s, _ in results.values() if s)
    total = len(results)

    for key, (success, message) in results.items():
        status = "PASS" if success else "FAIL"
        logger.info(f"  {status} {key}: {message}")

    logger.info("")
    logger.info(f"Total: {passed}/{total} passed")

    return 0 if passed == total else 1


if __name__ == "__main__":
    exit(main())
