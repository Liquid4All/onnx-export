#!/usr/bin/env python3
"""
Batch verification for all LFM2-VL ONNX exports.

Verifies numerical correctness across all model sizes and quantization variants.

Available models: 450M, 1.6B, 3B
Available variants: FP32, B4V4, B4V8, B8V8

Usage:
    # Verify all models and variants
    uv run verify_all_vl.py

    # Verify specific models
    uv run verify_all_vl.py --models 450M 1.6B

    # Verify specific variants
    uv run verify_all_vl.py --variants FP32 B4V8

    # Verify single combination
    uv run verify_all_vl.py --models 450M --variants B4V8

    # Custom tolerances for quantized models
    uv run verify_all_vl.py --quant-atol 0.1 --quant-rtol 0.1

    # Use specific image for vision tests
    uv run verify_all_vl.py --image cardinal.jpg
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

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


def get_onnx_dir(size: str, variant: str) -> str:
    """Get ONNX directory name for a model/variant combination."""
    if VARIANTS[variant] is None:
        return f"LFM2-VL-{size}-ONNX"
    else:
        backbone_bits, vision_bits = VARIANTS[variant]
        return f"LFM2-VL-{size}-ONNX-B{backbone_bits}V{vision_bits}"


def verify_model(
    size: str,
    variant: str,
    output_dir: Path,
    model_path: str,
    image: str | None,
    atol: float,
    rtol: float,
) -> tuple[bool, str]:
    """Verify a single model/variant combination."""
    onnx_dir = output_dir / get_onnx_dir(size, variant)

    if not onnx_dir.exists():
        return False, f"Not found: {onnx_dir}"

    cmd = [
        sys.executable, "verify_vl.py",
        "--model", model_path,
        "--onnx", str(onnx_dir),
        "--atol", str(atol),
        "--rtol", str(rtol),
    ]

    if image:
        cmd.extend(["--image", image])

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Extract summary from output
    summary = ""
    for line in result.stdout.split('\n'):
        if 'SUMMARY:' in line:
            summary = line.strip()
            break

    if result.returncode == 0:
        return True, summary
    else:
        # Try to get error info
        error = result.stderr.strip().split('\n')[-1] if result.stderr else "Unknown error"
        return False, error


def main():
    parser = argparse.ArgumentParser(
        description="Batch verification for LFM2-VL ONNX exports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
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

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(message)s"
    )

    results = {}

    logger.info("=" * 70)
    logger.info("LFM2-VL BATCH VERIFICATION")
    logger.info("=" * 70)
    logger.info(f"Models: {args.models}")
    logger.info(f"Variants: {args.variants}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info("")

    for size in args.models:
        model_path = MODELS[size]
        logger.info(f"--- Model: {size} ({model_path}) ---")

        for variant in args.variants:
            # Use different tolerances for FP32 vs quantized
            if variant == "FP32":
                atol, rtol = args.fp32_atol, args.fp32_rtol
            else:
                atol, rtol = args.quant_atol, args.quant_rtol

            key = f"{size}_{variant}"
            onnx_name = get_onnx_dir(size, variant)

            logger.info(f"  Verifying {variant} ({onnx_name})...")
            success, message = verify_model(
                size, variant, args.output_dir, model_path, args.image, atol, rtol
            )
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
