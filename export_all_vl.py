#!/usr/bin/env python3
"""
Export all LFM2-VL models to ONNX with quantization.

Available variants:
- FP32: Original precision (largest, reference quality)
- B4V4: Backbone Q4, Vision Q4 (smallest)
- B4V8: Backbone Q4, Vision Q8 (balanced)
- B8V8: Backbone Q8, Vision Q8 (best quantized quality)

Quantized variants keep lm_head in FP32 for output quality.

Usage:
    # Export all VL models (all 4 variants)
    uv run export_all_vl.py

    # Export specific models
    uv run export_all_vl.py --models 450M 1.6B

    # Export specific variants only
    uv run export_all_vl.py --variants FP32 B4V8

    # Export only quantized variants (no FP32)
    uv run export_all_vl.py --variants B4V4 B4V8 B8V8

    # Skip FP32 export, only quantize existing models
    uv run export_all_vl.py --skip-export

    # Custom output directory
    uv run export_all_vl.py --output-dir ./my_models
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


def get_fp32_name(size: str) -> str:
    """Get FP32 output directory name."""
    return f"LFM2-VL-{size}-ONNX"


def get_quant_name(size: str, backbone_bits: int, vision_bits: int) -> str:
    """Get quantized output directory name."""
    return f"LFM2-VL-{size}-ONNX-B{backbone_bits}V{vision_bits}"


def export_model(size: str, model_path: str, output_dir: Path) -> bool:
    """Export a single VL model to ONNX (FP32)."""
    output_path = output_dir / get_fp32_name(size)

    logger.info(f"Exporting {size} to {output_path}...")

    cmd = [
        sys.executable, "lfm2_vl.py",
        "--model", model_path,
        "--output", str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"Export failed for {size}:")
        logger.error(result.stderr)
        return False

    # Print size info
    for line in result.stdout.split('\n'):
        if 'Total ONNX size:' in line or 'embed_' in line or 'decoder' in line:
            logger.info(f"  {line.strip()}")

    return True


def quantize_model(size: str, output_dir: Path, backbone_bits: int, vision_bits: int) -> bool:
    """Quantize a VL model with specified backbone and vision bits."""
    input_path = output_dir / get_fp32_name(size)
    output_path = output_dir / get_quant_name(size, backbone_bits, vision_bits)

    if not input_path.exists():
        logger.error(f"Input model not found: {input_path}")
        return False

    logger.info(f"Quantizing {size} -> B{backbone_bits}V{vision_bits} (backbone=Q{backbone_bits}, vision=Q{vision_bits}, lm_head=FP32)...")

    cmd = [
        sys.executable, "quantize_vl.py",
        "--input", str(input_path),
        "--output", str(output_path),
        "--bits", str(backbone_bits),
        "--vision-bits", str(vision_bits),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"Quantization failed for {size}:")
        logger.error(result.stderr)
        return False

    # Print compression info
    for line in result.stdout.split('\n'):
        if 'Total:' in line or '->' in line:
            logger.info(f"  {line.strip()}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Export all LFM2-VL models to ONNX with quantization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS.keys()),
        default=list(MODELS.keys()),
        help="Model sizes to export (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Output directory (default: current directory)",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=list(VARIANTS.keys()),
        default=list(VARIANTS.keys()),
        help="Quantization variants to export (default: all)",
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Skip FP32 export, only run quantization on existing models",
    )
    parser.add_argument(
        "--keep-fp32",
        action="store_true",
        help="Keep the intermediate FP32 models after quantization",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(message)s"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Track results
    results = {"export": {}, "quantize": {}}

    # Export FP32 models
    if not args.skip_export:
        logger.info("=" * 60)
        logger.info("EXPORTING VL MODELS (FP32)")
        logger.info("=" * 60)

        for size in args.models:
            model_path = MODELS[size]
            success = export_model(size, model_path, args.output_dir)
            results["export"][size] = success
            if success:
                logger.info(f"  {size}: OK")
            else:
                logger.error(f"  {size}: FAILED")

    # Quantize models (skip FP32 variant - it's already exported)
    quant_variants = [v for v in args.variants if VARIANTS[v] is not None]

    if quant_variants:
        logger.info("")
        logger.info("=" * 60)
        logger.info("QUANTIZING MODELS")
        logger.info("=" * 60)

        for variant_name in quant_variants:
            backbone_bits, vision_bits = VARIANTS[variant_name]
            logger.info(f"\n--- {variant_name} (backbone=Q{backbone_bits}, vision=Q{vision_bits}, lm_head=FP32) ---")
            for size in args.models:
                key = f"{size}_{variant_name}"
                success = quantize_model(size, args.output_dir, backbone_bits, vision_bits)
                results["quantize"][key] = success
                if success:
                    logger.info(f"  {size}: OK -> {get_quant_name(size, backbone_bits, vision_bits)}")
                else:
                    logger.error(f"  {size}: FAILED")

    # Cleanup FP32 models if not requested as output variant
    keep_fp32 = args.keep_fp32 or "FP32" in args.variants
    if not keep_fp32 and not args.skip_export:
        logger.info("")
        logger.info("Cleaning up intermediate FP32 models...")
        import shutil
        for size in args.models:
            fp32_path = args.output_dir / get_fp32_name(size)
            if fp32_path.exists():
                shutil.rmtree(fp32_path)
                logger.info(f"  Removed {fp32_path}")

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    if not args.skip_export:
        export_ok = sum(1 for v in results["export"].values() if v)
        export_total = len(results["export"])
        logger.info(f"Export: {export_ok}/{export_total} succeeded")

    quant_ok = sum(1 for v in results["quantize"].values() if v)
    quant_total = len(results["quantize"])
    logger.info(f"Quantize: {quant_ok}/{quant_total} succeeded")

    # List output directories
    logger.info("")
    logger.info("Output directories:")
    for size in args.models:
        for variant_name in args.variants:
            variant_bits = VARIANTS[variant_name]
            if variant_bits is None:
                # FP32 variant
                out_dir = args.output_dir / get_fp32_name(size)
            else:
                backbone_bits, vision_bits = variant_bits
                out_dir = args.output_dir / get_quant_name(size, backbone_bits, vision_bits)
            if out_dir.exists():
                # Get total size
                total_size = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file())
                logger.info(f"  {out_dir} ({total_size/1e9:.2f} GB)")

    return 0


if __name__ == "__main__":
    exit(main())
