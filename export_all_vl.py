#!/usr/bin/env python3
"""
Export all LFM2-VL models to ONNX with optional quantization.

Usage:
    # Export all VL models (FP32)
    uv run export_all_vl.py

    # Export specific models
    uv run export_all_vl.py --models 450M 1.6B

    # Export and quantize (Q8 vision, Q4 decoder, lm_head FP32)
    uv run export_all_vl.py --quantize

    # Export and quantize with Q8 decoder
    uv run export_all_vl.py --quantize --decoder-bits 8

    # Export with Q4 vision encoder (not recommended for quality)
    uv run export_all_vl.py --quantize --vision-bits 4

    # Skip export, only quantize existing models
    uv run export_all_vl.py --quantize --skip-export

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


def get_output_name(size: str, quantize: bool, decoder_bits: int) -> str:
    """Get output directory name for a model."""
    base = f"LFM2-VL-{size}-ONNX-builder"
    if quantize:
        return f"{base}-Q{decoder_bits}-fp32head"
    return base


def export_model(size: str, model_path: str, output_dir: Path) -> bool:
    """Export a single VL model to ONNX."""
    output_path = output_dir / f"LFM2-VL-{size}-ONNX-builder"

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
        if 'Model size:' in line or 'embed_images:' in line or 'decoder:' in line:
            logger.info(line.strip())

    return True


def quantize_model(size: str, output_dir: Path, vision_bits: int,
                   decoder_bits: int) -> bool:
    """Quantize a VL model."""
    input_path = output_dir / f"LFM2-VL-{size}-ONNX-builder"
    output_path = output_dir / f"LFM2-VL-{size}-ONNX-builder-Q{decoder_bits}-fp32head"

    if not input_path.exists():
        logger.error(f"Input model not found: {input_path}")
        return False

    logger.info(f"Quantizing {size} (vision=Q{vision_bits}, decoder=Q{decoder_bits})...")

    cmd = [
        sys.executable, "quantize_vl.py",
        "--input", str(input_path),
        "--output", str(output_path),
        "--bits", str(decoder_bits),
        "--vision-bits", str(vision_bits),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"Quantization failed for {size}:")
        logger.error(result.stderr)
        return False

    # Print compression info
    for line in result.stdout.split('\n'):
        if 'Summary' in line or '->' in line or 'Compression' in line:
            logger.info(line.strip())

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Export all LFM2-VL models to ONNX",
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
        "--quantize",
        action="store_true",
        help="Quantize models after export",
    )
    parser.add_argument(
        "--vision-bits",
        type=int,
        choices=[4, 8],
        default=8,
        help="Quantization bits for vision encoder (default: 8, recommended for quality)",
    )
    parser.add_argument(
        "--decoder-bits",
        type=int,
        choices=[4, 8],
        default=4,
        help="Quantization bits for decoder/backbone (default: 4)",
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Skip export, only run quantization on existing models",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(message)s"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Track results
    results = {"export": {}, "quantize": {}}

    # Export models
    if not args.skip_export:
        logger.info("=" * 60)
        logger.info("EXPORTING VL MODELS")
        logger.info("=" * 60)

        for size in args.models:
            model_path = MODELS[size]
            success = export_model(size, model_path, args.output_dir)
            results["export"][size] = success
            if success:
                logger.info(f"  {size}: OK")
            else:
                logger.error(f"  {size}: FAILED")

    # Quantize models
    if args.quantize:
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"QUANTIZING (vision=Q{args.vision_bits}, decoder=Q{args.decoder_bits}, lm_head=FP32)")
        logger.info("=" * 60)

        for size in args.models:
            success = quantize_model(size, args.output_dir, args.vision_bits,
                                     args.decoder_bits)
            results["quantize"][size] = success
            if success:
                logger.info(f"  {size}: OK")
            else:
                logger.error(f"  {size}: FAILED")

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    if not args.skip_export:
        export_ok = sum(1 for v in results["export"].values() if v)
        export_total = len(results["export"])
        logger.info(f"Export: {export_ok}/{export_total} succeeded")

    if args.quantize:
        quant_ok = sum(1 for v in results["quantize"].values() if v)
        quant_total = len(results["quantize"])
        logger.info(f"Quantize: {quant_ok}/{quant_total} succeeded")

    # List output directories
    logger.info("")
    logger.info("Output directories:")
    for size in args.models:
        if not args.skip_export:
            fp32_dir = args.output_dir / f"LFM2-VL-{size}-ONNX-builder"
            if fp32_dir.exists():
                logger.info(f"  {fp32_dir}")

        if args.quantize:
            quant_name = get_output_name(size, True, args.decoder_bits)
            quant_dir = args.output_dir / quant_name
            if quant_dir.exists():
                logger.info(f"  {quant_dir}")


if __name__ == "__main__":
    main()
