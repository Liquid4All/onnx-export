#!/usr/bin/env python3
"""
Export all LFM2 models to ONNX with optional quantization.

Usage:
    # Export all models (FP32)
    uv run export_all.py

    # Export specific models
    uv run export_all.py --models 350M 1.2B

    # Export and quantize to Q4
    uv run export_all.py --quantize q4

    # Export and quantize to Q8
    uv run export_all.py --quantize q8

    # Export with custom output directory
    uv run export_all.py --output-dir ./my_models

    # Skip export, only quantize existing models
    uv run export_all.py --quantize q4 --skip-export
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

MODELS = {
    "350M": "LiquidAI/LFM2-350M",
    "700M": "LiquidAI/LFM2-700M",
    "1.2B": "LiquidAI/LFM2-1.2B",
    "2.6B": "LiquidAI/LFM2-2.6B",
}


def get_output_name(size: str, quantize: str | None) -> str:
    """Get output directory name for a model."""
    base = f"LFM2-{size}-ONNX-builder"
    if quantize == "q4":
        return f"{base}-Q4-fp32head"
    elif quantize == "q8":
        return f"{base}-Q8"
    return base


def export_model(size: str, model_path: str, output_dir: Path) -> bool:
    """Export a single model to ONNX."""
    output_path = output_dir / f"LFM2-{size}-ONNX-builder"

    logger.info(f"Exporting {size} to {output_path}...")

    cmd = [
        sys.executable, "lfm2.py",
        "--model", model_path,
        "--output", str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"Export failed for {size}:")
        logger.error(result.stderr)
        return False

    # Print the last line (model size)
    for line in result.stdout.split('\n'):
        if 'Model size:' in line:
            logger.info(line.strip())

    return True


def quantize_model(size: str, output_dir: Path, bits: int) -> bool:
    """Quantize a model to INT4 or INT8."""
    input_path = output_dir / f"LFM2-{size}-ONNX-builder"

    if bits == 4:
        output_path = output_dir / f"LFM2-{size}-ONNX-builder-Q4-fp32head"
    else:
        output_path = output_dir / f"LFM2-{size}-ONNX-builder-Q8"

    if not input_path.exists():
        logger.error(f"Input model not found: {input_path}")
        return False

    logger.info(f"Quantizing {size} to Q{bits}...")

    cmd = [
        sys.executable, "quantize.py",
        "--input", str(input_path),
        "--output", str(output_path),
        "--bits", str(bits),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"Quantization failed for {size}:")
        logger.error(result.stderr)
        return False

    # Print compression info
    for line in result.stdout.split('\n'):
        if 'Quantized:' in line or 'Compression:' in line:
            logger.info(line.strip())

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Export all LFM2 models to ONNX",
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
        choices=["q4", "q8"],
        help="Quantize models after export (q4=INT4, q8=INT8)",
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
        logger.info("EXPORTING MODELS")
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
        bits = 4 if args.quantize == "q4" else 8

        logger.info("")
        logger.info("=" * 60)
        logger.info(f"QUANTIZING TO Q{bits}")
        logger.info("=" * 60)

        for size in args.models:
            success = quantize_model(size, args.output_dir, bits)
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
            fp32_dir = args.output_dir / f"LFM2-{size}-ONNX-builder"
            if fp32_dir.exists():
                logger.info(f"  {fp32_dir}")

        if args.quantize:
            quant_name = get_output_name(size, args.quantize)
            quant_dir = args.output_dir / quant_name
            if quant_dir.exists():
                logger.info(f"  {quant_dir}")


if __name__ == "__main__":
    main()
