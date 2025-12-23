#!/usr/bin/env python3
"""
Export all LFM2-VL models to ONNX with quantization.

Vision Input Formats:
- -T (tiled): Input [B, N, 768] with pre-extracted patches (HuggingFace style)
- -C (conv2d): Input [B, 3, H, W] with raw image (simpler, llama.cpp style)

Available quantization variants:
- FP32: Original precision (largest, reference quality)
- B4V4: Backbone Q4, Vision Q4 (smallest)
- B4V8: Backbone Q4, Vision Q8 (balanced)
- B8V8: Backbone Q8, Vision Q8 (best quantized quality)

Quantized variants keep lm_head in FP32 for output quality.

Usage:
    # Export all VL models with tiled format (default)
    uv run export_all_vl.py -T

    # Export all VL models with conv2d format
    uv run export_all_vl.py -C

    # Export both formats
    uv run export_all_vl.py -T -C

    # Export specific models
    uv run export_all_vl.py -T --models 450M 1.6B

    # Export specific variants only
    uv run export_all_vl.py -T --variants FP32 B4V8

    # Export only quantized variants (no FP32)
    uv run export_all_vl.py -T --variants B4V4 B4V8 B8V8

    # Skip FP32 export, only quantize existing models
    uv run export_all_vl.py -T --skip-export

    # Custom output directory
    uv run export_all_vl.py -T --output-dir ./my_models
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

# Vision input formats
FORMATS = {
    "T": "tiled",   # [B, N, 768] pre-extracted patches
    "C": "conv2d",  # [B, 3, H, W] raw image
}


def get_format_suffix(format_key: str) -> str:
    """Get format suffix for directory name."""
    return f"-{format_key}"


def get_fp32_name(size: str, format_key: str) -> str:
    """Get FP32 output directory name."""
    return f"LFM2-VL-{size}-ONNX{get_format_suffix(format_key)}"


def get_quant_name(size: str, backbone_bits: int, vision_bits: int, format_key: str) -> str:
    """Get quantized output directory name."""
    return f"LFM2-VL-{size}-ONNX-B{backbone_bits}V{vision_bits}{get_format_suffix(format_key)}"


def export_model(size: str, model_path: str, output_dir: Path, format_key: str) -> bool:
    """Export a single VL model to ONNX (FP32).

    Args:
        size: Model size (e.g., "450M", "1.6B", "3B")
        model_path: HuggingFace model path
        output_dir: Output directory
        format_key: "T" for tiled or "C" for conv2d
    """
    output_path = output_dir / get_fp32_name(size, format_key)

    logger.info(f"Exporting {size} ({format_key}) to {output_path}...")

    cmd = [
        sys.executable, "lfm2_vl.py",
        "--model", model_path,
        "--output", str(output_path),
        f"-{format_key}",  # -T or -C
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"Export failed for {size} ({format_key}):")
        logger.error(result.stderr)
        return False

    # Print size info
    for line in result.stdout.split('\n'):
        if 'Total ONNX size:' in line or 'embed_' in line or 'decoder' in line:
            logger.info(f"  {line.strip()}")

    return True


def quantize_model(size: str, output_dir: Path, backbone_bits: int, vision_bits: int, format_key: str) -> bool:
    """Quantize a VL model with specified backbone and vision bits.

    Args:
        size: Model size (e.g., "450M", "1.6B", "3B")
        output_dir: Output directory
        backbone_bits: Quantization bits for backbone (4 or 8)
        vision_bits: Quantization bits for vision encoder (4 or 8)
        format_key: "T" for tiled or "C" for conv2d
    """
    input_path = output_dir / get_fp32_name(size, format_key)
    output_path = output_dir / get_quant_name(size, backbone_bits, vision_bits, format_key)

    if not input_path.exists():
        logger.error(f"Input model not found: {input_path}")
        return False

    logger.info(f"Quantizing {size} ({format_key}) -> B{backbone_bits}V{vision_bits} (backbone=Q{backbone_bits}, vision=Q{vision_bits}, lm_head=FP32)...")

    cmd = [
        sys.executable, "quantize_vl.py",
        "--input", str(input_path),
        "--output", str(output_path),
        "--bits", str(backbone_bits),
        "--vision-bits", str(vision_bits),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"Quantization failed for {size} ({format_key}):")
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
    # Vision input format (at least one required)
    parser.add_argument(
        "-T", "--tiled",
        action="store_true",
        help="Export with tiled input format [B, N, 768] (HuggingFace style)",
    )
    parser.add_argument(
        "-C", "--conv2d",
        action="store_true",
        help="Export with conv2d input format [B, 3, H, W] (llama.cpp style)",
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
        "--cleanup-fp32",
        action="store_true",
        help="Remove FP32 models after quantization (default: keep them)",
    )
    args = parser.parse_args()

    # Determine which formats to export
    format_keys = []
    if args.tiled:
        format_keys.append("T")
    if args.conv2d:
        format_keys.append("C")

    if not format_keys:
        # Default to tiled if nothing specified
        logger.warning("No format specified, defaulting to -T (tiled)")
        format_keys = ["T"]

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(message)s"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Track results
    results = {"export": {}, "quantize": {}}

    # Export FP32 models for each format
    if not args.skip_export:
        logger.info("=" * 60)
        logger.info("EXPORTING VL MODELS (FP32)")
        logger.info("=" * 60)

        for format_key in format_keys:
            format_name = FORMATS[format_key]
            logger.info(f"\n--- Format: {format_key} ({format_name}) ---")
            for size in args.models:
                model_path = MODELS[size]
                key = f"{size}_{format_key}"
                success = export_model(size, model_path, args.output_dir, format_key)
                results["export"][key] = success
                if success:
                    logger.info(f"  {size}: OK -> {get_fp32_name(size, format_key)}")
                else:
                    logger.error(f"  {size}: FAILED")

    # Quantize models (skip FP32 variant - it's already exported)
    quant_variants = [v for v in args.variants if VARIANTS[v] is not None]

    if quant_variants:
        logger.info("")
        logger.info("=" * 60)
        logger.info("QUANTIZING MODELS")
        logger.info("=" * 60)

        for format_key in format_keys:
            format_name = FORMATS[format_key]
            logger.info(f"\n=== Format: {format_key} ({format_name}) ===")

            for variant_name in quant_variants:
                backbone_bits, vision_bits = VARIANTS[variant_name]
                logger.info(f"\n--- {variant_name} (backbone=Q{backbone_bits}, vision=Q{vision_bits}, lm_head=FP32) ---")
                for size in args.models:
                    key = f"{size}_{variant_name}_{format_key}"
                    success = quantize_model(size, args.output_dir, backbone_bits, vision_bits, format_key)
                    results["quantize"][key] = success
                    if success:
                        logger.info(f"  {size}: OK -> {get_quant_name(size, backbone_bits, vision_bits, format_key)}")
                    else:
                        logger.error(f"  {size}: FAILED")

    # Cleanup FP32 models only if explicitly requested
    if args.cleanup_fp32 and not args.skip_export:
        logger.info("")
        logger.info("Cleaning up intermediate FP32 models...")
        import shutil
        for format_key in format_keys:
            for size in args.models:
                fp32_path = args.output_dir / get_fp32_name(size, format_key)
                if fp32_path.exists():
                    shutil.rmtree(fp32_path)
                    logger.info(f"  Removed {fp32_path}")

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    logger.info(f"Formats exported: {', '.join(format_keys)}")

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
    for format_key in format_keys:
        for size in args.models:
            for variant_name in args.variants:
                variant_bits = VARIANTS[variant_name]
                if variant_bits is None:
                    # FP32 variant
                    out_dir = args.output_dir / get_fp32_name(size, format_key)
                else:
                    backbone_bits, vision_bits = variant_bits
                    out_dir = args.output_dir / get_quant_name(size, backbone_bits, vision_bits, format_key)
                if out_dir.exists():
                    # Get total size
                    total_size = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file())
                    logger.info(f"  {out_dir} ({total_size/1e9:.2f} GB)")

    return 0


if __name__ == "__main__":
    exit(main())
