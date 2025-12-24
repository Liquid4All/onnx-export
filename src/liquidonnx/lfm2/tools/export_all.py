#!/usr/bin/env python3
"""
Export all LFM2 models to ONNX with optional quantization.

Usage:
    # Export all models (FP32)
    lfm2-export-all

    # Export specific models
    lfm2-export-all --models 350M 1.2B

    # Export and quantize to Q4
    lfm2-export-all --quantize q4

    # Export and quantize to Q8
    lfm2-export-all --quantize q8

    # Export with custom output directory
    lfm2-export-all --output-dir ./my_models

    # Skip export, only quantize existing models
    lfm2-export-all --quantize q4 --skip-export
"""

import argparse
import logging
from pathlib import Path

from liquidonnx.lfm2.export import export_model
from liquidonnx.lfm2.quantize import quantize_int4, quantize_int8

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
        return f"{base}-Q8-fp32head"
    return base


def do_export(size: str, model_path: str, output_dir: Path) -> bool:
    """Export a single model to ONNX."""
    output_path = output_dir / f"LFM2-{size}-ONNX-builder"

    logger.info(f"Exporting {size} to {output_path}...")

    try:
        export_model(model_path, str(output_path))
        return True
    except Exception as e:
        logger.error(f"Export failed for {size}: {e}")
        return False


def do_quantize(size: str, output_dir: Path, bits: int) -> bool:
    """Quantize a model to INT4 or INT8."""
    input_path = output_dir / f"LFM2-{size}-ONNX-builder"

    if bits == 4:
        output_path = output_dir / f"LFM2-{size}-ONNX-builder-Q4-fp32head"
    else:
        output_path = output_dir / f"LFM2-{size}-ONNX-builder-Q8-fp32head"

    if not input_path.exists():
        logger.error(f"Input model not found: {input_path}")
        return False

    logger.info(f"Quantizing {size} to Q{bits}...")

    try:
        input_model = input_path / "onnx" / "model.onnx"
        if not input_model.exists():
            input_model = input_path / "model.onnx"

        output_onnx_dir = output_path / "onnx"
        output_onnx_dir.mkdir(parents=True, exist_ok=True)
        output_model = output_onnx_dir / "model.onnx"

        if bits == 4:
            quantize_int4(input_model, output_model, quantize_lm_head=False)
        else:
            quantize_int8(input_model, output_model, quantize_lm_head=False)

        # Copy config files
        import shutil
        for cfg in ["config.json", "tokenizer.json", "tokenizer_config.json",
                    "special_tokens_map.json", "genai_config.json", "generation_config.json"]:
            src = input_path / cfg
            if src.exists():
                shutil.copy(src, output_path / cfg)

        return True
    except Exception as e:
        logger.error(f"Quantization failed for {size}: {e}")
        return False


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
            success = do_export(size, model_path, args.output_dir)
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
            success = do_quantize(size, args.output_dir, bits)
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
