#!/usr/bin/env python3
"""
Export all LFM2 models to ONNX with optional quantization.

Output Structure (Transformers.js compatible):
    exports/
    └── LFM2-{size}-ONNX/
        ├── config.json
        ├── tokenizer.json
        ├── ...
        └── onnx/
            ├── decoder_fp32.onnx      # FP32 (reference)
            ├── decoder_fp32.onnx_data
            ├── decoder_q4.onnx        # INT4 quantized
            ├── decoder_q4.onnx_data
            ├── decoder_q8.onnx        # INT8 quantized
            └── decoder_q8.onnx_data

Usage:
    # Export all models (FP32 only)
    lfm2-export-all

    # Export specific models
    lfm2-export-all --models 350M 1.2B

    # Export with Q4 quantization
    lfm2-export-all --quantize q4

    # Export with Q8 quantization
    lfm2-export-all --quantize q8

    # Export with both Q4 and Q8
    lfm2-export-all --quantize q4 q8

    # Skip export, only quantize existing models
    lfm2-export-all --quantize q4 --skip-export

    # Custom output directory
    lfm2-export-all --output-dir ./my_models
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


def get_output_dir(size: str, output_base: Path) -> Path:
    """Get output directory for a model (single directory per model size)."""
    return output_base / "exports" / f"LFM2-{size}-ONNX"


def do_export(size: str, model_path: str, output_base: Path) -> bool:
    """Export a single model to ONNX (FP32)."""
    output_path = get_output_dir(size, output_base)

    logger.info(f"Exporting {size} to {output_path}...")

    try:
        export_model(model_path, str(output_path))
        return True
    except Exception as e:
        logger.error(f"Export failed for {size}: {e}")
        return False


def do_quantize(size: str, output_base: Path, bits: int) -> bool:
    """Quantize a model to INT4 or INT8 (in-place, same directory)."""
    model_dir = get_output_dir(size, output_base)
    onnx_dir = model_dir / "onnx"

    # Find FP32 model
    input_model = onnx_dir / "decoder_fp32.onnx"
    if not input_model.exists():
        # Fallback to old naming
        input_model = onnx_dir / "model.onnx"
    if not input_model.exists():
        logger.error(f"Input model not found: {onnx_dir}")
        return False

    output_model = onnx_dir / f"decoder_q{bits}.onnx"

    logger.info(f"Quantizing {size} to Q{bits}...")

    try:
        if bits == 4:
            quantize_int4(input_model, output_model, quantize_lm_head=False)
        else:
            quantize_int8(input_model, output_model, quantize_lm_head=False)
        return True
    except Exception as e:
        logger.error(f"Quantization failed for {size}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Export all LFM2 models to ONNX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
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
        help="Output base directory (default: current directory)",
    )
    parser.add_argument(
        "--quantize",
        nargs="+",
        choices=["q4", "q8"],
        help="Quantize models after export (can specify multiple: q4 q8)",
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Skip export, only run quantization on existing models",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Track results
    results = {"export": {}, "quantize": {}}

    # Export models
    if not args.skip_export:
        logger.info("=" * 60)
        logger.info("EXPORTING MODELS (FP32)")
        logger.info("=" * 60)

        for size in args.models:
            model_path = MODELS[size]
            success = do_export(size, model_path, args.output_dir)
            results["export"][size] = success
            if success:
                logger.info(f"  {size}: OK")
            else:
                logger.error(f"  {size}: FAILED")

    # Quantize models (all variants in same directory)
    if args.quantize:
        for quant in args.quantize:
            bits = 4 if quant == "q4" else 8

            logger.info("")
            logger.info("=" * 60)
            logger.info(f"QUANTIZING TO Q{bits}")
            logger.info("=" * 60)

            for size in args.models:
                key = f"{size}_q{bits}"
                success = do_quantize(size, args.output_dir, bits)
                results["quantize"][key] = success
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
        out_dir = get_output_dir(size, args.output_dir)
        if out_dir.exists():
            onnx_dir = out_dir / "onnx"
            files = list(onnx_dir.glob("decoder_*.onnx"))
            file_names = ", ".join(f.name for f in sorted(files))
            total_size = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file())
            logger.info(f"  {out_dir} ({total_size / 1e9:.2f} GB)")
            logger.info(f"    Files: {file_names}")


if __name__ == "__main__":
    main()
