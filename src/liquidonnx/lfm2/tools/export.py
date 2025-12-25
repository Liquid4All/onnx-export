#!/usr/bin/env python3
"""
Export LFM2 model to ONNX with optional quantization.

Output Structure (Transformers.js compatible):
    exports/
    └── LFM2-{size}-ONNX/
        ├── config.json
        ├── tokenizer.json
        └── onnx/
            ├── model.onnx           # FP32
            ├── model.onnx_data
            ├── model_q4.onnx        # INT4 quantized
            ├── model_q4.onnx_data
            ├── model_q8.onnx        # INT8 quantized
            └── model_q8.onnx_data

Usage:
    # Export FP32 only
    uv run lfm2-export --model LiquidAI/LFM2-1.2B --output ./exports/LFM2-1.2B-ONNX

    # Export and quantize to Q4
    uv run lfm2-export --model LiquidAI/LFM2-1.2B --output ./exports/LFM2-1.2B-ONNX --quantize q4

    # Export and quantize to both Q4 and Q8
    uv run lfm2-export --model LiquidAI/LFM2-1.2B --output ./exports/LFM2-1.2B-ONNX --quantize q4 q8

    # Quantize existing export only (skip FP32 export)
    uv run lfm2-export --output ./exports/LFM2-1.2B-ONNX --skip-export --quantize q4

    # Quantize with lm_head included
    uv run lfm2-export --output ./exports/LFM2-1.2B-ONNX --skip-export --quantize q4 --no-exclude-lm-head
"""

import argparse
import logging
import pathlib

from liquidonnx.lfm2.export import export_model
from liquidonnx.lfm2.quantize import get_model_size, quantize_int4, quantize_int8

logger = logging.getLogger(__name__)


def do_export(model_path: str, output_path: pathlib.Path):
    """Export model to ONNX (FP32)."""
    logger.info(f"Exporting {model_path} to {output_path}...")
    export_model(model_path, str(output_path))


def do_quantize(onnx_dir: pathlib.Path, quant_type: str, exclude_lm_head: bool, block_size: int):
    """Quantize model to INT4 or INT8.

    Args:
        onnx_dir: Directory containing ONNX files
        quant_type: "q4" or "q8"
        exclude_lm_head: Whether to exclude lm_head from quantization
        block_size: Block size for quantization
    """
    bits = int(quant_type.replace("q", ""))

    # Find FP32 model
    input_model = onnx_dir / "model.onnx"
    if not input_model.exists():
        input_model = onnx_dir / "decoder_fp32.onnx"
    if not input_model.exists():
        raise FileNotFoundError(f"No model.onnx or decoder_fp32.onnx found in {onnx_dir}")

    output_model = onnx_dir / f"model_{quant_type}.onnx"

    if output_model.exists():
        logger.info(f"Skipping {quant_type} (already exists)")
        return

    _, orig_mb = get_model_size(input_model)

    logger.info(f"Quantizing to {quant_type.upper()}...")
    if bits == 4:
        quantize_int4(input_model, output_model, block_size, exclude_lm_head)
    else:
        quantize_int8(input_model, output_model, block_size, exclude_lm_head)

    _, quant_mb = get_model_size(output_model)
    if orig_mb > 0:
        logger.info(f"  {orig_mb:.1f} MB -> {quant_mb:.1f} MB ({orig_mb / quant_mb:.1f}x)")


def main():
    parser = argparse.ArgumentParser(
        description="Export LFM2 model to ONNX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        help="HuggingFace model path (e.g., LiquidAI/LFM2-1.2B)",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        required=True,
        help="Output directory for ONNX files",
    )

    # Export options
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Skip FP32 export, only run quantization",
    )

    # Quantization options
    parser.add_argument(
        "--quantize",
        nargs="*",
        metavar="QUANT",
        help="Quantize: q4, q8, or multiple (e.g., --quantize q4 q8)",
    )
    parser.add_argument(
        "--no-exclude-lm-head",
        action="store_true",
        help="Quantize lm_head layer (by default kept in FP32)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=32,
        help="Block size for quantization (default: 32)",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    # Validate arguments
    if not args.skip_export and not args.model:
        parser.error("--model is required unless --skip-export is specified")

    # Parse quantization options
    quant_types = []
    if args.quantize is not None:
        if len(args.quantize) == 0:
            quant_types = ["q4", "q8"]
        else:
            for q in args.quantize:
                q = q.lower()
                if not q.startswith("q"):
                    q = f"q{q}"
                if q not in ("q4", "q8"):
                    parser.error(f"Invalid quantization: {q}. Use q4 or q8.")
                quant_types.append(q)

    exclude_lm_head = not args.no_exclude_lm_head

    # Export FP32
    if not args.skip_export:
        do_export(args.model, args.output)

    # Quantize
    if quant_types:
        onnx_dir = args.output / "onnx"
        if not onnx_dir.exists():
            raise FileNotFoundError(f"ONNX directory not found: {onnx_dir}")

        for quant_type in quant_types:
            do_quantize(onnx_dir, quant_type, exclude_lm_head, args.block_size)

    # Summary
    onnx_dir = args.output / "onnx"
    if onnx_dir.exists():
        logger.info("")
        logger.info("Output files:")
        for f in sorted(onnx_dir.glob("*.onnx")):
            size_mb = f.stat().st_size / 1e6
            data_file = f.with_suffix(".onnx_data")
            if data_file.exists():
                data_mb = data_file.stat().st_size / 1e6
                logger.info(f"  {f.name}: {size_mb:.1f} MB + {data_mb:.1f} MB data")
            else:
                logger.info(f"  {f.name}: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
