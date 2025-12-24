#!/usr/bin/env python3
"""
Export LFM2-VL models to ONNX with optional quantization.

Vision Input Formats:
- --tiled: Input [B, N, 768] with pre-extracted patches (HuggingFace style)
- --conv2d: Input [B, 3, H, W] with raw image (simpler, llama.cpp style)

Output Structure (Transformers.js compatible):
    exports/
    └── LFM2-VL-{size}-ONNX-{tiled|conv2d}/
        ├── config.json
        ├── tokenizer.json
        └── onnx/
            ├── embed_tokens.onnx
            ├── embed_images_fp32.onnx
            ├── embed_images_q{4,8}.onnx
            ├── decoder_fp32.onnx
            └── decoder_q{4,8}.onnx

Usage:
    # Export all predefined models (both formats by default)
    lfm2-vl-export --sizes all

    # Export specific sizes
    lfm2-vl-export --sizes 450M 1.6B

    # Export only tiled format
    lfm2-vl-export --sizes 450M --tiled

    # Export only conv2d format
    lfm2-vl-export --sizes 450M --conv2d

    # Export with quantization
    lfm2-vl-export --sizes 450M --decoder-bits 4 --vision-bits 8

    # Skip FP32 export, only quantize existing models
    lfm2-vl-export --sizes 450M --skip-export --decoder-bits 4 --vision-bits 8

    # Custom output directory
    lfm2-vl-export --sizes all --output-dir ./my_models
"""

import argparse
import logging
import pathlib

from liquidonnx.lfm2_vl import MODELS, FORMATS
from liquidonnx.lfm2_vl.export import export_vl_model
from liquidonnx.lfm2_vl.quantize import quantize_model

logger = logging.getLogger(__name__)


def get_output_dir(size: str, fmt: str, output_base: pathlib.Path) -> pathlib.Path:
    """Get output directory for a model."""
    return output_base / "exports" / f"LFM2-VL-{size}-ONNX-{fmt}"


def do_export(model_path: str, output_path: pathlib.Path, fmt: str):
    """Export a single VL model to ONNX (FP32)."""
    logger.info(f"Exporting {model_path} to {output_path}...")
    export_vl_model(model_path, str(output_path), vision_input_format=fmt)


def do_quantize(onnx_dir: pathlib.Path, decoder_bits: int, vision_bits: int):
    """Quantize a VL model (in-place)."""
    if not onnx_dir.exists():
        raise FileNotFoundError(f"ONNX directory not found: {onnx_dir}")

    logger.info(f"Quantizing -> decoder_q{decoder_bits}, embed_images_q{vision_bits}...")

    # Quantize embed_images
    embed_fp32 = onnx_dir / "embed_images_fp32.onnx"
    if not embed_fp32.exists():
        embed_fp32 = onnx_dir / "embed_images.onnx"

    if embed_fp32.exists():
        embed_output = onnx_dir / f"embed_images_q{vision_bits}.onnx"
        quantize_model(embed_fp32, embed_output, bits=vision_bits, exclude_lm_head=False)

    # Quantize decoder
    decoder_fp32 = onnx_dir / "decoder_fp32.onnx"
    if not decoder_fp32.exists():
        decoder_fp32 = onnx_dir / "decoder.onnx"

    if decoder_fp32.exists():
        decoder_output = onnx_dir / f"decoder_q{decoder_bits}.onnx"
        quantize_model(decoder_fp32, decoder_output, bits=decoder_bits, exclude_lm_head=True)


def main():
    parser = argparse.ArgumentParser(
        description="Export LFM2-VL models to ONNX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Model selection
    parser.add_argument(
        "--sizes",
        nargs="+",
        required=True,
        help="Model sizes: 450M, 1.6B, 3B, or 'all'",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("."),
        help="Output base directory (default: current directory)",
    )

    # Vision input format
    parser.add_argument(
        "--tiled",
        action="store_true",
        help="Tiled input format [B, N, 768] (HuggingFace style)",
    )
    parser.add_argument(
        "--conv2d",
        action="store_true",
        help="Conv2d input format [B, 3, H, W] (llama.cpp style)",
    )

    # Quantization
    parser.add_argument(
        "--decoder-bits",
        type=int,
        choices=[4, 8],
        help="Quantize decoder to INT4 or INT8 (requires --vision-bits)",
    )
    parser.add_argument(
        "--vision-bits",
        type=int,
        choices=[4, 8],
        help="Quantize vision encoder to INT4 or INT8 (requires --decoder-bits)",
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Skip FP32 export, only run quantization",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    # Validate arguments
    if (args.decoder_bits is None) != (args.vision_bits is None):
        parser.error("Both --decoder-bits and --vision-bits must be specified together")

    formats = [f for f in FORMATS if getattr(args, f)] or FORMATS
    sizes = list(MODELS.keys()) if "all" in args.sizes else args.sizes

    for s in sizes:
        if s not in MODELS:
            parser.error(f"Unknown size: {s}. Available: {', '.join(MODELS.keys())}")

    # Export
    if not args.skip_export:
        for fmt in formats:
            for size in sizes:
                do_export(MODELS[size], get_output_dir(size, fmt, args.output_dir), fmt)

    # Quantize
    if args.decoder_bits:
        for fmt in formats:
            for size in sizes:
                onnx_dir = get_output_dir(size, fmt, args.output_dir) / "onnx"
                do_quantize(onnx_dir, args.decoder_bits, args.vision_bits)


if __name__ == "__main__":
    main()
