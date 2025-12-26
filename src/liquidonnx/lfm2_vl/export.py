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
        ├── tokenizer_config.json
        └── onnx/
            ├── embed_tokens.onnx
            ├── embed_images.onnx
            ├── embed_images_q4.onnx
            ├── embed_images_q8.onnx
            ├── decoder.onnx
            ├── decoder_q4.onnx
            └── decoder_q8.onnx

Usage:
    # Export FP32 only (all sizes, both formats)
    uv run lfm2-vl-export --sizes all

    # Export with all quantizations (q4, q8 decoder, q8 vision)
    uv run lfm2-vl-export --sizes all --quantize

    # Export with Q4 vision for WebGPU
    uv run lfm2-vl-export --sizes 450M --quantize q4 --vision-quantize 4

    # Export with specific quantization
    uv run lfm2-vl-export --sizes 450M --quantize q4
    uv run lfm2-vl-export --sizes 450M --quantize q4 q8

    # Quantize existing exports (skip FP32 export)
    uv run lfm2-vl-export --sizes all --quantize --skip-export

    # Export only tiled format
    uv run lfm2-vl-export --sizes 450M --tiled --quantize
"""

import argparse
import logging
import pathlib

from liquidonnx.lfm2_vl import MODELS, VISION_MODES
from liquidonnx.lfm2_vl.builder import export_vl_model
from liquidonnx.quantize import get_model_size, quantize_model

logger = logging.getLogger(__name__)


def get_output_dir(size: str, fmt: str, output_base: pathlib.Path) -> pathlib.Path:
    """Get output directory for a model."""
    return output_base / "exports" / f"LFM2-VL-{size}-ONNX-{fmt}"


def do_export(model_path: str, output_path: pathlib.Path, fmt: str):
    """Export a single VL model to ONNX (FP32)."""
    logger.info(f"Exporting {model_path} to {output_path}...")
    export_vl_model(model_path, str(output_path), vision_input_format=fmt)


def do_quantize(
    onnx_dir: pathlib.Path, decoder_bits: int, vision_bits: int = 8, block_size: int = 32
):
    """Quantize a VL model."""
    if not onnx_dir.exists():
        raise FileNotFoundError(f"ONNX directory not found: {onnx_dir}")

    logger.info(
        f"Quantizing {onnx_dir.parent.name} -> decoder=q{decoder_bits}, vision=q{vision_bits}..."
    )

    # Quantize embed_images
    embed_fp32 = onnx_dir / "embed_images.onnx"

    embed_output = onnx_dir / f"embed_images_q{vision_bits}.onnx"
    if embed_fp32.exists() and not embed_output.exists():
        _, embed_orig_mb = get_model_size(embed_fp32)
        quantize_model(
            embed_fp32, embed_output, bits=vision_bits, block_size=block_size, exclude_lm_head=False
        )
        _, embed_quant_mb = get_model_size(embed_output)
        logger.info(
            f"  embed_images: {embed_orig_mb:.1f} -> {embed_quant_mb:.1f} MB "
            f"({embed_orig_mb / embed_quant_mb:.1f}x)"
        )

    # Quantize decoder
    decoder_fp32 = onnx_dir / "decoder.onnx"

    decoder_output = onnx_dir / f"decoder_q{decoder_bits}.onnx"
    if decoder_fp32.exists() and not decoder_output.exists():
        _, decoder_orig_mb = get_model_size(decoder_fp32)
        quantize_model(
            decoder_fp32,
            decoder_output,
            bits=decoder_bits,
            block_size=block_size,
            exclude_lm_head=True,
        )
        _, decoder_quant_mb = get_model_size(decoder_output)
        logger.info(
            f"  decoder: {decoder_orig_mb:.1f} -> {decoder_quant_mb:.1f} MB "
            f"({decoder_orig_mb / decoder_quant_mb:.1f}x)"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Export LFM2-VL models to ONNX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
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
        "--quantize",
        nargs="*",
        metavar="BITS",
        help="Quantize decoder: q4, q8, or both (default if no args)",
    )
    parser.add_argument(
        "--vision-quantize",
        type=int,
        choices=[4, 8],
        default=8,
        help="Vision encoder quantization bits (default: 8)",
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Skip FP32 export, only run quantization",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=32,
        help="Block size for quantization (default: 32)",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    # Parse quantization options
    quant_bits = []
    if args.quantize is not None:
        if len(args.quantize) == 0:
            quant_bits = [4, 8]
        else:
            for q in args.quantize:
                q = q.lower().replace("q", "")
                if q not in ("4", "8"):
                    parser.error(f"Invalid quantization: {q}. Use q4 or q8.")
                quant_bits.append(int(q))

    formats = [m for m in VISION_MODES if getattr(args, m)] or VISION_MODES
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
    for bits in quant_bits:
        for fmt in formats:
            for size in sizes:
                onnx_dir = get_output_dir(size, fmt, args.output_dir) / "onnx"
                do_quantize(onnx_dir, bits, args.vision_quantize, args.block_size)


if __name__ == "__main__":
    main()
