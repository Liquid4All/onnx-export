#!/usr/bin/env python3
"""
Export LFM2 models to ONNX with optional quantization.

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
    # Export single model (FP32 only)
    uv run lfm2-export --sizes 350M

    # Export all models
    uv run lfm2-export --sizes all

    # Export with Q4 quantization
    uv run lfm2-export --sizes 350M --quantize q4

    # Export with both Q4 and Q8
    uv run lfm2-export --sizes 1.2B --quantize q4 q8

    # Export with all quantizations (default when --quantize has no args)
    uv run lfm2-export --sizes all --quantize

    # Quantize existing exports (skip FP32 export)
    uv run lfm2-export --sizes all --quantize --skip-export

    # Quantize with lm_head included
    uv run lfm2-export --sizes 350M --quantize q4 --no-exclude-lm-head
"""

import argparse
import json
import logging
import os
import pathlib

import onnx
from transformers import AutoConfig, AutoTokenizer

from liquidonnx.lfm2 import MODELS
from liquidonnx.lfm2.builder import LFM2Builder, LFM2Config
from liquidonnx.quantize import get_model_size, quantize_model

logger = logging.getLogger(__name__)


def get_output_dir(size: str, output_base: pathlib.Path) -> pathlib.Path:
    """Get output directory for a model."""
    return output_base / "exports" / f"LFM2-{size}-ONNX"


def export_model(model_path: str, output_dir: str):
    """Export LFM2 model to ONNX.

    Creates output structure:
        output_dir/
        ├── config.json
        ├── tokenizer.json
        ├── generation_config.json
        ├── chat_template.jinja
        └── onnx/
            ├── model.onnx
            └── model.onnx_data
    """
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    lfm2_config = LFM2Config.from_hf_config(config)

    builder = LFM2Builder(lfm2_config)
    model = builder.build(model_path)

    os.makedirs(output_dir, exist_ok=True)
    onnx_dir = os.path.join(output_dir, "onnx")
    os.makedirs(onnx_dir, exist_ok=True)

    output_path = os.path.join(onnx_dir, "model.onnx")

    # Remove existing external data file to avoid appending
    external_data_path = os.path.join(onnx_dir, "model.onnx_data")
    if os.path.exists(external_data_path):
        os.remove(external_data_path)

    onnx.save_model(
        model,
        output_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="model.onnx_data",
    )

    logger.info(f"Model saved to {output_path}")

    # Copy tokenizer and config
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)
    config.save_pretrained(output_dir)

    # Create generation_config.json (required by Transformers.js)
    gen_config = {
        "_from_model_config": True,
        "bos_token_id": config.bos_token_id,
        "eos_token_id": config.eos_token_id,
        "pad_token_id": getattr(config, "pad_token_id", 0),
        "transformers_version": "4.54.0",
    }
    gen_config_path = os.path.join(output_dir, "generation_config.json")
    with open(gen_config_path, "w") as f:
        json.dump(gen_config, f, indent=2)

    # Add transformers.js_config to config.json (for external data support)
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path) as f:
        cfg = json.load(f)
    cfg["transformers.js_config"] = {
        "kv_cache_dtype": {"fp32": "float32"},
        "use_external_data_format": True,
    }
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)

    # Save chat_template if available
    tokenizer_config_path = os.path.join(output_dir, "tokenizer_config.json")
    chat_template_path = os.path.join(output_dir, "chat_template.jinja")

    if tokenizer.chat_template:
        with open(chat_template_path, "w") as f:
            f.write(tokenizer.chat_template)

        if os.path.exists(tokenizer_config_path):
            with open(tokenizer_config_path) as f:
                tok_cfg = json.load(f)
            if "chat_template" not in tok_cfg:
                tok_cfg["chat_template"] = tokenizer.chat_template
                with open(tokenizer_config_path, "w") as f:
                    json.dump(tok_cfg, f, indent=2)

    # Print summary
    size_mb = os.path.getsize(output_path) / 1e6
    data_path = os.path.join(onnx_dir, "model.onnx_data")
    data_size_gb = os.path.getsize(data_path) / 1e9 if os.path.exists(data_path) else 0
    logger.info(f"Model size: {size_mb:.2f} MB + {data_size_gb:.2f} GB data")

    return output_path


def do_quantize(onnx_dir: pathlib.Path, bits: int, exclude_lm_head: bool, block_size: int):
    """Quantize model to INT4 or INT8.

    Args:
        onnx_dir: Directory containing ONNX files
        bits: 4 for INT4, 8 for INT8
        exclude_lm_head: Whether to exclude lm_head from quantization
        block_size: Block size for quantization
    """
    input_model = onnx_dir / "model.onnx"
    if not input_model.exists():
        raise FileNotFoundError(f"model.onnx not found in {onnx_dir}")

    output_model = onnx_dir / f"model_q{bits}.onnx"

    if output_model.exists():
        logger.info(f"Skipping q{bits} (already exists)")
        return

    _, orig_mb = get_model_size(input_model)

    logger.info(f"Quantizing to Q{bits}...")
    quantize_model(
        input_model, output_model, bits=bits, block_size=block_size, exclude_lm_head=exclude_lm_head
    )

    _, quant_mb = get_model_size(output_model)
    if orig_mb > 0:
        logger.info(f"  {orig_mb:.1f} MB -> {quant_mb:.1f} MB ({orig_mb / quant_mb:.1f}x)")


def main():
    parser = argparse.ArgumentParser(
        description="Export LFM2 models to ONNX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Model selection
    parser.add_argument(
        "--sizes",
        nargs="+",
        required=True,
        help="Model sizes: 350M, 700M, 1.2B, 2.6B, or 'all'",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("."),
        help="Output base directory (default: current directory)",
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
        metavar="BITS",
        help="Quantize: q4, q8, or both (default if no args)",
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

    # Parse sizes
    sizes = list(MODELS.keys()) if "all" in args.sizes else args.sizes
    for s in sizes:
        if s not in MODELS:
            parser.error(f"Unknown size: {s}. Available: {', '.join(MODELS.keys())}")

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

    exclude_lm_head = not args.no_exclude_lm_head

    # Export FP32
    if not args.skip_export:
        logger.info("=" * 60)
        logger.info("EXPORTING MODELS (FP32)")
        logger.info("=" * 60)

        for size in sizes:
            output_path = get_output_dir(size, args.output_dir)
            logger.info(f"Exporting {MODELS[size]} to {output_path}...")
            try:
                export_model(MODELS[size], str(output_path))
                logger.info(f"  {size}: OK")
            except Exception as e:
                logger.error(f"  {size}: FAILED - {e}")

    # Quantize
    for bits in quant_bits:
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"QUANTIZING TO Q{bits}")
        logger.info("=" * 60)

        for size in sizes:
            onnx_dir = get_output_dir(size, args.output_dir) / "onnx"
            try:
                do_quantize(onnx_dir, bits, exclude_lm_head, args.block_size)
                logger.info(f"  {size}: OK")
            except Exception as e:
                logger.error(f"  {size}: FAILED - {e}")

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("OUTPUT")
    logger.info("=" * 60)

    for size in sizes:
        out_dir = get_output_dir(size, args.output_dir)
        if out_dir.exists():
            onnx_dir = out_dir / "onnx"
            files = list(onnx_dir.glob("model*.onnx"))
            file_names = ", ".join(f.name for f in sorted(files))
            total_size = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file())
            logger.info(f"  {out_dir} ({total_size / 1e9:.2f} GB)")
            logger.info(f"    Files: {file_names}")


if __name__ == "__main__":
    main()
