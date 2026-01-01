#!/usr/bin/env python3
"""
Export LFM2 models to ONNX with optional quantization and FP16 conversion.

Output Structure (Transformers.js compatible):
    exports/
    └── LFM2-{size}-ONNX/
        ├── config.json
        ├── tokenizer.json
        └── onnx/
            ├── model.onnx           # FP32
            ├── model.onnx_data
            ├── model_fp16.onnx      # --precision fp16
            ├── model_fp16.onnx_data
            ├── model_q4.onnx        # --precision q4
            ├── model_q4.onnx_data
            ├── model_q8.onnx        # --precision q8
            └── model_q8.onnx_data

Usage:
    # Export single model (FP32 only)
    uv run lfm2-export --sizes 350M

    # Export all models
    uv run lfm2-export --sizes all

    # Export with all precisions (fp16, q4, q8)
    uv run lfm2-export --sizes all --precision

    # Export with specific precisions
    uv run lfm2-export --sizes 350M --precision q4
    uv run lfm2-export --sizes 350M --precision fp16 q4 q8

    # Convert existing exports (skip FP32 export)
    uv run lfm2-export --sizes all --precision --skip-export

    # Quantize with lm_head included
    uv run lfm2-export --sizes 350M --precision q4 --no-exclude-lm-head
"""

import argparse
import json
import logging
import pathlib

import onnx
from transformers import AutoConfig, AutoTokenizer

from liquidonnx.lfm2 import MODELS
from liquidonnx.lfm2.builder import LFM2Builder, LFM2Config
from liquidonnx.quantize import get_model_size, get_total_model_size_mb, quantize_model

logger = logging.getLogger(__name__)


def convert_to_fp16(
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    keep_io_types: bool = True,
):
    """Convert ONNX model from FP32 to FP16.

    Args:
        input_path: Path to FP32 ONNX model
        output_path: Path for FP16 output model
        keep_io_types: Keep inputs/outputs as FP32 for compatibility
    """
    from onnx.external_data_helper import load_external_data_for_model
    from onnxruntime.transformers.float16 import convert_float_to_float16

    logger.info(f"Converting {input_path.name} to FP16...")

    model = onnx.load(str(input_path), load_external_data=False)
    load_external_data_for_model(model, str(input_path.parent))

    model_fp16 = convert_float_to_float16(
        model,
        keep_io_types=keep_io_types,
        force_fp16_initializers=True,
        disable_shape_infer=True,
    )

    output_data_path = output_path.parent / f"{output_path.stem}.onnx_data"
    if output_data_path.exists():
        output_data_path.unlink()

    onnx.save_model(
        model_fp16,
        str(output_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=f"{output_path.stem}.onnx_data",
    )

    orig_mb = get_total_model_size_mb(input_path)
    fp16_mb = get_total_model_size_mb(output_path)
    ratio = orig_mb / fp16_mb if fp16_mb > 0 else 0
    logger.info(f"  {input_path.name}: {orig_mb:.1f} -> {fp16_mb:.1f} MB ({ratio:.1f}x)")

    return output_path


def get_output_dir(size: str, output_base: pathlib.Path) -> pathlib.Path:
    return output_base / "exports" / f"LFM2-{size}-ONNX"


def export_model(model_path: str, output_dir: pathlib.Path | str):
    """Export LFM2 model to ONNX.

    Creates output structure:
        output_dir/
        ├── config.json
        ├── tokenizer.json
        ├── tokenizer_config.json
        ├── generation_config.json
        ├── chat_template.jinja
        └── onnx/
            ├── model.onnx
            └── model.onnx_data
    """
    output_dir = pathlib.Path(output_dir)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    lfm2_config = LFM2Config.from_hf_config(config)

    builder = LFM2Builder(lfm2_config)
    model = builder.build(model_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_dir = output_dir / "onnx"
    onnx_dir.mkdir(exist_ok=True)

    output_path = onnx_dir / "model.onnx"

    external_data_path = onnx_dir / "model.onnx_data"
    if external_data_path.exists():
        external_data_path.unlink()

    onnx.save_model(
        model,
        str(output_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="model.onnx_data",
    )

    logger.info(f"Model saved to {output_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)
    config.save_pretrained(output_dir)

    gen_config = {
        "_from_model_config": True,
        "bos_token_id": config.bos_token_id,
        "eos_token_id": config.eos_token_id,
        "pad_token_id": getattr(config, "pad_token_id", 0),
        "transformers_version": "4.54.0",
    }
    gen_config_path = output_dir / "generation_config.json"
    gen_config_path.write_text(json.dumps(gen_config, indent=2))

    config_path = output_dir / "config.json"
    cfg = json.loads(config_path.read_text())
    cfg["transformers.js_config"] = {
        "kv_cache_dtype": {"fp32": "float32"},
        "use_external_data_format": True,
    }
    config_path.write_text(json.dumps(cfg, indent=2))

    tokenizer_config_path = output_dir / "tokenizer_config.json"
    chat_template_path = output_dir / "chat_template.jinja"

    if tokenizer.chat_template:
        chat_template_path.write_text(tokenizer.chat_template)

        if tokenizer_config_path.exists():
            tok_cfg = json.loads(tokenizer_config_path.read_text())
            if "chat_template" not in tok_cfg:
                tok_cfg["chat_template"] = tokenizer.chat_template
                tokenizer_config_path.write_text(json.dumps(tok_cfg, indent=2))

    size_mb = output_path.stat().st_size / 1e6
    data_path = onnx_dir / "model.onnx_data"
    data_size_gb = data_path.stat().st_size / 1e9 if data_path.exists() else 0
    logger.info(f"Model size: {size_mb:.2f} MB + {data_size_gb:.2f} GB data")

    return output_path


def do_quantize(onnx_dir: pathlib.Path, bits: int, exclude_lm_head: bool, block_size: int):
    """Quantize model to INT4 or INT8."""
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


def do_fp16(onnx_dir: pathlib.Path):
    """Convert model to FP16."""
    if not onnx_dir.exists():
        raise FileNotFoundError(f"ONNX directory not found: {onnx_dir}")

    model_fp32 = onnx_dir / "model.onnx"
    model_fp16 = onnx_dir / "model_fp16.onnx"

    if model_fp16.exists():
        logger.info("Skipping fp16 (already exists)")
        return

    if model_fp32.exists():
        convert_to_fp16(model_fp32, model_fp16)


def main():
    parser = argparse.ArgumentParser(
        description="Export LFM2 models to ONNX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--sizes",
        nargs="+",
        required=True,
        help="Model sizes: 350M, 700M, 1.2B, 2.6B, or 'all'",
    )

    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("."),
        help="Output base directory (default: current directory)",
    )

    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Skip FP32 export, only run precision conversion",
    )

    parser.add_argument(
        "--precision",
        nargs="*",
        metavar="PRECISION",
        help="Output precisions: fp16, q4, q8, or all (default if no args)",
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

    sizes = list(MODELS.keys()) if "all" in args.sizes else args.sizes
    for s in sizes:
        if s not in MODELS:
            parser.error(f"Unknown size: {s}. Available: {', '.join(MODELS.keys())}")

    quant_bits = []
    do_fp16_conversion = False
    if args.precision is not None:
        if len(args.precision) == 0:
            quant_bits = [4, 8]
            do_fp16_conversion = True
        else:
            for p in args.precision:
                p = p.lower()
                if p == "fp16":
                    do_fp16_conversion = True
                elif p in ("q4", "q8"):
                    quant_bits.append(int(p[1]))
                else:
                    parser.error(f"Invalid precision: {p}. Use fp16, q4, or q8.")

    exclude_lm_head = not args.no_exclude_lm_head

    if not args.skip_export:
        logger.info("=" * 60)
        logger.info("Exporting models (FP32)")
        logger.info("=" * 60)

        for size in sizes:
            output_path = get_output_dir(size, args.output_dir)
            logger.info(f"Exporting {MODELS[size]} to {output_path}...")
            try:
                export_model(MODELS[size], str(output_path))
                logger.info(f"  {size}: OK")
            except Exception as e:
                logger.error(f"  {size}: FAILED - {e}")

    if do_fp16_conversion:
        logger.info("=" * 60)
        logger.info("Converting to FP16")
        logger.info("=" * 60)

        for size in sizes:
            onnx_dir = get_output_dir(size, args.output_dir) / "onnx"
            try:
                do_fp16(onnx_dir)
                logger.info(f"  {size}: OK")
            except Exception as e:
                logger.error(f"  {size}: FAILED - {e}")

    for bits in quant_bits:
        logger.info("=" * 60)
        logger.info(f"Quantizing to Q{bits}")
        logger.info("=" * 60)

        for size in sizes:
            onnx_dir = get_output_dir(size, args.output_dir) / "onnx"
            try:
                do_quantize(onnx_dir, bits, exclude_lm_head, args.block_size)
                logger.info(f"  {size}: OK")
            except Exception as e:
                logger.error(f"  {size}: FAILED - {e}")

    logger.info("=" * 60)
    logger.info("Output summary")
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
