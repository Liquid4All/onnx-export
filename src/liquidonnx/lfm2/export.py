#!/usr/bin/env python3
"""
Export LFM2 models to ONNX with optional quantization and FP16 conversion.

Output Structure (Transformers.js compatible):
    {output-dir}/
    └── {model-name}-ONNX/
        ├── config.json
        ├── tokenizer.json
        └── onnx/
            ├── model.onnx              # FP32
            ├── model.onnx_data
            ├── model_fp16.onnx         # --precision fp16
            ├── model_fp16.onnx_data
            ├── model_q4.onnx           # --precision q4 (GatherBlockQuantized)
            ├── model_q4.onnx_data
            ├── model_q4f32.onnx        # --precision q4f32 (FP32 embedding)
            ├── model_q4f32.onnx_data
            ├── model_q8.onnx           # --precision q8
            └── model_q8.onnx_data

Usage:
    # Export from HuggingFace
    uv run lfm2-export LiquidAI/LFM2-350M
    uv run lfm2-export LiquidAI/LFM2.5-1.2B-Instruct

    # Export from local path
    uv run lfm2-export /path/to/local/model

    # Export with custom output directory
    uv run lfm2-export LiquidAI/LFM2-350M --output-dir ./exports

    # Export with specific precisions
    uv run lfm2-export LiquidAI/LFM2-350M --precision fp16 q4

    # Export with all precisions
    uv run lfm2-export LiquidAI/LFM2-350M --precision

    # Convert existing export (skip FP32 export)
    uv run lfm2-export LiquidAI/LFM2-350M --precision --skip-export
"""

import argparse
import json
import logging
import pathlib

import onnx
from transformers import AutoConfig, AutoTokenizer

from liquidonnx.external_data import split_external_data
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


def get_model_name(model_path: str) -> str:
    """Extract model name from HF slug or local path."""
    # Handle HF slugs like "LiquidAI/LFM2-350M" -> "LFM2-350M"
    if "/" in model_path:
        return model_path.split("/")[-1]
    # Handle local paths
    return pathlib.Path(model_path).name


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


def export_model_q4(
    model_path: str,
    output_dir: pathlib.Path | str,
    block_size: int = 32,
    symmetric: bool = True,
):
    """Export LFM2 model with Q4 quantization (GatherBlockQuantized + MatMulNBits).

    Builds the model with INT4 embedding (GatherBlockQuantized) and lm_head
    (MatMulNBits), then post-export quantizes remaining FP32 MatMul layers.
    """
    output_dir = pathlib.Path(output_dir)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    lfm2_config = LFM2Config.from_hf_config(config)

    builder = LFM2Builder(lfm2_config, use_q4=True, q4_block_size=block_size)
    model = builder.build(model_path)

    onnx_dir = output_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    # Save intermediate model (embedding+lm_head quantized, other layers FP32)
    intermediate_path = onnx_dir / "model_q4_intermediate.onnx"
    intermediate_data = onnx_dir / "model_q4_intermediate.onnx_data"
    if intermediate_data.exists():
        intermediate_data.unlink()

    onnx.save_model(
        model,
        str(intermediate_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="model_q4_intermediate.onnx_data",
    )

    # Post-export quantization for remaining FP32 MatMul nodes
    output_path = onnx_dir / "model_q4.onnx"
    if output_path.exists():
        logger.info("Skipping q4 (already exists)")
        intermediate_path.unlink(missing_ok=True)
        intermediate_data.unlink(missing_ok=True)
        return

    quantize_model(
        intermediate_path,
        output_path,
        bits=4,
        block_size=block_size,
        exclude_lm_head=True,
        symmetric=symmetric,
    )

    # Clean up intermediate
    intermediate_path.unlink(missing_ok=True)
    intermediate_data.unlink(missing_ok=True)

    _, q4_mb = get_model_size(output_path)
    logger.info(f"  Q4 model: {q4_mb:.1f} MB")


def do_quantize(
    onnx_dir: pathlib.Path,
    bits: int,
    exclude_lm_head: bool,
    block_size: int,
    symmetric: bool = False,
    output_name: str | None = None,
    source: str = "model.onnx",
):
    """Quantize model to INT4 or INT8.

    Args:
        onnx_dir: Directory containing source model
        bits: Quantization bits (4 or 8)
        exclude_lm_head: Keep lm_head in FP32
        block_size: Block size for quantization
        symmetric: Use symmetric quantization (no zero points, better JSEP compatibility)
        output_name: Override output filename (default: model_q{bits}.onnx)
        source: Source model filename to quantize from (default: model.onnx)
    """
    input_model = onnx_dir / source
    if not input_model.exists():
        raise FileNotFoundError(f"{source} not found in {onnx_dir}")

    output_model = onnx_dir / (output_name or f"model_q{bits}.onnx")

    if output_model.exists():
        logger.info(f"Skipping {output_model.name} (already exists)")
        return

    _, orig_mb = get_model_size(input_model)

    quant_type = "symmetric" if symmetric else "asymmetric"
    logger.info(f"Quantizing {source} to Q{bits} ({quant_type})...")
    quantize_model(
        input_model,
        output_model,
        bits=bits,
        block_size=block_size,
        exclude_lm_head=exclude_lm_head,
        symmetric=symmetric,
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
        "model",
        help="HuggingFace model ID or local path (e.g., LiquidAI/LFM2-350M)",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("."),
        help="Output base directory (default: current directory)",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        help="Output folder name (default: {model-name}-ONNX)",
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Skip FP32 base model export. Q4 always rebuilds from HF weights.",
    )
    parser.add_argument(
        "--precision",
        nargs="*",
        metavar="PRECISION",
        help="Output precisions: fp16, q4, q4f32, q8, or all (default if no args)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=32,
        help="Block size for quantization (default: 32)",
    )
    parser.add_argument(
        "--q4-asymmetric",
        action="store_true",
        help="Use asymmetric Q4 quantization. Default is symmetric (required for WebGPU)",
    )
    parser.add_argument(
        "--split-data",
        type=float,
        default=2.0,
        metavar="GB",
        help="Split external data into chunks (default: 2GB per chunk)",
    )
    parser.add_argument(
        "--no-split-data",
        action="store_true",
        help="Disable external data splitting",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Derive output name from model path
    model_name = get_model_name(args.model)
    output_name = args.output_name or f"{model_name}-ONNX"
    output_dir = args.output_dir / "exports" / output_name

    do_fp16_conversion = False
    do_q4 = False
    do_q4f32 = False
    do_q8 = False
    if args.precision is not None:
        if len(args.precision) == 0:
            do_fp16_conversion = True
            do_q4 = True
            do_q4f32 = True
            do_q8 = True
        else:
            for p in args.precision:
                p = p.lower()
                if p == "fp16":
                    do_fp16_conversion = True
                elif p == "q4":
                    do_q4 = True
                elif p == "q4f32":
                    do_q4f32 = True
                elif p == "q8":
                    do_q8 = True
                else:
                    parser.error(f"Invalid precision: {p}. Use fp16, q4, q4f32, or q8.")

    onnx_dir = output_dir / "onnx"

    if not args.skip_export:
        logger.info("=" * 60)
        logger.info("Exporting model (FP32)")
        logger.info("=" * 60)
        logger.info(f"Exporting {args.model} to {output_dir}...")
        export_model(args.model, str(output_dir))
        logger.info(f"  {model_name}: OK")

    if do_fp16_conversion:
        logger.info("=" * 60)
        logger.info("Converting to FP16")
        logger.info("=" * 60)
        do_fp16(onnx_dir)
        logger.info(f"  {model_name}: OK")

    if do_q4:
        logger.info("=" * 60)
        logger.info("Exporting Q4 (GatherBlockQuantized)")
        logger.info("=" * 60)
        export_model_q4(
            args.model,
            output_dir,
            block_size=args.block_size,
            symmetric=not args.q4_asymmetric,
        )
        logger.info(f"  {model_name}: OK")

    if do_q4f32:
        logger.info("=" * 60)
        logger.info("Quantizing to Q4F32")
        logger.info("=" * 60)
        symmetric = not args.q4_asymmetric
        do_quantize(
            onnx_dir,
            bits=4,
            exclude_lm_head=True,
            block_size=args.block_size,
            symmetric=symmetric,
            output_name="model_q4f32.onnx",
        )
        logger.info(f"  {model_name}: OK")

    if do_q8:
        logger.info("=" * 60)
        logger.info("Quantizing to Q8")
        logger.info("=" * 60)
        do_quantize(
            onnx_dir,
            bits=8,
            exclude_lm_head=True,
            block_size=args.block_size,
            symmetric=False,
            output_name="model_q8.onnx",
        )
        logger.info(f"  {model_name}: OK")

    if not args.no_split_data:
        chunk_size_bytes = int(args.split_data * 1024 * 1024 * 1024)
        for onnx_file in onnx_dir.glob("model*.onnx"):
            data_file = onnx_file.with_suffix(".onnx_data")
            if data_file.exists() and data_file.stat().st_size > chunk_size_bytes:
                logger.info("=" * 60)
                logger.info(f"Splitting external data ({args.split_data:.1f} GB chunks)")
                logger.info("=" * 60)
                logger.info(f"  Splitting {onnx_file.name}...")
                split_external_data(onnx_file, chunk_size=chunk_size_bytes)

    logger.info("=" * 60)
    logger.info("Output summary")
    logger.info("=" * 60)
    if output_dir.exists():
        files = list(onnx_dir.glob("model*.onnx"))
        file_names = ", ".join(f.name for f in sorted(files))
        total_size = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())
        logger.info(f"  {output_dir} ({total_size / 1e9:.2f} GB)")
        logger.info(f"    Files: {file_names}")


if __name__ == "__main__":
    main()
