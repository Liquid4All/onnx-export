#!/usr/bin/env python3
"""
Export LFM2 and LFM2-MoE models to ONNX for onnxruntime-genai.

The decoder comes from the onnxruntime-genai model builder (fp32, CPU EP, see
liquidonnx.genai_builder); every other precision is derived from it by liquidonnx.quantize.

Output Structure:
    {output-dir}/exports/{model-name}-ONNX/
        ├── genai_config.json        # decoder points at the default precision
        ├── config.json
        ├── tokenizer.json
        ├── tokenizer_config.json
        ├── chat_template.jinja
        └── onnx/
            ├── model.onnx           # fp32
            ├── model_fp16.onnx      # fp16 weights, activations and caches; fp32 logits
            ├── model_q4.onnx        # int4; embedding and tied lm_head share one int4 table
            ├── model_q4f16.onnx     # q4 with fp16 activations
            ├── model_q4f32.onnx     # int4 MatMuls; fp32 embedding and lm_head
            └── model_q8.onnx        # int8

    MoE experts become QMoE int4 (q4*) or int8 (q8); the routers stay fp32.

genai_config.json uses the first exported precision of q4, q4f16, q8, fp16, q4f32, fp32. Load
another one by overriding model.decoder.filename, e.g. with onnxruntime_genai.Config.overlay.

Usage:
    # Export from HuggingFace (fp32 only)
    uv run lfm2-export LiquidAI/LFM2.5-1.2B-Instruct

    # Export with all precisions
    uv run lfm2-export LiquidAI/LFM2.5-1.2B-Instruct --precision

    # Export with specific precisions
    uv run lfm2-export LiquidAI/LFM2.5-350M --precision fp16 q4

    # MoE checkpoints
    uv run lfm2-moe-export LiquidAI/LFM2.5-8B-A1B --precision q4 q8

    # Derive precisions from an existing fp32 export
    uv run lfm2-export LiquidAI/LFM2.5-350M --precision q8 --skip-export
"""

import argparse
import json
import logging
import pathlib
import shutil
import tempfile

import onnx

from liquidonnx.external_data import split_external_data
from liquidonnx.genai_builder import build_decoder, resolve_checkpoint
from liquidonnx.quantize import (
    DEFAULT_BLOCK_SIZE,
    convert_to_fp16,
    find_router_nodes,
    get_total_model_size_mb,
    load_model,
    moe_to_qmoe,
    quantize_matmuls,
    save_model,
    tie_embedding_int4,
)

logger = logging.getLogger(__name__)

TEXT_PRECISIONS = ("fp16", "q4", "q4f32", "q8")
MOE_PRECISIONS = ("fp16", "q4", "q4f16", "q8")
ALL_PRECISIONS = ("fp16", "q4", "q4f16", "q4f32", "q8")
DEFAULT_ORDER = ("q4", "q4f16", "q8", "fp16", "q4f32")
CHECKPOINT_FILES = ("config.json", "generation_config.json")


def get_model_name(model_path: str) -> str:
    """Extract model name from HF slug or local path."""
    return pathlib.Path(model_path).name


def pin_kv_head_size(model: onnx.ModelProto, head_size: int):
    """Replace the builder's symbolic `kv_cache_dim` with the head size on the cache I/O.

    genai leaves it symbolic so one graph can serve quantized KV caches; fixing it lets plain
    onnxruntime callers allocate empty caches from the graph signature.
    """
    for value in [*model.graph.input, *model.graph.output]:
        for dim in value.type.tensor_type.shape.dim:
            if dim.dim_param == "kv_cache_dim":
                dim.Clear()
                dim.dim_value = head_size


def export_model(model_path: str, output_dir: pathlib.Path) -> pathlib.Path:
    """Build the fp32 decoder into output_dir/onnx/model.onnx plus genai_config and tokenizer."""
    onnx_dir = output_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=output_dir, prefix=".genai-build-") as tmp:
        build_dir = pathlib.Path(tmp)
        built = build_decoder(model_path, build_dir)

        genai_config = json.loads((build_dir / "genai_config.json").read_text())
        model = load_model(built)
        pin_kv_head_size(model, genai_config["model"]["decoder"]["head_size"])
        output_path = save_model(model, onnx_dir / "model.onnx")
        del model

        for f in build_dir.iterdir():
            if f.is_file() and not f.name.startswith("model.onnx"):
                shutil.copy2(f, output_dir / f.name)

    checkpoint = resolve_checkpoint(model_path)
    for name in CHECKPOINT_FILES:
        if (checkpoint / name).exists():
            shutil.copy2(checkpoint / name, output_dir / name)

    logger.info(f"Model saved to {output_path} ({get_total_model_size_mb(output_path):.1f} MB)")
    return output_path


def derive_precision(
    onnx_dir: pathlib.Path,
    precision: str,
    block_size: int = DEFAULT_BLOCK_SIZE,
    q4_symmetric: bool = True,
) -> pathlib.Path:
    """Write onnx_dir/model_{precision}.onnx from onnx_dir/model.onnx."""
    base = onnx_dir / "model.onnx"
    output_path = onnx_dir / f"model_{precision}.onnx"

    if precision == "fp16":
        return convert_to_fp16(base, output_path)

    if precision == "q4f16":
        q4 = onnx_dir / "model_q4.onnx"
        if q4.exists():
            return convert_to_fp16(q4, output_path)
        with tempfile.TemporaryDirectory(dir=onnx_dir) as tmp:
            q4 = _quantize(
                base, pathlib.Path(tmp) / "model_q4.onnx", "q4", block_size, q4_symmetric
            )
            return convert_to_fp16(q4, output_path)

    return _quantize(base, output_path, precision, block_size, q4_symmetric)


def _quantize(
    base: pathlib.Path,
    output_path: pathlib.Path,
    precision: str,
    block_size: int,
    q4_symmetric: bool,
) -> pathlib.Path:
    model = load_model(base)
    exclude = ["/lm_head/MatMul", *find_router_nodes(model)]
    if precision == "q8":
        experts = moe_to_qmoe(model, bits=8, block_size=block_size)
        model = quantize_matmuls(
            model, bits=8, block_size=block_size, symmetric=False, exclude=exclude
        )
    elif precision in ("q4", "q4f32"):
        tied = precision == "q4" and tie_embedding_int4(model, block_size)
        experts = moe_to_qmoe(model, bits=4, block_size=block_size)
        model = quantize_matmuls(
            model, bits=4, block_size=block_size, symmetric=q4_symmetric, exclude=exclude
        )
        if tied:
            logger.info("  embedding and lm_head share one int4 table")
    else:
        raise ValueError(f"Unknown precision: {precision}")
    if experts:
        logger.info(f"  {experts} MoE layers -> QMoE")
    save_model(model, output_path)
    logger.info(f"  {output_path.name}: {get_total_model_size_mb(output_path):.1f} MB")
    return output_path


def set_default_decoder(output_dir: pathlib.Path):
    """Point genai_config.json at the preferred precision that was exported."""
    config_path = output_dir / "genai_config.json"
    onnx_dir = output_dir / "onnx"
    filename = next(
        (f"model_{p}.onnx" for p in DEFAULT_ORDER if (onnx_dir / f"model_{p}.onnx").exists()),
        "model.onnx",
    )
    config = json.loads(config_path.read_text())
    config["model"]["decoder"]["filename"] = f"onnx/{filename}"
    config_path.write_text(json.dumps(config, indent=4))
    logger.info(f"genai_config.json decoder -> onnx/{filename}")


def main(default_precisions: tuple[str, ...] = TEXT_PRECISIONS, description: str | None = None):
    parser = argparse.ArgumentParser(
        description=description or "Export LFM2 models to ONNX for onnxruntime-genai",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "model",
        help="HuggingFace model ID or local path (e.g., LiquidAI/LFM2.5-350M)",
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
        help="Reuse the existing fp32 onnx/model.onnx instead of rebuilding it",
    )
    parser.add_argument(
        "--precision",
        nargs="*",
        metavar="PRECISION",
        help=f"Output precisions: {', '.join(ALL_PRECISIONS)} "
        f"(no value: {' '.join(default_precisions)})",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=DEFAULT_BLOCK_SIZE,
        help=f"Block size for quantization (default: {DEFAULT_BLOCK_SIZE})",
    )
    parser.add_argument(
        "--q4-asymmetric",
        action="store_true",
        help="Use asymmetric int4 for MatMul weights. Default is symmetric",
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

    model_name = get_model_name(args.model)
    output_dir = args.output_dir / "exports" / (args.output_name or f"{model_name}-ONNX")
    onnx_dir = output_dir / "onnx"

    precisions = [] if args.precision is None else [p.lower() for p in args.precision]
    if args.precision == []:
        precisions = list(default_precisions)
    for p in precisions:
        if p not in ALL_PRECISIONS:
            parser.error(f"Invalid precision: {p}. Use {', '.join(ALL_PRECISIONS)}.")
    # q4f16 reuses model_q4.onnx when it is exported too.
    precisions.sort(key=ALL_PRECISIONS.index)

    if args.skip_export:
        if not (onnx_dir / "model.onnx").exists():
            parser.error(f"--skip-export needs an existing {onnx_dir / 'model.onnx'}")
    else:
        logger.info("=" * 60)
        logger.info(f"Exporting {args.model} (fp32) to {output_dir}")
        logger.info("=" * 60)
        output_dir.mkdir(parents=True, exist_ok=True)
        export_model(args.model, output_dir)

    for precision in precisions:
        logger.info("=" * 60)
        logger.info(f"Deriving {precision}")
        logger.info("=" * 60)
        derive_precision(
            onnx_dir, precision, block_size=args.block_size, q4_symmetric=not args.q4_asymmetric
        )

    set_default_decoder(output_dir)

    if not args.no_split_data:
        chunk_size_bytes = int(args.split_data * 1024 * 1024 * 1024)
        for onnx_file in onnx_dir.glob("model*.onnx"):
            data_file = onnx_file.with_suffix(".onnx_data")
            if data_file.exists() and data_file.stat().st_size > chunk_size_bytes:
                logger.info(f"Splitting {onnx_file.name} ({args.split_data:.1f} GB chunks)")
                split_external_data(onnx_file, chunk_size=chunk_size_bytes)

    logger.info("=" * 60)
    logger.info("Output summary")
    logger.info("=" * 60)
    files = ", ".join(f.name for f in sorted(onnx_dir.glob("model*.onnx")))
    total_size = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())
    logger.info(f"  {output_dir} ({total_size / 1e9:.2f} GB)")
    logger.info(f"    Files: {files}")


if __name__ == "__main__":
    main()
