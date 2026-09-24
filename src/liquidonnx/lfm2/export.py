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

from liquidonnx.external_data import split_external_data
from liquidonnx.genai_builder import export_decoder
from liquidonnx.quantize import DEFAULT_BLOCK_SIZE, derive_precision

logger = logging.getLogger(__name__)

TEXT_PRECISIONS = ("fp16", "q4", "q4f32", "q8")
MOE_PRECISIONS = ("fp16", "q4", "q4f16", "q8")
ALL_PRECISIONS = ("fp16", "q4", "q4f16", "q4f32", "q8")
DEFAULT_ORDER = ("q4", "q4f16", "q8", "fp16", "q4f32")


def get_model_name(model_path: str) -> str:
    """Extract model name from HF slug or local path."""
    return pathlib.Path(model_path).name


def set_default_decoder(output_dir: pathlib.Path, precisions: list[str]):
    """Point genai_config.json at the preferred one of precisions (fp32 if none)."""
    config_path = output_dir / "genai_config.json"
    default = next((p for p in DEFAULT_ORDER if p in precisions), None)
    filename = f"model_{default}.onnx" if default else "model.onnx"
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
        for required in (onnx_dir / "model.onnx", output_dir / "genai_config.json"):
            if not required.exists():
                parser.error(f"--skip-export needs an existing {required}")
    else:
        logger.info("=" * 60)
        logger.info(f"Exporting {args.model} (fp32) to {output_dir}")
        logger.info("=" * 60)
        output_dir.mkdir(parents=True, exist_ok=True)
        export_decoder(args.model, output_dir)

    for precision in precisions:
        logger.info("=" * 60)
        logger.info(f"Deriving {precision}")
        logger.info("=" * 60)
        derive_precision(
            onnx_dir, precision, block_size=args.block_size, q4_symmetric=not args.q4_asymmetric
        )

    # Rebuilding fp32 makes precisions from earlier runs stale; --skip-export keeps them valid.
    available = precisions
    if args.skip_export:
        available = [p for p in ALL_PRECISIONS if (onnx_dir / f"model_{p}.onnx").exists()]
    set_default_decoder(output_dir, available)

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
