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

from liquidonnx.export_cli import (
    add_export_arguments,
    finish,
    log_step,
    output_dir,
    parse_precisions,
)
from liquidonnx.genai_builder import export_decoder
from liquidonnx.quantize import derive_precision

logger = logging.getLogger(__name__)

TEXT_PRECISIONS = ("fp16", "q4", "q4f32", "q8")
MOE_PRECISIONS = ("fp16", "q4", "q4f16", "q8")
ALL_PRECISIONS = ("fp16", "q4", "q4f16", "q4f32", "q8")
DEFAULT_ORDER = ("q4", "q4f16", "q8", "fp16", "q4f32")


def model_file(precision: str) -> str:
    """Decoder file (in onnx/) of a precision."""
    return "model.onnx" if precision == "fp32" else f"model_{precision}.onnx"


def genai_files(precision: str) -> dict:
    """genai_config.json model entries that load one precision."""
    return {"decoder": {"filename": f"onnx/{model_file(precision)}"}}


def set_default_decoder(output_dir: pathlib.Path, precisions: list[str]):
    """Point genai_config.json at the preferred one of precisions (fp32 if none)."""
    config_path = output_dir / "genai_config.json"
    default = next((p for p in DEFAULT_ORDER if p in precisions), "fp32")
    config = json.loads(config_path.read_text())
    config["model"]["decoder"].update(genai_files(default)["decoder"])
    config_path.write_text(json.dumps(config, indent=4))
    logger.info(f"genai_config.json decoder -> onnx/{model_file(default)}")


def main(default_precisions: tuple[str, ...] = TEXT_PRECISIONS, description: str | None = None):
    parser = argparse.ArgumentParser(
        description=description or "Export LFM2 models to ONNX for onnxruntime-genai",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    add_export_arguments(parser, ALL_PRECISIONS, default_precisions, "LiquidAI/LFM2.5-350M")
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Reuse the existing fp32 onnx/model.onnx instead of rebuilding it",
    )
    parser.add_argument(
        "--q4-asymmetric",
        action="store_true",
        help="Use asymmetric int4 for MatMul weights. Default is symmetric",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    precisions = parse_precisions(parser, args, ALL_PRECISIONS, default_precisions)
    export_dir = output_dir(args)
    onnx_dir = export_dir / "onnx"

    if args.skip_export:
        for required in (onnx_dir / "model.onnx", export_dir / "genai_config.json"):
            if not required.exists():
                parser.error(f"--skip-export needs an existing {required}")
    else:
        log_step(f"Exporting {args.model} (fp32) to {export_dir}")
        export_dir.mkdir(parents=True, exist_ok=True)
        export_decoder(args.model, export_dir)

    for precision in precisions:
        log_step(f"Deriving {precision}")
        derive_precision(
            onnx_dir,
            precision,
            block_size=args.block_size,
            q4_symmetric=not args.q4_asymmetric,
            reuse_q4="q4" in precisions,
        )

    # Rebuilding fp32 makes precisions from earlier runs stale; --skip-export keeps them valid.
    available = precisions
    if args.skip_export:
        available = [p for p in ALL_PRECISIONS if (onnx_dir / model_file(p)).exists()]
    set_default_decoder(export_dir, available)

    finish(args, export_dir)


if __name__ == "__main__":
    main()
