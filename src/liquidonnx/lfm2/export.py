#!/usr/bin/env python3
"""
Export LFM2 and LFM2-MoE models to ONNX for onnxruntime-genai.

The onnxruntime-genai model builder (CPU EP, see liquidonnx.genai_builder) builds the fp32, q4,
q4f32 and q8 decoders, one run each; fp16 and q4f16 are the fp32 and q4 decoders converted to fp16.

Output Structure:
    {output-dir}/exports/{model-name}-ONNX/
        ├── genai_config.json        # decoder points at the default precision
        ├── config.json, generation_config.json
        ├── tokenizer.json, tokenizer_config.json, chat_template.jinja
        └── onnx/
            ├── model.onnx           # fp32
            ├── model_fp16.onnx      # fp16 weights, activations and caches; fp32 logits
            ├── model_q4.onnx        # int4 (k_quant); int8 lm_head, tied table, sensitive layers
            ├── model_q4f16.onnx     # q4 with fp16 activations
            ├── model_q4f32.onnx     # int4 MatMuls; fp32 embedding and lm_head
            ├── model_q8.onnx        # int8, lm_head and tied embedding table included
            └── *.onnx_data          # weights in --split-data chunks: .onnx_data, .onnx_data_1, ...

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

    # Add precisions to an existing export without rebuilding fp32; q4f16 converts the export's
    # model_q4.onnx when it is of the same checkpoint and --block-size (see reusable_q4)
    uv run lfm2-export LiquidAI/LFM2.5-350M --precision fp16 q4f16 --skip-export
"""

import argparse
import json
import logging
import pathlib

import numpy as np
import onnx
from onnx import numpy_helper

from liquidonnx.export_cli import (
    add_export_arguments,
    finish,
    log_step,
    output_dir,
    parse_precisions,
)
from liquidonnx.genai_builder import export_decoder, export_precision

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


def reusable_q4(onnx_dir: pathlib.Path, block_size: int) -> bool:
    """Whether q4f16 can convert the existing onnx/model_q4.onnx instead of building a fresh q4.

    --skip-export trusts the folder's fp32 decoder, so the q4 is reused when it matches that one:
    the tensors both keep unquantized (norms, conv kernels, rotary caches) are equal, which a q4
    left over from another checkpoint fails, and its blocks are block_size. Like model.onnx, the
    q4 is trusted to come from this exporter; its version is not checked.
    """
    q4_path, fp32_path = onnx_dir / model_file("q4"), onnx_dir / model_file("fp32")
    if not q4_path.exists():
        return False
    q4, fp32 = (onnx.load(path, load_external_data=False).graph for path in (q4_path, fp32_path))
    sizes = {a.i for node in q4.node for a in node.attribute if a.name == "block_size"}
    fp32_tensors = {t.name: t for t in fp32.initializer}
    shared = [(t, fp32_tensors[t.name]) for t in q4.initializer if t.name in fp32_tensors]

    def value(tensor: onnx.TensorProto) -> np.ndarray:
        return numpy_helper.to_array(tensor, str(onnx_dir))

    try:
        if sizes != {block_size}:
            reason = f"has block sizes {sorted(sizes)}, not {block_size}"
        elif not shared or not all(np.array_equal(value(a), value(b)) for a, b in shared):
            reason = f"has unquantized tensors that differ from {fp32_path.name}"
        else:
            logger.info(f"Converting the existing {q4_path.name} to q4f16")
            return True
    except onnx.checker.ValidationError as error:
        # onnx loads no external data file with several hard links (a `cp -al` copy); such a
        # folder keeps the fresh q4 build it had before q4 reuse rather than failing.
        reason = f"is not loadable: {error}"
    logger.info(f"Building a fresh q4 for q4f16: {q4_path.name} {reason}")
    return False


def main(
    default_precisions: tuple[str, ...] = TEXT_PRECISIONS,
    description: str | None = None,
    family: str = "lfm2",
):
    parser = argparse.ArgumentParser(
        description=description or "Export LFM2 models to ONNX for onnxruntime-genai",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    add_export_arguments(parser, ALL_PRECISIONS, default_precisions, "LiquidAI/LFM2.5-350M")
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Reuse the existing fp32 onnx/model.onnx instead of rebuilding it; q4f16 also "
        "converts an existing onnx/model_q4.onnx of the same checkpoint and --block-size",
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

    # A q4 from an earlier run is stale once fp32 is rebuilt, so q4f16 converts only a q4 built in
    # this run; --skip-export trusts the folder, so it also converts a q4 that matches it.
    reuse_q4 = "q4" in precisions or (
        args.skip_export and "q4f16" in precisions and reusable_q4(onnx_dir, args.block_size)
    )
    for precision in precisions:
        log_step(f"Exporting {precision}")
        export_precision(
            args.model,
            export_dir,
            family,
            precision,
            block_size=args.block_size,
            reuse_q4=reuse_q4,
        )

    # Rebuilding fp32 makes precisions from earlier runs stale; --skip-export keeps them valid.
    available = precisions
    if args.skip_export:
        available = [p for p in ALL_PRECISIONS if (onnx_dir / model_file(p)).exists()]
    set_default_decoder(export_dir, available)

    finish(args, export_dir)


if __name__ == "__main__":
    main()
