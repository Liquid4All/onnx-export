#!/usr/bin/env python3
"""
Export d1 (LiquidAI/d1-2B, LiquidAI/d1-Omni-2B) to ONNX for onnxruntime: LFM3-VL's vision encoder
(lfm3_vl.vision, unchanged), d1-Omni's speech encoder (lfm3_audio.encoder, unchanged) and a
one-pass decoder under d1's set attention (d1.decoder). d1-infer runs it, with the checkpoint's
own host code (prompts, packing, readouts, calibration), which the folder carries.

    {output-dir}/exports/{model-name}-ONNX/
    ├── config.json, processor_config.json, tokenizer.json, tokenizer_config.json,
    │   chat_template.jinja, calibration.json, LICENSE, *.py   (the checkpoint's, verbatim)
    └── onnx/
        ├── decoder.onnx, vision_encoder.onnx, audio_encoder.onnx (d1-Omni)   fp32
        ├── decoder_fp16.onnx, vision_encoder_fp16.onnx, audio_encoder_fp16.onnx (d1-Omni)
        │                                                   --precision fp16 (and the CUDA bundles)
        ├── decoder_q8.onnx                                 --precision q8
        ├── decoder_q4.onnx, vision_encoder_q8.onnx         --precision q4
        └── decoder_q8f16.onnx, decoder_q4f16.onnx          --precision q8f16 / q4f16: fp16
                                                            activations for CUDA

session.BUNDLES names the files of each precision, as LFM3-VL's and LFM3-Omni's; derive writes
only those the requested precisions need. d1-Omni's depth decoder, codebook embedding and speech
codec are not exported: d1 never generates. The q4 decoders keep only the MLP gate_proj and
up_proj in int4 (lfm3.quantize.gate_up_int4_int8_nodes, README 3.1f).

Usage:
    uv run d1-export LiquidAI/d1-2B --precision
    uv run d1-export LiquidAI/d1-Omni-2B --precision q4 --skip-export
"""

import argparse
import logging
import pathlib
import shutil

from liquidonnx import export_cli
from liquidonnx.d1.config import REVISIONS, load_config
from liquidonnx.d1.decoder import KEEP_IO_TYPES, D1DecoderBuilder
from liquidonnx.d1.session import BUNDLES
from liquidonnx.genai_builder import resolve_checkpoint
from liquidonnx.lfm3.export import (
    BLOCK_SIZE_HELP,
    check_q4_block_size,
    convert_to_fp16,
    derive_quantized,
)
from liquidonnx.lfm3.quantize import gate_up_int4_int8_nodes
from liquidonnx.lfm3_audio.encoder import LFM3AudioEncoderBuilder
from liquidonnx.lfm3_audio.weights import load_weights as load_audio_weights
from liquidonnx.lfm3_vl.export import derive_vision
from liquidonnx.lfm3_vl.vision import LFM3VLVisionBuilder
from liquidonnx.lfm3_vl.weights import load_tensors
from liquidonnx.quantize import convert_to_fp16 as convert_side_graph_to_fp16
from liquidonnx.quantize import save_model

logger = logging.getLogger(__name__)

PRECISIONS = ("fp16", "q4", "q8", "q4f16", "q8f16")
# The host code (*.py) and its data run beside the graphs (d1.session)
COPIED_PATTERNS = ("*.json", "*.jinja", "*.py", "LICENSE")
CHECKPOINT_PATTERNS = [*COPIED_PATTERNS, "model*.safetensors"]


def fp32_files(onnx_dir: pathlib.Path) -> list[str]:
    files = ["decoder.onnx", "vision_encoder.onnx"]
    return files + ["audio_encoder.onnx"] * (load_config(onnx_dir.parent).audio is not None)


def export_fp32(checkpoint: pathlib.Path, onnx_dir: pathlib.Path):
    config = load_config(checkpoint)
    logger.info("Building vision_encoder.onnx...")
    if config.audio is None:
        weights = load_tensors(checkpoint, "vision_")
    else:
        weights = load_audio_weights(checkpoint, "vision", config.audio)
    save_model(LFM3VLVisionBuilder(config.vision).build(weights), onnx_dir / "vision_encoder.onnx")
    del weights
    builders = {"decoder.onnx": D1DecoderBuilder(config)}
    if config.audio is not None:
        builders["audio_encoder.onnx"] = LFM3AudioEncoderBuilder(config.audio)
    for name, builder in builders.items():
        logger.info(f"Building {name}...")
        save_model(builder.build(checkpoint), onnx_dir / name)


def derive(onnx_dir: pathlib.Path, precisions: list[str], block_size: int):
    """The fp16 and quantized files of the requested bundles (session.BUNDLES) that the folder's
    fp32 graphs make: as lfm3_vl.export.derive and lfm3_audio.export.derive do, with every d1
    checkpoint's q4 decoders on gate_up_int4_int8_nodes."""
    files = {name for precision in precisions for name in BUNDLES[precision].values()}
    decoder = onnx_dir / "decoder.onnx"
    if "decoder_fp16.onnx" in files:
        convert_to_fp16(decoder, onnx_dir / "decoder_fp16.onnx", keep_io_types=KEEP_IO_TYPES)
    derive_vision(onnx_dir, files)
    audio = onnx_dir / "audio_encoder.onnx"
    if "audio_encoder_fp16.onnx" in files and audio.exists():
        convert_side_graph_to_fp16(audio, onnx_dir / "audio_encoder_fp16.onnx", keep_io=True)
    for bits in (4, 8):
        quantized, f16 = f"decoder_q{bits}.onnx", f"decoder_q{bits}f16.onnx"
        if files & {quantized, f16}:
            derive_quantized(
                decoder,
                onnx_dir / quantized,
                bits,
                block_size,
                keep=quantized in files,
                fp16_activations=f16 in files,
                int8_nodes=gate_up_int4_int8_nodes,
            )


def copy_checkpoint_files(checkpoint: pathlib.Path, output_dir: pathlib.Path):
    for pattern in COPIED_PATTERNS:
        for path in checkpoint.glob(pattern):
            # not the cache's read-only modes, which would fail the next run's copy
            shutil.copyfile(path, output_dir / path.name)


def main():
    parser = argparse.ArgumentParser(
        description="Export d1 to ONNX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    export_cli.add_export_arguments(
        parser, PRECISIONS, PRECISIONS, "LiquidAI/d1-2B", block_size_help=BLOCK_SIZE_HELP
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Reuse the folder's fp32 graphs instead of rebuilding them",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    precisions = export_cli.parse_precisions(parser, args, PRECISIONS, PRECISIONS)
    output_dir = export_cli.output_dir(args)
    onnx_dir = output_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    # The weights are read for the fp32 graphs only
    patterns = [
        p for p in CHECKPOINT_PATTERNS if not (args.skip_export and p.endswith(".safetensors"))
    ]
    repo, revision = args.model, REVISIONS.get(args.model)
    checkpoint = resolve_checkpoint(repo, patterns, revision=revision)
    copy_checkpoint_files(checkpoint, output_dir)
    if args.skip_export:
        missing = [f for f in fp32_files(onnx_dir) if not (onnx_dir / f).exists()]
        if missing:
            parser.error(f"--skip-export needs {', '.join(missing)} in {onnx_dir}")
    else:
        export_cli.log_step(f"Exporting {args.model} (fp32)")
        export_fp32(checkpoint, onnx_dir)
    if {"q4", "q4f16"} & set(precisions):
        # Up front with --skip-export, else before any other precision is derived
        check_q4_block_size(
            parser, onnx_dir / "decoder.onnx", args.block_size, gate_up_int4_int8_nodes
        )
    if precisions:
        export_cli.log_step(f"Deriving {', '.join(precisions)}")
    derive(onnx_dir, precisions, args.block_size)
    export_cli.finish(args, output_dir, precisions, REVISIONS)


if __name__ == "__main__":
    main()
