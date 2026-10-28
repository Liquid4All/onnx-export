#!/usr/bin/env python3
"""
Export LFM3-Omni (LiquidAI/LFM3-Omni-2B) to ONNX for onnxruntime: LFM3-Audio's graphs
(lfm3_audio.export: speech encoder, depth decoder, codec decoder, and a decoder that also scatters
image and video features) and LFM3-VL's vision encoder (lfm3_vl.vision, unchanged).
onnxruntime-genai has no LFM3 model type, so the folder has no genai_config.json; lfm3-omni-infer
runs it.

    {output-dir}/exports/{model-name}-ONNX/
    ├── config.json, generation_config.json, tokenizer.json, tokenizer_config.json,
    │   chat_template.jinja, processor_config.json, video_preprocessor_config.json,
    │   audio_tokenizer/config.json                      (the checkpoint's, verbatim)
    ├── preprocessor_config.json                         (processor_config.json's image_processor)
    └── onnx/
        ├── decoder.onnx, vision_encoder.onnx, audio_encoder.onnx, depth_decoder.onnx,
        │   codec_decoder.onnx                              fp32
        ├── decoder_fp16.onnx, vision_encoder_fp16.onnx, audio_encoder_fp16.onnx,
        │   codec_decoder_fp16.onnx                         --precision fp16 (and the CUDA bundles)
        ├── decoder_q8.onnx                                 --precision q8
        ├── decoder_q4.onnx, vision_encoder_q8.onnx         --precision q4
        └── decoder_q8f16.onnx, decoder_q4f16.onnx          --precision q8f16 / q4f16: fp16
                                                            activations for CUDA

session.BUNDLES names the files of each precision; derive writes only those the requested
precisions need. The mel front end's settings stay in processor_config.json's feature_extractor,
from which the runtime computes the filters (lfm3_audio.session.mel_filters).

Usage:
    uv run lfm3-omni-export LiquidAI/LFM3-Omni-2B --precision
    uv run lfm3-omni-export LiquidAI/LFM3-Omni-2B --precision q4 --skip-export
"""

import argparse
import json
import logging
import pathlib
import shutil

from liquidonnx import export_cli
from liquidonnx.genai_builder import resolve_checkpoint
from liquidonnx.lfm3.export import BLOCK_SIZE_HELP, check_q4_block_size
from liquidonnx.lfm3_audio import export as audio_export
from liquidonnx.lfm3_audio.config import REVISIONS
from liquidonnx.lfm3_audio.weights import load_weights
from liquidonnx.lfm3_omni.config import load_config
from liquidonnx.lfm3_omni.session import BUNDLES
from liquidonnx.lfm3_vl.export import derive_vision
from liquidonnx.lfm3_vl.vision import LFM3VLVisionBuilder
from liquidonnx.quantize import save_model

logger = logging.getLogger(__name__)

PRECISIONS = audio_export.PRECISIONS
FP32_FILES = (*audio_export.FP32_FILES, "vision_encoder.onnx")


def export_fp32(checkpoint: pathlib.Path, onnx_dir: pathlib.Path):
    config = load_config(checkpoint)
    logger.info("Building vision_encoder.onnx...")
    weights = load_weights(checkpoint, "vision", config.audio)
    model = LFM3VLVisionBuilder(config.vision).build(weights)
    save_model(model, onnx_dir / "vision_encoder.onnx")
    del model, weights
    audio_export.export_fp32(checkpoint, onnx_dir)


def derive(onnx_dir: pathlib.Path, precisions: list[str], block_size: int):
    """The fp16 and quantized files of the requested bundles (session.BUNDLES), from the fp32
    graphs: lfm3_audio.export.derive's and lfm3_vl.export.derive_vision's."""
    audio_export.derive(onnx_dir, precisions, block_size)
    derive_vision(onnx_dir, {BUNDLES[precision]["vision_encoder"] for precision in precisions})


def copy_checkpoint_files(checkpoint: pathlib.Path, output_dir: pathlib.Path):
    """lfm3_audio.export.copy_checkpoint_files', video_preprocessor_config.json, and a flat
    preprocessor_config.json of processor_config.json's image_processor, the file
    lfm3_vl.session.image_processor reads."""
    audio_export.copy_checkpoint_files(checkpoint, output_dir)
    if (checkpoint / "video_preprocessor_config.json").exists():
        shutil.copy2(checkpoint / "video_preprocessor_config.json", output_dir)
    if not (checkpoint / "preprocessor_config.json").exists():
        processor = json.loads((checkpoint / "processor_config.json").read_text())
        flat = json.dumps(processor["image_processor"], indent=2) + "\n"
        (output_dir / "preprocessor_config.json").write_text(flat)


def main():
    parser = argparse.ArgumentParser(
        description="Export LFM3-Omni to ONNX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    export_cli.add_export_arguments(
        parser,
        PRECISIONS,
        PRECISIONS,
        "LiquidAI/LFM3-Omni-2B",
        block_size_help=BLOCK_SIZE_HELP,
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
    if args.skip_export:
        missing = [f for f in FP32_FILES if not (onnx_dir / f).exists()]
        if missing:
            parser.error(f"--skip-export needs {', '.join(missing)} in {onnx_dir}")
    # The weights are read for the fp32 graphs only
    patterns = [
        p
        for p in audio_export.CHECKPOINT_PATTERNS
        if not (args.skip_export and p.endswith(".safetensors"))
    ]
    repo, revision = args.model, REVISIONS.get(args.model)
    checkpoint = resolve_checkpoint(repo, patterns, revision=revision)

    if not args.skip_export:
        export_cli.log_step(f"Exporting {args.model} (fp32)")
        export_fp32(checkpoint, onnx_dir)
    if {"q4", "q4f16"} & set(precisions):
        # Up front with --skip-export, else before any other precision is derived
        check_q4_block_size(parser, onnx_dir / "decoder.onnx", args.block_size)
    copy_checkpoint_files(checkpoint, output_dir)
    if precisions:
        export_cli.log_step(f"Deriving {', '.join(precisions)}")
    derive(onnx_dir, precisions, args.block_size)
    export_cli.finish(args, output_dir, precisions, REVISIONS)


if __name__ == "__main__":
    main()
