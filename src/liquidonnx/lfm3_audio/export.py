#!/usr/bin/env python3
"""
Export LFM3-Audio to ONNX for onnxruntime: a speech encoder, an LFM3 decoder that scatters speech
features and speech frames into its token embeddings, a MaskGIT depth decoder and the Liquid codec
decoder. onnxruntime-genai has no LFM3 model type, so the folder has no genai_config.json;
lfm3-audio-infer runs it.

    {output-dir}/exports/{model-name}-ONNX/
    ├── config.json, generation_config.json, tokenizer.json, tokenizer_config.json,
    │   special_tokens_map.json, chat_template.jinja, preprocessor_config.json,
    │   processor_config.json, {audio_detokenizer|audio_tokenizer}/config.json
    │                                                   (the checkpoint's, verbatim)
    └── onnx/
        ├── decoder.onnx, audio_encoder.onnx, depth_decoder.onnx, codec_decoder.onnx   fp32
        ├── decoder_fp16.onnx, audio_encoder_fp16.onnx,
        │   codec_decoder_fp16.onnx                         --precision fp16 (and the CUDA bundles)
        ├── decoder_q8.onnx                                 --precision q8
        ├── decoder_q4.onnx                                 --precision q4
        └── decoder_q8f16.onnx, decoder_q4f16.onnx          --precision q8f16 / q4f16: fp16
                                                            activations for CUDA

session.BUNDLES names the files of each precision (every bundle runs the fp32 depth decoder);
derive writes only those the requested precisions need. The decoders are derived from
decoder.onnx as LFM3-VL's are (lfm3.export.convert_to_fp16 with decoder.KEEP_IO_TYPES,
lfm3.export.derive_quantized); the encoder and codec with
liquidonnx.quantize.convert_to_fp16(keep_io=True).

Usage:
    uv run lfm3-audio-export LiquidAI/LFM3-Audio-2B --precision
    uv run lfm3-audio-export LiquidAI/LFM3-Omni-2B --precision q4 q4f16
    uv run lfm3-audio-export LiquidAI/LFM3-Audio-2B --precision q4 --skip-export
"""

import argparse
import json
import logging
import pathlib
import shutil

from liquidonnx import export_cli
from liquidonnx.genai_builder import resolve_checkpoint
from liquidonnx.lfm3.export import (
    BLOCK_SIZE_HELP,
    check_q4_block_size,
    convert_to_fp16,
    derive_quantized,
)
from liquidonnx.lfm3_audio.codec import LFM3AudioCodecBuilder
from liquidonnx.lfm3_audio.config import REVISIONS, codec_dir, load_config
from liquidonnx.lfm3_audio.decoder import KEEP_IO_TYPES, LFM3AudioDecoderBuilder
from liquidonnx.lfm3_audio.depth import LFM3AudioDepthBuilder
from liquidonnx.lfm3_audio.encoder import LFM3AudioEncoderBuilder
from liquidonnx.lfm3_audio.session import BUNDLES
from liquidonnx.quantize import convert_to_fp16 as convert_side_graph_to_fp16
from liquidonnx.quantize import save_model

logger = logging.getLogger(__name__)

PRECISIONS = ("fp16", "q4", "q8", "q4f16", "q8f16")
# fnmatch's "*" crosses "/": "*.json" also takes the codec folder's config.json
CHECKPOINT_PATTERNS = ["*.json", "*.jinja", "model*.safetensors", "*/model*.safetensors"]
COPIED_FILES = (
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)
# The depth decoder's q8 (LFM2.5-Audio's depthformer recipe; its code tables and heads stay
# float). No bundle takes it (session.BUNDLES); test_depth.py scores it.
DEPTH_QUANT = {"bits": 8, "symmetric": True, "accuracy_level": 4}
# The fp32 graphs every precision derives from
FP32_FILES = ("decoder.onnx", "audio_encoder.onnx", "depth_decoder.onnx", "codec_decoder.onnx")


def export_fp32(checkpoint: pathlib.Path, onnx_dir: pathlib.Path):
    config = load_config(checkpoint)
    # lfm3_omni's decoder also scatters image and video features
    raw = json.loads((checkpoint / "config.json").read_text())
    vision = [raw[key] for key in ("image_token_id", "video_token_id") if key in raw]
    builders = {
        "audio_encoder.onnx": LFM3AudioEncoderBuilder(config),
        "depth_decoder.onnx": LFM3AudioDepthBuilder(config),
        "codec_decoder.onnx": LFM3AudioCodecBuilder(config),
        "decoder.onnx": LFM3AudioDecoderBuilder(config, *vision),
    }
    for name, builder in builders.items():
        logger.info(f"Building {name}...")
        model = builder.build(checkpoint)
        save_model(model, onnx_dir / name)
        del model


def derive(onnx_dir: pathlib.Path, precisions: list[str], block_size: int):
    """The fp16 and quantized files of the requested bundles (session.BUNDLES), from the fp32
    graphs. No other file of the folder changes: the q4 or q8 decoder that q4f16 or q8f16 is
    converted from is a scratch file unless its own bundle is requested (derive_quantized)."""
    files = {name for precision in precisions for name in BUNDLES[precision].values()}
    decoder = onnx_dir / "decoder.onnx"
    if "decoder_fp16.onnx" in files:
        # fp16 caches: they stay in the graph's dtype, so the KV buffers can be shared in place
        convert_to_fp16(decoder, onnx_dir / "decoder_fp16.onnx", keep_io_types=KEEP_IO_TYPES)
    for graph in ("audio_encoder", "codec_decoder"):
        if f"{graph}_fp16.onnx" in files:
            convert_side_graph_to_fp16(
                onnx_dir / f"{graph}.onnx", onnx_dir / f"{graph}_fp16.onnx", keep_io=True
            )
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
            )


def copy_checkpoint_files(checkpoint: pathlib.Path, output_dir: pathlib.Path):
    """COPIED_FILES that exist, and the codec folder's config.json under the same folder name, so
    config.load_config works on the export folder."""
    for name in COPIED_FILES:
        if (checkpoint / name).exists():
            shutil.copy2(checkpoint / name, output_dir / name)
    codec = codec_dir(checkpoint)
    (output_dir / codec.name).mkdir(exist_ok=True)
    shutil.copy2(codec / "config.json", output_dir / codec.name / "config.json")


def main():
    parser = argparse.ArgumentParser(
        description="Export LFM3-Audio to ONNX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    export_cli.add_export_arguments(
        parser,
        PRECISIONS,
        PRECISIONS,
        "LiquidAI/LFM3-Audio-2B",
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
        p for p in CHECKPOINT_PATTERNS if not (args.skip_export and p.endswith(".safetensors"))
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
