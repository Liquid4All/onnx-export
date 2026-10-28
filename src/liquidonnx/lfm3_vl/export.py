#!/usr/bin/env python3
"""
Export LFM3-VL to ONNX for onnxruntime: a vision encoder and an LFM3 decoder that scatters the
visual features into its token embeddings. onnxruntime-genai has no LFM3 model type, so the folder
has no genai_config.json; lfm3-vl-infer runs it.

    {output-dir}/exports/{model-name}-ONNX/
    ├── config.json, generation_config.json, tokenizer.json, tokenizer_config.json,
    │   special_tokens_map.json, chat_template.jinja, preprocessor_config.json,
    │   video_preprocessor_config.json, processor_config.json     (the checkpoint's, verbatim)
    └── onnx/
        ├── decoder.onnx, vision_encoder.onnx                      fp32
        ├── decoder_fp16.onnx, vision_encoder_fp16.onnx            --precision fp16
        ├── decoder_q8.onnx                                        --precision q8 (with fp16 vision)
        ├── decoder_q4.onnx, vision_encoder_q8.onnx                --precision q4
        └── decoder_q8f16.onnx, decoder_q4f16.onnx                 --precision q8f16 / q4f16: fp16
                                                                   activations for CUDA (fp16 vision)

The quantized decoders have an int8 LM head, whose weights are also their token table. The rest
of q8 and q8f16 is int8; q4 and q4f16 are k_quant int4 with the layers most sensitive to int4 in
int8, the recipe of the LFM2 q4 decoders (liquidonnx.lfm3.quantize).

Usage:
    uv run lfm3-vl-export LiquidAI/LFM3-VL-2B --precision
    uv run lfm3-vl-export LiquidAI/LFM3-VL-2B --precision q4 --skip-export
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
from liquidonnx.lfm3_vl.config import REVISIONS, parse_config
from liquidonnx.lfm3_vl.decoder import LFM3VLDecoderBuilder
from liquidonnx.lfm3_vl.session import BUNDLES
from liquidonnx.lfm3_vl.vision import LFM3VLVisionBuilder
from liquidonnx.lfm3_vl.weights import load_tensors
from liquidonnx.quantize import convert_to_fp16 as convert_vision_to_fp16
from liquidonnx.quantize import quantize_model, save_model

logger = logging.getLogger(__name__)

PRECISIONS = ("fp16", "q4", "q8", "q4f16", "q8f16")
CHECKPOINT_PATTERNS = ["*.json", "*.jinja", "model*.safetensors"]
COPIED_FILES = (
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "processor_config.json",
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)
# The q4 bundle's vision encoder: int8 weights in blocks of 128, symmetric, with float activations
# (accuracy_level 0), the values of LFM2-VL's. Its answer-token KL from the fp32 reference, with
# the fp32 decoder, is far above fp16 vision's and hidden by the q4 decoder's own error on
# test_decoder's image prompts. int8 activations (accuracy_level 4) double it and drop the worst
# image token to a low cosine.
VISION_QUANT = {"bits": 8, "block_size": 128, "symmetric": True, "accuracy_level": 0}


def load_config(checkpoint: pathlib.Path):
    return parse_config(json.loads((checkpoint / "config.json").read_text()))


def export_fp32(checkpoint: pathlib.Path, onnx_dir: pathlib.Path):
    language, vision, tokens = load_config(checkpoint)
    logger.info("Building the vision encoder...")
    vision_model = LFM3VLVisionBuilder(vision).build(load_tensors(checkpoint, "vision_"))
    save_model(vision_model, onnx_dir / "vision_encoder.onnx")
    del vision_model
    logger.info("Building the decoder...")
    decoder = LFM3VLDecoderBuilder(language, tokens.image, tokens.video).build(checkpoint)
    save_model(decoder, onnx_dir / "decoder.onnx")


def derive(onnx_dir: pathlib.Path, precisions: list[str], block_size: int):
    """The fp16 and quantized files of the requested bundles, from the fp32 graphs. No other
    file of the folder changes: the q4 or q8 decoder that q4f16 or q8f16 is converted from is a
    scratch file unless its own bundle is requested (derive_quantized)."""
    files = {name for precision in precisions for name in BUNDLES[precision]}
    decoder = onnx_dir / "decoder.onnx"
    if "decoder_fp16.onnx" in files:
        # fp16 caches: they stay in the graph's dtype, so the KV buffers can be shared in place
        convert_to_fp16(
            decoder,
            onnx_dir / "decoder_fp16.onnx",
            keep_io_types=["image_features", "video_features", "logits"],
        )
    derive_vision(onnx_dir, files)
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


def derive_vision(onnx_dir: pathlib.Path, files: set[str]):
    """vision_encoder_fp16.onnx and vision_encoder_q8.onnx from vision_encoder.onnx, those of
    them in files."""
    vision = onnx_dir / "vision_encoder.onnx"
    if "vision_encoder_fp16.onnx" in files:
        convert_vision_to_fp16(vision, onnx_dir / "vision_encoder_fp16.onnx", keep_io=True)
    if "vision_encoder_q8.onnx" in files:
        quantize_model(
            vision, onnx_dir / "vision_encoder_q8.onnx", exclude_lm_head=False, **VISION_QUANT
        )


def copy_checkpoint_files(checkpoint: pathlib.Path, output_dir: pathlib.Path):
    for name in COPIED_FILES:
        if (checkpoint / name).exists():
            shutil.copy2(checkpoint / name, output_dir / name)


def main():
    parser = argparse.ArgumentParser(
        description="Export LFM3-VL to ONNX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    export_cli.add_export_arguments(
        parser,
        PRECISIONS,
        PRECISIONS,
        "LiquidAI/LFM3-VL-2B",
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
        missing = [
            f for f in ("decoder.onnx", "vision_encoder.onnx") if not (onnx_dir / f).exists()
        ]
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
