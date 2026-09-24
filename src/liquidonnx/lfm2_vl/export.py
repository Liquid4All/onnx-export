#!/usr/bin/env python3
"""
Export LFM2-VL models to ONNX for onnxruntime-genai.

The decoder comes from the onnxruntime-genai model builder (fp32, CPU EP, inputs_embeds in; see
liquidonnx.genai_builder). This repository builds the vision encoder and the embedding model that
splices image features into the token embeddings, and derives every precision.

Output Structure:
    {output-dir}/exports/{model-name}-ONNX/
        ├── genai_config.json            # decoder, embedding and vision at the default precision
        ├── genai_processor_config.json  # onnxruntime-genai image preprocessing
        ├── config.json, processor_config.json
        ├── tokenizer.json, tokenizer_config.json, chat_template.jinja
        └── onnx/
            ├── decoder.onnx             # fp32; decoder_{fp16,q4,q8}.onnx as in lfm2-export
            ├── vision_encoder.onnx      # fp32; vision_encoder_{fp16,q4,q8}.onnx
            ├── embeddings.onnx          # token table + image feature scatter (fp32 table)
            └── embeddings_fp16.onnx     # fp16 table, used with every other precision

genai_config.json uses the first exported precision of q4, q8, fp16, fp32; bundle() lists the
files each precision loads. onnxruntime-genai needs the tiled vision encoder, so a conv2d export
has no genai_config.json.

Usage:
    # Export from HuggingFace (fp32 only)
    uv run lfm2-vl-export LiquidAI/LFM2.5-VL-1.6B

    # Export with all precisions (fp16, q4, q8)
    uv run lfm2-vl-export LiquidAI/LFM2.5-VL-1.6B --precision

    # Export with specific precisions
    uv run lfm2-vl-export LiquidAI/LFM2-VL-450M --precision fp16 q4

    # Derive precisions from an existing fp32 export
    uv run lfm2-vl-export LiquidAI/LFM2-VL-450M --precision q8 --skip-export

    # Export with conv2d vision format (instead of default tiled), for plain onnxruntime
    uv run lfm2-vl-export LiquidAI/LFM2-VL-450M --vision-format conv2d
"""

import argparse
import gc
import json
import logging
import pathlib

import onnx

from liquidonnx.embeddings import build_embeddings, embeddings_to_fp16
from liquidonnx.external_data import split_external_data
from liquidonnx.genai_builder import export_decoder
from liquidonnx.lfm2_vl import VISION_MODE_CONV2D, VISION_MODE_TILED
from liquidonnx.lfm2_vl.builder import LFM2VLConfig, VisionEmbedBuilder
from liquidonnx.quantize import (
    DEFAULT_BLOCK_SIZE,
    derive_precision,
    get_total_model_size_mb,
    load_model,
    quantize_model,
    save_model,
)

logger = logging.getLogger(__name__)

PRECISIONS = ("fp16", "q4", "q8")
DEFAULT_ORDER = ("q4", "q8", "fp16")
GENAI_PROCESSOR_CONFIG = "genai_processor_config.json"
INTERPOLATION = {2: "LINEAR", 3: "CUBIC"}  # PIL resample -> onnxruntime-extensions Resize
# LFM2.5-VL-3B's pre-tokenizer pattern, which onnxruntime-extensions cannot parse, and the
# equivalent one the rest of the line uses (same tokens on chat prompts).
UNSUPPORTED_PATTERN = (
    r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]|\s+(?!\S)|\s"
)
SUPPORTED_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+|\s+(?!\S)|\s+"
)


def bundle(precision: str) -> dict[str, str]:
    """ONNX files (in onnx/) that make up one precision."""
    suffix = "" if precision == "fp32" else f"_{precision}"
    return {
        "decoder": f"decoder{suffix}.onnx",
        "vision": f"vision_encoder{suffix}.onnx",
        "embedding": "embeddings.onnx" if precision == "fp32" else "embeddings_fp16.onnx",
    }


def convert_vision_to_fp16(input_path: pathlib.Path, output_path: pathlib.Path) -> pathlib.Path:
    """fp16 vision encoder with fp32 inputs and outputs."""
    from onnxruntime.transformers.float16 import convert_float_to_float16

    logger.info(f"Converting {input_path.name} to FP16...")
    model = load_model(input_path)

    # The converter casts the empty roi/scales inputs of Resize, which breaks shape inference;
    # restore them after conversion.
    resize_orig_inputs = {n.name: list(n.input) for n in model.graph.node if n.op_type == "Resize"}
    empty_initializers = {}
    for init in model.graph.initializer:
        if "empty" in init.name.lower():
            empty_initializers[init.name] = onnx.TensorProto()
            empty_initializers[init.name].CopyFrom(init)

    # disable_shape_infer=True is required for models with com.microsoft custom ops
    model_fp16 = convert_float_to_float16(
        model, keep_io_types=True, force_fp16_initializers=True, disable_shape_infer=True
    )

    for node in model_fp16.graph.node:
        orig_inputs = resize_orig_inputs.get(node.name) if node.op_type == "Resize" else None
        if not orig_inputs:
            continue
        for index in (1, 2):  # roi, scales
            if (
                len(node.input) > index
                and len(orig_inputs) > index
                and "cast" in node.input[index].lower()
                and "empty" in orig_inputs[index].lower()
            ):
                node.input[index] = orig_inputs[index]
    for init in model_fp16.graph.initializer:
        if init.name in empty_initializers:
            init.CopyFrom(empty_initializers[init.name])

    save_model(model_fp16, output_path)
    orig_mb = get_total_model_size_mb(input_path)
    fp16_mb = get_total_model_size_mb(output_path)
    logger.info(f"  {input_path.name}: {orig_mb:.1f} -> {fp16_mb:.1f} MB")
    return output_path


def get_model_name(model_path: str) -> str:
    return pathlib.Path(model_path).name


def export_vl_model(
    model_path: str,
    output_dir: pathlib.Path,
    vision_input_format: str = VISION_MODE_TILED,
):
    """fp32 decoder, vision encoder and embedding model, plus config, processor and tokenizer."""
    import torch
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

    onnx_dir = output_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    vl_config = LFM2VLConfig.from_hf_config(config)

    logger.info(f"Loading weights from {model_path}...")
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, torch_dtype=torch.float32, trust_remote_code=True
    )
    weights = {name: param.detach().numpy() for name, param in model.named_parameters()}
    del model
    gc.collect()

    # === 1. Vision encoder (SigLIP2 + projector) ===
    logger.info(f"Exporting vision_encoder [{vision_input_format} mode]...")
    vision_builder = VisionEmbedBuilder(vl_config, vision_input_format=vision_input_format)
    vision_builder.load_weights(weights)
    save_model(vision_builder.build(), onnx_dir / "vision_encoder.onnx")
    del vision_builder
    gc.collect()

    # === 2. Embedding model ===
    logger.info("Exporting embeddings...")
    build_embeddings(
        weights["model.language_model.embed_tokens.weight"],
        vl_config.image_token_id,
        "image_features",
        onnx_dir / "embeddings.onnx",
    )
    weights.clear()
    gc.collect()

    # === 3. Decoder (onnxruntime-genai model builder) ===
    export_decoder(model_path, output_dir, "decoder.onnx", {"exclude_embeds": "true"})

    # === 4. Processor and tokenizer ===
    AutoProcessor.from_pretrained(model_path, trust_remote_code=True).save_pretrained(output_dir)
    fix_tokenizer_pattern(output_dir / "tokenizer.json")


def fix_tokenizer_pattern(tokenizer_path: pathlib.Path):
    """Swap a pre-tokenizer pattern onnxruntime-extensions cannot parse for its equivalent."""
    text = tokenizer_path.read_text()
    unsupported = json.dumps(UNSUPPORTED_PATTERN)
    if unsupported in text:
        tokenizer_path.write_text(text.replace(unsupported, json.dumps(SUPPORTED_PATTERN)))
        logger.info("Replaced the pre-tokenizer pattern onnxruntime-extensions cannot parse")


def derive_precision_files(
    onnx_dir: pathlib.Path,
    precision: str,
    block_size: int = DEFAULT_BLOCK_SIZE,
    q4_symmetric: bool = True,
):
    """Write the files bundle(precision) loads."""
    files = bundle(precision)
    derive_precision(
        onnx_dir, precision, name="decoder", block_size=block_size, q4_symmetric=q4_symmetric
    )

    vision = onnx_dir / files["vision"]
    if precision == "fp16":
        convert_vision_to_fp16(onnx_dir / "vision_encoder.onnx", vision)
    else:
        bits = int(precision[1])
        quantize_model(
            onnx_dir / "vision_encoder.onnx",
            vision,
            bits=bits,
            block_size=block_size,
            exclude_lm_head=False,
            symmetric=bits == 4 and q4_symmetric,
        )

    embeddings_to_fp16(onnx_dir / "embeddings.onnx", onnx_dir / files["embedding"])


def genai_processor_config(image_processor) -> dict:
    """onnxruntime-genai's lfm2_vl image processor, configured like the upstream processor.

    Images are resized once (no tiling), to between min_image_tokens and max_image_tokens.
    """
    patch = image_processor.encoder_patch_size
    merge = image_processor.downsample_factor
    pixels_per_token = patch**2 * merge**2
    resize = {
        "height": 512,
        "width": 512,
        "interpolation": INTERPOLATION[int(image_processor.resample)],
        "smart_resize": 1,
        "min_pixels": image_processor.min_image_tokens * pixels_per_token,
        "max_pixels": image_processor.max_image_tokens * pixels_per_token,
        "patch_size": patch,
        "merge_size": merge,
    }
    transforms = [
        ("decode_image", "DecodeImage", {"color_space": "RGB"}),
        ("resize", "Resize", resize),
        ("rescale", "Rescale", {"rescale_factor": image_processor.rescale_factor}),
        (
            "normalize",
            "Normalize",
            {"mean": list(image_processor.image_mean), "std": list(image_processor.image_std)},
        ),
        ("to_channel_first", "Permute3D", {"dims": [2, 0, 1]}),
        ("image_sizes", "PixtralImageSizes", None),
    ]
    return {
        "processor": {
            "name": "lfm2_vl_image_processor",
            "transforms": [
                {"operation": {"name": name, "type": kind, **({"attrs": attrs} if attrs else {})}}
                for name, kind, attrs in transforms
            ],
        }
    }


def write_genai_config(output_dir: pathlib.Path, precision: str):
    """Point genai_config.json at one precision and add the embedding and vision sections."""
    from transformers import AutoProcessor

    image_processor = AutoProcessor.from_pretrained(output_dir).image_processor
    (output_dir / GENAI_PROCESSOR_CONFIG).write_text(
        json.dumps(genai_processor_config(image_processor), indent=4)
    )

    files = {k: f"onnx/{v}" for k, v in bundle(precision).items()}
    config_path = output_dir / "genai_config.json"
    config = json.loads(config_path.read_text())
    model = config["model"]
    model["decoder"]["filename"] = files["decoder"]
    model["embedding"] = {
        "filename": files["embedding"],
        "inputs": {"input_ids": "input_ids", "image_features": "image_features"},
        "outputs": {"inputs_embeds": "inputs_embeds"},
    }
    merge = image_processor.downsample_factor
    model["vision"] = {
        "filename": files["vision"],
        "config_filename": GENAI_PROCESSOR_CONFIG,
        "patch_size": image_processor.encoder_patch_size,
        "spatial_merge_size": merge,
        "max_num_patches": image_processor.max_image_tokens * merge**2,
        "inputs": {
            "pixel_values": "pixel_values",
            "attention_mask": "pixel_attention_mask",
            "image_sizes": "spatial_shapes",
        },
        "outputs": {"image_features": "image_features"},
    }
    config_path.write_text(json.dumps(config, indent=4))
    logger.info(f"genai_config.json -> {precision}: {', '.join(files.values())}")


def main():
    parser = argparse.ArgumentParser(
        description="Export LFM2-VL models to ONNX for onnxruntime-genai",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "model",
        help="HuggingFace model ID or local path (e.g., LiquidAI/LFM2-VL-450M)",
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
        "--precision",
        nargs="*",
        metavar="PRECISION",
        help=f"Output precisions: {', '.join(PRECISIONS)} (no value: all)",
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Reuse the existing fp32 graphs instead of rebuilding them",
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
        "--vision-format",
        choices=[VISION_MODE_TILED, VISION_MODE_CONV2D],
        default=VISION_MODE_TILED,
        help="Vision encoder format: tiled (default) or conv2d (plain onnxruntime only)",
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

    precisions = [] if args.precision is None else [p.lower() for p in args.precision]
    if args.precision == []:
        precisions = list(PRECISIONS)
    for p in precisions:
        if p not in PRECISIONS:
            parser.error(f"Invalid precision: {p}. Use {', '.join(PRECISIONS)}.")

    model_name = get_model_name(args.model)
    vision_suffix = f"-{args.vision_format}" if args.vision_format != VISION_MODE_TILED else ""
    output_name = args.output_name or f"{model_name}-ONNX{vision_suffix}"
    output_dir = args.output_dir / "exports" / output_name
    onnx_dir = output_dir / "onnx"

    if args.skip_export:
        required = [onnx_dir / f for f in bundle("fp32").values()]
        if args.vision_format == VISION_MODE_TILED:
            required.append(output_dir / "genai_config.json")
        for path in required:
            if not path.exists():
                parser.error(f"--skip-export needs an existing {path}")
    else:
        logger.info("=" * 60)
        logger.info(f"Exporting {args.model} (fp32) to {output_dir}")
        logger.info("=" * 60)
        export_vl_model(args.model, output_dir, args.vision_format)

    for precision in precisions:
        logger.info("=" * 60)
        logger.info(f"Deriving {precision}")
        logger.info("=" * 60)
        derive_precision_files(
            onnx_dir, precision, block_size=args.block_size, q4_symmetric=not args.q4_asymmetric
        )

    genai_config = output_dir / "genai_config.json"
    if args.vision_format == VISION_MODE_TILED:
        # Rebuilding fp32 makes precisions from earlier runs stale; --skip-export keeps them valid.
        available = precisions
        if args.skip_export:
            available = [
                p for p in PRECISIONS if all((onnx_dir / f).exists() for f in bundle(p).values())
            ]
        write_genai_config(output_dir, next((p for p in DEFAULT_ORDER if p in available), "fp32"))
    elif genai_config.exists():
        genai_config.unlink()
        logger.warning("onnxruntime-genai needs the tiled vision encoder; no genai_config.json")

    if not args.no_split_data:
        chunk_size_bytes = int(args.split_data * 1024 * 1024 * 1024)
        for onnx_file in onnx_dir.glob("*.onnx"):
            data_file = onnx_file.with_suffix(".onnx_data")
            if data_file.exists() and data_file.stat().st_size > chunk_size_bytes:
                logger.info(f"Splitting {onnx_file.name} ({args.split_data:.1f} GB chunks)")
                split_external_data(onnx_file, chunk_size=chunk_size_bytes)

    logger.info("=" * 60)
    logger.info("Output summary")
    logger.info("=" * 60)
    files = ", ".join(f.name for f in sorted(onnx_dir.glob("*.onnx")))
    total_size = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())
    logger.info(f"  {output_dir} ({total_size / 1e9:.2f} GB)")
    logger.info(f"    Files: {files}")


if __name__ == "__main__":
    main()
