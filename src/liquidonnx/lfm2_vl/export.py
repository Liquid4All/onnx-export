#!/usr/bin/env python3
"""
Export LFM2-VL models to ONNX with optional quantization and FP16 conversion.

Output Structure (Transformers.js compatible):
    exports/
    └── LFM2-VL-{size}-ONNX/
        ├── config.json
        ├── tokenizer.json
        ├── tokenizer_config.json
        └── onnx/
            ├── embed_tokens.onnx          # FP32
            ├── embed_tokens_fp16.onnx     # --precision fp16
            ├── embed_images.onnx          # FP32
            ├── embed_images_fp16.onnx     # --precision fp16
            ├── embed_images_q4.onnx       # --precision q4
            ├── embed_images_q8.onnx       # --precision q8
            ├── decoder.onnx               # FP32
            ├── decoder_fp16.onnx          # --precision fp16
            ├── decoder_q4.onnx            # --precision q4
            └── decoder_q8.onnx            # --precision q8

Usage:
    # Export from local model path
    uv run lfm2-vl-export --model-path ./LFM2-VL-1.6B-3102461
    uv run lfm2-vl-export --model-path ./LFM2-VL-1.6B-3102461 --precision

    # Export FP32 only (all sizes)
    uv run lfm2-vl-export --sizes all

    # Export with all precisions (fp16, q4, q8)
    uv run lfm2-vl-export --sizes all --precision

    # Export with specific precisions
    uv run lfm2-vl-export --sizes 450M --precision q4
    uv run lfm2-vl-export --sizes 450M --precision fp16 q4 q8

    # Convert existing exports (skip FP32 export)
    uv run lfm2-vl-export --sizes all --precision --skip-export

    # Convert to FP16 (2x smaller, matches onnx-community format)
    uv run lfm2-vl-export --model-path ./LFM2-VL-1.6B-3102461 --precision fp16

    # Export with conv2d vision format (instead of default tiled)
    uv run lfm2-vl-export --sizes 450M --vision-format conv2d
"""

import argparse
import gc
import json
import logging
import pathlib

import onnx
from onnx import TensorProto, helper

from liquidonnx.lfm2.builder import LFM2Builder
from liquidonnx.lfm2_vl import MODELS, VISION_MODE_CONV2D, VISION_MODE_TILED
from liquidonnx.lfm2_vl.builder import EmbedTokensBuilder, LFM2VLConfig, VisionEmbedBuilder
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

    # Load model with external data
    model = onnx.load(str(input_path), load_external_data=False)
    load_external_data_for_model(model, str(input_path.parent))

    # Save original Resize inputs and empty tensor initializers before conversion
    # (converter modifies in-place)
    resize_orig_inputs = {}
    empty_initializers = {}
    for node in model.graph.node:
        if node.op_type == "Resize":
            resize_orig_inputs[node.name] = list(node.input)
    for init in model.graph.initializer:
        if "empty" in init.name.lower():
            empty_initializers[init.name] = onnx.TensorProto()
            empty_initializers[init.name].CopyFrom(init)

    # Convert to FP16
    # disable_shape_infer=True is required for models with com.microsoft custom ops
    model_fp16 = convert_float_to_float16(
        model,
        keep_io_types=keep_io_types,
        force_fp16_initializers=True,
        disable_shape_infer=True,
    )

    # Fix Resize nodes: the converter adds Cast nodes for empty scales/roi tensors
    # which breaks shape inference. Restore original empty tensor inputs and initializers.
    for node in model_fp16.graph.node:
        if node.op_type == "Resize" and node.name in resize_orig_inputs:
            orig_inputs = resize_orig_inputs[node.name]
            # Restore scales input (index 2) if it was an empty tensor
            if len(node.input) > 2 and len(orig_inputs) > 2:
                if "cast" in node.input[2].lower() and "empty" in orig_inputs[2].lower():
                    node.input[2] = orig_inputs[2]
            # Restore roi input (index 1) if it was an empty tensor
            if len(node.input) > 1 and len(orig_inputs) > 1:
                if "cast" in node.input[1].lower() and "empty" in orig_inputs[1].lower():
                    node.input[1] = orig_inputs[1]

    # Restore FP32 empty tensor initializers
    for i, init in enumerate(model_fp16.graph.initializer):
        if init.name in empty_initializers:
            model_fp16.graph.initializer[i].CopyFrom(empty_initializers[init.name])

    # Save with external data
    output_data_path = output_path.parent / f"{output_path.stem}.onnx_data"
    if output_data_path.exists():
        output_data_path.unlink()

    onnx.save_model(
        model_fp16,
        str(output_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=f"{output_path.stem}.onnx_data",
        size_threshold=1024,
    )

    # Get sizes for logging
    orig_mb = get_total_model_size_mb(input_path)
    fp16_mb = get_total_model_size_mb(output_path)
    ratio = orig_mb / fp16_mb if fp16_mb > 0 else 0
    logger.info(f"  {input_path.name}: {orig_mb:.1f} -> {fp16_mb:.1f} MB ({ratio:.1f}x)")

    return output_path


def get_output_dir(
    size: str, output_base: pathlib.Path, vision_format: str = VISION_MODE_TILED
) -> pathlib.Path:
    suffix = f"-{vision_format}" if vision_format != VISION_MODE_TILED else ""
    return output_base / "exports" / f"LFM2-VL-{size}-ONNX{suffix}"


def export_vl_model(
    model_path: str,
    output_dir: pathlib.Path | str,
    vision_input_format: str = VISION_MODE_TILED,
    use_integrated_rope: bool = False,
):
    """Export LFM2-VL model to ONNX (embed_tokens + embed_images + decoder).

    Args:
        model_path: HuggingFace model path
        output_dir: Output directory for ONNX files
        vision_input_format: "tiled" for [B, N, 768] or "conv2d" for [B, 3, H, W]
        use_integrated_rope: Use RoPE integrated in GroupQueryAttention (better precision)
    """
    import torch
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

    output_dir = pathlib.Path(output_dir)
    logger.info(f"Vision input format: {vision_input_format}")

    # Load config
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    vl_config = LFM2VLConfig.from_hf_config(config)

    # Load model weights
    logger.info(f"Loading weights from {model_path}...")
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, torch_dtype=torch.float32, trust_remote_code=True
    )

    weights = {}
    for name, param in model.named_parameters():
        weights[name] = param.detach().numpy()
        logger.debug(f"Loaded: {name} {param.shape}")

    logger.info(f"Loaded {len(weights)} total weights")

    del model
    gc.collect()

    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_dir = output_dir / "onnx"
    onnx_dir.mkdir(exist_ok=True)

    # === 1. Export embed_tokens ===
    logger.info("Exporting embed_tokens...")
    embed_tokens_builder = EmbedTokensBuilder(vl_config)
    embed_tokens_builder.load_weights(weights)
    embed_tokens_model = embed_tokens_builder.build()

    embed_tokens_path = onnx_dir / "embed_tokens.onnx"
    onnx.save_model(embed_tokens_model, str(embed_tokens_path))
    logger.info(f"embed_tokens saved to {embed_tokens_path}")
    del embed_tokens_model
    del embed_tokens_builder

    # === 2. Export embed_images ===
    logger.info(
        f"Exporting embed_images (vision encoder + projector) [{vision_input_format} mode]..."
    )
    vision_builder = VisionEmbedBuilder(vl_config, vision_input_format=vision_input_format)
    vision_builder.load_weights(weights)
    vision_model = vision_builder.build()

    vision_path = onnx_dir / "embed_images.onnx"
    vision_data_path = onnx_dir / "embed_images.onnx_data"
    if vision_data_path.exists():
        vision_data_path.unlink()
    onnx.save_model(
        vision_model,
        str(vision_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="embed_images.onnx_data",
        size_threshold=1024,
    )
    logger.info(f"embed_images saved to {vision_path}")

    del vision_model
    del vision_builder
    gc.collect()

    # === 3. Export decoder ===
    logger.info("Exporting decoder (with inputs_embeds input)...")
    if use_integrated_rope:
        logger.info("Using integrated RoPE in GroupQueryAttention")

    text_builder = LFM2Builder(vl_config.text_config, use_integrated_rope=use_integrated_rope)

    # Filter text model weights (they have "model.language_model." prefix in VL model)
    for name, weight in weights.items():
        if name.startswith("model.language_model."):
            new_name = name.replace("model.language_model.", "model.")
            text_builder.weights[new_name] = weight
        elif name.startswith("language_model."):
            new_name = name.replace("language_model.", "model.")
            text_builder.weights[new_name] = weight

    weights.clear()
    gc.collect()

    # Build custom inputs: inputs_embeds instead of input_ids
    H = vl_config.text_config.hidden_size

    text_builder.inputs.append(
        helper.make_tensor_value_info(
            "inputs_embeds", TensorProto.FLOAT, ["batch_size", "sequence_length", H]
        )
    )
    text_builder.inputs.append(
        helper.make_tensor_value_info(
            "attention_mask", TensorProto.INT64, ["batch_size", "total_sequence_length"]
        )
    )
    text_builder.inputs.append(
        helper.make_tensor_value_info(
            "position_ids", TensorProto.INT64, ["batch_size", "sequence_length"]
        )
    )

    # Conv caches
    for idx in text_builder.conv_indices:
        text_builder.inputs.append(
            helper.make_tensor_value_info(
                f"past_conv.{idx}",
                TensorProto.FLOAT,
                ["batch_size", H, vl_config.text_config.conv_L_cache],
            )
        )

    # KV caches
    for idx in text_builder.attn_indices:
        text_builder.inputs.append(
            helper.make_tensor_value_info(
                f"past_key_values.{idx}.key",
                TensorProto.FLOAT,
                [
                    "batch_size",
                    vl_config.text_config.num_key_value_heads,
                    "past_sequence_length",
                    text_builder.head_dim,
                ],
            )
        )
        text_builder.inputs.append(
            helper.make_tensor_value_info(
                f"past_key_values.{idx}.value",
                TensorProto.FLOAT,
                [
                    "batch_size",
                    vl_config.text_config.num_key_value_heads,
                    "past_sequence_length",
                    text_builder.head_dim,
                ],
            )
        )

    text_builder.build_outputs()
    text_builder.build_rope_cache()
    text_builder.build_attention_mask_subgraph()

    # Skip build_embedding() - use inputs_embeds directly as hidden_state
    # But we still need embed_tokens weight for lm_head (tied weights)
    text_builder.add_initializer(
        "model.embed_tokens.weight", text_builder.weights["model.embed_tokens.weight"]
    )
    hidden_state = "inputs_embeds"

    for layer_idx in range(vl_config.text_config.num_hidden_layers):
        layer_type = vl_config.text_config.layer_types[layer_idx]
        logger.info(f"Building text layer {layer_idx} ({layer_type})...")

        # Prepare weights for this layer (handles transposition, adds initializers)
        text_builder.prepare_layer_weights(layer_idx, layer_type)

        if layer_type == "conv":
            hidden_state = text_builder.build_conv_layer(layer_idx, hidden_state)
        else:
            hidden_state = text_builder.build_attention_layer(layer_idx, hidden_state)

    text_builder.build_lm_head(hidden_state)

    text_builder.weights.clear()
    gc.collect()

    logger.info("Building decoder graph...")
    text_graph = helper.make_graph(
        text_builder.nodes,
        "decoder",
        text_builder.inputs,
        text_builder.outputs,
        text_builder.initializers,
    )

    text_model = helper.make_model(
        text_graph,
        opset_imports=[
            helper.make_opsetid("", 21),
            helper.make_opsetid("com.microsoft", 1),
        ],
        ir_version=9,
    )
    text_model.producer_name = "lfm2-vl-builder"

    text_builder.nodes.clear()
    text_builder.initializers.clear()
    gc.collect()

    decoder_path = onnx_dir / "decoder.onnx"
    decoder_data_path = onnx_dir / "decoder.onnx_data"
    if decoder_data_path.exists():
        decoder_data_path.unlink()

    logger.info("Saving decoder (this may take a while for large models)...")
    onnx.save_model(
        text_model,
        str(decoder_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="decoder.onnx_data",
        size_threshold=1024,
    )
    logger.info(f"decoder saved to {decoder_path}")

    del text_model
    gc.collect()

    # Copy tokenizer and config
    try:
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        processor.save_pretrained(output_dir)
    except Exception as e:
        logger.warning(f"Could not save processor: {e}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer.save_pretrained(output_dir)

    config.save_pretrained(output_dir)

    # Create generation_config.json
    gen_config = {
        "_from_model_config": True,
        "bos_token_id": config.text_config.bos_token_id
        if hasattr(config.text_config, "bos_token_id")
        else 1,
        "eos_token_id": config.text_config.eos_token_id
        if hasattr(config.text_config, "eos_token_id")
        else 7,
        "pad_token_id": 0,
        "transformers_version": "4.57.0",
    }
    gen_config_path = output_dir / "generation_config.json"
    gen_config_path.write_text(json.dumps(gen_config, indent=2))

    # Print summary
    total_size = 0
    for fpath in onnx_dir.iterdir():
        if fpath.is_file():
            size = fpath.stat().st_size
            total_size += size
            logger.info(f"  {fpath.name}: {size / 1e6:.1f} MB")

    logger.info(f"Total ONNX size: {total_size / 1e9:.2f} GB")
    logger.info(f"Output directory: {output_dir}")

    return output_dir


def do_export(
    model_path: str,
    output_path: pathlib.Path,
    vision_format: str = VISION_MODE_TILED,
    use_integrated_rope: bool = False,
):
    """Export a single VL model to ONNX (FP32)."""
    logger.info(f"Exporting {model_path} to {output_path}...")
    export_vl_model(
        model_path,
        output_path,
        vision_input_format=vision_format,
        use_integrated_rope=use_integrated_rope,
    )


def do_quantize(
    onnx_dir: pathlib.Path, decoder_bits: int, vision_bits: int = 8, block_size: int = 32
):
    if not onnx_dir.exists():
        raise FileNotFoundError(f"ONNX directory not found: {onnx_dir}")

    logger.info(
        f"Quantizing {onnx_dir.parent.name} -> decoder=q{decoder_bits}, vision=q{vision_bits}..."
    )

    # Quantize embed_images
    embed_fp32 = onnx_dir / "embed_images.onnx"

    embed_output = onnx_dir / f"embed_images_q{vision_bits}.onnx"
    if embed_fp32.exists() and not embed_output.exists():
        _, embed_orig_mb = get_model_size(embed_fp32)
        quantize_model(
            embed_fp32, embed_output, bits=vision_bits, block_size=block_size, exclude_lm_head=False
        )
        _, embed_quant_mb = get_model_size(embed_output)
        logger.info(
            f"  embed_images: {embed_orig_mb:.1f} -> {embed_quant_mb:.1f} MB "
            f"({embed_orig_mb / embed_quant_mb:.1f}x)"
        )

    # Quantize decoder
    decoder_fp32 = onnx_dir / "decoder.onnx"

    decoder_output = onnx_dir / f"decoder_q{decoder_bits}.onnx"
    if decoder_fp32.exists() and not decoder_output.exists():
        _, decoder_orig_mb = get_model_size(decoder_fp32)
        quantize_model(
            decoder_fp32,
            decoder_output,
            bits=decoder_bits,
            block_size=block_size,
            exclude_lm_head=True,
        )
        _, decoder_quant_mb = get_model_size(decoder_output)
        logger.info(
            f"  decoder: {decoder_orig_mb:.1f} -> {decoder_quant_mb:.1f} MB "
            f"({decoder_orig_mb / decoder_quant_mb:.1f}x)"
        )


def do_fp16(onnx_dir: pathlib.Path):
    """Convert VL model to FP16 (matching onnx-community naming).

    Creates:
    - embed_tokens_fp16.onnx
    - embed_images_fp16.onnx
    - decoder_fp16.onnx
    """
    if not onnx_dir.exists():
        raise FileNotFoundError(f"ONNX directory not found: {onnx_dir}")

    logger.info(f"Converting {onnx_dir.parent.name} to FP16...")

    # Convert embed_tokens
    embed_tokens_fp32 = onnx_dir / "embed_tokens.onnx"
    embed_tokens_fp16 = onnx_dir / "embed_tokens_fp16.onnx"
    if embed_tokens_fp32.exists() and not embed_tokens_fp16.exists():
        convert_to_fp16(embed_tokens_fp32, embed_tokens_fp16)

    # Convert embed_images
    embed_images_fp32 = onnx_dir / "embed_images.onnx"
    embed_images_fp16 = onnx_dir / "embed_images_fp16.onnx"
    if embed_images_fp32.exists() and not embed_images_fp16.exists():
        convert_to_fp16(embed_images_fp32, embed_images_fp16)

    # Convert decoder
    decoder_fp32 = onnx_dir / "decoder.onnx"
    decoder_fp16 = onnx_dir / "decoder_fp16.onnx"
    if decoder_fp32.exists() and not decoder_fp16.exists():
        convert_to_fp16(decoder_fp32, decoder_fp16)


def main():
    parser = argparse.ArgumentParser(
        description="Export LFM2-VL models to ONNX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Model selection
    parser.add_argument(
        "--sizes",
        nargs="+",
        help="Model sizes: 450M, 1.6B, 3B, or 'all'",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="Local model path (alternative to --sizes for local models)",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("."),
        help="Output base directory (default: current directory)",
    )

    # Output precision
    parser.add_argument(
        "--precision",
        nargs="*",
        metavar="PRECISION",
        help="Output precisions: fp16, q4, q8, or all (default if no args)",
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Skip FP32 export, only run quantization",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=32,
        help="Block size for quantization (default: 32)",
    )
    parser.add_argument(
        "--integrated-rope",
        action="store_true",
        help="Use RoPE integrated in GroupQueryAttention (better precision)",
    )
    parser.add_argument(
        "--vision-format",
        choices=[VISION_MODE_TILED, VISION_MODE_CONV2D],
        default=VISION_MODE_TILED,
        help="Vision encoder format: tiled (default) or conv2d",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    # Parse output precisions
    quant_bits = []
    do_fp16_conversion = False
    if args.precision is not None:
        if len(args.precision) == 0:
            quant_bits = [4, 8]
            do_fp16_conversion = True
        else:
            for p in args.precision:
                p = p.lower()
                if p == "fp16":
                    do_fp16_conversion = True
                elif p in ("q4", "q8"):
                    quant_bits.append(int(p[1]))
                else:
                    parser.error(f"Invalid precision: {p}. Use fp16, q4, or q8.")

    # Validate args
    if args.model_path and args.sizes:
        parser.error("Cannot specify both --model-path and --sizes")
    if not args.model_path and not args.sizes:
        parser.error("Must specify either --model-path or --sizes")

    # Build list of (model_path, output_dir) pairs
    vision_suffix = f"-{args.vision_format}" if args.vision_format != VISION_MODE_TILED else ""
    exports = []
    if args.model_path:
        model_name = pathlib.Path(args.model_path).name
        output_dir = args.output_dir / "exports" / f"{model_name}-ONNX{vision_suffix}"
        exports.append((args.model_path, output_dir))
    else:
        sizes = list(MODELS.keys()) if "all" in args.sizes else args.sizes
        for s in sizes:
            if s not in MODELS:
                parser.error(f"Unknown size: {s}. Available: {', '.join(MODELS.keys())}")
        for size in sizes:
            exports.append((MODELS[size], get_output_dir(size, args.output_dir, args.vision_format)))

    # Export
    if not args.skip_export:
        for model_path, output_dir in exports:
            do_export(model_path, output_dir, args.vision_format, args.integrated_rope)

    # Quantize
    for bits in quant_bits:
        for _, output_dir in exports:
            onnx_dir = output_dir / "onnx"
            do_quantize(onnx_dir, bits, bits, args.block_size)

    # FP16 conversion
    if do_fp16_conversion:
        for _, output_dir in exports:
            onnx_dir = output_dir / "onnx"
            do_fp16(onnx_dir)


if __name__ == "__main__":
    main()
