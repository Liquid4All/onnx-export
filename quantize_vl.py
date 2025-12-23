#!/usr/bin/env python3
"""
INT4/INT8 Quantization for LFM2-VL ONNX models.

Quantizes:
- embed_images.onnx (vision + projector): Always Q8 for quality
- decoder.onnx (LFM2 backbone): Q4 or Q8 (user choice)

Usage:
    # Quantize with Q4 decoder (default)
    uv run quantize_vl.py --input LFM2-VL-450M-ONNX-builder

    # Quantize with Q8 decoder
    uv run quantize_vl.py --input LFM2-VL-450M-ONNX-builder --bits 8

    # Custom output path
    uv run quantize_vl.py --input LFM2-VL-450M-ONNX-builder --output my-output
"""

import argparse
import shutil
from pathlib import Path

import onnx
from onnxruntime.quantization.matmul_nbits_quantizer import (
    MatMulNBitsQuantizer,
    DefaultWeightOnlyQuantConfig,
)


def find_lm_head_node(model) -> str | None:
    """Find the lm_head MatMul node name."""
    for node in model.graph.node:
        if node.op_type == "MatMul":
            # Check if any input contains lm_head weight
            for inp in node.input:
                if "lm_head" in inp.lower():
                    return node.name
    return None


def quantize_model(model_path: Path, output_path: Path, bits: int = 4,
                   block_size: int = 32, exclude_lm_head: bool = False) -> Path:
    """Quantize a single ONNX model to INT4 or INT8."""
    print(f"Loading {model_path}...")
    model = onnx.load(str(model_path))

    # Load external data if present
    external_data = model_path.with_suffix(".onnx_data")
    if external_data.exists():
        onnx.load_external_data_for_model(model, str(model_path.parent))

    # Find nodes to exclude (lm_head for decoder)
    nodes_to_exclude = None
    if exclude_lm_head:
        lm_head_node = find_lm_head_node(model)
        if lm_head_node:
            nodes_to_exclude = [lm_head_node]
            print(f"Keeping lm_head in FP32 (excluding: {lm_head_node})")
        else:
            print("Warning: Could not find lm_head node")

    print(f"Quantizing to INT{bits} (block_size={block_size})...")

    if bits == 4:
        quantizer = MatMulNBitsQuantizer(
            model,
            block_size=block_size,
            is_symmetric=True,
            accuracy_level=4,
            nodes_to_exclude=nodes_to_exclude,
        )
    else:
        algo_config = DefaultWeightOnlyQuantConfig(
            block_size=block_size,
            is_symmetric=True,
            accuracy_level=4,
            bits=8,
        )
        quantizer = MatMulNBitsQuantizer(
            model,
            block_size=block_size,
            is_symmetric=True,
            accuracy_level=4,
            nodes_to_exclude=nodes_to_exclude,
            algo_config=algo_config,
        )

    quantizer.process()

    print(f"Saving to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    quantized_model = quantizer.model.model

    # Remove any existing external data file
    external_data_path = output_path.parent / (output_path.stem + ".onnx_data")
    if external_data_path.exists():
        external_data_path.unlink()

    onnx.save_model(
        quantized_model,
        str(output_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=output_path.stem + ".onnx_data",
        size_threshold=1024,
    )

    return output_path


def get_model_size(path: Path) -> tuple[float, float]:
    """Return (model_mb, data_mb)."""
    model_size = path.stat().st_size / 1e6 if path.exists() else 0
    data_path = path.with_suffix(".onnx_data")
    data_size = data_path.stat().st_size / 1e6 if data_path.exists() else 0
    return model_size, data_size


def main():
    parser = argparse.ArgumentParser(description="Quantize LFM2-VL ONNX models")
    parser.add_argument("--input", type=Path, required=True,
                        help="Input ONNX model directory")
    parser.add_argument("--output", type=Path,
                        help="Output directory (auto-generated if not specified)")
    parser.add_argument("--bits", type=int, choices=[4, 8], default=4,
                        help="Quantization bits for decoder (default: 4)")
    parser.add_argument("--block-size", type=int, default=32,
                        help="Block size for quantization")
    parser.add_argument("--vision-bits", type=int, choices=[4, 8], default=8,
                        help="Quantization bits for vision encoder (default: 8)")
    args = parser.parse_args()

    # Find models
    onnx_dir = args.input / "onnx"
    embed_images_path = onnx_dir / "embed_images.onnx"
    decoder_path = onnx_dir / "decoder.onnx"

    if not embed_images_path.exists():
        raise FileNotFoundError(f"embed_images.onnx not found in {onnx_dir}")
    if not decoder_path.exists():
        raise FileNotFoundError(f"decoder.onnx not found in {onnx_dir}")

    # Auto-generate output name
    if args.output is None:
        input_name = args.input.name
        # Remove -ONNX suffix if present
        base_name = input_name.replace("-ONNX", "")
        output_name = f"{base_name}-ONNX-B{args.bits}V{args.vision_bits}"
        args.output = args.input.parent / output_name

    output_onnx_dir = args.output / "onnx"
    output_onnx_dir.mkdir(parents=True, exist_ok=True)

    # Get original sizes
    embed_orig_mb, embed_orig_data = get_model_size(embed_images_path)
    decoder_orig_mb, decoder_orig_data = get_model_size(decoder_path)
    print(f"Original embed_images: {embed_orig_mb:.1f} MB + {embed_orig_data:.1f} MB data")
    print(f"Original decoder: {decoder_orig_mb:.1f} MB + {decoder_orig_data:.1f} MB data")
    print()

    # Quantize embed_images (vision + projector) - always Q8 for quality
    print(f"=== Quantizing embed_images to Q{args.vision_bits} ===")
    embed_output = output_onnx_dir / "embed_images.onnx"
    quantize_model(embed_images_path, embed_output, bits=args.vision_bits,
                   block_size=args.block_size)

    embed_quant_mb, embed_quant_data = get_model_size(embed_output)
    print(f"Quantized: {embed_quant_mb:.1f} MB + {embed_quant_data:.1f} MB data")
    if embed_orig_data > 0:
        ratio = embed_orig_data / embed_quant_data
        print(f"Compression: {ratio:.1f}x")
    print()

    # Quantize decoder (LFM2 backbone) - keep lm_head in FP32
    print(f"=== Quantizing decoder to Q{args.bits} ===")
    decoder_output = output_onnx_dir / "decoder.onnx"
    quantize_model(decoder_path, decoder_output, bits=args.bits,
                   block_size=args.block_size, exclude_lm_head=True)

    decoder_quant_mb, decoder_quant_data = get_model_size(decoder_output)
    print(f"Quantized: {decoder_quant_mb:.1f} MB + {decoder_quant_data:.1f} MB data")
    if decoder_orig_data > 0:
        ratio = decoder_orig_data / decoder_quant_data
        print(f"Compression: {ratio:.1f}x")
    print()

    # Copy embed_tokens.onnx (no quantization needed - it's just embedding lookup)
    embed_tokens_src = onnx_dir / "embed_tokens.onnx"
    if embed_tokens_src.exists():
        embed_tokens_dst = output_onnx_dir / "embed_tokens.onnx"
        shutil.copy(embed_tokens_src, embed_tokens_dst)
        print(f"Copied embed_tokens.onnx ({embed_tokens_src.stat().st_size / 1e6:.1f} MB)")
    print()

    # Copy config files
    for cfg in ["config.json", "tokenizer.json", "tokenizer_config.json",
                "special_tokens_map.json", "generation_config.json",
                "chat_template.jinja", "preprocessor_config.json"]:
        src = args.input / cfg
        if src.exists():
            shutil.copy(src, args.output / cfg)

    # Print summary
    total_orig = embed_orig_data + decoder_orig_data
    total_quant = embed_quant_data + decoder_quant_data
    print("=== Summary ===")
    print(f"Vision (V{args.vision_bits}): {embed_orig_data:.1f} MB -> {embed_quant_data:.1f} MB")
    print(f"Backbone (B{args.bits}): {decoder_orig_data:.1f} MB -> {decoder_quant_data:.1f} MB (lm_head FP32)")
    print(f"Total: {total_orig:.1f} MB -> {total_quant:.1f} MB ({total_orig/total_quant:.1f}x)")
    print(f"\nOutput: {args.output}")


if __name__ == "__main__":
    main()
