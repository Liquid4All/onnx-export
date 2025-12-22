#!/usr/bin/env python3
"""
INT4 Quantization for LFM2 ONNX models.

Usage:
    # Quantize a single model
    python quantize.py --input LFM2-1.2B-ONNX-builder --output LFM2-1.2B-ONNX-builder-Q4

    # Quantize with custom block size
    python quantize.py --input LFM2-1.2B-ONNX-builder --output out --block-size 64

    # INT8 quantization
    python quantize.py --input LFM2-1.2B-ONNX-builder --output out --bits 8
"""

import argparse
import shutil
from pathlib import Path

import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer


def quantize_int4(model_path: Path, output_path: Path, block_size: int = 32):
    """Quantize model to INT4 using MatMulNBits."""
    print(f"Loading {model_path}...")
    model = onnx.load(str(model_path))

    # Load external data if present
    external_data = model_path.with_suffix(".onnx_data")
    if external_data.exists():
        onnx.load_external_data_for_model(model, str(model_path.parent))

    print(f"Quantizing to INT4 (block_size={block_size})...")
    quantizer = MatMulNBitsQuantizer(
        model,
        block_size=block_size,
        is_symmetric=True,
        accuracy_level=4,
    )
    quantizer.process()

    print(f"Saving to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    quantizer.model.save_model_to_file(str(output_path), use_external_data_format=True)

    return output_path


def quantize_int8(model_path: Path, output_path: Path):
    """Quantize model to INT8 using dynamic quantization."""
    print(f"Loading {model_path}...")

    external_data = model_path.with_suffix(".onnx_data")
    if external_data.exists():
        model = onnx.load(str(model_path))
        onnx.load_external_data_for_model(model, str(model_path.parent))
        temp_path = model_path.parent / "temp_for_quant.onnx"
        onnx.save_model(model, str(temp_path), save_as_external_data=False)
        model_to_quant = str(temp_path)
    else:
        model_to_quant = str(model_path)

    print("Quantizing to INT8 (dynamic)...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    quantize_dynamic(
        model_input=model_to_quant,
        model_output=str(output_path),
        weight_type=QuantType.QInt8,
        extra_options={
            "MatMulConstBOnly": True,
            "DefaultTensorType": onnx.TensorProto.FLOAT,
        }
    )

    if external_data.exists():
        temp_path = model_path.parent / "temp_for_quant.onnx"
        if temp_path.exists():
            temp_path.unlink()

    return output_path


def get_model_size(path: Path) -> tuple[float, float]:
    """Return (model_mb, data_gb)."""
    model_size = path.stat().st_size / 1e6 if path.exists() else 0
    data_path = path.with_suffix(".onnx_data")
    data_size = data_path.stat().st_size / 1e9 if data_path.exists() else 0
    return model_size, data_size


def main():
    parser = argparse.ArgumentParser(description="Quantize LFM2 ONNX models")
    parser.add_argument("--input", type=Path, required=True, help="Input ONNX model directory")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--bits", type=int, choices=[4, 8], default=4, help="Quantization bits")
    parser.add_argument("--block-size", type=int, default=32, help="Block size for INT4")
    args = parser.parse_args()

    # Find model.onnx
    input_model = args.input / "onnx" / "model.onnx"
    if not input_model.exists():
        input_model = args.input / "model.onnx"
    if not input_model.exists():
        raise FileNotFoundError(f"No model.onnx found in {args.input}")

    output_dir = args.output / "onnx"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_model = output_dir / "model.onnx"

    # Get original size
    orig_model_mb, orig_data_gb = get_model_size(input_model)
    print(f"Original: {orig_model_mb:.1f} MB + {orig_data_gb:.2f} GB data")

    # Quantize
    if args.bits == 4:
        quantize_int4(input_model, output_model, args.block_size)
    else:
        quantize_int8(input_model, output_model)

    # Get quantized size
    quant_model_mb, quant_data_gb = get_model_size(output_model)
    print(f"Quantized: {quant_model_mb:.1f} MB + {quant_data_gb:.2f} GB data")

    # Copy config files
    for cfg in ["config.json", "tokenizer.json", "tokenizer_config.json",
                "special_tokens_map.json", "genai_config.json"]:
        src = args.input / cfg
        if src.exists():
            shutil.copy(src, args.output / cfg)

    # Compression ratio
    orig_total = orig_model_mb / 1000 + orig_data_gb
    quant_total = quant_model_mb / 1000 + quant_data_gb
    if orig_total > 0:
        ratio = orig_total / quant_total
        print(f"Compression: {ratio:.1f}x")

    print(f"\nOutput: {args.output}")


if __name__ == "__main__":
    main()
