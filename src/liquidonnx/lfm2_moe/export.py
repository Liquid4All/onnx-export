#!/usr/bin/env python3
"""
Export LFM2-MoE models to ONNX with optional quantization and FP16 conversion.

Output Structure (Transformers.js compatible):
    {output-dir}/
    └── {model-name}-ONNX/
        ├── config.json
        ├── tokenizer.json
        └── onnx/
            ├── model.onnx           # FP32
            ├── model.onnx_data
            ├── model_fp16.onnx      # --precision fp16
            ├── model_fp16.onnx_data
            ├── model_q4.onnx        # --precision q4
            ├── model_q4.onnx_data
            ├── model_q4f16.onnx     # --precision q4f16
            └── model_q4f16.onnx_data

Usage:
    # Export from HuggingFace
    uv run lfm2-moe-export LiquidAI/LFM2-MoE-8B-A1B

    # Export from local path
    uv run lfm2-moe-export /path/to/local/model

    # Export with all precisions (fp16, q4, q4f16)
    uv run lfm2-moe-export LiquidAI/LFM2-MoE-8B-A1B --precision

    # Export with specific precisions
    uv run lfm2-moe-export LiquidAI/LFM2-MoE-8B-A1B --precision q4
    uv run lfm2-moe-export LiquidAI/LFM2-MoE-8B-A1B --precision fp16 q4 q4f16

    # Convert existing export (skip FP32 export)
    uv run lfm2-moe-export LiquidAI/LFM2-MoE-8B-A1B --precision --skip-export

    # Quantize with lm_head included
    uv run lfm2-moe-export LiquidAI/LFM2-MoE-8B-A1B --precision q4 --no-exclude-lm-head
"""

import argparse
import json
import logging
import pathlib

import onnx
from transformers import AutoConfig, AutoTokenizer

from liquidonnx.lfm2_moe.builder import LFM2MoEBuilder, LFM2MoEConfig
from liquidonnx.quantize import get_model_size, get_total_model_size_mb, quantize_model

logger = logging.getLogger(__name__)


def convert_to_fp16(
    input_path: pathlib.Path,
    output_path: pathlib.Path,
):
    """Convert ONNX model from FP32 to FP16.

    Matches community convention:
    - All float32 weights become float16
    - KV cache inputs/outputs become float16
    - logits output stays float32 (added Cast node)
    - input_ids and attention_mask stay int64

    Args:
        input_path: Path to FP32 ONNX model
        output_path: Path for FP16 output model
    """
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper
    from onnx.external_data_helper import load_external_data_for_model

    logger.info(f"Converting {input_path.name} to FP16...")

    model = onnx.load(str(input_path), load_external_data=False)
    load_external_data_for_model(model, str(input_path.parent))

    graph = model.graph

    # === 1. Convert all float32 initializers to float16 ===
    # Track renamed constants for node input updates
    renamed_constants = {}
    fp16_min = np.finfo(np.float16).min  # -65504.0
    fp16_max = np.finfo(np.float16).max  # 65504.0
    new_initializers = []
    for init in graph.initializer:
        if init.data_type == TensorProto.FLOAT:
            arr = numpy_helper.to_array(init)
            # Clamp to FP16 range before conversion to avoid -inf/+inf
            arr_clamped = np.clip(arr, fp16_min, fp16_max)
            arr_fp16 = arr_clamped.astype(np.float16)
            new_name = init.name

            # Rename /model/constants/FLOAT/... to /model/constants/FLOAT16/... with FP16 value
            if init.name.startswith("/model/constants/FLOAT/"):
                # Format the FP16 value to match community naming
                value_str = str(arr_fp16.tolist())
                new_name = f"/model/constants/FLOAT16/{value_str}"
                renamed_constants[init.name] = new_name

            new_init = numpy_helper.from_array(arr_fp16, new_name)
            new_initializers.append(new_init)
        else:
            new_initializers.append(init)

    del graph.initializer[:]
    graph.initializer.extend(new_initializers)

    # Update node inputs that reference renamed constants
    if renamed_constants:
        for node in graph.node:
            for i, inp in enumerate(node.input):
                if inp in renamed_constants:
                    node.input[i] = renamed_constants[inp]

    # === 2. Convert KV cache inputs to FP16 (keep int64 inputs) ===
    for inp in graph.input:
        if inp.type.tensor_type.elem_type == TensorProto.FLOAT:
            inp.type.tensor_type.elem_type = TensorProto.FLOAT16

    # === 3. Convert KV cache outputs to FP16 (except logits) ===
    for out in graph.output:
        if out.type.tensor_type.elem_type == TensorProto.FLOAT:
            if out.name != "logits":
                out.type.tensor_type.elem_type = TensorProto.FLOAT16

    # === 4. Add Cast node for logits (fp16 internal -> fp32 output) ===
    for output in graph.output:
        if output.name == "logits":
            cast_input = "logits_fp16"
            cast_node = helper.make_node(
                "Cast",
                inputs=[cast_input],
                outputs=["logits"],
                to=TensorProto.FLOAT,
            )

            # Find the node producing logits and rename its output
            for node in graph.node:
                for j, out in enumerate(node.output):
                    if out == "logits":
                        node.output[j] = cast_input
                        break

            graph.node.append(cast_node)
            break

    output_data_path = output_path.parent / f"{output_path.stem}.onnx_data"
    if output_data_path.exists():
        output_data_path.unlink()

    onnx.save_model(
        model,
        str(output_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=f"{output_path.stem}.onnx_data",
    )

    orig_mb = get_total_model_size_mb(input_path)
    fp16_mb = get_total_model_size_mb(output_path)
    ratio = orig_mb / fp16_mb if fp16_mb > 0 else 0
    logger.info(f"  {input_path.name}: {orig_mb:.1f} -> {fp16_mb:.1f} MB ({ratio:.1f}x)")

    return output_path


def convert_q4_to_fp16(
    input_path: pathlib.Path,
    output_path: pathlib.Path,
):
    """Convert Q4 ONNX model to Q4F16 (FP16 non-quantized weights).

    Matches community Q4F16 convention:
    - Quantization scales stay float32 (needed for precision)
    - Zero points stay uint8
    - Quant weights stay uint8
    - LayerNorm weights become float16
    - RoPE caches become float16
    - Conv weights become float16
    - Expert biases become float16
    - KV cache inputs/outputs become float16
    - logits output stays float32 (added Cast node)

    Args:
        input_path: Path to Q4 ONNX model
        output_path: Path for Q4F16 output model
    """
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper
    from onnx.external_data_helper import load_external_data_for_model

    logger.info(f"Converting {input_path.name} to Q4F16...")

    model = onnx.load(str(input_path), load_external_data=False)
    load_external_data_for_model(model, str(input_path.parent))

    graph = model.graph

    # Convert float32 initializers to FP16, except quantization scales (keep FP32 for precision)
    fp16_min = np.finfo(np.float16).min  # -65504.0
    fp16_max = np.finfo(np.float16).max  # 65504.0
    new_initializers = []
    for init in graph.initializer:
        if init.data_type == TensorProto.FLOAT:
            name = init.name
            # Keep quantization scales in FP32 for precision (matches community)
            if "_scales" in name:
                new_initializers.append(init)
                continue
            arr = numpy_helper.to_array(init)
            # Clamp to FP16 range before conversion to avoid -inf/+inf
            arr_clamped = np.clip(arr, fp16_min, fp16_max)
            arr_fp16 = arr_clamped.astype(np.float16)
            new_init = numpy_helper.from_array(arr_fp16, name)
            new_initializers.append(new_init)
        else:
            # Keep int64, uint8 as-is
            new_initializers.append(init)

    # Replace initializers
    del graph.initializer[:]
    graph.initializer.extend(new_initializers)

    # Update float constants name (FLOAT -> FLOAT16) with FP16 value to match community
    for init in graph.initializer:
        if "/model/constants/FLOAT/" in init.name and init.data_type == TensorProto.FLOAT16:
            old_name = init.name
            # Get the FP16 value and format it to match community naming
            arr = numpy_helper.to_array(init)
            value_str = str(arr.tolist())
            new_name = f"/model/constants/FLOAT16/{value_str}"
            init.name = new_name
            # Update all node references
            for node in graph.node:
                for i, inp in enumerate(node.input):
                    if inp == old_name:
                        node.input[i] = new_name

    # Convert KV cache inputs to FP16
    for inp in graph.input:
        if inp.type.tensor_type.elem_type == TensorProto.FLOAT:
            if "past_" in inp.name or "key" in inp.name or "value" in inp.name:
                inp.type.tensor_type.elem_type = TensorProto.FLOAT16

    # Convert KV cache outputs to FP16 (except logits)
    for out in graph.output:
        if out.type.tensor_type.elem_type == TensorProto.FLOAT:
            if out.name != "logits":
                out.type.tensor_type.elem_type = TensorProto.FLOAT16

    # Add Cast node for logits (fp16 -> fp32) matching community
    for output in graph.output:
        if output.name == "logits":
            cast_input = "logits_fp16"
            cast_node = helper.make_node(
                "Cast",
                inputs=[cast_input],
                outputs=["logits"],
                to=TensorProto.FLOAT,
            )

            # Find node producing logits and rename its output
            for node in graph.node:
                for j, out in enumerate(node.output):
                    if out == "logits":
                        node.output[j] = cast_input
                        break

            graph.node.append(cast_node)
            break

    output_data_path = output_path.parent / f"{output_path.stem}.onnx_data"
    if output_data_path.exists():
        output_data_path.unlink()

    onnx.save_model(
        model,
        str(output_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=f"{output_path.stem}.onnx_data",
    )

    orig_mb = get_total_model_size_mb(input_path)
    fp16_mb = get_total_model_size_mb(output_path)
    logger.info(f"  {input_path.name}: {orig_mb:.1f} -> {fp16_mb:.1f} MB")

    return output_path


def get_model_name(model_path: str) -> str:
    """Extract model name from HF slug or local path."""
    # Handle HF slugs like "LiquidAI/LFM2-MoE-8B-A1B" -> "LFM2-MoE-8B-A1B"
    if "/" in model_path:
        return model_path.split("/")[-1]
    # Handle local paths
    return pathlib.Path(model_path).name


def export_model(
    model_path: str,
    output_dir: pathlib.Path | str,
    integrated_rope: bool = False,
    use_qmoe: bool = False,
    qmoe_block_size: int = 32,
    use_q4: bool = False,
):
    """Export LFM2-MoE model to ONNX.

    Creates output structure:
        output_dir/
        ├── config.json
        ├── tokenizer.json
        ├── tokenizer_config.json
        ├── generation_config.json
        ├── chat_template.jinja
        └── onnx/
            ├── model.onnx (or model_q4.onnx in Q4 mode)
            └── model.onnx_data

    Args:
        model_path: HuggingFace model path
        output_dir: Output directory
        integrated_rope: Use RoPE integrated in GQA (matches onnx-community style)
        use_qmoe: Use QMoE operator with INT4 quantized expert weights
        qmoe_block_size: Block size for QMoE quantization
        use_q4: Full Q4 quantization matching onnx-community Q4 structure
    """
    output_dir = pathlib.Path(output_dir)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    lfm2_config = LFM2MoEConfig.from_hf_config(config)

    # Always use integrated RoPE to match community structure
    # Community models use RoPE integrated in GroupQueryAttention (do_rotary=1)
    effective_integrated_rope = True

    builder = LFM2MoEBuilder(
        lfm2_config,
        use_integrated_rope=effective_integrated_rope,
        use_qmoe=use_qmoe,
        qmoe_block_size=qmoe_block_size,
        use_q4=use_q4,
    )
    model = builder.build(model_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_dir = output_dir / "onnx"
    onnx_dir.mkdir(exist_ok=True)

    # Use model_q4.onnx for Q4 mode to match community naming
    model_name = "model_q4" if use_q4 else "model"
    output_path = onnx_dir / f"{model_name}.onnx"

    external_data_path = onnx_dir / f"{model_name}.onnx_data"
    if external_data_path.exists():
        external_data_path.unlink()

    onnx.save_model(
        model,
        str(output_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=f"{model_name}.onnx_data",
    )

    logger.info(f"Model saved to {output_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)
    config.save_pretrained(output_dir)

    gen_config = {
        "_from_model_config": True,
        "bos_token_id": config.bos_token_id,
        "eos_token_id": config.eos_token_id,
        "pad_token_id": getattr(config, "pad_token_id", 0),
        "transformers_version": "4.54.0",
    }
    gen_config_path = output_dir / "generation_config.json"
    gen_config_path.write_text(json.dumps(gen_config, indent=2))

    config_path = output_dir / "config.json"
    cfg = json.loads(config_path.read_text())
    cfg["transformers.js_config"] = {
        "kv_cache_dtype": {"fp32": "float32"},
        "use_external_data_format": True,
    }
    config_path.write_text(json.dumps(cfg, indent=2))

    tokenizer_config_path = output_dir / "tokenizer_config.json"
    chat_template_path = output_dir / "chat_template.jinja"

    if tokenizer.chat_template:
        chat_template_path.write_text(tokenizer.chat_template)

        if tokenizer_config_path.exists():
            tok_cfg = json.loads(tokenizer_config_path.read_text())
            if "chat_template" not in tok_cfg:
                tok_cfg["chat_template"] = tokenizer.chat_template
                tokenizer_config_path.write_text(json.dumps(tok_cfg, indent=2))

    size_mb = output_path.stat().st_size / 1e6
    data_path = onnx_dir / "model.onnx_data"
    data_size_gb = data_path.stat().st_size / 1e9 if data_path.exists() else 0
    logger.info(f"Model size: {size_mb:.2f} MB + {data_size_gb:.2f} GB data")

    return output_path


def do_quantize(onnx_dir: pathlib.Path, bits: int, exclude_lm_head: bool, block_size: int):
    """Quantize model to INT4 or INT8."""
    input_model = onnx_dir / "model.onnx"
    if not input_model.exists():
        raise FileNotFoundError(f"model.onnx not found in {onnx_dir}")

    output_model = onnx_dir / f"model_q{bits}.onnx"

    if output_model.exists():
        logger.info(f"Skipping q{bits} (already exists)")
        return

    _, orig_mb = get_model_size(input_model)

    logger.info(f"Quantizing to Q{bits}...")
    quantize_model(
        input_model, output_model, bits=bits, block_size=block_size, exclude_lm_head=exclude_lm_head
    )

    _, quant_mb = get_model_size(output_model)
    if orig_mb > 0:
        logger.info(f"  {orig_mb:.1f} MB -> {quant_mb:.1f} MB ({orig_mb / quant_mb:.1f}x)")


def do_fp16(onnx_dir: pathlib.Path):
    """Convert FP32 model to FP16."""
    if not onnx_dir.exists():
        raise FileNotFoundError(f"ONNX directory not found: {onnx_dir}")

    model_fp32 = onnx_dir / "model.onnx"
    model_fp16 = onnx_dir / "model_fp16.onnx"

    if model_fp16.exists():
        logger.info("Skipping fp16 (already exists)")
        return

    if model_fp32.exists():
        convert_to_fp16(model_fp32, model_fp16)


def do_q4f16(onnx_dir: pathlib.Path):
    """Convert Q4 model to Q4F16."""
    if not onnx_dir.exists():
        raise FileNotFoundError(f"ONNX directory not found: {onnx_dir}")

    model_q4 = onnx_dir / "model_q4.onnx"
    model_q4f16 = onnx_dir / "model_q4f16.onnx"

    if model_q4f16.exists():
        logger.info("Skipping q4f16 (already exists)")
        return

    if not model_q4.exists():
        raise FileNotFoundError(f"model_q4.onnx not found in {onnx_dir}")

    convert_q4_to_fp16(model_q4, model_q4f16)


def main():
    parser = argparse.ArgumentParser(
        description="Export LFM2-MoE models to ONNX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "model",
        help="HuggingFace model ID or local path (e.g., LiquidAI/LFM2-MoE-8B-A1B)",
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
        "--skip-export",
        action="store_true",
        help="Skip FP32 export, only run precision conversion",
    )
    parser.add_argument(
        "--precision",
        nargs="*",
        metavar="PRECISION",
        help="Output precisions: fp16, q4, q4f16, or all (default if no args)",
    )
    parser.add_argument(
        "--no-exclude-lm-head",
        action="store_true",
        help="Quantize lm_head layer (by default kept in FP32)",
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
        help="Use RoPE integrated in GQA (matches onnx-community structure)",
    )
    parser.add_argument(
        "--qmoe",
        action="store_true",
        help="Use QMoE operator with INT4 quantized expert weights (matches onnx-community Q4)",
    )
    parser.add_argument(
        "--qmoe-block-size",
        type=int,
        default=32,
        help="Block size for QMoE quantization (default: 32)",
    )
    parser.add_argument(
        "--q4",
        action="store_true",
        help="Full Q4 quantization matching onnx-community Q4 structure (includes QMoE)",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    # Derive output name from model path
    model_name = get_model_name(args.model)
    output_name = args.output_name or f"{model_name}-ONNX"
    output_dir = args.output_dir / "exports" / output_name
    onnx_dir = output_dir / "onnx"

    quant_bits = []
    do_fp16_conversion = False
    do_q4f16_conversion = False
    if args.precision is not None:
        if len(args.precision) == 0:
            quant_bits = [4]
            do_fp16_conversion = True
            do_q4f16_conversion = True
        else:
            for p in args.precision:
                p = p.lower()
                if p == "fp16":
                    do_fp16_conversion = True
                elif p == "q4f16":
                    do_q4f16_conversion = True
                elif p == "q4":
                    quant_bits.append(4)
                else:
                    parser.error(f"Invalid precision: {p}. Use fp16, q4, or q4f16.")

    exclude_lm_head = not args.no_exclude_lm_head

    if not args.skip_export:
        logger.info("=" * 60)
        if args.q4:
            precision_label = "Q4 (full INT4 quantization)"
        elif args.qmoe:
            precision_label = "QMoE (INT4 experts only)"
        else:
            precision_label = "FP32"
        logger.info(f"Exporting model ({precision_label})")
        logger.info("=" * 60)
        logger.info(f"Exporting {args.model} to {output_dir}...")
        export_model(
            args.model,
            str(output_dir),
            integrated_rope=args.integrated_rope,
            use_qmoe=args.qmoe,
            qmoe_block_size=args.qmoe_block_size,
            use_q4=args.q4,
        )
        logger.info(f"  {model_name}: OK")

    if do_fp16_conversion:
        logger.info("=" * 60)
        logger.info("Converting to FP16")
        logger.info("=" * 60)
        do_fp16(onnx_dir)
        logger.info(f"  {model_name}: OK")

    for bits in quant_bits:
        logger.info("=" * 60)
        logger.info(f"Quantizing to Q{bits}")
        logger.info("=" * 60)
        do_quantize(onnx_dir, bits, exclude_lm_head, args.block_size)
        logger.info(f"  {model_name}: OK")

    if do_q4f16_conversion:
        logger.info("=" * 60)
        logger.info("Converting Q4 to Q4F16")
        logger.info("=" * 60)
        do_q4f16(onnx_dir)
        logger.info(f"  {model_name}: OK")

    logger.info("=" * 60)
    logger.info("Output summary")
    logger.info("=" * 60)
    if output_dir.exists():
        files = list(onnx_dir.glob("model*.onnx"))
        file_names = ", ".join(f.name for f in sorted(files))
        total_size = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())
        logger.info(f"  {output_dir} ({total_size / 1e9:.2f} GB)")
        logger.info(f"    Files: {file_names}")


if __name__ == "__main__":
    main()
