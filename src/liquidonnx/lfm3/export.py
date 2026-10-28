#!/usr/bin/env python3
"""
Export LFM3 models to ONNX with optional quantization and FP16 conversion.

Output Structure (Transformers.js compatible):
    {output-dir}/
    └── {model-name}-ONNX/
        ├── config.json
        ├── tokenizer.json
        └── onnx/
            ├── model.onnx              # FP32
            ├── model.onnx_data
            ├── model_fp16.onnx         # --precision fp16
            ├── model_fp16.onnx_data
            ├── model_q4.onnx           # --precision q4 (k_quant INT4; INT8 lm_head, which is
            ├── model_q4.onnx_data      #   also the embedding table, and INT8 sensitive layers)
            ├── model_q4f16.onnx        # --precision q4f16 (q4 with FP16 activations, for CUDA)
            ├── model_q4f16.onnx_data
            ├── model_q4f32.onnx        # --precision q4f32 (INT4 MatMuls; FP32 embedding, lm_head)
            ├── model_q4f32.onnx_data
            ├── model_q8.onnx           # --precision q8 (INT8 lm_head, also the embedding table)
            ├── model_q8.onnx_data
            ├── model_q8f16.onnx        # --precision q8f16 (q8 with FP16 activations, for CUDA)
            └── model_q8f16.onnx_data

Usage:
    # Export from HuggingFace
    uv run lfm3-export LiquidAI/LFM3-1.7B

    # Export from local path
    uv run lfm3-export /path/to/local/model

    # Export with custom output directory
    uv run lfm3-export <model_id> --output-dir ./exports

    # Export with specific precisions
    uv run lfm3-export <model_id> --precision fp16 q4

    # Export with all precisions
    uv run lfm3-export <model_id> --precision

    # Add precisions to an existing export, or rebuild them (skip FP32 export)
    uv run lfm3-export <model_id> --precision --skip-export
"""

import argparse
import functools
import json
import logging
import pathlib
import re
import tempfile
from collections.abc import Callable

import onnx

from liquidonnx.export_cli import apply_umask, write_export_info
from liquidonnx.external_data import split_external_data
from liquidonnx.genai_builder import resolve_checkpoint
from liquidonnx.lfm3.builder import LFM3Builder, LFM3Config
from liquidonnx.lfm3.quantize import (
    q4_block_size_error,
    q4_int8_nodes,
    quantize_decoder_q4,
    quantize_decoder_q8,
)
from liquidonnx.lfm3.session import CHECKPOINT_PATTERNS, REVISIONS, load_tokenizer
from liquidonnx.quantize import (
    DEFAULT_BLOCK_SIZE,
    get_total_model_size_mb,
    quantize_model,
    save_model,
)

logger = logging.getLogger(__name__)

BLOCK_SIZE_HELP = (
    f"Block size for quantization (default: {DEFAULT_BLOCK_SIZE}). For q4 and q4f16 (k_quant int4) "
    "it must split the input size of every int4 MatMul into an even number of blocks"
)

# The caches of the LFM3 decoders (liquidonnx.lfm3.session), fp16 in the fp16-activation graphs
CACHE_INPUT = re.compile(r"past_conv\.\d+|past_key_values\.\d+\.(?:key|value)")
CACHE_OUTPUT = re.compile(r"present_conv\.\d+|present\.\d+\.(?:key|value)")


def convert_to_fp16(
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    keep_io_types: bool | list[str] = True,
):
    """Convert ONNX model from FP32 to FP16.

    Args:
        input_path: Path to FP32 ONNX model
        output_path: Path for FP16 output model
        keep_io_types: Inputs/outputs kept FP32: all of them (True) or the named ones
    """
    from onnx.external_data_helper import load_external_data_for_model
    from onnxruntime.transformers.float16 import convert_float_to_float16

    logger.info(f"Converting {input_path.name} to FP16...")

    model = onnx.load(str(input_path), load_external_data=False)
    load_external_data_for_model(model, str(input_path.parent))

    model_fp16 = convert_float_to_float16(
        model,
        keep_io_types=keep_io_types,
        force_fp16_initializers=True,
        disable_shape_infer=True,
    )

    save_model(model_fp16, output_path)

    orig_mb = get_total_model_size_mb(input_path)
    fp16_mb = get_total_model_size_mb(output_path)
    ratio = orig_mb / fp16_mb if fp16_mb > 0 else 0
    logger.info(f"  {input_path.name}: {orig_mb:.1f} -> {fp16_mb:.1f} MB ({ratio:.1f}x)")

    return output_path


def convert_quantized_to_fp16(input_path: pathlib.Path, output_path: pathlib.Path) -> pathlib.Path:
    """Convert a MatMulNBits (q4 or q8) ONNX model to FP16 activations (q4f16, q8f16).

    The mix for CUDA, whose GroupQueryAttention has no float32 kernel:
      - Quant weights (uint8 packed) and zero points (uint8) are unchanged.
      - Scales and all other float32 initializers become float16 — MatMulNBits
        requires its T1 (input) / scales / output to share dtype.
      - KV cache and conv-state inputs / outputs (CACHE_INPUT, CACHE_OUTPUT) become float16.
      - Other float32 inputs (LFM3-VL's image/video features, a vision encoder's
        pixel_values) stay float32, with a Cast(FLOAT -> FLOAT16) node inserted after them.
      - Other float32 outputs (logits, a vision encoder's image_features) stay float32, with a
        Cast(FLOAT16 -> FLOAT) node inserted before them.
      - value_info is cleared so ORT re-infers intermediate dtypes from the
        FP16 activations (otherwise stale FLOAT annotations conflict).
    """
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper
    from onnx.external_data_helper import load_external_data_for_model

    logger.info(f"Converting {input_path.name} to FP16 activations...")

    model = onnx.load(str(input_path), load_external_data=False)
    load_external_data_for_model(model, str(input_path.parent))
    graph = model.graph

    fp16_min = np.finfo(np.float16).min
    fp16_max = np.finfo(np.float16).max
    new_initializers = []
    for init in graph.initializer:
        if init.data_type == TensorProto.FLOAT:
            arr = numpy_helper.to_array(init)
            arr_fp16 = np.clip(arr, fp16_min, fp16_max).astype(np.float16)
            new_initializers.append(numpy_helper.from_array(arr_fp16, init.name))
        else:
            new_initializers.append(init)
    del graph.initializer[:]
    graph.initializer.extend(new_initializers)

    del graph.value_info[:]

    casts = []
    for inp in graph.input:
        if inp.type.tensor_type.elem_type != TensorProto.FLOAT:
            continue
        if CACHE_INPUT.fullmatch(inp.name):
            inp.type.tensor_type.elem_type = TensorProto.FLOAT16
            continue
        cast_output = f"{inp.name}_fp16"
        for node in graph.node:
            for j, name in enumerate(node.input):
                if name == inp.name:
                    node.input[j] = cast_output
        casts.append(helper.make_node("Cast", [inp.name], [cast_output], to=TensorProto.FLOAT16))
    if casts:
        nodes = [*casts, *graph.node]
        del graph.node[:]
        graph.node.extend(nodes)

    for out in graph.output:
        if out.type.tensor_type.elem_type != TensorProto.FLOAT:
            continue
        if CACHE_OUTPUT.fullmatch(out.name):
            out.type.tensor_type.elem_type = TensorProto.FLOAT16
            continue
        cast_input = f"{out.name}_fp16"
        for node in graph.node:
            for names in (node.input, node.output):
                for j, name in enumerate(names):
                    if name == out.name:
                        names[j] = cast_input
        graph.node.append(helper.make_node("Cast", [cast_input], [out.name], to=TensorProto.FLOAT))

    save_model(model, output_path)

    orig_mb = get_total_model_size_mb(input_path)
    fp16_mb = get_total_model_size_mb(output_path)
    logger.info(f"  {input_path.name}: {orig_mb:.1f} -> {fp16_mb:.1f} MB")
    return output_path


def get_model_name(model_path: str) -> str:
    """Extract model name from HF slug or local path."""
    # Handle HF slugs like "LiquidAI/LFM3-1.7B" -> "LFM3-1.7B"
    if "/" in model_path:
        return model_path.split("/")[-1]
    # Handle local paths
    return pathlib.Path(model_path).name


def export_model(model_path: str, output_dir: pathlib.Path | str):
    """Export LFM3 model to ONNX.

    Creates output structure:
        output_dir/
        ├── config.json
        ├── tokenizer.json
        ├── tokenizer_config.json
        ├── generation_config.json
        ├── chat_template.jinja
        └── onnx/
            ├── model.onnx
            └── model.onnx_data
    """
    output_dir = pathlib.Path(output_dir)
    repo, revision = model_path, REVISIONS.get(model_path)
    checkpoint = resolve_checkpoint(repo, CHECKPOINT_PATTERNS, revision=revision)
    hf_config = json.loads((checkpoint / "config.json").read_text())
    lfm3_config = LFM3Config.from_hf_config(hf_config)

    builder = LFM3Builder(lfm3_config)
    model = builder.build(checkpoint)

    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_dir = output_dir / "onnx"
    onnx_dir.mkdir(exist_ok=True)

    output_path = save_model(model, onnx_dir / "model.onnx")
    logger.info(f"Model saved to {output_path}")

    tokenizer = load_tokenizer(checkpoint)
    tokenizer.save_pretrained(output_dir)
    # The checkpoint's config.json as is: a save_pretrained of the loaded config would write
    # the layer types transformers renamed (5.15-5.17: "conv" -> "linear_attention").
    for path in checkpoint.glob("configuration_*.py"):
        (output_dir / path.name).write_text(path.read_text())

    gen_config = {
        "_from_model_config": True,
        "bos_token_id": hf_config.get("bos_token_id"),
        "eos_token_id": hf_config.get("eos_token_id"),
        "pad_token_id": hf_config.get("pad_token_id", 0),
        "transformers_version": "4.54.0",
    }
    gen_config_path = output_dir / "generation_config.json"
    gen_config_path.write_text(json.dumps(gen_config, indent=2))

    hf_config["transformers.js_config"] = {
        "kv_cache_dtype": {
            "fp32": "float32",
            "fp16": "float16",
            "q4f16": "float16",
            "q8f16": "float16",
        },
        "use_external_data_format": True,
    }
    (output_dir / "config.json").write_text(json.dumps(hf_config, indent=2))

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


def do_quantize(
    onnx_dir: pathlib.Path,
    bits: int,
    exclude_lm_head: bool,
    block_size: int,
    symmetric: bool = False,
    output_name: str | None = None,
    source: str = "model.onnx",
):
    """Quantize model to INT4 or INT8.

    Args:
        onnx_dir: Directory containing source model
        bits: Quantization bits (4 or 8)
        exclude_lm_head: Keep lm_head in FP32
        block_size: Block size for quantization
        symmetric: Use symmetric quantization (no zero points, better JSEP compatibility)
        output_name: Override output filename (default: model_q{bits}.onnx)
        source: Source model filename to quantize from (default: model.onnx)
    """
    input_model = onnx_dir / source
    if not input_model.exists():
        raise FileNotFoundError(f"{source} not found in {onnx_dir}")

    output_model = onnx_dir / (output_name or f"model_q{bits}.onnx")

    orig_mb = get_total_model_size_mb(input_model)

    quant_type = "symmetric" if symmetric else "asymmetric"
    logger.info(f"Quantizing {source} to Q{bits} ({quant_type})...")
    quantize_model(
        input_model,
        output_model,
        bits=bits,
        block_size=block_size,
        exclude_lm_head=exclude_lm_head,
        symmetric=symmetric,
    )

    quant_mb = get_total_model_size_mb(output_model)
    if orig_mb > 0:
        logger.info(f"  {orig_mb:.1f} MB -> {quant_mb:.1f} MB ({orig_mb / quant_mb:.1f}x)")


def derive_quantized(
    source: pathlib.Path,
    target: pathlib.Path,
    bits: int,
    block_size: int,
    keep: bool,
    fp16_activations: bool,
    int8_nodes: Callable[[onnx.ModelProto], list[str]] = q4_int8_nodes,
):
    """fp32 decoder (source) -> q{bits} (target), with the INT8 lm_head whose weights are the
    embedding (liquidonnx.lfm3.quantize), and with fp16_activations its q{bits}f16 version next
    to it ({target.stem}f16.onnx). int8_nodes: the MatMuls q4 keeps in int8.

    Without keep, the q{bits} graph is only what q{bits}f16 is converted from. It is then built in
    a scratch folder, as lfm2-export builds the q4 of a q4f16: the run leaves no file it wasn't
    asked for, and a target of an earlier run stays as it is.
    """
    q4 = functools.partial(quantize_decoder_q4, int8_nodes=int8_nodes)
    quantize = {4: q4, 8: quantize_decoder_q8}[bits]
    f16_target = target.with_name(f"{target.stem}f16.onnx")
    if keep:
        quantize(source, target, block_size)
        if fp16_activations:
            convert_quantized_to_fp16(target, f16_target)
    elif fp16_activations:
        with tempfile.TemporaryDirectory(dir=target.parent, prefix=f".q{bits}-") as scratch:
            intermediate = quantize(source, pathlib.Path(scratch) / target.name, block_size)
            convert_quantized_to_fp16(intermediate, f16_target)


def check_q4_block_size(
    parser: argparse.ArgumentParser,
    decoder: pathlib.Path,
    block_size: int,
    int8_nodes: Callable[[onnx.ModelProto], list[str]] = q4_int8_nodes,
):
    """parser.error before any quantization when block_size can't make the q4 of the fp32 decoder
    (q4_block_size_error), which reads the graph without its external data."""
    model = onnx.load(str(decoder), load_external_data=False)
    if error := q4_block_size_error(model, block_size, int8_nodes):
        parser.error(f"--block-size {block_size} can't make q4 or q4f16. {error}")


def do_fp16(onnx_dir: pathlib.Path):
    """Convert model to FP16."""
    # fp16 caches: they stay in the graph's dtype, so the KV buffers can be shared in place
    convert_to_fp16(onnx_dir / "model.onnx", onnx_dir / "model_fp16.onnx", keep_io_types=["logits"])


def main():
    parser = argparse.ArgumentParser(
        description="Export LFM3 models to ONNX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "model",
        help="HuggingFace model ID or local path (e.g., LiquidAI/LFM3-1.7B)",
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
        help="Skip the FP32 export: derive the precisions from the folder's onnx/model.onnx",
    )
    parser.add_argument(
        "--precision",
        nargs="*",
        metavar="PRECISION",
        help="Output precisions: fp16, q4, q4f16, q4f32, q8, q8f16, or all (default if no args)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=DEFAULT_BLOCK_SIZE,
        help=BLOCK_SIZE_HELP,
    )
    parser.add_argument(
        "--q4-asymmetric",
        action="store_true",
        help="Asymmetric INT4 for q4f32 (default: symmetric). q4 and q4f16 are k_quant, "
        "which always writes zero points",
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

    # Derive output name from model path
    model_name = get_model_name(args.model)
    output_name = args.output_name or f"{model_name}-ONNX"
    output_dir = args.output_dir / "exports" / output_name

    do_fp16_conversion = False
    do_q4 = False
    do_q4f16_conversion = False
    do_q4f32 = False
    do_q8 = False
    do_q8f16 = False
    if args.precision is not None:
        if len(args.precision) == 0:
            do_fp16_conversion = True
            do_q4 = True
            do_q4f16_conversion = True
            do_q4f32 = True
            do_q8 = True
            do_q8f16 = True
        else:
            for p in args.precision:
                p = p.lower()
                if p == "fp16":
                    do_fp16_conversion = True
                elif p == "q4":
                    do_q4 = True
                elif p == "q4f16":
                    do_q4f16_conversion = True
                elif p == "q4f32":
                    do_q4f32 = True
                elif p == "q8":
                    do_q8 = True
                elif p == "q8f16":
                    do_q8f16 = True
                else:
                    parser.error(
                        f"Invalid precision: {p}. Use fp16, q4, q4f16, q4f32, q8, or q8f16."
                    )

    onnx_dir = output_dir / "onnx"
    if args.skip_export and not (onnx_dir / "model.onnx").exists():
        parser.error(f"--skip-export needs model.onnx in {onnx_dir}")

    if not args.skip_export:
        logger.info("=" * 60)
        logger.info("Exporting model (FP32)")
        logger.info("=" * 60)
        logger.info(f"Exporting {args.model} to {output_dir}...")
        export_model(args.model, str(output_dir))
        logger.info(f"  {model_name}: OK")
    if do_q4 or do_q4f16_conversion:
        # Up front with --skip-export, else before any other precision is derived
        check_q4_block_size(parser, onnx_dir / "model.onnx", args.block_size)

    if do_fp16_conversion:
        logger.info("=" * 60)
        logger.info("Converting to FP16")
        logger.info("=" * 60)
        do_fp16(onnx_dir)
        logger.info(f"  {model_name}: OK")

    for bits, keep, fp16_activations in ((4, do_q4, do_q4f16_conversion), (8, do_q8, do_q8f16)):
        names = [name for name, on in ((f"Q{bits}", keep), (f"Q{bits}F16", fp16_activations)) if on]
        if not names:
            continue
        logger.info("=" * 60)
        logger.info(f"Quantizing to {' and '.join(names)}")
        logger.info("=" * 60)
        derive_quantized(
            onnx_dir / "model.onnx",
            onnx_dir / f"model_q{bits}.onnx",
            bits,
            args.block_size,
            keep=keep,
            fp16_activations=fp16_activations,
        )
        logger.info(f"  {model_name}: OK")

    if do_q4f32:
        logger.info("=" * 60)
        logger.info("Quantizing to Q4F32")
        logger.info("=" * 60)
        symmetric = not args.q4_asymmetric
        do_quantize(
            onnx_dir,
            bits=4,
            exclude_lm_head=True,
            block_size=args.block_size,
            symmetric=symmetric,
            output_name="model_q4f32.onnx",
        )
        logger.info(f"  {model_name}: OK")

    built = [
        precision
        for precision, on in (
            ("fp32", not args.skip_export),
            ("fp16", do_fp16_conversion),
            ("q4", do_q4),
            ("q4f16", do_q4f16_conversion),
            ("q4f32", do_q4f32),
            ("q8", do_q8),
            ("q8f16", do_q8f16),
        )
        if on
    ]
    repo, revision = args.model, REVISIONS.get(args.model)
    write_export_info(output_dir, repo, built, revision)

    if not args.no_split_data:
        chunk_size_bytes = int(args.split_data * 1024 * 1024 * 1024)
        for onnx_file in onnx_dir.glob("model*.onnx"):
            data_file = onnx_file.with_suffix(".onnx_data")
            if data_file.exists() and data_file.stat().st_size > chunk_size_bytes:
                logger.info("=" * 60)
                logger.info(f"Splitting external data ({args.split_data:.1f} GB chunks)")
                logger.info("=" * 60)
                logger.info(f"  Splitting {onnx_file.name}...")
                split_external_data(onnx_file, chunk_size=chunk_size_bytes)
    apply_umask(output_dir)

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
