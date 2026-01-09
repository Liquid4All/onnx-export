#!/usr/bin/env python3
"""
Export LFM2.5-Audio models to ONNX with optional quantization.

Output Structure:
    {output-dir}/
    └── {model-name}-ONNX/
        ├── config.json
        ├── tokenizer.json
        └── onnx/
            ├── embed_tokens.onnx      # Text token embeddings
            ├── audio_encoder.onnx     # Conformer + adapter
            ├── decoder.onnx           # LFM2 backbone
            ├── depthformer.onnx       # Audio codebook prediction
            └── audio_detokenizer.onnx # Audio synthesis (optional)

Usage:
    uv run lfm2-audio-export LiquidAI/LFM2.5-Audio-1.5B
    uv run lfm2-audio-export LiquidAI/LFM2.5-Audio-1.5B --precision q4
"""

import argparse
import gc
import json
import logging
import pathlib

import numpy as np
import onnx
from onnx import TensorProto, helper

from liquidonnx.external_data import split_external_data
from liquidonnx.lfm2.builder import LFM2Builder, LFM2Config
from liquidonnx.quantize import get_model_size, quantize_model

logger = logging.getLogger(__name__)


def get_model_name(model_path: str) -> str:
    """Extract model name from HF slug or local path."""
    if "/" in model_path:
        return model_path.split("/")[-1]
    return pathlib.Path(model_path).name


def load_audio_model_weights(model_path: str) -> dict[str, np.ndarray]:
    """Load all weights from HuggingFace audio model."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    logger.info(f"Loading weights from {model_path}...")

    # Download safetensors file
    safetensors_path = hf_hub_download(model_path, "model.safetensors")

    weights = {}
    with safe_open(safetensors_path, framework="np", device="cpu") as f:
        for key in f.keys():
            weights[key] = f.get_tensor(key)

    logger.info(f"Loaded {len(weights)} weights")
    return weights


def load_audio_config(model_path: str) -> dict:
    """Load config.json from HuggingFace model."""
    from huggingface_hub import hf_hub_download

    config_path = hf_hub_download(model_path, "config.json")
    with open(config_path) as f:
        return json.load(f)


class EmbedTokensBuilder:
    """Simple token embedding builder for audio model."""

    def __init__(self, vocab_size: int, hidden_size: int):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embed_weight: np.ndarray | None = None

    def load_weights(self, weights: dict[str, np.ndarray]):
        if "lfm.embed_tokens.weight" in weights:
            self.embed_weight = weights["lfm.embed_tokens.weight"].astype(np.float32)
        else:
            raise ValueError("Could not find embed_tokens weight")

    def build(self) -> onnx.ModelProto:
        nodes = []
        inputs = [
            helper.make_tensor_value_info(
                "input_ids", TensorProto.INT64, ["batch_size", "sequence_length"]
            )
        ]
        outputs = [
            helper.make_tensor_value_info(
                "inputs_embeds",
                TensorProto.FLOAT,
                ["batch_size", "sequence_length", self.hidden_size],
            )
        ]

        initializers = [
            onnx.numpy_helper.from_array(self.embed_weight, "model.embed_tokens.weight")
        ]

        nodes.append(
            helper.make_node(
                "Gather",
                ["model.embed_tokens.weight", "input_ids"],
                ["inputs_embeds"],
                name="/model/embed_tokens/Gather",
                axis=0,
            )
        )

        graph = helper.make_graph(nodes, "embed_tokens", inputs, outputs, initializers)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)], ir_version=10)
        model.producer_name = "liquidonnx"
        return model


def export_embed_tokens(
    weights: dict[str, np.ndarray], config: dict, onnx_dir: pathlib.Path
) -> pathlib.Path:
    """Export embed_tokens.onnx."""
    logger.info("Exporting embed_tokens...")

    lfm_config = config.get("lfm", {})
    vocab_size = lfm_config.get("vocab_size", 65536)
    hidden_size = lfm_config.get("hidden_size", 2048)

    builder = EmbedTokensBuilder(vocab_size, hidden_size)
    builder.load_weights(weights)
    model = builder.build()

    output_path = onnx_dir / "embed_tokens.onnx"
    onnx.save_model(model, str(output_path))
    logger.info(f"embed_tokens saved to {output_path}")
    return output_path


def export_decoder(
    weights: dict[str, np.ndarray], config: dict, onnx_dir: pathlib.Path
) -> pathlib.Path:
    """Export decoder.onnx (LFM2 backbone with inputs_embeds)."""
    logger.info("Exporting decoder...")

    lfm_config = config.get("lfm", {})
    lfm2_config = LFM2Config.from_hf_config(type("Config", (), lfm_config)())

    builder = LFM2Builder(lfm2_config, use_integrated_rope=True, vl_naming=True)

    # Load LFM weights (prefixed with "lfm.")
    for name, weight in weights.items():
        if name.startswith("lfm."):
            new_name = "model." + name[4:]  # Remove "lfm." prefix, add "model."
            builder.weights[new_name] = weight

    H = lfm2_config.hidden_size

    # Build custom inputs: inputs_embeds instead of input_ids
    builder.inputs.append(
        helper.make_tensor_value_info(
            "inputs_embeds", TensorProto.FLOAT, ["batch_size", "sequence_length", H]
        )
    )
    builder.inputs.append(
        helper.make_tensor_value_info(
            "attention_mask", TensorProto.INT64, ["batch_size", "total_sequence_length"]
        )
    )

    # Cache inputs
    conv_set = set(builder.conv_indices)
    attn_set = set(builder.attn_indices)
    for idx in range(lfm2_config.num_hidden_layers):
        if idx in conv_set:
            builder.inputs.append(
                helper.make_tensor_value_info(
                    f"past_conv.{idx}",
                    TensorProto.FLOAT,
                    ["batch_size", H, lfm2_config.conv_L_cache],
                )
            )
        elif idx in attn_set:
            builder.inputs.append(
                helper.make_tensor_value_info(
                    f"past_key_values.{idx}.key",
                    TensorProto.FLOAT,
                    [
                        "batch_size",
                        lfm2_config.num_key_value_heads,
                        "past_sequence_length",
                        builder.head_dim,
                    ],
                )
            )
            builder.inputs.append(
                helper.make_tensor_value_info(
                    f"past_key_values.{idx}.value",
                    TensorProto.FLOAT,
                    [
                        "batch_size",
                        lfm2_config.num_key_value_heads,
                        "past_sequence_length",
                        builder.head_dim,
                    ],
                )
            )

    builder.build_outputs()
    builder.build_rope_cache()
    builder.build_attention_mask_subgraph()

    # Add embed_tokens weight for tied lm_head
    builder.add_initializer(
        "model.embed_tokens.weight", builder.weights["model.embed_tokens.weight"]
    )
    hidden_state = "inputs_embeds"

    # Build layers
    for layer_idx in range(lfm2_config.num_hidden_layers):
        layer_type = lfm2_config.layer_types[layer_idx]
        logger.info(f"Building decoder layer {layer_idx} ({layer_type})...")
        builder.prepare_layer_weights(layer_idx, layer_type)

        if layer_type == "conv":
            hidden_state = builder.build_conv_layer(layer_idx, hidden_state)
        else:
            hidden_state = builder.build_attention_layer(layer_idx, hidden_state)

    builder.build_lm_head(hidden_state)
    builder.build_value_info()

    # Build graph
    graph = helper.make_graph(
        builder.nodes,
        "decoder",
        builder.inputs,
        builder.outputs,
        builder.initializers,
        value_info=builder.value_info,
    )

    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 21),
            helper.make_opsetid("com.microsoft", 1),
        ],
        ir_version=10,
    )
    model.producer_name = "liquidonnx"

    output_path = onnx_dir / "decoder.onnx"
    output_data = onnx_dir / "decoder.onnx_data"
    if output_data.exists():
        output_data.unlink()

    onnx.save_model(
        model,
        str(output_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="decoder.onnx_data",
        size_threshold=1024,
    )
    logger.info(f"decoder saved to {output_path}")
    return output_path


def export_audio_embedding(
    weights: dict[str, np.ndarray], config: dict, onnx_dir: pathlib.Path
) -> pathlib.Path:
    """Export audio_embedding.onnx (audio token embedding lookup).

    This is for the audio tokens (8 codebooks × 2049 vocab = 16392 total).
    """
    logger.info("Exporting audio_embedding...")

    nodes = []
    hidden_size = config.get("lfm", {}).get("hidden_size", 2048)

    # Audio embedding: [16392, 2048]
    embed_weight = weights["audio_embedding.embedding.weight"].astype(np.float32)
    norm_weight = weights["audio_embedding.embedding_norm.weight"].astype(np.float32)

    inputs = [
        helper.make_tensor_value_info(
            "audio_codes", TensorProto.INT64, ["batch_size", "audio_length"]
        )
    ]
    outputs = [
        helper.make_tensor_value_info(
            "audio_embeds",
            TensorProto.FLOAT,
            ["batch_size", "audio_length", hidden_size],
        )
    ]

    initializers = [
        onnx.numpy_helper.from_array(embed_weight, "audio_embedding.weight"),
        onnx.numpy_helper.from_array(norm_weight, "audio_embedding_norm.weight"),
    ]

    # Gather embeddings
    nodes.append(
        helper.make_node(
            "Gather",
            ["audio_embedding.weight", "audio_codes"],
            ["/audio_embedding/Gather/output_0"],
            axis=0,
        )
    )

    # LayerNorm
    nodes.append(
        helper.make_node(
            "SimplifiedLayerNormalization",
            ["/audio_embedding/Gather/output_0", "audio_embedding_norm.weight"],
            ["audio_embeds"],
            epsilon=1e-5,
        )
    )

    graph = helper.make_graph(nodes, "audio_embedding", inputs, outputs, initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)], ir_version=10)
    model.producer_name = "liquidonnx"

    output_path = onnx_dir / "audio_embedding.onnx"
    onnx.save_model(model, str(output_path))
    logger.info(f"audio_embedding saved to {output_path}")
    return output_path


def convert_to_fp16(input_path: pathlib.Path, output_path: pathlib.Path):
    """Convert ONNX model from FP32 to FP16."""
    from onnx.external_data_helper import load_external_data_for_model
    from onnxruntime.transformers.float16 import convert_float_to_float16

    logger.info(f"Converting {input_path.name} to FP16...")

    model = onnx.load(str(input_path), load_external_data=False)
    load_external_data_for_model(model, str(input_path.parent))

    model_fp16 = convert_float_to_float16(
        model,
        keep_io_types=True,
        force_fp16_initializers=True,
        disable_shape_infer=True,
    )

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


def do_quantize(onnx_dir: pathlib.Path, bits: int, block_size: int, symmetric: bool):
    """Quantize decoder model."""
    decoder_fp32 = onnx_dir / "decoder.onnx"
    decoder_output = onnx_dir / f"decoder_q{bits}.onnx"

    if decoder_fp32.exists() and not decoder_output.exists():
        _, orig_mb = get_model_size(decoder_fp32)
        quantize_model(
            decoder_fp32,
            decoder_output,
            bits=bits,
            block_size=block_size,
            exclude_lm_head=True,
            symmetric=symmetric,
        )
        _, quant_mb = get_model_size(decoder_output)
        logger.info(f"  decoder: {orig_mb:.1f} -> {quant_mb:.1f} MB ({orig_mb / quant_mb:.1f}x)")


def do_fp16(onnx_dir: pathlib.Path):
    """Convert models to FP16."""
    for model_name in ["embed_tokens", "audio_embedding", "decoder"]:
        fp32_path = onnx_dir / f"{model_name}.onnx"
        fp16_path = onnx_dir / f"{model_name}_fp16.onnx"
        if fp32_path.exists() and not fp16_path.exists():
            convert_to_fp16(fp32_path, fp16_path)


def export_audio_model(model_path: str, output_dir: pathlib.Path):
    """Export LFM2.5-Audio model to ONNX."""
    from huggingface_hub import hf_hub_download

    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_dir = output_dir / "onnx"
    onnx_dir.mkdir(exist_ok=True)

    # Load config and weights
    config = load_audio_config(model_path)
    weights = load_audio_model_weights(model_path)

    # Export components
    export_embed_tokens(weights, config, onnx_dir)
    export_audio_embedding(weights, config, onnx_dir)
    export_decoder(weights, config, onnx_dir)

    # Clean up weights to save memory
    weights.clear()
    gc.collect()

    # Copy config and tokenizer
    for filename in ["config.json", "tokenizer.json", "tokenizer_config.json"]:
        try:
            src = hf_hub_download(model_path, filename)
            dst = output_dir / filename
            import shutil

            shutil.copy(src, dst)
        except Exception as e:
            logger.warning(f"Could not copy {filename}: {e}")

    # Print summary
    total_size = 0
    for fpath in onnx_dir.iterdir():
        if fpath.is_file():
            size = fpath.stat().st_size
            total_size += size
            logger.info(f"  {fpath.name}: {size / 1e6:.1f} MB")

    logger.info(f"Total ONNX size: {total_size / 1e9:.2f} GB")
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Export LFM2.5-Audio models to ONNX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "model",
        help="HuggingFace model ID (e.g., LiquidAI/LFM2.5-Audio-1.5B)",
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
        "--q4-asymmetric",
        action="store_true",
        help="Use asymmetric Q4 quantization",
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

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    # Parse precisions
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
                    parser.error(f"Invalid precision: {p}")

    # Derive output paths
    model_name = get_model_name(args.model)
    output_name = args.output_name or f"{model_name}-ONNX"
    output_dir = args.output_dir / "exports" / output_name
    onnx_dir = output_dir / "onnx"

    # Export
    if not args.skip_export:
        logger.info("=" * 60)
        logger.info("Exporting model (FP32)")
        logger.info("=" * 60)
        export_audio_model(args.model, output_dir)

    # Quantize
    for bits in quant_bits:
        logger.info("=" * 60)
        logger.info(f"Quantizing to Q{bits}")
        logger.info("=" * 60)
        symmetric = (bits == 4) and not args.q4_asymmetric
        do_quantize(onnx_dir, bits, args.block_size, symmetric)

    # FP16
    if do_fp16_conversion:
        logger.info("=" * 60)
        logger.info("Converting to FP16")
        logger.info("=" * 60)
        do_fp16(onnx_dir)

    # Split data
    if not args.no_split_data:
        chunk_size_bytes = int(args.split_data * 1024 * 1024 * 1024)
        for onnx_file in onnx_dir.glob("*.onnx"):
            data_file = onnx_file.with_suffix(".onnx_data")
            if data_file.exists() and data_file.stat().st_size > chunk_size_bytes:
                logger.info(f"Splitting {onnx_file.name}...")
                split_external_data(onnx_file, chunk_size=chunk_size_bytes)

    logger.info("=" * 60)
    logger.info("Export complete!")
    logger.info("=" * 60)
    logger.info(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
