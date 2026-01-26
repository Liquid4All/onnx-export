#!/usr/bin/env python3
"""
ONNX export for LFM2.5-Audio model supporting all 3 modes:
- ASR (Automatic Speech Recognition): Audio -> Text
- TTS (Text-to-Speech): Text -> Audio
- Interleaved: Mixed text and audio I/O

Exports the following ONNX models:
1. decoder.onnx - LFM2 backbone with text embeddings (input_ids -> logits/hidden_states)
2. audio_encoder.onnx - Conformer encoder for ASR (mel-spectrogram -> audio embeddings)
3. audio_embedding.onnx - Audio code embeddings for TTS/interleaved
4. audio_detokenizer.onnx - Neural vocoder for TTS (codes -> STFT features)

Note: Depthformer (audio codebook prediction) uses PyTorch at inference time for
autoregressive generation, which produces higher quality audio than parallel ONNX.

Usage:
    uv run lfm2-audio-export LiquidAI/LFM2.5-Audio-1.5B
    uv run lfm2-audio-export LiquidAI/LFM2.5-Audio-1.5B --precision fp16
    uv run lfm2-audio-export LiquidAI/LFM2.5-Audio-1.5B --precision q4
"""

import argparse
import gc
import json
import logging
import pathlib
import shutil

import numpy as np
import onnx
from onnx import TensorProto, helper

from liquidonnx.external_data import split_external_data
from liquidonnx.lfm2.builder import LFM2Builder, LFM2Config
from liquidonnx.lfm2_audio.builder.config import ConformerConfig
from liquidonnx.lfm2_audio.builder.conformer_builder import ConformerEncoderBuilder
from liquidonnx.lfm2_audio.builder.depthformer_builder import (
    export_vocoder_depthformer,
)
from liquidonnx.lfm2_audio.builder.detokenizer_builder import (
    export_audio_detokenizer_builder,
)
from liquidonnx.quantize import get_model_size, get_total_model_size_mb, quantize_model

logger = logging.getLogger(__name__)


def get_model_name(model_path: str) -> str:
    if "/" in model_path:
        return model_path.split("/")[-1]
    return pathlib.Path(model_path).name


def load_audio_model_weights(model_path: str) -> dict[str, np.ndarray]:
    """Load all weights from HuggingFace audio model."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    logger.info(f"Loading weights from {model_path}...")
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


# === 1. Audio Encoder Export (builder) ===


def export_audio_encoder_builder(
    model_path: str, config: dict, onnx_dir: pathlib.Path
) -> pathlib.Path:
    """Export Conformer audio encoder to ONNX using ONNX builder (no torch.onnx.export).

    Args:
        model_path: HuggingFace model ID or local path
        config: Model configuration dict
        onnx_dir: Output directory for ONNX models

    Returns:
        Path to exported audio_encoder.onnx
    """
    logger.info("Exporting audio_encoder.onnx (builder)...")

    # Create conformer config from model config
    encoder_config = config.get("encoder", {})
    conformer_config = ConformerConfig.from_hf_config(encoder_config)

    # Get adapter output dimension from LFM config
    adapter_output_dim = config.get("lfm", {}).get("hidden_size", 2048)

    # Build the model
    builder = ConformerEncoderBuilder(conformer_config, adapter_output_dim)
    model = builder.build(model_path)

    output_path = onnx_dir / "audio_encoder.onnx"
    onnx.save(model, str(output_path))

    logger.info(f"audio_encoder saved to {output_path}")
    return output_path


# === 2. Audio Embedding Export (builder) ===
# Note: embed_tokens is NOT exported separately - it's included in decoder.onnx
# and extracted at inference time for text embedding lookup.


def export_audio_embedding(
    weights: dict[str, np.ndarray], config: dict, onnx_dir: pathlib.Path
) -> pathlib.Path:
    """Export audio_embedding.onnx (audio token embedding lookup).

    Note: The embedding lookup does NOT apply normalization.
    The embedding_norm (RMSNorm) is only used in get_logits() for the inverse
    projection (embedding -> logits), not for the forward embedding lookup.

    Reference: liquid_audio/model/transformer.py SharedEmbedding.embed()
    """
    logger.info("Exporting audio_embedding.onnx...")

    nodes = []
    hidden_size = config.get("lfm", {}).get("hidden_size", 2048)

    embed_weight = weights["audio_embedding.embedding.weight"].astype(np.float32)

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
    ]

    # Just do embedding lookup - no normalization
    nodes.append(
        helper.make_node(
            "Gather",
            ["audio_embedding.weight", "audio_codes"],
            ["audio_embeds"],
            axis=0,
        )
    )

    graph = helper.make_graph(nodes, "audio_embedding", inputs, outputs, initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)], ir_version=10)
    model.producer_name = "liquidonnx"

    output_path = onnx_dir / "audio_embedding.onnx"
    onnx.save_model(model, str(output_path))
    logger.info(f"audio_embedding saved to {output_path}")
    return output_path


# === 3b. Audio Embedding Binary Export ===
#
# Export audio_embedding.weight as raw binary for direct lookup.
# This eliminates the ONNX model call overhead (352 calls per generation).


def export_audio_embedding_binary(
    weights: dict[str, np.ndarray], config: dict, onnx_dir: pathlib.Path
) -> pathlib.Path:
    """Export audio_embedding weight as raw binary.

    The audio embedding table has shape [num_codebooks * codebook_vocab, hidden_size]
    where num_codebooks=8 and codebook_vocab=2049 (including end-of-audio token).

    Token index = codebook_idx * 2049 + code_value

    Saves as:
    1. audio_embedding.bin - raw float32 binary
    2. audio_embedding.json - metadata (num_codebooks, codebook_vocab, hidden_size)

    This enables direct numpy/JS indexing instead of ONNX model calls.
    """
    embed_weight = weights["audio_embedding.embedding.weight"]  # [16392, hidden_size]
    vocab_size, hidden_size = embed_weight.shape

    # Validate expected shape
    num_codebooks = 8
    codebook_vocab = 2049
    expected_vocab = num_codebooks * codebook_vocab
    if vocab_size != expected_vocab:
        logger.warning(
            f"audio_embedding vocab_size={vocab_size}, expected {expected_vocab}"
        )

    logger.info(f"audio_embedding weight shape: {embed_weight.shape}")

    # Save as raw binary (float32, little-endian)
    bin_path = onnx_dir / "audio_embedding.bin"
    embed_weight.astype(np.float32).tofile(bin_path)
    logger.info(f"audio_embedding.bin saved ({bin_path.stat().st_size / 1e6:.1f} MB)")

    # Save metadata
    meta_path = onnx_dir / "audio_embedding.json"
    with open(meta_path, "w") as f:
        json.dump(
            {
                "vocab_size": int(vocab_size),
                "hidden_size": int(hidden_size),
                "num_codebooks": num_codebooks,
                "codebook_vocab": codebook_vocab,
                "dtype": "float32",
                "byte_order": "little",
            },
            f,
            indent=2,
        )
    logger.info("audio_embedding.json saved")

    return bin_path


# === 3c. Text Embedding Export ===
#
# Why export embed_tokens separately?
#
# The embed_tokens.weight is already stored in decoder.onnx (for the tied LM head),
# but we export it as a standalone binary file for cross-platform consistency:
#
# - Python CAN extract weights from ONNX via `onnx.load()` + graph.initializer
# - JavaScript CANNOT - ONNX Runtime Web only exposes inference APIs, not internals
#
# By exporting as raw binary, both Python and JS use the same artifact and code path:
#   weight = load_binary("embed_tokens.bin")
#   embedding = weight[token_id]
#
# This avoids maintaining two different extraction methods and ensures identical behavior.


def export_embed_tokens(
    weights: dict[str, np.ndarray], config: dict, onnx_dir: pathlib.Path
) -> pathlib.Path:
    """Export embed_tokens weight as raw binary.

    Saves embed_tokens.weight as:
    1. embed_tokens.bin - raw float32 binary (vocab_size * hidden_size * 4 bytes)
    2. embed_tokens.json - metadata (vocab_size, hidden_size, dtype)

    Both Python and JavaScript load this the same way for consistency.
    """
    embed_weight = weights["lfm.embed_tokens.weight"]  # [vocab_size, hidden_size]
    vocab_size, hidden_size = embed_weight.shape
    logger.info(f"embed_tokens weight shape: {embed_weight.shape}")

    # Save as raw binary (float32, little-endian)
    bin_path = onnx_dir / "embed_tokens.bin"
    embed_weight.astype(np.float32).tofile(bin_path)
    logger.info(f"embed_tokens.bin saved ({bin_path.stat().st_size / 1e6:.1f} MB)")

    # Save metadata
    meta_path = onnx_dir / "embed_tokens.json"
    with open(meta_path, "w") as f:
        json.dump(
            {
                "vocab_size": int(vocab_size),
                "hidden_size": int(hidden_size),
                "dtype": "float32",
                "byte_order": "little",
            },
            f,
            indent=2,
        )
    logger.info(f"embed_tokens.json saved")

    return bin_path


# === 4. Decoder Export (builder) ===


def export_decoder(
    weights: dict[str, np.ndarray], config: dict, onnx_dir: pathlib.Path
) -> pathlib.Path:
    """Export decoder.onnx (LFM2 backbone with inputs_embeds).

    Outputs both logits and hidden_states for audio generation.
    """
    logger.info("Exporting decoder.onnx...")

    lfm_config = config.get("lfm", {})
    lfm2_config = LFM2Config.from_hf_config(type("Config", (), lfm_config)())

    builder = LFM2Builder(lfm2_config, use_integrated_rope=True, vl_naming=True)

    # Load LFM weights (prefixed with "lfm.")
    for name, weight in weights.items():
        if name.startswith("lfm."):
            new_name = "model." + name[4:]
            builder.weights[new_name] = weight

    H = lfm2_config.hidden_size

    # Build inputs
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

    # Add hidden_states output for audio generation
    builder.outputs.append(
        helper.make_tensor_value_info(
            "hidden_states", TensorProto.FLOAT, ["batch_size", "sequence_length", H]
        )
    )

    builder.build_rope_cache()
    builder.build_attention_mask_subgraph()

    builder.add_initializer(
        "model.embed_tokens.weight", builder.weights["model.embed_tokens.weight"]
    )
    hidden_state = "inputs_embeds"

    for layer_idx in range(lfm2_config.num_hidden_layers):
        layer_type = lfm2_config.layer_types[layer_idx]
        logger.info(f"Building decoder layer {layer_idx} ({layer_type})...")
        builder.prepare_layer_weights(layer_idx, layer_type)

        if layer_type == "conv":
            hidden_state = builder.build_conv_layer(layer_idx, hidden_state)
        else:
            hidden_state = builder.build_attention_layer(layer_idx, hidden_state)

    # Build lm_head and capture hidden states
    # The build_lm_head applies final norm then lm_head projection
    # We need the normed hidden states before projection
    builder.build_lm_head(hidden_state)

    # Add Identity node to output hidden states (final norm output)
    # The final norm output is at /model/layers.{num_layers}/final_norm_layernorm/output_0
    num_layers = lfm2_config.num_hidden_layers
    final_norm_output = f"/model/layers.{num_layers}/final_norm_layernorm/output_0"
    builder.nodes.append(
        helper.make_node(
            "Identity",
            [final_norm_output],
            ["hidden_states"],
            name="/hidden_states/Identity",
        )
    )

    builder.build_value_info()

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


# === Quantization ===


def do_quantize(onnx_dir: pathlib.Path, bits: int, block_size: int, symmetric: bool):
    """Quantize all exportable models to specified precision."""
    # Models to quantize: (relative_path, exclude_lm_head)
    models_to_quantize = [
        ("decoder", True),
        ("audio_encoder", False),
        ("audio_embedding", False),
        ("audio_detokenizer", False),
        ("vocoder_depthformer", False),
    ]

    for model_path, exclude_lm_head in models_to_quantize:
        fp32_path = onnx_dir / f"{model_path}.onnx"
        quant_path = onnx_dir / f"{model_path}_q{bits}.onnx"

        if not fp32_path.exists():
            continue
        if quant_path.exists():
            logger.info(f"  {model_path}_q{bits}.onnx already exists, skipping")
            continue

        _, orig_mb = get_model_size(fp32_path)
        quantize_model(
            fp32_path,
            quant_path,
            bits=bits,
            block_size=block_size,
            exclude_lm_head=exclude_lm_head,
            symmetric=symmetric,
        )
        _, quant_mb = get_model_size(quant_path)
        logger.info(f"  {model_path}: {orig_mb:.1f} -> {quant_mb:.1f} MB")


# === FP16 Conversion ===


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

    model = onnx.load(str(input_path), load_external_data=False)
    load_external_data_for_model(model, str(input_path.parent))

    model_fp16 = convert_float_to_float16(
        model,
        keep_io_types=keep_io_types,
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
    )

    orig_mb = get_total_model_size_mb(input_path)
    fp16_mb = get_total_model_size_mb(output_path)
    ratio = orig_mb / fp16_mb if fp16_mb > 0 else 0
    logger.info(f"  {input_path.name}: {orig_mb:.1f} -> {fp16_mb:.1f} MB ({ratio:.1f}x)")


def do_fp16(onnx_dir: pathlib.Path):
    """Convert all models to FP16."""
    models = [
        "decoder",
        "audio_encoder",
        "audio_embedding",
        "audio_detokenizer",
        "vocoder_depthformer",
    ]

    for model_name in models:
        fp32_path = onnx_dir / f"{model_name}.onnx"
        fp16_path = onnx_dir / f"{model_name}_fp16.onnx"

        if not fp32_path.exists():
            continue
        if fp16_path.exists():
            logger.info(f"  {model_name}_fp16.onnx already exists, skipping")
            continue

        convert_to_fp16(fp32_path, fp16_path)


# === 7. Audio Detokenizer Export ===


def save_mel_config(onnx_dir: pathlib.Path):
    """Save mel spectrogram configuration for ASR preprocessing.

    The mel filterbank and window are generated at runtime using librosa,
    making inference compatible with transformers.js which cannot load numpy files.

    Parameters match liquid_audio's AudioToMelSpectrogramPreprocessor config.
    """
    # Mel spectrogram parameters from LFM2.5-Audio config
    mel_config = {
        "sample_rate": 16000,
        "n_fft": 512,
        "win_length": 400,  # window_size (0.025) * sample_rate
        "hop_length": 160,  # window_stride (0.01) * sample_rate
        "n_mels": 128,
        "fmin": 0,
        "fmax": 8000,  # sample_rate / 2
        "preemph": 0.97,
        "log_zero_guard": 5.960464477539063e-08,  # 2^-24
        "normalize": "per_feature",
        "mel_norm": "slaney",
    }

    # Save config only - filterbank and window generated at runtime via librosa
    config_path = onnx_dir / "mel_config.json"
    with open(config_path, "w") as f:
        json.dump(mel_config, f, indent=2)
    logger.info(f"Mel config saved to {config_path}")


# === Main Export ===


def export_full_model(model_path: str, output_dir: pathlib.Path):
    """Export all components of LFM2.5-Audio to ONNX.

    Exports:
    - decoder.onnx: LFM2 backbone (includes embed_tokens.weight for text embedding)
    - audio_encoder.onnx: Conformer encoder for ASR
    - audio_embedding.onnx: Audio code embeddings for TTS
    - audio_detokenizer.onnx: Neural vocoder for TTS
    - vocoder_depthformer.onnx: Autoregressive audio codebook prediction

    Note: embed_tokens.weight is stored in decoder.onnx (used for tied LM head).
    Inference extracts this weight for text embedding lookup.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_dir = output_dir / "onnx"
    onnx_dir.mkdir(exist_ok=True)

    # Load config and weights
    config = load_audio_config(model_path)
    weights = load_audio_model_weights(model_path)

    # === Builder-based exports (no PyTorch model needed) ===
    export_audio_embedding(weights, config, onnx_dir)
    export_audio_embedding_binary(weights, config, onnx_dir)  # For direct lookup
    export_embed_tokens(weights, config, onnx_dir)  # For web inference
    export_decoder(weights, config, onnx_dir)
    export_audio_encoder_builder(model_path, config, onnx_dir)
    export_vocoder_depthformer(model_path, onnx_dir)
    export_audio_detokenizer_builder(model_path, onnx_dir)
    save_mel_config(onnx_dir)

    # Clean up weights after builder exports
    weights.clear()
    gc.collect()

    # Copy config and tokenizer
    from huggingface_hub import hf_hub_download

    for filename in ["config.json", "tokenizer.json", "tokenizer_config.json"]:
        try:
            src = hf_hub_download(model_path, filename)
            shutil.copy(src, output_dir / filename)
        except Exception as e:
            logger.warning(f"Could not copy {filename}: {e}")

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("Export Summary")
    logger.info("=" * 60)
    total_size = 0
    for fpath in sorted(onnx_dir.iterdir()):
        if fpath.is_file():
            size = fpath.stat().st_size
            total_size += size
            logger.info(f"  {fpath.name}: {size / 1e6:.1f} MB")
    logger.info(f"Total: {total_size / 1e9:.2f} GB")

    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="ONNX export for LFM2.5-Audio (ASR, TTS, Interleaved)",
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
        help="Output base directory",
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
        help="Output precisions: fp16, q4, q8 (default if no args: fp16, q4, q8)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=32,
        help="Block size for quantization (default: 32)",
    )
    parser.add_argument(
        "--split-data",
        type=float,
        default=2.0,
        metavar="GB",
        help="Split external data into chunks (default: 2GB per chunk)",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    model_name = get_model_name(args.model)
    output_name = args.output_name or f"{model_name}-ONNX"
    output_dir = args.output_dir / "exports" / output_name
    onnx_dir = output_dir / "onnx"

    logger.info("=" * 60)
    logger.info("ONNX Export for LFM2.5-Audio")
    logger.info("=" * 60)

    export_full_model(args.model, output_dir)

    # Parse precision options
    do_fp16_conversion = False
    quant_bits = []
    if args.precision is not None:
        if len(args.precision) == 0:
            # Default: export all precisions
            do_fp16_conversion = True
            quant_bits = [4, 8]
        else:
            for p in args.precision:
                p = p.lower()
                if p == "fp16":
                    do_fp16_conversion = True
                elif p in ("q4", "q8"):
                    quant_bits.append(int(p[1]))

    # FP16 conversion
    if do_fp16_conversion:
        logger.info("=" * 60)
        logger.info("Converting to FP16")
        logger.info("=" * 60)
        do_fp16(onnx_dir)

    # Quantization
    for bits in quant_bits:
        logger.info("=" * 60)
        logger.info(f"Quantizing to Q{bits}")
        logger.info("=" * 60)
        do_quantize(onnx_dir, bits, args.block_size, symmetric=(bits == 4))

    # Split data
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
