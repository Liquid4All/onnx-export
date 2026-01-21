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
    export_depth_linear_builder,
    export_depthformer_unified_builder,
)
from liquidonnx.lfm2_audio.builder.detokenizer_builder import (
    export_audio_detokenizer_builder,
)
from liquidonnx.quantize import get_model_size, quantize_model

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


# === 2. Embed Tokens Export ===


def export_embed_tokens(
    weights: dict[str, np.ndarray], config: dict, onnx_dir: pathlib.Path
) -> pathlib.Path:
    """Export embed_tokens.onnx and embed_tokens.npy."""
    logger.info("Exporting embed_tokens.onnx...")

    lfm_config = config.get("lfm", {})
    hidden_size = lfm_config.get("hidden_size", 2048)

    if "lfm.embed_tokens.weight" not in weights:
        raise ValueError("Could not find embed_tokens weight")
    embed_weight = weights["lfm.embed_tokens.weight"].astype(np.float32)

    # Build simple Gather graph
    inputs = [
        helper.make_tensor_value_info(
            "input_ids", TensorProto.INT64, ["batch_size", "sequence_length"]
        )
    ]
    outputs = [
        helper.make_tensor_value_info(
            "inputs_embeds", TensorProto.FLOAT, ["batch_size", "sequence_length", hidden_size]
        )
    ]
    initializers = [onnx.numpy_helper.from_array(embed_weight, "model.embed_tokens.weight")]
    nodes = [
        helper.make_node(
            "Gather",
            ["model.embed_tokens.weight", "input_ids"],
            ["inputs_embeds"],
            name="/model/embed_tokens/Gather",
            axis=0,
        )
    ]

    graph = helper.make_graph(nodes, "embed_tokens", inputs, outputs, initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)], ir_version=10)
    model.producer_name = "liquidonnx"

    output_path = onnx_dir / "embed_tokens.onnx"
    onnx.save_model(model, str(output_path))
    logger.info(f"embed_tokens saved to {output_path}")

    # Also save numpy weights for PyTorch-free inference
    numpy_path = onnx_dir / "embed_tokens.npy"
    np.save(numpy_path, embed_weight)
    logger.info(f"embed_tokens.npy saved to {numpy_path}")

    return output_path


# === 3. Audio Embedding Export (builder) ===


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
        ("vocoder_projection", False),
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


# === 7. Audio Detokenizer Export ===


def save_mel_config(onnx_dir: pathlib.Path):
    """Save mel spectrogram configuration and filterbank for numpy-based preprocessing.

    This enables pure numpy mel spectrogram computation without PyTorch/torchaudio,
    making it portable to AMD NPU, Qualcomm NPU, etc.

    Parameters match liquid_audio's AudioToMelSpectrogramPreprocessor config.
    """
    import librosa

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

    # Generate mel filterbank matrix using librosa (same as NeMo/liquid_audio)
    mel_filterbank = librosa.filters.mel(
        sr=mel_config["sample_rate"],
        n_fft=mel_config["n_fft"],
        n_mels=mel_config["n_mels"],
        fmin=mel_config["fmin"],
        fmax=mel_config["fmax"],
        norm=mel_config["mel_norm"],
    ).astype(np.float32)

    # Generate hann window
    hann_window = np.hanning(mel_config["win_length"]).astype(np.float32)

    # Save config
    config_path = onnx_dir / "mel_config.json"
    with open(config_path, "w") as f:
        json.dump(mel_config, f, indent=2)
    logger.info(f"Mel config saved to {config_path}")

    # Save filterbank matrix [n_mels, n_fft//2+1] = [128, 257]
    filterbank_path = onnx_dir / "mel_filterbank.npy"
    np.save(filterbank_path, mel_filterbank)
    logger.info(f"Mel filterbank saved to {filterbank_path} {mel_filterbank.shape}")

    # Save hann window
    window_path = onnx_dir / "mel_window.npy"
    np.save(window_path, hann_window)
    logger.info(f"Mel window saved to {window_path} {hann_window.shape}")


# === Main Export ===


def export_full_model(model_path: str, output_dir: pathlib.Path):
    """Export all components of LFM2.5-Audio to ONNX.

    Exports:
    - embed_tokens.onnx/.npy: Text token embeddings
    - decoder.onnx: LFM2 backbone with text embeddings
    - audio_encoder.onnx: Conformer encoder for ASR
    - audio_embedding.onnx: Audio code embeddings for TTS
    - audio_detokenizer.onnx: Neural vocoder for TTS
    - vocoder_projection.onnx: Projects hidden states to depthformer space
    - vocoder_depthformer.onnx: Autoregressive audio codebook prediction
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_dir = output_dir / "onnx"
    onnx_dir.mkdir(exist_ok=True)

    # Load config and weights
    config = load_audio_config(model_path)
    weights = load_audio_model_weights(model_path)

    # === Builder-based exports (no PyTorch model needed) ===
    export_embed_tokens(weights, config, onnx_dir)
    export_audio_embedding(weights, config, onnx_dir)
    export_decoder(weights, config, onnx_dir)
    export_audio_encoder_builder(model_path, config, onnx_dir)
    export_depth_linear_builder(model_path, onnx_dir)
    export_depthformer_unified_builder(model_path, onnx_dir)
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
        help="Output precisions: q4, q8 (default if no args)",
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

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    model_name = get_model_name(args.model)
    output_name = args.output_name or f"{model_name}-ONNX"
    output_dir = args.output_dir / "exports" / output_name
    onnx_dir = output_dir / "onnx"

    logger.info("=" * 60)
    logger.info("ONNX Export for LFM2.5-Audio")
    logger.info("=" * 60)

    export_full_model(args.model, output_dir)

    # Quantize
    quant_bits = []
    if args.precision is not None:
        if len(args.precision) == 0:
            quant_bits = [4, 8]
        else:
            for p in args.precision:
                p = p.lower()
                if p in ("q4", "q8"):
                    quant_bits.append(int(p[1]))

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
