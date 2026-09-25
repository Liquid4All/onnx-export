#!/usr/bin/env python3
"""
ONNX export of LFM2.5-Audio for onnxruntime-genai, covering all three modes:
- ASR (Automatic Speech Recognition): Audio -> Text
- TTS (Text-to-Speech): Text -> Audio
- Interleaved: Mixed text and audio I/O

The decoder comes from the onnxruntime-genai model builder (fp32, CPU EP, inputs_embeds in,
logits and hidden states out; see liquidonnx.genai_builder). This repository builds the other
graphs and derives every precision.

Output Structure:
    {output-dir}/exports/{model-name}-ONNX/
        ├── genai_config.json          # lfm2_audio pipeline at the default precision
        ├── config.json, tokenizer.json, tokenizer_config.json
        └── onnx/
            ├── decoder.onnx               # LFM2 backbone (inputs_embeds -> logits, hidden_states)
            ├── embeddings.onnx            # token table + audio feature scatter (fp32 table)
            ├── embeddings_fp16.onnx       # fp16 table, used with every other precision
            ├── audio_encoder.onnx         # Conformer: mel-spectrogram -> audio features
            ├── audio_embedding.onnx       # audio codes -> decoder input (+ .bin/.json table)
            ├── vocoder_depthformer.onnx   # decoder hidden state -> frame of 8 audio codes
            ├── audio_detokenizer.onnx     # audio codes -> STFT features (outside the runtime)
            ├── embed_tokens.bin/.json     # text embedding table for web runtimes
            └── mel_config.json

    fp16, q4 and q8 add {graph}_{precision}.onnx; bundle() lists what each precision loads. The
    depthformer and audio embedding stay fp16 in the q4 and q8 bundles: quantizing the depthformer
    changes most audio codes.

genai_config.json uses the first exported precision of q4, q8, fp16, fp32.

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

import numpy as np
import onnx
from onnx import TensorProto, helper

from liquidonnx.embeddings import build_embeddings, embeddings_to_fp16
from liquidonnx.export_cli import (
    add_export_arguments,
    finish,
    log_step,
    output_dir,
    parse_precisions,
)
from liquidonnx.genai_builder import export_decoder
from liquidonnx.lfm2_audio.builder.config import ConformerConfig
from liquidonnx.lfm2_audio.builder.conformer_builder import ConformerEncoderBuilder
from liquidonnx.lfm2_audio.builder.depthformer_builder import export_vocoder_depthformer
from liquidonnx.lfm2_audio.builder.detokenizer_builder import (
    export_audio_detokenizer_builder,
)
from liquidonnx.quantize import (
    convert_to_fp16,
    derive_precision,
    get_model_size,
    quantize_model,
)

logger = logging.getLogger(__name__)

PRECISIONS = ("fp16", "q4", "q8")
DEFAULT_ORDER = ("q4", "q8", "fp16")
DECODER_OPTIONS = {"exclude_embeds": "true", "include_hidden_states": "true"}
# Token ids this export hard-codes; write_genai_config checks them against tokenizer.json.
SPECIAL_TOKEN_IDS = {"<|reserved_123|>": 133, "<|audio_start|>": 128, "<|text_end|>": 130}
# The placeholder the embedding model replaces with audio features. The onnxruntime-genai audio
# processor writes one per encoder frame.
AUDIO_TOKEN_ID = SPECIAL_TOKEN_IDS["<|reserved_123|>"]
# The model turns to speech here, so these only end generation in a text-only pipeline.
MODALITY_SWITCH_TOKEN_IDS = (
    SPECIAL_TOKEN_IDS["<|audio_start|>"],
    SPECIAL_TOKEN_IDS["<|text_end|>"],
)


def bundle(precision: str) -> dict[str, str]:
    """ONNX files (in onnx/) that make up one precision."""
    suffix = "" if precision == "fp32" else f"_{precision}"
    fp16 = "" if precision == "fp32" else "_fp16"
    return {
        "decoder": f"decoder{suffix}.onnx",
        "embedding": f"embeddings{fp16}.onnx",
        "speech": f"audio_encoder{suffix}.onnx",
        "depthformer": f"vocoder_depthformer{fp16}.onnx",
        "audio_embedding": f"audio_embedding{fp16}.onnx",
        "detokenizer": f"audio_detokenizer{suffix}.onnx",
    }


def genai_files(precision: str) -> dict:
    """genai_config.json model entries that load one precision (the detokenizer runs outside)."""
    files = {name: {"filename": f"onnx/{file}"} for name, file in bundle(precision).items()}
    return {
        "decoder": files["decoder"],
        "embedding": files["embedding"],
        "speech": files["speech"],
        "audio_output": {
            "depthformer": files["depthformer"],
            "embedding": files["audio_embedding"],
        },
    }


def load_audio_model_weights(model_path: str) -> dict[str, np.ndarray]:
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
        logger.warning(f"audio_embedding vocab_size={vocab_size}, expected {expected_vocab}")

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
# onnxruntime-genai looks up text through embeddings.onnx. Web runtimes, which cannot read ONNX
# initializers, index this raw table instead:
#   weight = load_binary("embed_tokens.bin")
#   embedding = weight[token_id]


def export_embed_tokens(
    weights: dict[str, np.ndarray], config: dict, onnx_dir: pathlib.Path
) -> pathlib.Path:
    """Export embed_tokens weight as raw binary.

    Saves embed_tokens.weight as:
    1. embed_tokens.bin - raw float32 binary (vocab_size * hidden_size * 4 bytes)
    2. embed_tokens.json - metadata (vocab_size, hidden_size, dtype)
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
    logger.info("embed_tokens.json saved")

    return bin_path


# === 4. Precisions ===


def derive_precision_files(onnx_dir: pathlib.Path, precision: str, block_size: int):
    """Write the files bundle(precision) loads."""
    derive_precision(onnx_dir, precision, name="decoder", block_size=block_size)

    for name in ("audio_encoder", "audio_detokenizer"):
        fp32_path = onnx_dir / f"{name}.onnx"
        output_path = onnx_dir / f"{name}_{precision}.onnx"
        if precision == "fp16":
            convert_to_fp16(fp32_path, output_path, keep_io=True)
            continue
        bits = int(precision[1])
        _, orig_mb = get_model_size(fp32_path)
        quantize_model(
            fp32_path,
            output_path,
            bits=bits,
            block_size=block_size,
            exclude_lm_head=False,
            symmetric=bits == 4,
        )
        _, quant_mb = get_model_size(output_path)
        logger.info(f"  {name}: {orig_mb:.1f} -> {quant_mb:.1f} MB")

    # The depthformer and audio embedding stay fp16 in every non-fp32 bundle; the audio
    # embedding is a Gather, which weight-only quantization leaves at full size anyway.
    for name in ("vocoder_depthformer", "audio_embedding"):
        convert_to_fp16(onnx_dir / f"{name}.onnx", onnx_dir / f"{name}_fp16.onnx", keep_io=True)
    embeddings_to_fp16(onnx_dir / "embeddings.onnx", onnx_dir / "embeddings_fp16.onnx")


def write_genai_config(output_dir: pathlib.Path, precision: str):
    """Point genai_config.json at one precision and add the speech input and output sections."""
    tokenizer = json.loads((output_dir / "tokenizer.json").read_text())
    ids = {token["content"]: token["id"] for token in tokenizer["added_tokens"]}
    if any(ids.get(name) != i for name, i in SPECIAL_TOKEN_IDS.items()):
        found = {name: ids.get(name) for name in SPECIAL_TOKEN_IDS}
        raise ValueError(f"tokenizer.json has {found}; the export assumes {SPECIAL_TOKEN_IDS}")

    files = genai_files(precision)
    config_path = output_dir / "genai_config.json"
    config = json.loads(config_path.read_text())
    model = config["model"]
    model["decoder"].update(files["decoder"])
    model["audio_token_id"] = AUDIO_TOKEN_ID
    eos = (
        model["eos_token_id"]
        if isinstance(model["eos_token_id"], list)
        else [model["eos_token_id"]]
    )
    model["eos_token_id"] = [t for t in eos if t not in MODALITY_SWITCH_TOKEN_IDS]
    model["embedding"] = {
        **files["embedding"],
        "inputs": {"input_ids": "input_ids", "audio_features": "audio_features"},
        "outputs": {"inputs_embeds": "inputs_embeds"},
    }
    model["speech"] = {
        **files["speech"],
        "inputs": {
            "audio_embeds": "mel_spectrogram",
            "audio_lengths": "mel_lengths",
            "audio_sizes": "audio_sizes",
        },
        "outputs": {"audio_features": "audio_embeddings"},
    }
    model["audio_output"] = files["audio_output"]
    config_path.write_text(json.dumps(config, indent=4))
    logger.info(f"genai_config.json -> {precision}")


# === 5. Mel Config ===


def save_mel_config(onnx_dir: pathlib.Path):
    """Mel front-end parameters of liquid-audio's AudioToMelSpectrogramPreprocessor, for web
    runtimes (onnxruntime-genai has the same settings built in)."""
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

    config_path = onnx_dir / "mel_config.json"
    with open(config_path, "w") as f:
        json.dump(mel_config, f, indent=2)
    logger.info(f"Mel config saved to {config_path}")


# === Main Export ===


def export_full_model(model_path: str, output_dir: pathlib.Path):
    """Export every fp32 graph of LFM2.5-Audio, plus config and tokenizer."""
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_dir = output_dir / "onnx"
    onnx_dir.mkdir(exist_ok=True)

    config = load_audio_config(model_path)
    weights = load_audio_model_weights(model_path)

    export_audio_embedding(weights, config, onnx_dir)
    export_audio_embedding_binary(weights, config, onnx_dir)
    export_embed_tokens(weights, config, onnx_dir)
    build_embeddings(
        weights["lfm.embed_tokens.weight"],
        AUDIO_TOKEN_ID,
        "audio_features",
        onnx_dir / "embeddings.onnx",
    )
    weights.clear()
    gc.collect()

    export_decoder(model_path, output_dir, "decoder.onnx", DECODER_OPTIONS)
    export_audio_encoder_builder(model_path, config, onnx_dir)
    export_vocoder_depthformer(model_path, onnx_dir)
    export_audio_detokenizer_builder(model_path, onnx_dir)
    save_mel_config(onnx_dir)
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="ONNX export for LFM2.5-Audio (ASR, TTS, Interleaved)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    add_export_arguments(parser, PRECISIONS, PRECISIONS, "LiquidAI/LFM2.5-Audio-1.5B")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    precisions = parse_precisions(parser, args, PRECISIONS, PRECISIONS)
    export_dir = output_dir(args)
    onnx_dir = export_dir / "onnx"

    log_step(f"Exporting {args.model} (fp32) to {export_dir}")
    export_full_model(args.model, export_dir)

    for precision in precisions:
        log_step(f"Deriving {precision}")
        derive_precision_files(onnx_dir, precision, args.block_size)

    write_genai_config(export_dir, next((p for p in DEFAULT_ORDER if p in precisions), "fp32"))

    finish(args, export_dir)


if __name__ == "__main__":
    main()
