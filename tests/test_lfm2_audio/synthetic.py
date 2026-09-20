"""Synthetic LFM2.5-Audio export for tests that must not download a checkpoint.

Builds every artifact `LFM2AudioInference` needs - decoder, conformer encoder,
detokenizer, depthformer, embedding binaries, mel config, tokenizer and config.json -
from small random weights. Shapes that infer.py hard-codes (8 codebooks, 2049 codebook
vocab, depthformer dim 1024 / 6 layers / 8 KV heads, detokenizer output 1282) are kept;
everything else is shrunk.
"""

import json
import pathlib

import numpy as np
import onnx
import scipy.io.wavfile
from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from liquidonnx.lfm2_audio.builder.config import ConformerConfig
from liquidonnx.lfm2_audio.builder.conformer_builder import ConformerEncoderBuilder
from liquidonnx.lfm2_audio.builder.depthformer_builder import DepthformerUnifiedBuilder
from liquidonnx.lfm2_audio.builder.detokenizer_builder import AudioDetokenizerBuilder
from liquidonnx.lfm2_audio.export import (
    export_audio_embedding_binary,
    export_decoder,
    export_embed_tokens,
    save_mel_config,
)

# Decoder (LFM2 backbone) - small everything
HIDDEN = 256
VOCAB = 256
NUM_HEADS = 8
NUM_KV_HEADS = 4
LAYER_TYPES = ["conv", "full_attention"]

# Ids infer.py relies on (LFM2AudioInference.*_TOKEN); must match the tokenizer below
SPECIAL_TOKENS = {
    "<|pad|>": 0,
    "<|startoftext|>": 1,
    "<|endoftext|>": 2,
    "[UNK]": 3,
    "<|im_start|>": 6,
    "<|im_end|>": 7,
    "<|audio_start|>": 128,
    "<|text_start|>": 129,
    "<|text_end|>": 130,
    "<|mixed_start|>": 131,
    "<|mixed_end|>": 132,
}
CHAR_BASE = 8  # printable ASCII 32..126 → 8..102, "\n" → 103

DETOK_CONFIG = {
    "hidden_size": 64,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "intermediate_size": 128,
    "output_size": 1282,
    "norm_eps": 1e-5,
    "sliding_window": 30,
    "rope_theta": 1000000.0,
    "layer_types": ["conv", "conv", "sliding_attention", "conv"],
}

AUDIO_VOCAB = 8 * 2049


def _w(rng: np.random.Generator, *shape: int) -> np.ndarray:
    return (rng.standard_normal(shape) * 0.02).astype(np.float32)


def char_id(ch: str) -> int:
    return CHAR_BASE + 95 if ch == "\n" else CHAR_BASE + (ord(ch) - 32)


# === Weights ===


def conformer_weights(cfg: ConformerConfig, adapter_out: int, rng) -> dict[str, np.ndarray]:
    C, D = cfg.subsampling_conv_channels, cfg.d_model
    hd, ff = D // cfg.n_heads, D * cfg.ff_expansion_factor
    freq = cfg.feat_in
    for _ in range(3):
        freq = (freq + 2 - 3) // 2 + 1
    ws = {
        "conformer.pre_encode.conv.0.weight": _w(rng, C, 1, 3, 3),
        "conformer.pre_encode.conv.0.bias": _w(rng, C),
        "conformer.pre_encode.conv.2.weight": _w(rng, C, 1, 3, 3),
        "conformer.pre_encode.conv.2.bias": _w(rng, C),
        "conformer.pre_encode.conv.3.weight": _w(rng, C, C, 1, 1),
        "conformer.pre_encode.conv.3.bias": _w(rng, C),
        "conformer.pre_encode.conv.5.weight": _w(rng, C, 1, 3, 3),
        "conformer.pre_encode.conv.5.bias": _w(rng, C),
        "conformer.pre_encode.conv.6.weight": _w(rng, C, C, 1, 1),
        "conformer.pre_encode.conv.6.bias": _w(rng, C),
        "conformer.pre_encode.out.weight": _w(rng, D, C * freq),
        "conformer.pre_encode.out.bias": _w(rng, D),
        "audio_adapter.model.0.weight": _w(rng, D),
        "audio_adapter.model.0.bias": _w(rng, D),
        "audio_adapter.model.1.weight": _w(rng, adapter_out, D),
        "audio_adapter.model.1.bias": _w(rng, adapter_out),
        "audio_adapter.model.3.weight": _w(rng, adapter_out, adapter_out),
        "audio_adapter.model.3.bias": _w(rng, adapter_out),
    }
    norms = ["norm_feed_forward1", "norm_feed_forward2", "norm_self_att", "norm_conv", "norm_out"]
    for i in range(cfg.n_layers):
        p = f"conformer.layers.{i}"
        for n in norms:
            ws[f"{p}.{n}.weight"] = _w(rng, D)
            ws[f"{p}.{n}.bias"] = _w(rng, D)
        for n in ["feed_forward1", "feed_forward2"]:
            ws[f"{p}.{n}.linear1.weight"] = _w(rng, ff, D)
            ws[f"{p}.{n}.linear1.bias"] = _w(rng, ff)
            ws[f"{p}.{n}.linear2.weight"] = _w(rng, D, ff)
            ws[f"{p}.{n}.linear2.bias"] = _w(rng, D)
        for n in ["q", "k", "v", "out"]:
            ws[f"{p}.self_attn.linear_{n}.weight"] = _w(rng, D, D)
            ws[f"{p}.self_attn.linear_{n}.bias"] = _w(rng, D)
        ws[f"{p}.self_attn.linear_pos.weight"] = _w(rng, D, D)
        ws[f"{p}.self_attn.pos_bias_u"] = _w(rng, cfg.n_heads, hd)
        ws[f"{p}.self_attn.pos_bias_v"] = _w(rng, cfg.n_heads, hd)
        ws[f"{p}.conv.pointwise_conv1.weight"] = _w(rng, 2 * D, D, 1)
        ws[f"{p}.conv.pointwise_conv1.bias"] = _w(rng, 2 * D)
        ws[f"{p}.conv.depthwise_conv.weight"] = _w(rng, D, 1, cfg.conv_kernel_size)
        ws[f"{p}.conv.depthwise_conv.bias"] = _w(rng, D)
        ws[f"{p}.conv.pointwise_conv2.weight"] = _w(rng, D, D, 1)
        ws[f"{p}.conv.pointwise_conv2.bias"] = _w(rng, D)
        ws[f"{p}.conv.batch_norm.weight"] = np.ones(D, dtype=np.float32)
        ws[f"{p}.conv.batch_norm.bias"] = _w(rng, D)
        ws[f"{p}.conv.batch_norm.running_mean"] = _w(rng, D)
        ws[f"{p}.conv.batch_norm.running_var"] = np.ones(D, dtype=np.float32)
    return ws


def detokenizer_weights(cfg: dict, rng) -> dict[str, np.ndarray]:
    H, inter = cfg["hidden_size"], cfg["intermediate_size"]
    nh, nkv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    hd = H // nh
    ws = {
        "emb.emb.weight": _w(rng, 8 * 2048, H),
        "lfm.embedding_norm.weight": _w(rng, H),
        "lin.weight": _w(rng, cfg["output_size"], H),
        "lin.bias": _w(rng, cfg["output_size"]),
    }
    for i, t in enumerate(cfg["layer_types"]):
        p = f"lfm.layers.{i}"
        ws[f"{p}.operator_norm.weight"] = _w(rng, H)
        ws[f"{p}.ffn_norm.weight"] = _w(rng, H)
        ws[f"{p}.feed_forward.w1.weight"] = _w(rng, inter, H)
        ws[f"{p}.feed_forward.w2.weight"] = _w(rng, H, inter)
        ws[f"{p}.feed_forward.w3.weight"] = _w(rng, inter, H)
        if t == "conv":
            ws[f"{p}.conv.in_proj.weight"] = _w(rng, 3 * H, H)
            ws[f"{p}.conv.conv.weight"] = _w(rng, H, 1, 3)
            ws[f"{p}.conv.out_proj.weight"] = _w(rng, H, H)
        else:
            ws[f"{p}.self_attn.q_proj.weight"] = _w(rng, nh * hd, H)
            ws[f"{p}.self_attn.k_proj.weight"] = _w(rng, nkv * hd, H)
            ws[f"{p}.self_attn.v_proj.weight"] = _w(rng, nkv * hd, H)
            ws[f"{p}.self_attn.out_proj.weight"] = _w(rng, H, nh * hd)
            ws[f"{p}.self_attn.q_layernorm.weight"] = _w(rng, hd)
            ws[f"{p}.self_attn.k_layernorm.weight"] = _w(rng, hd)
    return ws


def depthformer_weights(b: DepthformerUnifiedBuilder, rng) -> dict[str, np.ndarray]:
    ws = {
        "depth_linear.weight": _w(rng, b.num_codebooks * b.dim, b.input_hidden_size),
        "depth_linear.bias": _w(rng, b.num_codebooks * b.dim),
    }
    for i in range(b.num_codebooks):
        ws[f"depth_embeddings.{i}.embedding.weight"] = _w(rng, b.vocab_size, b.dim)
        ws[f"depth_embeddings.{i}.to_logits.weight"] = _w(rng, b.vocab_size, b.dim)
        ws[f"depth_embeddings.{i}.embedding_norm.weight"] = _w(rng, b.dim)
    qkv = (b.num_heads + 2 * b.num_kv_heads) * b.head_dim
    for i in range(b.num_layers):
        p = f"depthformer.layers.{i}"
        ws[f"{p}.operator_norm.weight"] = _w(rng, b.dim)
        ws[f"{p}.ffn_norm.weight"] = _w(rng, b.dim)
        ws[f"{p}.operator.qkv_proj.weight"] = _w(rng, qkv, b.dim)
        ws[f"{p}.operator.out_proj.weight"] = _w(rng, b.dim, b.num_heads * b.head_dim)
        ws[f"{p}.operator.bounded_attention.q_layernorm.weight"] = _w(rng, b.head_dim)
        ws[f"{p}.operator.bounded_attention.k_layernorm.weight"] = _w(rng, b.head_dim)
        ws[f"{p}.feed_forward.w1.weight"] = _w(rng, b.intermediate_size, b.dim)
        ws[f"{p}.feed_forward.w2.weight"] = _w(rng, b.dim, b.intermediate_size)
        ws[f"{p}.feed_forward.w3.weight"] = _w(rng, b.intermediate_size, b.dim)
    return ws


def decoder_weights(rng) -> dict[str, np.ndarray]:
    """LFM2 backbone weights with the `lfm.` prefix export_decoder() expects."""
    H, hd = HIDDEN, HIDDEN // NUM_HEADS
    inter = decoder_config()["lfm"]["intermediate_size"]
    ws = {
        "lfm.embed_tokens.weight": _w(rng, VOCAB, H),
        "lfm.embedding_norm.weight": _w(rng, H),
        "audio_embedding.embedding.weight": _w(rng, AUDIO_VOCAB, H),
    }
    for i, t in enumerate(LAYER_TYPES):
        p = f"lfm.layers.{i}"
        ws[f"{p}.operator_norm.weight"] = _w(rng, H)
        ws[f"{p}.ffn_norm.weight"] = _w(rng, H)
        ws[f"{p}.feed_forward.w1.weight"] = _w(rng, inter, H)
        ws[f"{p}.feed_forward.w2.weight"] = _w(rng, H, inter)
        ws[f"{p}.feed_forward.w3.weight"] = _w(rng, inter, H)
        if t == "conv":
            ws[f"{p}.conv.in_proj.weight"] = _w(rng, 3 * H, H)
            ws[f"{p}.conv.conv.weight"] = _w(rng, H, 1, 3)
            ws[f"{p}.conv.out_proj.weight"] = _w(rng, H, H)
        else:
            ws[f"{p}.self_attn.q_proj.weight"] = _w(rng, NUM_HEADS * hd, H)
            ws[f"{p}.self_attn.k_proj.weight"] = _w(rng, NUM_KV_HEADS * hd, H)
            ws[f"{p}.self_attn.v_proj.weight"] = _w(rng, NUM_KV_HEADS * hd, H)
            ws[f"{p}.self_attn.out_proj.weight"] = _w(rng, H, NUM_HEADS * hd)
            ws[f"{p}.self_attn.q_layernorm.weight"] = _w(rng, hd)
            ws[f"{p}.self_attn.k_layernorm.weight"] = _w(rng, hd)
    return ws


def decoder_config() -> dict:
    return {
        "lfm": {
            "hidden_size": HIDDEN,
            "num_hidden_layers": len(LAYER_TYPES),
            "num_attention_heads": NUM_HEADS,
            "num_key_value_heads": NUM_KV_HEADS,
            "vocab_size": VOCAB,
            "layer_types": LAYER_TYPES,
            "intermediate_size": 512,
            "conv_L_cache": 3,
            "max_position_embeddings": 4096,
            "norm_eps": 1e-5,
            "rope_theta": 1000000.0,
        }
    }


# === Graph builders ===


def build_encoder(
    rng, cfg: ConformerConfig | None = None, adapter_out: int = HIDDEN
) -> onnx.ModelProto:
    """Defaults to the real LFM2.5-Audio encoder shape with 2 layers."""
    cfg = cfg or ConformerConfig(n_layers=2)
    builder = ConformerEncoderBuilder(cfg, adapter_output_dim=adapter_out)
    builder.weights = conformer_weights(cfg, adapter_out, rng)
    builder.load_weights = lambda _path: None
    return builder.build("synthetic")


def build_detokenizer(rng) -> onnx.ModelProto:
    return AudioDetokenizerBuilder(DETOK_CONFIG, detokenizer_weights(DETOK_CONFIG, rng)).build()


def build_depthformer(rng, num_layers: int = 6, input_hidden: int = HIDDEN) -> onnx.ModelProto:
    builder = DepthformerUnifiedBuilder()
    builder.num_layers = num_layers
    builder.input_hidden_size = input_hidden
    builder.intermediate_size = 64
    builder.weights = depthformer_weights(builder, rng)
    builder.load_weights = lambda _path: None
    return builder.build("synthetic")


# === Tokenizer ===


def save_tokenizer(model_dir: pathlib.Path) -> None:
    """Char-level tokenizer whose special-token ids match LFM2AudioInference."""
    vocab = dict(SPECIAL_TOKENS)
    for ch in map(chr, range(32, 127)):
        vocab[ch] = char_id(ch)
    vocab["\n"] = char_id("\n")

    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Split(Regex(r"[\s\S]"), "isolated")
    tok.decoder = decoders.Fuse()
    tok.add_special_tokens(list(SPECIAL_TOKENS))

    PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token="<|startoftext|>",
        eos_token="<|im_end|>",
        pad_token="<|pad|>",
        unk_token="[UNK]",
    ).save_pretrained(model_dir)


# === Full export directory ===


def build_model_dir(root: pathlib.Path, seed: int = 0) -> pathlib.Path:
    """Write a complete synthetic export under root and return its path."""
    rng = np.random.default_rng(seed)
    model_dir = root / "LFM2.5-Audio-synthetic-ONNX"
    onnx_dir = model_dir / "onnx"
    onnx_dir.mkdir(parents=True)

    config = decoder_config()
    weights = decoder_weights(rng)
    export_decoder(weights, config, onnx_dir)
    export_audio_embedding_binary(weights, config, onnx_dir)
    export_embed_tokens(weights, config, onnx_dir)
    save_mel_config(onnx_dir)

    small_encoder = ConformerConfig(
        n_layers=2, d_model=128, n_heads=4, subsampling_conv_channels=64
    )
    onnx.save(build_encoder(rng, small_encoder), str(onnx_dir / "audio_encoder.onnx"))
    onnx.save(build_detokenizer(rng), str(onnx_dir / "audio_detokenizer.onnx"))
    onnx.save(build_depthformer(rng), str(onnx_dir / "vocoder_depthformer.onnx"))

    with open(model_dir / "config.json", "w") as f:
        json.dump(config, f)
    save_tokenizer(model_dir)
    return model_dir


def write_wav(path: pathlib.Path, seconds: float = 0.6, sample_rate: int = 16000) -> pathlib.Path:
    """A short 440 Hz tone with noise, 16-bit mono."""
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    rng = np.random.default_rng(1)
    signal = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.05 * rng.standard_normal(t.shape)
    scipy.io.wavfile.write(str(path), sample_rate, (signal * 32767).astype(np.int16))
    return path
