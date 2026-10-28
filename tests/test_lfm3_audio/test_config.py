"""
LFM3-Audio config parsing (no download): the configs of LFM3-Audio-2B and LFM3-Omni-2B at their
pinned revisions, copied to assets/ (LFM3-Audio-2B's config.json without its tokenizer key, which
no export reads), and the tiny checkpoint's.

Run with:
    uv run pytest tests/test_lfm3_audio/test_config.py -v
"""

import copy
import dataclasses
import json

import pytest

from liquidonnx.lfm3_audio.config import (
    load_config,
    maskgit_schedule,
    parse_config,
)

from .tiny import ASSETS, TINY_CODEC_CONFIG, TINY_CONFIG

CHECKPOINTS = ("LFM3-Audio-2B", "LFM3-Omni-2B")


def configs(name: str) -> tuple[dict, dict, dict]:
    """(config.json, the codec's config.json, the feature extractor) of an asset checkpoint."""
    root = ASSETS / name
    codec = next(root.glob("audio_*/config.json"))
    processor = json.loads((root / "processor_config.json").read_text())
    return (
        json.loads((root / "config.json").read_text()),
        json.loads(codec.read_text()),
        processor["feature_extractor"],
    )


@pytest.mark.parametrize("name", CHECKPOINTS)
def test_real_config(name):
    config = load_config(ASSETS / name)
    language, encoder, depth, codec = config.language, config.encoder, config.depth, config.codec
    assert config.model_type == {"LFM3-Audio-2B": "lfm3_audio"}.get(name, "lfm3_omni")

    assert (language.hidden_size, language.num_hidden_layers, language.vocab_size) == (
        2048,
        24,
        128000,
    )
    assert language.layer_types.count("sliding_attention") == 5
    assert (language.sliding_window, language.swa_rope_theta) == (256, 10000.0)
    assert (language.partial_rotary_factor, language.rope_negative_power) == (0.5, False)
    assert language.use_peri_ln and language.gated_attention_style == "headwise"

    # The LFM2 stack: its 7 attention layers slide over 32 tokens; no gate, no sandwich norms
    attention = [i for i, t in enumerate(encoder.layer_types) if t != "conv"]
    assert attention == [2, 5, 8, 11, 14, 15, 16]
    assert {encoder.layer_types[i] for i in attention} == {"sliding_attention"}
    assert (encoder.hidden_size, encoder.head_dim, encoder.intermediate_size) == (768, 96, 1792)
    assert (encoder.num_attention_heads, encoder.num_key_value_heads) == (8, 4)
    assert (encoder.sliding_window, encoder.conv_L_cache, encoder.rope_theta) == (32, 3, 1e6)
    assert encoder.swa_rope_theta is None and encoder.partial_rotary_factor == 1.0
    assert not encoder.use_gated_attention and not encoder.use_peri_ln
    assert config.encoder_conv_channels == 256

    assert depth.layer_types == ["full_attention"] * 3
    assert (depth.hidden_size, depth.head_dim, depth.intermediate_size) == (1024, 32, 2816)
    assert (depth.num_attention_heads, depth.num_key_value_heads) == (32, 8)
    assert depth.use_gated_attention and depth.gated_attention_style == "headwise"
    assert not depth.use_peri_ln and depth.partial_rotary_factor == 1.0

    assert [i for i, t in enumerate(codec.layer_types) if t == "sliding_attention"] == [2, 4, 6]
    assert codec.layer_types.count("conv") == 5
    assert (codec.hidden_size, codec.head_dim, codec.intermediate_size) == (512, 64, 2304)
    assert (codec.sliding_window, codec.conv_L_cache) == (30, 7)
    assert codec.use_peri_ln and codec.gated_attention_style == "headwise"

    assert (config.num_codebooks, config.codebook_size, config.audio_eos_code) == (12, 2017, 2016)
    assert config.maskgit_schedule == [4, 4, 4]
    assert (config.interleaved_n_text, config.interleaved_n_audio) == (6, 12)
    assert config.fsq_levels == [8, 7, 6, 6] and config.codec_quantizer_dim == 256
    assert (config.codec_n_fft, config.codec_hop_length, config.frame_samples) == (1280, 320, 1920)
    assert config.codec_sampling_rate == 24000

    tokens = config.tokens
    assert (tokens.audio, tokens.audio_out, tokens.audio_start) == (125031, 125032, 125012)
    assert (tokens.text_end, tokens.mixed_start) == (125014, 125015)
    assert (tokens.bos, tokens.eos, tokens.pad) == (124894, 124900, 124893)

    mel = config.mel
    assert (mel.sampling_rate, mel.num_mel_bins) == (16000, 128)
    assert (mel.n_fft, mel.win_length, mel.hop_length) == (400, 400, 160)
    # A clip of 120456 samples: 752 mel frames, 94 encoder tokens; below n_fft one frame
    assert mel.num_frames(120456) == 752 and mel.num_tokens(752) == 94
    assert mel.num_frames(5920) == 36 and mel.num_frames(100) == 1


def test_formats_agree():
    """The two checkpoints differ in format and weights, not in their audio side."""
    audio, omni = (load_config(ASSETS / name) for name in CHECKPOINTS)
    assert dataclasses.replace(omni, model_type="lfm3_audio") == audio


def test_tiny_config():
    config = parse_config(TINY_CONFIG, TINY_CODEC_CONFIG, configs("LFM3-Audio-2B")[2])
    assert config.encoder.intermediate_size == 128 and config.encoder.head_dim == 16
    assert config.encoder.layer_types == ["conv", "sliding_attention"] * 2
    assert (config.encoder.sliding_window, config.codec.sliding_window) == (8, 8)
    assert (config.codec.hidden_size, config.codec_quantizer_dim) == (64, 16)
    assert config.maskgit_schedule == [4, 4, 4]


def test_maskgit_schedule():
    assert maskgit_schedule(3, 12) == [4, 4, 4]
    assert maskgit_schedule(5, 12) == [2, 2, 2, 2, 4]
    with pytest.raises(ValueError):
        maskgit_schedule(13, 12)


def edited(path: tuple[str, ...], value, cfg: dict | None = None) -> dict:
    cfg = copy.deepcopy(TINY_CONFIG if cfg is None else cfg)
    node = cfg
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    return cfg


FEATURES = configs("LFM3-Audio-2B")[2]
OMNI = configs("LFM3-Omni-2B")[0]


@pytest.mark.parametrize(
    "cfg, codec, features",
    [
        (edited(("new_knob",), 1), TINY_CODEC_CONFIG, FEATURES),
        (edited(("encoder", "new_knob"), 1), TINY_CODEC_CONFIG, FEATURES),
        (edited(("encoder", "lfm", "new_knob"), 1), TINY_CODEC_CONFIG, FEATURES),
        (edited(("depthformer", "new_knob"), 1), TINY_CODEC_CONFIG, FEATURES),
        (edited(("lfm", "new_knob"), 1), TINY_CODEC_CONFIG, FEATURES),
        (TINY_CONFIG, {**TINY_CODEC_CONFIG, "new_knob": 1}, FEATURES),
        (TINY_CONFIG, TINY_CODEC_CONFIG, {**FEATURES, "new_knob": 1}),
        (edited(("audio_encoder_config", "new_knob"), 1, OMNI), TINY_CODEC_CONFIG, FEATURES),
        # lfm3_omni's vision keys belong to lfm3_omni only
        (edited(("model_type",), "lfm3_audio", OMNI), TINY_CODEC_CONFIG, FEATURES),
    ],
)
def test_unknown_keys_raise(cfg, codec, features):
    with pytest.raises(NotImplementedError, match="does not know"):
        parse_config(cfg, codec, features)


@pytest.mark.parametrize(
    "cfg, codec, features",
    [
        (edited(("model_type",), "lfm3"), TINY_CODEC_CONFIG, FEATURES),
        (edited(("codebooks",), 8), TINY_CODEC_CONFIG, FEATURES),
        (edited(("audio_vocab_size",), 2049), TINY_CODEC_CONFIG, FEATURES),
        (edited(("encoder", "dim_out"), 32), TINY_CODEC_CONFIG, FEATURES),
        (TINY_CONFIG, {**TINY_CODEC_CONFIG, "fsq_levels": [8, 5, 5, 5]}, FEATURES),
        (TINY_CONFIG, TINY_CODEC_CONFIG, {**FEATURES, "feature_size": 80}),
    ],
)
def test_inconsistent_configs_raise(cfg, codec, features):
    with pytest.raises(ValueError):
        parse_config(cfg, codec, features)


@pytest.mark.parametrize(
    "cfg, codec, features",
    [
        (edited(("encoder", "lfm", "conv_bias"), True), TINY_CODEC_CONFIG, FEATURES),
        (edited(("depthformer", "layer_types"), ["conv"] * 3), TINY_CODEC_CONFIG, FEATURES),
        (TINY_CONFIG, {**TINY_CODEC_CONFIG, "stft_win_length": 1024}, FEATURES),
        (TINY_CONFIG, TINY_CODEC_CONFIG, {**FEATURES, "encoder_stride": 4}),
    ],
)
def test_unsupported_configs_raise(cfg, codec, features):
    with pytest.raises(NotImplementedError):
        parse_config(cfg, codec, features)
