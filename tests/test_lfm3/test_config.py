"""
LFM3Config.from_hf_config on both checkpoint formats (no download).

Run with:
    uv run pytest tests/test_lfm3/test_config.py -v
"""

import copy

import pytest

from liquidonnx.lfm3.builder import LFM3Config

LAYER_TYPES = ["conv", "conv", "full_attention", "conv", "sliding_attention", "conv"]

# Keys of the newer format, LFM3-1.7B's: rope_parameters per layer type, use_sandwich_norm
NEW_FORMAT = {
    "hidden_size": 2048,
    "num_hidden_layers": 6,
    "num_attention_heads": 16,
    "num_key_value_heads": 4,
    "vocab_size": 128000,
    "head_dim": 128,
    "intermediate_size": 7680,
    "conv_L_cache": 7,
    "conv_bias": False,
    "max_position_embeddings": 128000,
    "norm_eps": 1e-05,
    "layer_types": LAYER_TYPES,
    "rope_parameters": {
        "full_attention": {
            "partial_rotary_factor": 0.5,
            "rope_theta": 1000000.0,
            "rope_type": "default",
        },
        "sliding_attention": {"rope_theta": 10000, "rope_type": "default"},
    },
    "sliding_window": 256,
    "tie_word_embeddings": True,
    "use_sandwich_norm": True,
}

# Keys of the older format: flat RoPE keys, sliding_window_attention, use_peri_ln
OLD_FORMAT = {
    "hidden_size": 2048,
    "num_hidden_layers": 6,
    "num_attention_heads": 16,
    "num_key_value_heads": 4,
    "vocab_size": 128000,
    "head_dim": None,
    "intermediate_size": 7680,
    "conv_L_cache": 7,
    "conv_bias": False,
    "max_position_embeddings": 128000,
    "norm_eps": 1e-05,
    "layer_types": [
        t.replace("sliding_attention", "sliding_window_attention") for t in LAYER_TYPES
    ],
    "rope_parameters": {
        "partial_rotary_factor": 0.5,
        "rope_theta": 1000000.0,
        "rope_type": "default",
    },
    "swa_rope_parameters": {"rope_theta": 10000, "rope_type": "default"},
    "swa_num_key_value_heads": None,
    "swa_use_gated_attention": None,
    "swa_gated_attention_style": None,
    "sliding_window": 256,
    "tie_word_embeddings": True,
    "use_peri_ln": True,
    "use_gated_attention": True,
    "gated_attention_style": "headwise",
}

EXPECTED = LFM3Config(
    hidden_size=2048,
    num_hidden_layers=6,
    num_attention_heads=16,
    num_key_value_heads=4,
    vocab_size=128000,
    layer_types=LAYER_TYPES,
    head_dim=128,
    intermediate_size=7680,
    conv_L_cache=7,
    max_position_embeddings=128000,
    norm_eps=1e-05,
    rope_theta=1000000.0,
    partial_rotary_factor=0.5,
    swa_rope_theta=10000.0,
    swa_partial_rotary_factor=1.0,
    sliding_window=256,
    use_peri_ln=True,
    use_gated_attention=True,
    gated_attention_style="headwise",
)


@pytest.mark.parametrize("cfg", [NEW_FORMAT, OLD_FORMAT], ids=["new", "old"])
def test_formats_parse_to_the_same_config(cfg):
    assert LFM3Config.from_hf_config(cfg) == EXPECTED


def test_old_format_without_swa_rope_reuses_the_full_table():
    cfg = copy.deepcopy(OLD_FORMAT)
    del cfg["swa_rope_parameters"]
    assert LFM3Config.from_hf_config(cfg).swa_rope_theta is None


def test_layer_types_from_full_attn_idxs():
    cfg = copy.deepcopy(OLD_FORMAT)
    del cfg["layer_types"]
    cfg["full_attn_idxs"] = [2, 5]
    parsed = LFM3Config.from_hf_config(cfg)
    assert parsed.layer_types == [
        "conv",
        "conv",
        "full_attention",
        "conv",
        "conv",
        "full_attention",
    ]


def edited(base: dict, **changes) -> dict:
    cfg = copy.deepcopy(base)
    cfg.update(changes)
    return cfg


@pytest.mark.parametrize(
    "cfg",
    [
        # What transformers 5.15-5.17 turn "conv" into; never read it as a conv layer.
        edited(NEW_FORMAT, layer_types=["linear_attention", *LAYER_TYPES[1:]]),
        edited(
            NEW_FORMAT,
            rope_parameters={"full_attention": NEW_FORMAT["rope_parameters"]["full_attention"]},
        ),
        edited(NEW_FORMAT, sliding_window=None),
        edited(
            NEW_FORMAT,
            rope_parameters={
                "full_attention": {"rope_theta": 1e6, "rope_type": "yarn", "factor": 4.0},
                "sliding_attention": {"rope_theta": 1e4, "rope_type": "default"},
            },
        ),
        edited(NEW_FORMAT, tie_word_embeddings=False),
        edited(NEW_FORMAT, conv_bias=True),
        edited(OLD_FORMAT, swa_num_key_value_heads=2),
    ],
    ids=[
        "linear_attention",
        "no-swa-rope",
        "no-window",
        "yarn",
        "untied",
        "conv-bias",
        "swa-override",
    ],
)
def test_unsupported_configs_raise(cfg):
    with pytest.raises((NotImplementedError, ValueError)):
        LFM3Config.from_hf_config(cfg)
