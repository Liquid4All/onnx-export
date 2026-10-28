"""
LFM3-VL config.json parsing (no download).

Run with:
    uv run pytest tests/test_lfm3_vl/test_config.py -v
"""

import copy
import dataclasses

import pytest

from liquidonnx.lfm3.builder import LFM3Config
from liquidonnx.lfm3_vl.config import parse_config

from .tiny import TINY_CONFIG, port_config


def test_tiny_config_parses():
    language, vision, tokens = parse_config(TINY_CONFIG)
    assert language.layer_types == TINY_CONFIG["layer_types"]
    assert (language.rope_theta, language.partial_rotary_factor) == (1e6, 0.5)
    assert (language.swa_rope_theta, language.swa_partial_rotary_factor) == (1e4, 1.0)
    assert language.use_peri_ln and language.use_gated_attention and language.rope_negative_power
    assert language.gated_attention_style == "headwise"
    assert (vision.head_dim, vision.merge_size, vision.patch_dim) == (64, 2, 1536)
    assert (tokens.image, tokens.video, tokens.eos) == (125030, 125029, 124900)


def edited(**changes) -> dict:
    cfg = copy.deepcopy(TINY_CONFIG)
    cfg.update(changes)
    return cfg


@pytest.mark.parametrize(
    "cfg",
    [
        edited(some_new_knob=1),
        edited(model_type="lfm3"),
        edited(conv_bias=True),
        edited(use_sandwich_norm=False),
        edited(tie_word_embeddings=False),
        # what transformers 5.15-5.17 turn "conv" into
        edited(layer_types=["linear_attention", *TINY_CONFIG["layer_types"][1:]]),
        edited(vision_layer_types=["conv", "sliding_attention", "conv"]),
        edited(vision_patch_dim=768),
        edited(downsample_factor=4),
    ],
    ids=[
        "unknown-key",
        "model-type",
        "conv-bias",
        "no-sandwich",
        "untied",
        "linear-attention",
        "vision-layer-type",
        "patch-dim",
        "downsample",
    ],
)
def test_unsupported_configs_raise(cfg):
    with pytest.raises((NotImplementedError, ValueError)):
        parse_config(cfg)


def test_port_config_parses_as_the_flat_one():
    """LFM3-VL-2B's nested layout gives the flat config's model, with RoPE inv_freq as
    1 / theta ** x."""
    flat_language, flat_vision, flat_tokens = parse_config(TINY_CONFIG)
    language, vision, tokens = parse_config(port_config())
    assert not language.rope_negative_power and not vision.rope_negative_power
    assert dataclasses.replace(language, rope_negative_power=True) == flat_language
    assert dataclasses.replace(vision, rope_negative_power=True) == flat_vision
    assert tokens == flat_tokens


def edited_port(section: str | None, **changes) -> dict:
    cfg = port_config()
    (cfg[section] if section else cfg).update(changes)
    return cfg


@pytest.mark.parametrize(
    "cfg",
    [
        edited_port(None, some_new_knob=1),
        edited_port("text_config", some_new_knob=1),
        edited_port("vision_config", some_new_knob=1),
        edited_port(None, projector_bias=False),
        edited_port(None, projector_hidden_act="gelu_pytorch_tanh"),
        edited_port("vision_config", conv_kernel_size=5),
        edited_port("vision_config", norm_eps=1e-6),
        edited_port("vision_config", rope_parameters={"rope_theta": 1e4, "rope_type": "default"}),
        edited_port(
            "text_config",
            rope_parameters={
                "full_attention": {"rope_theta": 1e6, "rope_type": "yarn"},
                "sliding_attention": {"rope_theta": 1e4, "rope_type": "default"},
            },
        ),
    ],
    ids=[
        "unknown-key",
        "unknown-text-key",
        "unknown-vision-key",
        "projector-bias",
        "projector-act",
        "vision-kernel",
        "norm-eps",
        "vision-rope",
        "text-rope",
    ],
)
def test_unsupported_port_configs_raise(cfg):
    with pytest.raises((NotImplementedError, ValueError)):
        parse_config(cfg)


def test_text_parser_rejects_vl_config():
    with pytest.raises(ValueError, match="lfm3-vl-export"):
        LFM3Config.from_hf_config(TINY_CONFIG)
