"""
LFM3Config.from_hf_config on both checkpoint formats, and the files of the released checkpoints at
their pinned revisions, copied to assets/: LFM3-1.7B's and LFM3-1.7B-Base's config.json, and
LFM3-1.7B's safetensors header (names, dtypes and shapes; LFM3-1.7B-Base's is the same) (no
download).

Run with:
    uv run pytest tests/test_lfm3/test_config.py -v
"""

import copy
import dataclasses
import json

import numpy as np
import pytest

from liquidonnx.lfm3.builder import LFM3Builder, LFM3Config

from .tiny import ASSETS, RELEASED, tensor_shapes

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


# === Released checkpoints ===


def released_tensors() -> dict[str, list]:
    """name -> [dtype, shape] of LFM3-1.7B's model.safetensors."""
    path = ASSETS / "LFM3-1.7B" / "safetensors_headers.json"
    return json.loads(path.read_text())["model.safetensors"]


@pytest.mark.parametrize("name", ["LFM3-1.7B", "LFM3-1.7B-Base"])
def test_released_configs_parse(name):
    """The format NEW_FORMAT stands for, with 24 layers and LFM3-1.7B's context."""
    layer_types = ["conv", "conv", *["full_attention", "conv", "sliding_attention", "conv"] * 5]
    layer_types += ["full_attention", "conv"]
    expected = dataclasses.replace(
        EXPECTED, num_hidden_layers=24, layer_types=layer_types, max_position_embeddings=262144
    )
    cfg = json.loads((ASSETS / name / "config.json").read_text())
    assert LFM3Config.from_hf_config(cfg) == expected


def test_tiny_checkpoint_follows_the_released_tensors():
    """tiny.tensor_shapes, which the tiny checkpoint is written with, gives the names and shapes of
    LFM3-1.7B's header from its config."""
    tensors = released_tensors()
    assert {dtype for dtype, _ in tensors.values()} == {"BF16"}
    assert tensor_shapes(RELEASED) == {name: tuple(shape) for name, (_, shape) in tensors.items()}


def test_released_tensors_all_become_graph_weights():
    """LFM3Builder reads each tensor of LFM3-1.7B's header once, into a weight a node of the graph
    takes, and needs no other: a build on zero-stride stand-ins of the header's shapes."""
    tensors = released_tensors()
    read, weights = [], {}

    class Tensors(dict):
        def __getitem__(self, name):
            read.append(name)
            return super().__getitem__(name)

    stand_ins = Tensors(
        {name: np.broadcast_to(np.float32(0), shape) for name, (_, shape) in tensors.items()}
    )
    builder = LFM3Builder(LFM3Config.from_hf_config(RELEASED))
    builder.load_weights = lambda path: setattr(builder, "weights", stand_ins)
    builder.add_initializer = lambda name, tensor: weights.setdefault(name, tensor.shape)
    # Computed, not read: tables of 262144 positions
    builder.build_rope_cache = lambda: None
    graph = builder.build("LFM3-1.7B").graph
    assert sorted(read) == sorted(tensors)
    checkpoint_weights = {name for name in weights if not name.startswith("/model/constants/")}
    assert len(checkpoint_weights) == len(tensors)
    assert checkpoint_weights <= {name for node in graph.node for name in node.input}
