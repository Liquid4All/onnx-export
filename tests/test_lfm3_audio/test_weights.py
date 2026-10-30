"""
LFM3-Audio weight names (no download): the real checkpoints' safetensors headers (names, dtypes
and shapes, copied to assets/) map onto the internal names with every tensor accounted for, and
the tiny checkpoint's tensors load per component.

Run with:
    uv run pytest tests/test_lfm3_audio/test_weights.py -v
"""

import json
import re
import shutil

import numpy as np
import pytest

from liquidonnx.lfm3_audio.config import codec_dir, load_config
from liquidonnx.lfm3_audio.weights import (
    check_shapes,
    codec_name,
    expected_shapes,
    load_codec_weights,
    load_weights,
    normalize_name,
    split_name,
)

from .test_config import CHECKPOINTS
from .tiny import ASSETS


def headers(name: str) -> dict[str, dict[str, list]]:
    return json.loads((ASSETS / name / "safetensors_headers.json").read_text())


@pytest.mark.parametrize(
    "name, expected",
    [
        ("lfm.model.layers.3.conv.conv.weight", "model.language_model.layers.3.conv.conv.weight"),
        ("encoder.lfm.lfm.embedding_norm.weight", "model.audio_encoder.lfm.embedding_norm.weight"),
        ("encoder.downsample.lin_out.bias", "model.audio_encoder.downsample.lin_out.bias"),
        ("mlp.model.3.weight", "model.audio_projector.linear_2.weight"),
        ("mask_embedding", "model.depth_decoder.mask_embedding"),
        ("depthformer.model.layers.2.self_attn.gate_proj.weight",
         "model.depth_decoder.model.layers.2.self_attn.gate_proj.weight"),
        ("depth_embeddings.11.embedding.weight",
         "model.depth_decoder.depth_embeddings.11.embedding.weight"),
        ("depth_embeddings.4.to_logits.weight", None),
        ("model.depth_decoder.depth_embeddings.4.to_logits.weight", None),
        ("model.language_model.embed_tokens.weight", "model.language_model.embed_tokens.weight"),
    ],
)  # fmt: skip
def test_normalize_name(name, expected):
    assert normalize_name(name) == expected


@pytest.mark.parametrize(
    "name, expected",
    [
        ("model.language_model.layers.0.conv.conv.weight", ("language", "model.layers.0.conv.conv.weight")),
        ("model.codebook_embedding.weight", ("language", "codebook_embedding.weight")),
        ("model.audio_encoder.lfm.layers.2.self_attn.out_proj.weight",
         ("encoder", "model.layers.2.self_attn.out_proj.weight")),
        ("model.audio_encoder.downsample.conv_in.weight", ("encoder", "downsample.conv_in.weight")),
        ("model.audio_projector.layer_norm.bias", ("encoder", "projector.layer_norm.bias")),
        ("model.depth_decoder.model.embedding_norm.weight", ("depth", "model.embedding_norm.weight")),
        ("model.depth_decoder.depth_linear.bias", ("depth", "depth_linear.bias")),
        ("model.depth_decoder.mask_embedding", ("depth", "mask_embedding")),
        ("model.vision_tower.layers.0.conv.spatial_conv.weight",
         ("vision", "vision_encoder.layers.0.conv.spatial_conv")),
        ("model.multi_modal_projector.linear_1.weight", ("vision", "vision_projector.linear_1.weight")),
    ],
)  # fmt: skip
def test_split_name(name, expected):
    assert split_name(name) == expected


@pytest.mark.parametrize(
    "name", ["lm_head.weight", "model.new_module.weight", "lfm.lm_head.weight"]
)
def test_unknown_tensor_raises(name):
    with pytest.raises(KeyError):
        split_name(normalize_name(name))


def test_codec_name():
    assert codec_name("decoder.lfm.layers.2.self_attn.gate_proj.weight") == (
        "model.layers.2.self_attn.gate_proj.weight"
    )
    assert codec_name("decoder.lin.bias") == "lin.bias"
    assert codec_name("decoder.istft.window") == "istft.window"
    assert codec_name("quantizer.project_out.weight") == "quantizer.project_out.weight"
    for name in ("encoder.lfm.embedding_norm.weight", "frontend.conv.weight", "to_quantizer.bias"):
        assert codec_name(name) is None
    with pytest.raises(KeyError):
        codec_name("quantizer.new.weight")


@pytest.mark.parametrize("checkpoint", CHECKPOINTS)
def test_real_tensors_normalize(checkpoint):
    """Every tensor of the checkpoint lands in a component, each component has exactly the
    tensors its config asks for, and only the tied to_logits heads are dropped."""
    config = load_config(ASSETS / checkpoint)
    files = headers(checkpoint)
    shapes, dropped = {}, []
    for name, (dtype, shape) in files["model.safetensors"].items():
        assert dtype == "BF16", name
        hf_name = normalize_name(name)
        if hf_name is None:
            dropped.append(name)
            continue
        component, internal = split_name(hf_name)
        shapes.setdefault(component, {})[internal] = tuple(shape)
    for component in ("language", "encoder", "depth"):
        check_shapes(component, shapes[component], config)
    if config.model_type == "lfm3_omni":
        assert not dropped
        assert len(shapes["vision"]) == 138
    else:
        assert len(dropped) == config.num_codebooks
        assert all(re.fullmatch(r"depth_embeddings\.\d+\.to_logits\.weight", n) for n in dropped)
        assert "vision" not in shapes

    [codec_file] = [f for f in files if f != "model.safetensors"]
    assert codec_file.split("/")[0] == "audio_tokenizer"
    codec = {}
    for name, (dtype, shape) in files[codec_file].items():
        internal = codec_name(name)
        if internal is not None:
            assert dtype == ("F32" if internal == "istft.window" else "BF16"), name
            codec[internal] = tuple(shape)
    check_shapes("codec", codec, config)


def test_check_shapes_raises():
    config = load_config(ASSETS / "LFM3-Audio-2B")
    shapes = dict(expected_shapes(config)["depth"])
    check_shapes("depth", shapes, config)
    with pytest.raises(ValueError, match="mask_embedding: \\(1023,\\)"):
        check_shapes("depth", shapes | {"mask_embedding": (1023,)}, config)
    with pytest.raises(ValueError, match="unexpected \\['new.weight'\\]"):
        check_shapes("depth", shapes | {"new.weight": (1,)}, config)
    with pytest.raises(ValueError, match="missing \\['mask_embedding'\\]"):
        check_shapes("depth", {k: v for k, v in shapes.items() if k != "mask_embedding"}, config)


def test_tiny_weights_load(tiny_checkpoint):
    """bf16 trainer-named tensors load as float32 under the internal names, component by
    component, every tensor but the tied to_logits heads."""
    import torch
    from safetensors.torch import load_file

    config = load_config(tiny_checkpoint)
    raw = load_file(str(tiny_checkpoint / "model.safetensors"))
    loaded = {c: load_weights(tiny_checkpoint, c, config) for c in ("language", "encoder", "depth")}
    dtypes = {a.dtype for weights in loaded.values() for a in weights.values()}
    assert dtypes == {np.dtype(np.float32)}
    assert sum(map(len, loaded.values())) == len(raw) - config.num_codebooks
    np.testing.assert_array_equal(
        loaded["encoder"]["projector.linear_1.weight"],
        raw["mlp.model.1.weight"].to(torch.float32).numpy(),
    )
    with pytest.raises(KeyError, match="no vision"):
        load_weights(tiny_checkpoint, "vision", config)

    codec = load_codec_weights(tiny_checkpoint, config)
    raw = load_file(str(tiny_checkpoint / "audio_tokenizer" / "model.safetensors"))
    np.testing.assert_array_equal(codec["istft.window"], raw["decoder.istft.window"].numpy())


def test_tiny_codec_in_audio_detokenizer(tiny_checkpoint, tmp_path):
    """The earlier layout's audio_detokenizer/ codec folder loads as audio_tokenizer/ does."""
    shutil.copytree(tiny_checkpoint, tmp_path, dirs_exist_ok=True)
    (tmp_path / "audio_tokenizer").rename(tmp_path / "audio_detokenizer")
    assert codec_dir(tmp_path) == tmp_path / "audio_detokenizer"
    config = load_config(tmp_path)
    assert config == load_config(tiny_checkpoint)
    codec, expected = (load_codec_weights(d, config) for d in (tmp_path, tiny_checkpoint))
    assert codec.keys() == expected.keys()
    for name, value in expected.items():
        np.testing.assert_array_equal(codec[name], value)
