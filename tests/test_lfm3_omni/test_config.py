"""
LFM3-Omni config parsing and export files (no download): the config.json, processor_config.json and
safetensors header of LFM3-Omni-2B copied to tests/test_lfm3_audio/assets/.

Run with:
    uv run pytest tests/test_lfm3_omni/test_config.py -v
"""

import copy
import json

import numpy as np
import pytest

from liquidonnx.lfm3_audio.weights import normalize_name, split_name
from liquidonnx.lfm3_omni.config import load_config, parse_vision
from liquidonnx.lfm3_omni.export import copy_checkpoint_files
from liquidonnx.lfm3_omni.session import BUNDLES
from liquidonnx.lfm3_vl.session import BUNDLES as VL_BUNDLES
from liquidonnx.lfm3_vl.weights import fuse_port_tensors

from ..test_lfm3_audio.test_weights import headers
from ..test_lfm3_audio.tiny import ASSETS

OMNI = ASSETS / "LFM3-Omni-2B"


def omni_config() -> dict:
    return json.loads((OMNI / "config.json").read_text())


def test_real_config():
    config = load_config(OMNI)
    vision, tokens = config.vision, config.vision_tokens
    assert config.audio.model_type == "lfm3_omni"
    assert (vision.hidden_size, vision.intermediate_size, len(vision.layer_types)) == (
        1024,
        2560,
        14,
    )
    assert vision.layer_types.count("full_attention") == 6
    assert (vision.num_attention_heads, vision.num_key_value_heads, vision.head_dim) == (16, 8, 64)
    assert (vision.patch_dim, vision.merge_size, vision.rope_theta) == (1536, 2, 1e6)
    assert (vision.projector_hidden_size, vision.projector_eps) == (4096, 1e-6)
    assert vision.text_hidden_size == config.audio.language.hidden_size == 2048
    # the transformers port's RoPE: inv_freq = 1 / theta ** x
    assert not vision.rope_negative_power
    assert (tokens.image, tokens.video, tokens.vision_start, tokens.vision_end) == (
        125030,
        125029,
        125009,
        125010,
    )
    assert (tokens.bos, tokens.eos, tokens.pad) == (124894, 124900, 124893)


@pytest.mark.parametrize(
    "change, error",
    [
        (lambda c: c["vision_config"].update(new_knob=1), NotImplementedError),
        (lambda c: c.update(projector_hidden_act="gelu_tanh"), NotImplementedError),
        (lambda c: c["vision_config"]["rope_parameters"].update(rope_type="default"), NotImplementedError),
        (lambda c: c.update(model_type="lfm3_vl"), ValueError),
        (lambda c: c.pop("text_config"), NotImplementedError),
    ],
)  # fmt: skip
def test_rejected_config(change, error):
    cfg = copy.deepcopy(omni_config())
    change(cfg)
    with pytest.raises(error):
        parse_vision(cfg)


def test_vision_tensor_names():
    """The checkpoint's vision tensors under the names LFM3VLVisionBuilder reads: the vision
    tower's 14 layers and the projector, w1/w3 fused into gate_up."""
    tensors = {}
    for name, (_, shape) in headers("LFM3-Omni-2B")["model.safetensors"].items():
        component, internal = split_name(normalize_name(name))
        if component == "vision":
            tensors[internal] = np.zeros(shape, dtype=np.int8)
    fused = {name: tensor.shape for name, tensor in fuse_port_tensors(tensors).items()}
    assert fused["vision_encoder.patchifier.weight"] == (1024, 1536)
    assert fused["vision_encoder.layers.0.conv.spatial_conv"] == (1024, 1, 3, 3)
    assert fused["vision_encoder.layers.2.feed_forward.gate_up.weight"] == (5120, 1024)
    layers = {int(n.split(".")[2]) for n in fused if n.startswith("vision_encoder.layers.")}
    assert layers == set(range(14))
    assert {n for n in fused if n.endswith("gate_up.weight")} == {
        f"vision_encoder.layers.{i}.feed_forward.gate_up.weight" for i in range(14)
    }
    assert {n for n in fused if n.startswith("vision_projector.")} == {
        f"vision_projector.{part}.{kind}"
        for part in ("layer_norm", "linear_1", "linear_2")
        for kind in ("weight", "bias")
    }


def test_bundles():
    """LFM3-Audio's graphs and LFM3-VL's vision encoder per precision."""
    for precision, files in BUNDLES.items():
        assert files["vision_encoder"] == VL_BUNDLES[precision][1]
        assert set(files) == {
            "decoder",
            "vision_encoder",
            "audio_encoder",
            "depth_decoder",
            "codec_decoder",
        }


def test_copy_checkpoint_files(tmp_path):
    """The flat preprocessor_config.json is processor_config.json's image_processor, which
    lfm3_vl.session.image_processor builds Qwen2VLImageProcessor from."""
    from liquidonnx.lfm3_vl.session import image_processor

    copy_checkpoint_files(OMNI, tmp_path)
    processor = json.loads((OMNI / "processor_config.json").read_text())
    flat = json.loads((tmp_path / "preprocessor_config.json").read_text())
    assert flat == processor["image_processor"]
    assert (tmp_path / "audio_tokenizer" / "config.json").exists()
    assert (tmp_path / "config.json").read_text() == (OMNI / "config.json").read_text()
    built = image_processor(tmp_path)
    assert (built.patch_size, built.merge_size, built.temporal_patch_size) == (16, 2, 2)
    assert built.size == {"shortest_edge": 65536, "longest_edge": 2097152}
