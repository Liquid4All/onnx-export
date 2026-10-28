"""
d1 config parsing on the tiny LFM3-VL config with d1's keys (no download).

Run with:
    uv run pytest tests/test_d1/test_config.py -v
"""

import pytest

from liquidonnx.d1.config import base_config, parse_config
from liquidonnx.lfm3_vl.config import parse_config as parse_vl_config

from ..test_lfm3_vl.tiny import TINY_CONFIG, port_config

D1_SETTINGS = {
    "d1_attention": "set",
    "d1_codes": 2,
    "d1_numbers": True,
    "d1_set_size": 16,
    "d1_vision_max_pixels": 1638400,
}


def d1_config(cfg: dict = TINY_CONFIG, **changes) -> dict:
    return {**cfg, "architectures": ["D1Model"], **D1_SETTINGS, **changes}


@pytest.mark.parametrize("layout", ["flat", "port"])
def test_parses_as_its_base_model(layout):
    base = TINY_CONFIG if layout == "flat" else port_config()
    config = parse_config(d1_config(base))
    assert (config.language, config.vision, config.tokens) == parse_vl_config(base)
    assert config.audio is None


def test_base_config_drops_the_d1_keys():
    assert base_config(d1_config()) == {**TINY_CONFIG, "architectures": ["D1Model"]}


@pytest.mark.parametrize(
    "changes,error,match",
    [
        ({"architectures": ["Lfm3VlForConditionalGeneration"]}, ValueError, "not a d1 config"),
        ({"d1_attention": "prompt"}, NotImplementedError, "'set' only"),
        ({"d1_attention": None}, NotImplementedError, "'set' only"),
        ({"d1_window": 4}, NotImplementedError, "d1_window"),
    ],
)
def test_rejects(changes, error, match):
    with pytest.raises(error, match=match):
        parse_config(d1_config(**changes))


def test_omni_needs_its_feature_extractor():
    cfg = d1_config(model_type="lfm3_omni")
    with pytest.raises(ValueError, match="feature extractor"):
        parse_config(cfg)
