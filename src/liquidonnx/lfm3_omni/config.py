"""
LFM3-Omni config.json (model_type lfm3_omni: LiquidAI/LFM3-Omni-2B) -> the shared
LFM3AudioConfig (text decoder, speech encoder, depth decoder, codec, mel front end) and the
LFM3-VL vision encoder's config.

The audio side parses as liquidonnx.lfm3_audio.config does it. The vision keys sit at the root as
in LFM3-VL-2B's config.json: vision_config, projector_*, downsample_factor, image_token_id and
video_token_id, next to text_config. parse_vision hands them to
liquidonnx.lfm3_vl.config.parse_config as such a nested config, so the vision tower is the same
LFM3VLVisionConfig, with its RoPE convention (inv_freq 1 / theta ** x).

An lfm3_omni config.json in a flat legacy layout (trainer weight names, 2049-entry codebooks) is
not supported; lfm3_audio.config rejects it.
"""

import json
import pathlib
from dataclasses import dataclass

from liquidonnx.lfm3_audio.config import OMNI_VISION_KEYS, LFM3AudioConfig
from liquidonnx.lfm3_audio.config import load_config as load_audio_config
from liquidonnx.lfm3_vl.config import LFM3VLTokens, LFM3VLVisionConfig
from liquidonnx.lfm3_vl.config import parse_config as parse_vl_config


@dataclass
class LFM3OmniConfig:
    audio: LFM3AudioConfig
    vision: LFM3VLVisionConfig
    vision_tokens: LFM3VLTokens


def parse_vision(cfg: dict) -> tuple[LFM3VLVisionConfig, LFM3VLTokens]:
    """The vision encoder's config and the vision token ids of an lfm3_omni config.json."""
    if cfg.get("model_type") != "lfm3_omni":
        raise ValueError(f"not an LFM3-Omni config: model_type={cfg.get('model_type')!r}")
    if "text_config" not in cfg:
        raise NotImplementedError("a legacy (lfm/encoder/depthformer) lfm3_omni config")
    text = cfg["text_config"]
    port = {key: cfg[key] for key in (*OMNI_VISION_KEYS, "tie_word_embeddings") if key in cfg}
    port |= {key: text[key] for key in ("bos_token_id", "eos_token_id", "pad_token_id")}
    _, vision, tokens = parse_vl_config({"model_type": "lfm3_vl", "text_config": text, **port})
    return vision, tokens


def load_config(model_dir: str | pathlib.Path) -> LFM3OmniConfig:
    """LFM3OmniConfig of a checkpoint or export folder (lfm3_audio.config.load_config's files)."""
    raw = json.loads((pathlib.Path(model_dir) / "config.json").read_text())
    vision, tokens = parse_vision(raw)
    return LFM3OmniConfig(load_audio_config(model_dir), vision, tokens)
