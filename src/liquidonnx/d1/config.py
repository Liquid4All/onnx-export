"""
d1 config.json (LiquidAI/d1-2B, d1-Omni-2B) -> the configs of the graphs d1-export builds.

A d1 checkpoint is an LFM3-VL (model_type lfm3_vl) or LFM3-Omni (lfm3_omni) checkpoint post-trained
to answer named questions at answer slots in one forward pass. Its config.json is the base model's
with architectures ["D1Model"], its own auto_map and the d1_* keys its host code reads
(runner.SlotReader._serve: option codes, number readout, picture cap, set size, ...). Of those the
graphs depend only on d1_attention, which must be "set" (attention.py: the state and every
question read each other on the full-attention layers, a question never reads another); the rest
are stripped before lfm3_vl.config or lfm3_audio.config / lfm3_omni.config parse the base keys.

d1-Omni ships no speech codec (no audio_tokenizer/ folder): d1 never speaks, so its LFM3AudioConfig
is parsed with an empty codec config, whose defaults pass the codebook checks.
"""

import json
import pathlib
from dataclasses import dataclass

from liquidonnx.lfm3.builder import LFM3Config
from liquidonnx.lfm3_audio.config import LFM3AudioConfig
from liquidonnx.lfm3_audio.config import parse_config as parse_audio_config
from liquidonnx.lfm3_omni.config import parse_vision
from liquidonnx.lfm3_vl.config import LFM3VLTokens, LFM3VLVisionConfig
from liquidonnx.lfm3_vl.config import parse_config as parse_vl_config

# Revisions the exporter and tests were validated on (README 3.1f); other models follow main
REVISIONS = {
    "LiquidAI/d1-2B": "97c57047d5852d77e02dfac211f8768aff77a8b2",
    "LiquidAI/d1-Omni-2B": "4cccbfd5ba3405ee0f973134bd03e0d9d60ef1d8",
}

# runner.SlotReader._serve's keys
D1_KEYS = {
    "d1_attention",
    "d1_box_edges",
    "d1_codes",
    "d1_numbers",
    "d1_prompt",
    "d1_set_size",
    "d1_vision_max_pixels",
}


@dataclass
class D1Config:
    language: LFM3Config
    vision: LFM3VLVisionConfig
    tokens: LFM3VLTokens
    # d1-Omni: its audio encoder and audio token; None for d1 on LFM3-VL
    audio: LFM3AudioConfig | None = None


def base_config(cfg: dict) -> dict:
    """The base model's config.json of a d1 one: the d1_* keys checked and dropped."""
    if cfg.get("architectures") != ["D1Model"]:
        raise ValueError(f"not a d1 config: architectures={cfg.get('architectures')!r}")
    unknown = sorted(k for k in cfg if k.startswith("d1_") and k not in D1_KEYS)
    if unknown:
        raise NotImplementedError(f"d1 export does not know the config keys {unknown}")
    if cfg.get("d1_attention") != "set":
        raise NotImplementedError(
            f"d1 export builds d1_attention 'set' only, got {cfg.get('d1_attention')!r}"
        )
    return {k: v for k, v in cfg.items() if k not in D1_KEYS}


def parse_config(cfg: dict, feature_extractor: dict | None = None) -> D1Config:
    """config.json, and for lfm3_omni processor_config.json's feature_extractor."""
    base = base_config(cfg)
    if base.get("model_type") == "lfm3_omni":
        if feature_extractor is None:
            raise ValueError("an lfm3_omni d1 config needs its feature extractor settings")
        audio = parse_audio_config(base, {}, feature_extractor)
        vision, tokens = parse_vision(base)
        return D1Config(audio.language, vision, tokens, audio)
    return D1Config(*parse_vl_config(base))


def load_config(model_dir: str | pathlib.Path) -> D1Config:
    """D1Config of a checkpoint or export folder."""
    model_dir = pathlib.Path(model_dir)
    cfg = json.loads((model_dir / "config.json").read_text())
    feature_extractor = None
    if cfg.get("model_type") == "lfm3_omni":
        processor = json.loads((model_dir / "processor_config.json").read_text())
        feature_extractor = processor["feature_extractor"]
    return parse_config(cfg, feature_extractor)
