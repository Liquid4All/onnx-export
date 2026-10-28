"""
LFM3-VL config.json (model_type lfm3_vl) -> the language model's LFM3Config and a vision config.

The parser works on flat keys, unlike the text checkpoints: full_attention_rope_theta,
sliding_attention_rope_theta, partial_rotary_factor (full attention only), rms_norm_eps, and
vision_* / projector_* for the encoder. Unknown keys raise, so a new knob can't be silently
dropped from the graph.

LFM3-VL-2B's config.json (transformers' Lfm3VlForConditionalGeneration with
model.language_model / model.vision_tower / model.multi_modal_projector) nests the language and
vision keys in text_config and vision_config; flatten_nested maps them to the flat keys. A flat
config.json, an earlier layout, is read as it is. The nested checkpoints' code computes RoPE
inv_freq as 1 / theta ** x, the flat ones' as theta ** -x.
"""

from dataclasses import dataclass

from liquidonnx.lfm3.builder import SUPPORTED_LAYER_TYPES, LFM3Config

# Revisions the exporter and tests were validated on (README 3.1c); other models follow main
REVISIONS = {
    "LiquidAI/LFM3-VL-2B": "f83d2b0a2296aaa277f394990a9b8b949ddf9691",
}

KNOWN_KEYS = {
    "architectures",
    "auto_map",
    "bos_token_id",
    "conv_L_cache",
    "conv_bias",
    "downsample_factor",
    "dtype",
    "eos_token_id",
    "full_attention_rope_theta",
    "head_dim",
    "hidden_size",
    "image_token_id",
    "intermediate_size",
    "layer_types",
    "max_position_embeddings",
    "model_type",
    "num_attention_heads",
    "num_hidden_layers",
    "num_key_value_heads",
    "pad_token_id",
    "partial_rotary_factor",
    "projector_hidden_size",
    "projector_layer_norm_eps",
    "rms_norm_eps",
    "sliding_attention_rope_theta",
    "sliding_window",
    "tie_word_embeddings",
    "transformers_version",
    "use_cache",
    "use_sandwich_norm",
    "video_token_id",
    "vision_end_token_id",
    "vision_hidden_size",
    "vision_intermediate_size",
    "vision_layer_types",
    "vision_max_position_embeddings",
    "vision_num_attention_heads",
    "vision_num_channels",
    "vision_num_hidden_layers",
    "vision_num_key_value_heads",
    "vision_patch_dim",
    "vision_patch_size",
    "vision_rope_theta",
    "vision_spatial_merge_size",
    "vision_start_token_id",
    "vision_temporal_patch_size",
    "vocab_size",
}

# The nested format's keys besides the flat ones its root keeps (token ids, projector_*)
NESTED_KEYS = {"projector_bias", "projector_hidden_act", "text_config", "vision_config"}
TEXT_KEYS = {
    "bos_token_id",
    "conv_L_cache",
    "conv_bias",
    "dtype",
    "eos_token_id",
    "head_dim",
    "hidden_size",
    "initializer_range",
    "intermediate_size",
    "layer_types",
    "max_position_embeddings",
    "model_type",
    "norm_eps",
    "num_attention_heads",
    "num_hidden_layers",
    "num_key_value_heads",
    "pad_token_id",
    "rope_parameters",
    "sliding_window",
    "tie_word_embeddings",
    "use_cache",
    "use_sandwich_norm",
    "vocab_size",
}
VISION_KEYS = {
    "conv_kernel_size",
    "dtype",
    "hidden_size",
    "in_channels",
    "initializer_range",
    "intermediate_size",
    "layer_types",
    "model_type",
    "norm_eps",
    "num_attention_heads",
    "num_hidden_layers",
    "num_key_value_heads",
    "patch_size",
    "rope_parameters",
    "spatial_merge_size",
    "temporal_patch_size",
}
# text_config keys copied under the same flat name
TEXT_FLAT = (
    "conv_L_cache",
    "conv_bias",
    "head_dim",
    "hidden_size",
    "intermediate_size",
    "layer_types",
    "max_position_embeddings",
    "num_attention_heads",
    "num_hidden_layers",
    "num_key_value_heads",
    "sliding_window",
    "tie_word_embeddings",
    "use_sandwich_norm",
    "vocab_size",
)

VISION_LAYER_TYPES = {"conv", "full_attention"}

# Longest patch-grid side the vision RoPE tables cover. The processor caps an image at
# 16_777_216 pixels with an aspect ratio of at most 200, so a side is at most ~3620 patches.
MAX_GRID_SIDE = 4096


@dataclass
class LFM3VLVisionConfig:
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    layer_types: list[str]
    patch_dim: int
    merge_size: int
    rope_theta: float
    norm_eps: float
    projector_hidden_size: int
    projector_eps: float
    text_hidden_size: int
    # inv_freq as theta ** -x (flat checkpoints) or 1 / theta ** x (nested)
    rope_negative_power: bool = True

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


@dataclass
class LFM3VLTokens:
    image: int
    video: int
    vision_start: int
    vision_end: int
    bos: int
    eos: int
    pad: int


def check_keys(cfg: dict, known: set[str], where: str):
    unknown = sorted(set(cfg) - known)
    if unknown:
        raise NotImplementedError(f"LFM3-VL export does not know the {where} keys {unknown}")


def flatten_nested(cfg: dict) -> dict:
    """The flat config of a nested one (text_config, vision_config)."""
    text, vision = cfg["text_config"], cfg["vision_config"]
    check_keys(cfg, KNOWN_KEYS | NESTED_KEYS, "config")
    check_keys(text, TEXT_KEYS, "text_config")
    check_keys(vision, VISION_KEYS, "vision_config")
    if not cfg.get("projector_bias", True) or cfg.get("projector_hidden_act", "gelu") != "gelu":
        raise NotImplementedError("LFM3-VL export expects a biased projector with exact GELU")
    if vision.get("conv_kernel_size", 3) != 3:
        raise NotImplementedError("LFM3-VL export expects 3x3 vision convolutions")
    if text.get("norm_eps", 1e-5) != vision.get("norm_eps", 1e-5):
        raise NotImplementedError("LFM3-VL export expects one norm_eps for text and vision")
    rope = text.get("rope_parameters", {})
    full, swa = rope.get("full_attention", {}), rope.get("sliding_attention", {})
    if any(e.get("rope_type", "default") != "default" for e in (full, swa)):
        raise NotImplementedError(f"LFM3-VL export supports only default text RoPE, got {rope}")
    if swa.get("partial_rotary_factor", 1.0) != 1.0:
        raise NotImplementedError("LFM3-VL export expects full rotation on sliding layers")
    if vision["rope_parameters"].get("rope_type") != "axial":
        raise NotImplementedError(f"unsupported vision RoPE {vision['rope_parameters']}")

    flat = {k: v for k, v in cfg.items() if k not in NESTED_KEYS}
    flat.update({k: text[k] for k in TEXT_FLAT if k in text})
    flat["rms_norm_eps"] = text.get("norm_eps", 1e-5)
    for key, value in (
        ("full_attention_rope_theta", full.get("rope_theta")),
        ("partial_rotary_factor", full.get("partial_rotary_factor")),
        ("sliding_attention_rope_theta", swa.get("rope_theta")),
    ):
        if value is not None:
            flat[key] = value
    flat.update(
        vision_hidden_size=vision["hidden_size"],
        vision_intermediate_size=vision["intermediate_size"],
        vision_num_hidden_layers=vision["num_hidden_layers"],
        vision_layer_types=vision["layer_types"],
        vision_num_attention_heads=vision["num_attention_heads"],
        vision_num_key_value_heads=vision["num_key_value_heads"],
        vision_num_channels=vision.get("in_channels", 3),
        vision_patch_size=vision["patch_size"],
        vision_temporal_patch_size=vision["temporal_patch_size"],
        vision_spatial_merge_size=vision["spatial_merge_size"],
        vision_patch_dim=vision.get("in_channels", 3)
        * vision["temporal_patch_size"]
        * vision["patch_size"] ** 2,
        vision_rope_theta=vision["rope_parameters"]["rope_theta"],
    )
    return flat


def parse_config(cfg: dict) -> tuple[LFM3Config, LFM3VLVisionConfig, LFM3VLTokens]:
    if cfg.get("model_type") != "lfm3_vl":
        raise ValueError(f"not an LFM3-VL config: model_type={cfg.get('model_type')!r}")
    nested = "text_config" in cfg
    if nested:
        cfg = flatten_nested(cfg)
    unknown = sorted(set(cfg) - KNOWN_KEYS)
    if unknown:
        raise NotImplementedError(f"LFM3-VL export does not know the config keys {unknown}")
    if cfg.get("conv_bias", False):
        raise NotImplementedError("LFM3-VL export does not support conv_bias=True")
    if not cfg.get("use_sandwich_norm", True):
        raise NotImplementedError("LFM3-VL export expects the sandwich norms")
    if not cfg.get("tie_word_embeddings", True):
        raise NotImplementedError("LFM3-VL export ties lm_head to the embedding table")

    layer_types = cfg["layer_types"]
    unknown_types = sorted(set(layer_types) - SUPPORTED_LAYER_TYPES)
    if unknown_types or len(layer_types) != cfg["num_hidden_layers"]:
        raise NotImplementedError(
            f"unsupported language layer_types {unknown_types or layer_types}"
        )
    vision_types = cfg["vision_layer_types"]
    unknown_types = sorted(set(vision_types) - VISION_LAYER_TYPES)
    if unknown_types or len(vision_types) != cfg["vision_num_hidden_layers"]:
        raise NotImplementedError(f"unsupported vision_layer_types {unknown_types or vision_types}")

    merge = cfg["vision_spatial_merge_size"]
    if cfg.get("downsample_factor", merge) != merge:
        raise ValueError("downsample_factor and vision_spatial_merge_size differ")
    patch_dim = (
        cfg["vision_temporal_patch_size"]
        * cfg["vision_num_channels"]
        * cfg["vision_patch_size"] ** 2
    )
    if cfg["vision_patch_dim"] != patch_dim:
        raise ValueError(f"vision_patch_dim {cfg['vision_patch_dim']} != {patch_dim}")
    vision_head_dim = cfg["vision_hidden_size"] // cfg["vision_num_attention_heads"]
    if vision_head_dim % 4:
        raise ValueError(f"2D RoPE needs a vision head_dim divisible by 4, got {vision_head_dim}")

    norm_eps = cfg.get("rms_norm_eps", 1e-5)
    language = LFM3Config(
        hidden_size=cfg["hidden_size"],
        num_hidden_layers=cfg["num_hidden_layers"],
        num_attention_heads=cfg["num_attention_heads"],
        num_key_value_heads=cfg["num_key_value_heads"],
        vocab_size=cfg["vocab_size"],
        layer_types=list(layer_types),
        head_dim=cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"],
        intermediate_size=cfg["intermediate_size"],
        conv_L_cache=cfg.get("conv_L_cache", 7),
        max_position_embeddings=cfg.get("max_position_embeddings", 128000),
        norm_eps=norm_eps,
        rope_theta=float(cfg.get("full_attention_rope_theta", 1_000_000.0)),
        partial_rotary_factor=float(cfg.get("partial_rotary_factor", 0.5)),
        swa_rope_theta=float(cfg.get("sliding_attention_rope_theta", 10_000.0)),
        swa_partial_rotary_factor=1.0,
        sliding_window=cfg.get("sliding_window", 256),
        use_peri_ln=True,
        use_gated_attention=True,
        gated_attention_style="headwise",
        rope_negative_power=not nested,
    )
    vision = LFM3VLVisionConfig(
        hidden_size=cfg["vision_hidden_size"],
        intermediate_size=cfg["vision_intermediate_size"],
        num_attention_heads=cfg["vision_num_attention_heads"],
        num_key_value_heads=cfg["vision_num_key_value_heads"],
        layer_types=list(vision_types),
        patch_dim=patch_dim,
        merge_size=merge,
        rope_theta=float(cfg.get("vision_rope_theta", 1_000_000.0)),
        norm_eps=norm_eps,
        projector_hidden_size=cfg["projector_hidden_size"],
        projector_eps=cfg.get("projector_layer_norm_eps", 1e-6),
        text_hidden_size=cfg["hidden_size"],
        rope_negative_power=not nested,
    )
    tokens = LFM3VLTokens(
        image=cfg.get("image_token_id", 125030),
        video=cfg.get("video_token_id", 125029),
        vision_start=cfg.get("vision_start_token_id", 125009),
        vision_end=cfg.get("vision_end_token_id", 125010),
        bos=cfg["bos_token_id"],
        eos=cfg["eos_token_id"],
        pad=cfg.get("pad_token_id", 0),
    )
    return language, vision, tokens
