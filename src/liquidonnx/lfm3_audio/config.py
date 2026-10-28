"""
LFM3-Audio configuration: config.json, the codec's config.json and the feature extractor settings
-> one LFM3AudioConfig shared by the lfm3_audio and lfm3_omni exporters.

Two config.json formats parse into it:

    LFM3-Audio-2B (training export, flat legacy keys)         LFM3-Omni-2B (HF-native)
    ─────────────────────────────────────────────────────    ────────────────────────
    lfm                                                        text_config
    encoder {dim_in, dim_conv, dim_out, sliding_window, lfm}   audio_encoder_config {num_mel_bins,
                                                                 conv_channels, hidden_size,
                                                                 sliding_window, text_config}
    depthformer                                                depth_config
    codebooks, audio_vocab_size (eos = size - 1)               num_codebooks, codebook_size,
                                                                 audio_eos_code
    maskgit_steps (schedule [12 // steps] * steps)             maskgit_schedule
    token ids: the Lfm3AudioConfig defaults                    audio_token_id, audio_out_token_id, ...

as Lfm3AudioConfig._convert_legacy_config of the checkpoint's remote code reads them. The vision
keys of lfm3_omni (vision_config, image/video token ids, projector_*) are accepted here and parsed
by liquidonnx.lfm3_omni. Unknown keys raise, so a new knob can't be silently dropped from a graph.

The four transformer stacks are LFM3Configs, the configs LFM3Builder builds from:
    language  LFM3Config.from_hf_config(lfm / text_config)
    encoder   the LFM2 stack (transformers' Lfm2Model) as an LFM3 one: its attention layers are
              sliding_attention with the encoder's window (the reference masks them with
              create_sliding_window_causal_mask), no gate, no sandwich norms, head_dim = hidden /
              heads, and the FFN width LFM2's block_auto_adjust_ff_dim gives (2560 -> 1792)
    depth     LFM3Config.from_hf_config(depthformer / depth_config); the reference runs it
              bidirectionally (is_causal=False, no mask)
    codec     LiquidCodecConfig.lfm_config(decoder_window) of modeling_liquid_codec: hard-coded in
              the remote code except encoder_dim
"""

import json
import math
import pathlib
from dataclasses import dataclass

from liquidonnx.lfm3.builder import LFM3Config

# Revisions the exporter and tests were validated on (README 3.1d, 3.1e); other models follow main
REVISIONS = {
    "LiquidAI/LFM3-Audio-2B": "9d49701c372d6011d900f6abb40c6b296fb706dd",
    "LiquidAI/LFM3-Omni-2B": "ce2458cbce4ad970f92032ed94e3d18d30b04f9c",
}

# Subfolder of the Liquid codec: LFM3-Audio-2B's and LFM3-Omni-2B's audio_tokenizer, or
# audio_detokenizer in an earlier layout
CODEC_SUBFOLDERS = ("audio_detokenizer", "audio_tokenizer")

# Keys of every PretrainedConfig.to_dict() that no graph reads
CONFIG_NOISE = {
    "_name_or_path",
    "architectures",
    "chunk_size_feed_forward",
    "dtype",
    "id2label",
    "initializer_range",
    "is_encoder_decoder",
    "label2id",
    "output_attentions",
    "output_hidden_states",
    "problem_type",
    "return_dict",
    "transformers_version",
}
LFM3_KEYS = CONFIG_NOISE | {
    "bos_token_id",
    "conv_L_cache",
    "conv_bias",
    "eos_token_id",
    "head_dim",
    "hidden_size",
    "intermediate_size",
    "is_causal",
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
LFM2_KEYS = CONFIG_NOISE | {
    "block_auto_adjust_ff_dim",
    "block_ffn_dim_multiplier",
    "block_multiple_of",
    "bos_token_id",
    "conv_L_cache",
    "conv_bias",
    "eos_token_id",
    "full_attn_idxs",
    "hidden_size",
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
    "tie_word_embeddings",
    "use_cache",
    "vocab_size",
}
LEGACY_KEYS = {
    "architectures",
    "audio_vocab_size",
    "auto_map",
    "codebooks",
    "depthformer",
    "dtype",
    "encoder",
    "format_version",
    "interleaved_n_audio",
    "interleaved_n_text",
    "lfm",
    "maskgit_steps",
    "model_type",
    # The feature extractor's settings again, and the tokenizer's source repo: the reference
    # drops both (processor_config.json holds the extractor's)
    "preprocessor",
    "tokenizer",
    "transformers_version",
}
LEGACY_ENCODER_KEYS = {"dim_conv", "dim_in", "dim_out", "lfm", "sliding_window"}
NESTED_KEYS = {
    "architectures",
    "audio_encoder_config",
    "audio_eos_code",
    "audio_loss_weight",
    "audio_out_token_id",
    "audio_start_token_id",
    "audio_token_id",
    "auto_map",
    "codebook_size",
    "depth_config",
    "dtype",
    "initializer_range",
    "interleaved_n_audio",
    "interleaved_n_text",
    "maskgit_schedule",
    "maskgit_steps",
    "mixed_start_token_id",
    "model_type",
    "num_codebooks",
    "text_config",
    "text_end_token_id",
    "text_loss_weight",
    "tie_word_embeddings",
    "transformers_version",
}
NESTED_ENCODER_KEYS = {
    "conv_channels",
    # d1-Omni-2B's: the weights' dtype, which no graph reads
    "dtype",
    "hidden_size",
    "initializer_range",
    "model_type",
    "num_mel_bins",
    "sliding_window",
    "text_config",
}
# lfm3_omni's vision side, parsed by liquidonnx.lfm3_omni
OMNI_VISION_KEYS = {
    "downsample_factor",
    "image_token_id",
    "projector_bias",
    "projector_hidden_act",
    "projector_hidden_size",
    "projector_layer_norm_eps",
    "video_token_id",
    "vision_config",
}
CODEC_KEYS = CONFIG_NOISE | {
    "decoder_window",
    "encoder_conv_dim",
    "encoder_dim",
    "frame_samples",
    "fsq_levels",
    "model_type",
    "n_q",
    "quantizer_dim",
    "sample_rate",
    "sampling_rate",
    "stft_frame_stack",
    "stft_hop_length",
    "stft_n_fft",
    "stft_win_length",
}
FEATURE_EXTRACTOR_KEYS = {
    "auto_map",
    "encoder_stride",
    "feature_extractor_type",
    "feature_size",
    "hop_length",
    "log_floor",
    "n_fft",
    "normalization_divisor",
    "normalization_offset",
    "padding_side",
    "padding_value",
    "return_attention_mask",
    "sampling_rate",
    "win_length",
}
FEATURE_EXTRACTORS = {"Lfm3AudioFeatureExtractor", "Lfm3OmniFeatureExtractor"}

# Lfm3AudioConfig's defaults, which the legacy format doesn't store
DEFAULT_TOKENS = {
    "audio_token_id": 125031,
    "audio_out_token_id": 125032,
    "audio_start_token_id": 125012,
    "text_end_token_id": 125014,
    "mixed_start_token_id": 125015,
}
# The downsampler's three stride-2 convolutions: one encoder token per 8 mel frames
ENCODER_STRIDE = 8


@dataclass
class LFM3AudioTokens:
    audio: int  # <|audio|>: one per encoder token of an input clip
    audio_out: int  # <|audio_out|>: one per speech frame, embedded from its 12 codes
    audio_start: int  # <|audio_start|>: switches to speech
    text_end: int  # <|text_end|>: ends an interleaved answer's text, or the turn
    mixed_start: int  # <|mixed_start|>: switches to interleaved text and speech
    bos: int
    eos: int  # <|im_end|>
    pad: int


@dataclass
class MelConfig:
    """The log-mel front end (Lfm3AudioFeatureExtractor): right-pad to max(n_fft, n + (-(n -
    n_fft) mod hop)) samples, |STFT|² (center=False, periodic Hann of win_length), slaney mel
    filters (0 - sampling_rate / 2), log10 clamped at log_floor, then (x + offset) / divisor."""

    sampling_rate: int = 16000
    num_mel_bins: int = 128
    n_fft: int = 400
    win_length: int = 400
    hop_length: int = 160
    log_floor: float = -12.0
    normalization_offset: float = 4.0
    normalization_divisor: float = 4.0

    def num_frames(self, num_samples: int) -> int:
        padded = max(self.n_fft, num_samples + (-(num_samples - self.n_fft) % self.hop_length))
        return 1 + (padded - self.n_fft) // self.hop_length

    @staticmethod
    def num_tokens(num_frames: int) -> int:
        return -(num_frames // -ENCODER_STRIDE)


@dataclass
class LFM3AudioConfig:
    model_type: str  # "lfm3_audio" or "lfm3_omni"
    language: LFM3Config
    encoder: LFM3Config
    depth: LFM3Config
    codec: LFM3Config
    tokens: LFM3AudioTokens
    mel: MelConfig
    # Speech encoder downsampler
    encoder_conv_channels: int  # 256
    # Speech frames
    num_codebooks: int  # 12
    codebook_size: int  # 2017: the codec's 2016 codes and the end-of-audio code
    audio_eos_code: int  # 2016
    maskgit_schedule: list[int]  # codebooks revealed per depth pass, [4, 4, 4]
    interleaved_n_text: int  # 6
    interleaved_n_audio: int  # 12
    # Liquid codec decoder
    fsq_levels: list[int]  # [8, 7, 6, 6]
    codec_quantizer_dim: int  # 256
    codec_frame_stack: int  # 6 STFT frames per speech frame
    codec_n_fft: int  # 1280
    codec_hop_length: int  # 320
    codec_sampling_rate: int  # 24000

    @property
    def frame_samples(self) -> int:
        """Output samples per speech frame (1920)."""
        return self.codec_hop_length * self.codec_frame_stack


def check_keys(cfg: dict, known: set[str], where: str):
    unknown = sorted(set(cfg) - known)
    if unknown:
        raise NotImplementedError(f"LFM3-Audio export does not know the {where} keys {unknown}")


def lfm2_ffn_size(cfg: dict) -> int:
    """The SwiGLU width of transformers' Lfm2MLP (block_auto_adjust_ff_dim)."""
    size = cfg["intermediate_size"]
    if not cfg.get("block_auto_adjust_ff_dim", True):
        return size
    size = int(2 * size / 3)
    multiplier = cfg.get("block_ffn_dim_multiplier", 1.0)
    if multiplier is not None:
        size = int(multiplier * size)
    multiple = cfg.get("block_multiple_of", 256)
    return multiple * ((size + multiple - 1) // multiple)


def parse_lfm3(cfg: dict, where: str) -> LFM3Config:
    check_keys(cfg, LFM3_KEYS, where)
    if cfg.get("model_type", "lfm3") != "lfm3":
        raise NotImplementedError(f"{where} has model_type {cfg['model_type']!r}, expected lfm3")
    return LFM3Config.from_hf_config(cfg)


def parse_encoder(cfg: dict, legacy: bool) -> tuple[LFM3Config, int, int]:
    """(the LFM2 stack as an LFM3Config, num_mel_bins, conv_channels)"""
    if legacy:
        check_keys(cfg, LEGACY_ENCODER_KEYS, "encoder")
        mels, channels = cfg.get("dim_in", 128), cfg.get("dim_conv", 256)
        hidden, lfm = cfg.get("dim_out", 768), cfg["lfm"]
    else:
        check_keys(cfg, NESTED_ENCODER_KEYS, "audio_encoder_config")
        mels, channels = cfg.get("num_mel_bins", 128), cfg.get("conv_channels", 256)
        hidden, lfm = cfg.get("hidden_size", 768), cfg["text_config"]
    check_keys(lfm, LFM2_KEYS, "audio encoder LFM2")
    if lfm.get("model_type", "lfm2") != "lfm2":
        raise NotImplementedError(f"audio encoder model_type {lfm['model_type']!r}, expected lfm2")
    if lfm["hidden_size"] != hidden:
        raise ValueError(f"audio encoder width {hidden} != its LFM2's {lfm['hidden_size']}")
    if mels % ENCODER_STRIDE:
        raise ValueError(f"num_mel_bins {mels} is not a multiple of {ENCODER_STRIDE}")
    if lfm.get("conv_bias", False):
        raise NotImplementedError("LFM3-Audio export does not support an encoder conv_bias")
    rope = lfm.get("rope_parameters") or {}
    if rope.get("rope_type", "default") != "default" or "partial_rotary_factor" in rope:
        raise NotImplementedError(f"unsupported audio encoder RoPE {rope}")
    layer_types = lfm.get("layer_types")
    if layer_types is None or len(layer_types) != lfm["num_hidden_layers"]:
        raise NotImplementedError(f"audio encoder needs layer_types per layer, got {layer_types}")
    if set(layer_types) - {"conv", "full_attention"}:
        raise NotImplementedError(f"unsupported audio encoder layer_types {layer_types}")
    encoder = LFM3Config(
        hidden_size=hidden,
        num_hidden_layers=lfm["num_hidden_layers"],
        num_attention_heads=lfm["num_attention_heads"],
        num_key_value_heads=lfm["num_key_value_heads"],
        vocab_size=lfm.get("vocab_size", 2),
        layer_types=["sliding_attention" if t == "full_attention" else t for t in layer_types],
        head_dim=hidden // lfm["num_attention_heads"],
        intermediate_size=lfm2_ffn_size(lfm),
        conv_L_cache=lfm.get("conv_L_cache", 3),
        max_position_embeddings=lfm.get("max_position_embeddings", 128000),
        norm_eps=lfm.get("norm_eps", 1e-5),
        rope_theta=float(rope.get("rope_theta", 1_000_000.0)),
        sliding_window=cfg.get("sliding_window", 32),
    )
    return encoder, mels, channels


def parse_codec(cfg: dict) -> tuple[LFM3Config, dict]:
    """(the codec decoder's LFM3Config, its other settings)"""
    check_keys(cfg, CODEC_KEYS, "codec config")
    if cfg.get("model_type", "liquid_codec") != "liquid_codec":
        raise NotImplementedError(f"codec model_type {cfg['model_type']!r}, expected liquid_codec")
    n_fft, hop = cfg.get("stft_n_fft", 1280), cfg.get("stft_hop_length", 320)
    stack = cfg.get("stft_frame_stack", 6)
    if cfg.get("stft_win_length", n_fft) != n_fft:
        raise NotImplementedError("the numpy iSTFT expects stft_win_length == stft_n_fft")
    rate = cfg.get("sampling_rate", 24000)
    if cfg.get("sample_rate", rate) != rate or cfg.get("frame_samples", hop * stack) != hop * stack:
        raise ValueError("codec sample_rate / frame_samples disagree with its STFT settings")
    window = cfg.get("decoder_window", 30)
    # LiquidCodecConfig.lfm_config(decoder_window)
    codec = LFM3Config(
        hidden_size=cfg.get("encoder_dim", 512),
        num_hidden_layers=8,
        num_attention_heads=8,
        num_key_value_heads=2,
        vocab_size=1,
        layer_types=["sliding_attention" if i in (2, 4, 6) else "conv" for i in range(8)],
        head_dim=64,
        intermediate_size=2304,
        conv_L_cache=7,
        max_position_embeddings=131072,
        norm_eps=1e-5,
        rope_theta=1_000_000.0,
        sliding_window=window,
        use_peri_ln=True,
        use_gated_attention=True,
        gated_attention_style="headwise",
    )
    settings = {
        "fsq_levels": list(cfg.get("fsq_levels") or [8, 7, 6, 6]),
        "codec_quantizer_dim": cfg.get("quantizer_dim", 256),
        "codec_frame_stack": stack,
        "codec_n_fft": n_fft,
        "codec_hop_length": hop,
        "codec_sampling_rate": rate,
        "n_q": cfg.get("n_q", 12),
    }
    return codec, settings


def parse_feature_extractor(cfg: dict) -> MelConfig:
    """processor_config.json's feature_extractor (or a flat preprocessor_config.json)."""
    check_keys(cfg, FEATURE_EXTRACTOR_KEYS, "feature extractor")
    if cfg.get("feature_extractor_type", "Lfm3AudioFeatureExtractor") not in FEATURE_EXTRACTORS:
        raise NotImplementedError(f"unsupported feature extractor {cfg['feature_extractor_type']}")
    if cfg.get("encoder_stride", ENCODER_STRIDE) != ENCODER_STRIDE:
        raise NotImplementedError(f"the downsampler's stride is {ENCODER_STRIDE}")
    if cfg.get("padding_side", "right") != "right":
        raise NotImplementedError("LFM3-Audio export expects right padding")
    return MelConfig(
        sampling_rate=cfg.get("sampling_rate", 16000),
        num_mel_bins=cfg.get("feature_size", 128),
        n_fft=cfg.get("n_fft", 400),
        win_length=cfg.get("win_length", 400),
        hop_length=cfg.get("hop_length", 160),
        log_floor=float(cfg.get("log_floor", -12.0)),
        normalization_offset=float(cfg.get("normalization_offset", 4.0)),
        normalization_divisor=float(cfg.get("normalization_divisor", 4.0)),
    )


def maskgit_schedule(steps: int, num_codebooks: int) -> list[int]:
    """Lfm3AudioConfig's schedule for maskgit_steps: equal passes, the remainder in the last."""
    if not 1 <= steps <= num_codebooks:
        raise ValueError(f"maskgit_steps must be in [1, {num_codebooks}], got {steps}")
    schedule = [num_codebooks // steps] * steps
    schedule[-1] += num_codebooks - sum(schedule)
    return schedule


def parse_config(cfg: dict, codec_cfg: dict, feature_extractor: dict) -> LFM3AudioConfig:
    """config.json, the codec subfolder's config.json and the feature extractor settings."""
    model_type = cfg.get("model_type")
    if model_type not in ("lfm3_audio", "lfm3_omni"):
        raise ValueError(f"not an LFM3-Audio or LFM3-Omni config: model_type={model_type!r}")
    legacy = "lfm" in cfg
    if legacy:
        if model_type != "lfm3_audio":
            raise NotImplementedError(f"a legacy (lfm/encoder/depthformer) {model_type} config")
        check_keys(cfg, LEGACY_KEYS, "config")
        text, encoder, depth = cfg["lfm"], cfg["encoder"], cfg.get("depthformer")
        num_codebooks = cfg.get("codebooks", 12)
        codebook_size = cfg.get("audio_vocab_size", 2017)
        eos_code = codebook_size - 1
        schedule = maskgit_schedule(cfg.get("maskgit_steps", 3), num_codebooks)
        ids = DEFAULT_TOKENS
    else:
        known = NESTED_KEYS | (OMNI_VISION_KEYS if model_type == "lfm3_omni" else set())
        check_keys(cfg, known, "config")
        text, encoder, depth = cfg["text_config"], cfg["audio_encoder_config"], cfg["depth_config"]
        num_codebooks = cfg.get("num_codebooks", 12)
        codebook_size = cfg.get("codebook_size", 2017)
        eos_code = cfg.get("audio_eos_code", 2016)
        schedule = cfg.get("maskgit_schedule") or maskgit_schedule(
            cfg.get("maskgit_steps", 3), num_codebooks
        )
        ids = {key: cfg.get(key, value) for key, value in DEFAULT_TOKENS.items()}
    if depth is None:
        raise NotImplementedError("LFM3-Audio export needs the depth decoder's config")
    if not cfg.get("tie_word_embeddings", True):
        raise NotImplementedError("LFM3-Audio export ties lm_head to the embedding table")

    language = parse_lfm3(text, "text config")
    encoder, num_mel_bins, channels = parse_encoder(encoder, legacy)
    depth = parse_lfm3(depth, "depth config")
    if set(depth.layer_types) != {"full_attention"}:
        raise NotImplementedError(f"the depth decoder must be all attention: {depth.layer_types}")
    if depth.max_position_embeddings < num_codebooks:
        raise ValueError(f"depth max_position_embeddings < num_codebooks {num_codebooks}")
    codec, codec_settings = parse_codec(codec_cfg)
    mel = parse_feature_extractor(feature_extractor)

    if sum(schedule) != num_codebooks or min(schedule) < 1:
        raise ValueError(f"maskgit_schedule {schedule} must split {num_codebooks} codebooks")
    if codec_settings.pop("n_q") != num_codebooks:
        raise ValueError("the codec's n_q and the model's num_codebooks differ")
    if math.prod(codec_settings["fsq_levels"]) != eos_code or eos_code != codebook_size - 1:
        raise ValueError(
            f"codes 0..{eos_code - 1} must be the codec's and {eos_code} the last of "
            f"{codebook_size}: fsq_levels {codec_settings['fsq_levels']}"
        )
    if mel.num_mel_bins != num_mel_bins:
        raise ValueError(f"{mel.num_mel_bins} mel bins, the encoder takes {num_mel_bins}")
    for name, value in ids.items():
        if not 0 <= value < language.vocab_size:
            raise ValueError(f"{name}={value} is outside the vocabulary")

    tokens = LFM3AudioTokens(
        audio=ids["audio_token_id"],
        audio_out=ids["audio_out_token_id"],
        audio_start=ids["audio_start_token_id"],
        text_end=ids["text_end_token_id"],
        mixed_start=ids["mixed_start_token_id"],
        bos=text["bos_token_id"],
        eos=text["eos_token_id"],
        pad=text["pad_token_id"],
    )
    return LFM3AudioConfig(
        model_type=model_type,
        language=language,
        encoder=encoder,
        depth=depth,
        codec=codec,
        tokens=tokens,
        mel=mel,
        encoder_conv_channels=channels,
        num_codebooks=num_codebooks,
        codebook_size=codebook_size,
        audio_eos_code=eos_code,
        maskgit_schedule=list(schedule),
        interleaved_n_text=cfg.get("interleaved_n_text", 6),
        interleaved_n_audio=cfg.get("interleaved_n_audio", 12),
        **codec_settings,
    )


def codec_dir(model_dir: str | pathlib.Path) -> pathlib.Path:
    """The codec subfolder of a checkpoint or export folder."""
    for name in CODEC_SUBFOLDERS:
        path = pathlib.Path(model_dir) / name
        if path.is_dir():
            return path
    raise FileNotFoundError(f"{model_dir} has none of the codec folders {CODEC_SUBFOLDERS}")


def load_config(model_dir: str | pathlib.Path) -> LFM3AudioConfig:
    """LFM3AudioConfig of a checkpoint or export folder: config.json, the codec folder's
    config.json and processor_config.json's feature_extractor."""
    model_dir = pathlib.Path(model_dir)
    processor = json.loads((model_dir / "processor_config.json").read_text())
    return parse_config(
        json.loads((model_dir / "config.json").read_text()),
        json.loads((codec_dir(model_dir) / "config.json").read_text()),
        processor["feature_extractor"],
    )
