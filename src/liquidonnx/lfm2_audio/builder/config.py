"""
Configuration dataclasses for LFM2.5-Audio ONNX export.

Architecture Overview:
    Audio waveform → Mel-spectrogram → Conformer → Adapter → LFM2 → Logits
                                                            ↓
                                                    Depthformer → Audio codes
                                                            ↓
                                                    Detokenizer → Waveform
"""

from dataclasses import dataclass, field


@dataclass
class ConformerConfig:
    """FastConformer encoder configuration."""

    feat_in: int = 128  # Input mel features
    d_model: int = 512  # Model dimension
    n_layers: int = 17  # Number of conformer layers
    n_heads: int = 8  # Attention heads
    ff_expansion_factor: int = 4  # Feed-forward expansion
    conv_kernel_size: int = 9  # Depthwise conv kernel
    subsampling_factor: int = 8  # Temporal subsampling
    subsampling_conv_channels: int = 256  # Subsampling conv channels
    pos_emb_max_len: int = 5000  # Max position embeddings

    @classmethod
    def from_hf_config(cls, encoder_config: dict) -> "ConformerConfig":
        return cls(
            feat_in=encoder_config.get("feat_in", 128),
            d_model=encoder_config.get("d_model", 512),
            n_layers=encoder_config.get("n_layers", 17),
            n_heads=encoder_config.get("n_heads", 8),
            ff_expansion_factor=encoder_config.get("ff_expansion_factor", 4),
            conv_kernel_size=encoder_config.get("conv_kernel_size", 9),
            subsampling_factor=encoder_config.get("subsampling_factor", 8),
            subsampling_conv_channels=encoder_config.get("subsampling_conv_channels", 256),
            pos_emb_max_len=encoder_config.get("pos_emb_max_len", 5000),
        )


@dataclass
class LFM2Config:
    """LFM2 backbone configuration (same as text model)."""

    hidden_size: int = 2048
    num_hidden_layers: int = 16
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    vocab_size: int = 65536
    layer_types: list[str] = field(default_factory=list)
    intermediate_size: int | None = None
    conv_L_cache: int = 3
    max_position_embeddings: int = 128000
    norm_eps: float = 1e-5
    rope_theta: float = 1000000.0

    def __post_init__(self):
        if self.intermediate_size is None:
            self.intermediate_size = self.hidden_size * 9 // 2

    @classmethod
    def from_hf_config(cls, lfm_config: dict) -> "LFM2Config":
        intermediate_size = lfm_config.get("intermediate_size")
        if intermediate_size is not None and lfm_config.get("block_auto_adjust_ff_dim", False):
            intermediate_size = int(2 * intermediate_size / 3)
            multiplier = lfm_config.get("block_ffn_dim_multiplier")
            if multiplier is not None:
                intermediate_size = int(multiplier * intermediate_size)
                multiple_of = lfm_config.get("block_multiple_of", 256)
                intermediate_size = multiple_of * (
                    (intermediate_size + multiple_of - 1) // multiple_of
                )

        return cls(
            hidden_size=lfm_config.get("hidden_size", 2048),
            num_hidden_layers=lfm_config.get("num_hidden_layers", 16),
            num_attention_heads=lfm_config.get("num_attention_heads", 32),
            num_key_value_heads=lfm_config.get("num_key_value_heads", 8),
            vocab_size=lfm_config.get("vocab_size", 65536),
            layer_types=lfm_config.get("layer_types", []),
            intermediate_size=intermediate_size,
            conv_L_cache=lfm_config.get("conv_L_cache", 3),
            max_position_embeddings=lfm_config.get("max_position_embeddings", 128000),
            norm_eps=lfm_config.get("norm_eps", 1e-5),
            rope_theta=lfm_config.get("rope_theta", 1000000.0),
        )


@dataclass
class DepthformerConfig:
    """Depthformer configuration for audio codebook prediction."""

    dim: int = 1024
    layers: int = 6
    n_heads: int = 32  # Derived from qkv_proj shape
    head_dim: int = 32  # 1024 / 32 = 32
    intermediate_size: int = 2816  # From w1/w3 shape
    n_codebooks: int = 8
    codebook_vocab_size: int = 2049

    @classmethod
    def from_hf_config(cls, df_config: dict) -> "DepthformerConfig":
        return cls(
            dim=df_config.get("dim", 1024),
            layers=df_config.get("layers", 6),
        )


@dataclass
class DetokenizerConfig:
    """Audio detokenizer configuration."""

    hidden_size: int = 512
    num_hidden_layers: int = 8
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    layer_types: list[str] = field(default_factory=list)
    intermediate_size: int = 3328
    conv_L_cache: int = 3
    sliding_window: int = 30
    output_size: int = 1282  # Magnitude + phase
    vocab_size: int = 65536
    norm_eps: float = 1e-5
    rope_theta: float = 1000000.0

    @classmethod
    def from_hf_config(cls, config: dict) -> "DetokenizerConfig":
        return cls(
            hidden_size=config.get("hidden_size", 512),
            num_hidden_layers=config.get("num_hidden_layers", 8),
            num_attention_heads=config.get("num_attention_heads", 16),
            num_key_value_heads=config.get("num_key_value_heads", 8),
            layer_types=config.get("layer_types", []),
            intermediate_size=config.get("intermediate_size", 3328),
            conv_L_cache=config.get("conv_L_cache", 3),
            sliding_window=config.get("sliding_window", 30),
            output_size=config.get("output_size", 1282),
            vocab_size=config.get("vocab_size", 65536),
            norm_eps=config.get("norm_eps", 1e-5),
            rope_theta=config.get("rope_theta", 1000000.0),
        )


@dataclass
class LFM2AudioConfig:
    """Complete LFM2.5-Audio model configuration."""

    conformer: ConformerConfig
    lfm: LFM2Config
    depthformer: DepthformerConfig
    codebooks: int = 8
    audio_vocab_size: int = 16392  # 2049 * 8
    codebook_vocab_size: int = 2049

    @classmethod
    def from_hf_config(cls, config: dict) -> "LFM2AudioConfig":
        return cls(
            conformer=ConformerConfig.from_hf_config(config.get("encoder", {})),
            lfm=LFM2Config.from_hf_config(config.get("lfm", {})),
            depthformer=DepthformerConfig.from_hf_config(config.get("depthformer", {})),
            codebooks=config.get("codebooks", 8),
        )
