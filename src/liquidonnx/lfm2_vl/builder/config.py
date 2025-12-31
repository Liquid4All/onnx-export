"""Configuration classes for LFM2-VL models."""

from dataclasses import dataclass

from liquidonnx.lfm2.builder import LFM2Config


@dataclass
class SigLIP2Config:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    patch_size: int
    num_channels: int = 3
    layer_norm_eps: float = 1e-6
    hidden_act: str = "gelu_pytorch_tanh"

    @classmethod
    def from_hf_config(cls, vision_config) -> "SigLIP2Config":
        return cls(
            hidden_size=vision_config.hidden_size,
            intermediate_size=vision_config.intermediate_size,
            num_hidden_layers=vision_config.num_hidden_layers,
            num_attention_heads=vision_config.num_attention_heads,
            patch_size=vision_config.patch_size,
            num_channels=getattr(vision_config, "num_channels", 3),
            layer_norm_eps=getattr(vision_config, "layer_norm_eps", 1e-6),
            hidden_act=getattr(vision_config, "hidden_act", "gelu_pytorch_tanh"),
        )


@dataclass
class LFM2VLConfig:
    text_config: LFM2Config
    vision_config: SigLIP2Config
    projector_hidden_size: int
    projector_hidden_act: str = "gelu"
    projector_bias: bool = True
    projector_use_layernorm: bool = True
    downsample_factor: int = 2
    image_token_id: int = 396
    tile_size: int = 512
    max_tiles: int = 10

    @classmethod
    def from_hf_config(cls, config) -> "LFM2VLConfig":
        return cls(
            text_config=LFM2Config.from_hf_config(config.text_config),
            vision_config=SigLIP2Config.from_hf_config(config.vision_config),
            projector_hidden_size=config.projector_hidden_size,
            projector_hidden_act=getattr(config, "projector_hidden_act", "gelu"),
            projector_bias=getattr(config, "projector_bias", True),
            projector_use_layernorm=getattr(config, "projector_use_layernorm", True),
            downsample_factor=getattr(config, "downsample_factor", 2),
            image_token_id=getattr(config, "image_token_id", 396),
            tile_size=getattr(config, "tile_size", 512),
            max_tiles=getattr(config, "max_tiles", 10),
        )
