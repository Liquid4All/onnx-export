# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from huggingface_hub.dataclasses import strict

from transformers.configuration_utils import PreTrainedConfig
from transformers.models.auto import CONFIG_MAPPING, AutoConfig
from transformers.utils import auto_docstring

from .configuration_lfm3 import Lfm3Config


def _config_class(model_type):
    # the LFM3 configs ship with the checkpoint instead of the transformers registry
    remote = {"lfm3": Lfm3Config, "lfm3_vl_vision": Lfm3VlVisionConfig}
    return remote.get(model_type) or CONFIG_MAPPING[model_type]


@auto_docstring(checkpoint="LiquidAI/LFM3-VL-2B")
@strict
class Lfm3VlVisionConfig(PreTrainedConfig):
    r"""
    layer_types (`list[str]`, *optional*):
        Per-layer operator type, each one of `"full_attention"` or `"conv"`. Defaults to two gated convolution
        layers followed by alternating attention and convolution layers.
    patch_size (`int`, *optional*, defaults to 16):
        The spatial size of each image patch.
    temporal_patch_size (`int`, *optional*, defaults to 2):
        The number of video frames packed into one patch.
    spatial_merge_size (`int`, *optional*, defaults to 2):
        The side of the square block of patches merged into one language token.
    conv_kernel_size (`int`, *optional*, defaults to 3):
        The kernel size of the depthwise 2D convolution layers.
    norm_eps (`float`, *optional*, defaults to 1e-05):
        The epsilon used by the RMS normalization layers.
    """

    model_type = "lfm3_vl_vision"
    base_config_key = "vision_config"
    default_theta = 1_000_000.0
    default_rope_type = "axial"
    _attention_layer_types = ("full_attention",)

    hidden_size: int = 1024
    intermediate_size: int = 2560
    num_hidden_layers: int = 14
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    layer_types: list[str] | None = None
    patch_size: int = 16
    temporal_patch_size: int = 2
    in_channels: int = 3
    spatial_merge_size: int = 2
    conv_kernel_size: int = 3
    norm_eps: float = 1e-5
    initializer_range: float = 0.02
    rope_parameters: dict | None = None

    def __post_init__(self, **kwargs):
        if self.layer_types is None:
            self.layer_types = ["conv" if i < 2 or i % 2 else "full_attention" for i in range(self.num_hidden_layers)]
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        """Part of ``@strict``-powered validation."""
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"`layer_types` has {len(self.layer_types)} entries but `num_hidden_layers` is {self.num_hidden_layers}."
            )
        allowed_layer_types = (*self._attention_layer_types, "conv")
        invalid_layer_types = [layer_type for layer_type in self.layer_types if layer_type not in allowed_layer_types]
        if invalid_layer_types:
            raise ValueError(f"`layer_types` entries must be one of {allowed_layer_types}; got {invalid_layer_types}.")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("`hidden_size` must be a multiple of `num_attention_heads`.")


# The first checkpoints spell every text and `vision_*` field at the top level of the config
_LEGACY_TEXT_KEYS = (
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "layer_types",
    "conv_bias",
    "conv_L_cache",
    "sliding_window",
    "max_position_embeddings",
    "use_sandwich_norm",
    "use_cache",
)
_LEGACY_VISION_KEYS = {
    "vision_hidden_size": "hidden_size",
    "vision_intermediate_size": "intermediate_size",
    "vision_num_hidden_layers": "num_hidden_layers",
    "vision_num_attention_heads": "num_attention_heads",
    "vision_num_key_value_heads": "num_key_value_heads",
    "vision_layer_types": "layer_types",
    "vision_patch_size": "patch_size",
    "vision_temporal_patch_size": "temporal_patch_size",
    "vision_num_channels": "in_channels",
    "vision_spatial_merge_size": "spatial_merge_size",
}
_LEGACY_DROPPED_KEYS = ("vision_patch_dim", "vision_max_position_embeddings")


@auto_docstring(checkpoint="LiquidAI/LFM3-VL-2B")
@strict
class Lfm3VlConfig(PreTrainedConfig):
    r"""
    video_token_id (`int`, *optional*, defaults to 125029):
        The id of the placeholder token for video patches.
    vision_start_token_id (`int`, *optional*, defaults to 125009):
        The id of the token opening an image or a video frame.
    vision_end_token_id (`int`, *optional*, defaults to 125010):
        The id of the token closing an image or a video frame.
    projector_layer_norm_eps (`float`, *optional*, defaults to 1e-06):
        The epsilon of the layer normalization applied to the vision features before projection.
    downsample_factor (`int`, *optional*, defaults to 2):
        The side of the square block of vision tokens merged by the projector, equal to the vision encoder's
        `spatial_merge_size`.
    """

    model_type = "lfm3_vl"
    sub_configs = {"text_config": AutoConfig, "vision_config": Lfm3VlVisionConfig}

    text_config: dict | PreTrainedConfig | None = None
    vision_config: dict | PreTrainedConfig | None = None
    image_token_id: int = 125_030
    video_token_id: int = 125_029
    vision_start_token_id: int = 125_009
    vision_end_token_id: int = 125_010
    projector_hidden_act: str = "gelu"
    projector_hidden_size: int = 4096
    projector_bias: bool = True
    projector_layer_norm_eps: float = 1e-6
    downsample_factor: int = 2
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        if "vision_hidden_size" in kwargs:
            kwargs = self._convert_legacy_flat_config(kwargs)

        if isinstance(self.vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()

        if isinstance(self.text_config, dict):
            self.text_config = {"model_type": "lfm3", **self.text_config}
            self.text_config = _config_class(self.text_config["model_type"])(**self.text_config)
        elif self.text_config is None:
            self.text_config = _config_class("lfm3")()
        self.validate_vision_fields()

        super().__post_init__(**kwargs)

    def validate_vision_fields(self):
        """Part of `@strict`-powered validation, so `validate()` and `save_pretrained` refuse later edits too."""
        if self.downsample_factor != self.vision_config.spatial_merge_size:
            raise ValueError("downsample_factor must match vision_config.spatial_merge_size")
        # an `inputs_embeds` forward embeds both placeholder ids to find them
        vocab_size = self.text_config.vocab_size
        for name in ("image_token_id", "video_token_id"):
            if not 0 <= getattr(self, name) < vocab_size:
                raise ValueError(f"{name}={getattr(self, name)} must lie in [0, text_config.vocab_size={vocab_size})")

    def _convert_legacy_flat_config(self, kwargs: dict) -> dict:
        text_config = {key: kwargs.pop(key) for key in _LEGACY_TEXT_KEYS if key in kwargs}
        text_config["norm_eps"] = kwargs.pop("rms_norm_eps", 1e-5)
        text_config["rope_parameters"] = {
            "full_attention": {
                "rope_type": "default",
                "rope_theta": kwargs.pop("full_attention_rope_theta", 1_000_000.0),
                "partial_rotary_factor": kwargs.pop("partial_rotary_factor", 0.5),
            },
            "sliding_attention": {
                "rope_type": "default",
                "rope_theta": kwargs.pop("sliding_attention_rope_theta", 10_000.0),
            },
        }
        for key in ("pad_token_id", "bos_token_id", "eos_token_id"):
            if kwargs.get(key) is not None:
                text_config[key] = kwargs[key]
        text_config["tie_word_embeddings"] = self.tie_word_embeddings

        vision_config = {new: kwargs.pop(old) for old, new in _LEGACY_VISION_KEYS.items() if old in kwargs}
        vision_config["norm_eps"] = text_config["norm_eps"]
        vision_config["rope_parameters"] = {"rope_type": "axial", "rope_theta": kwargs.pop("vision_rope_theta", 1e6)}
        for key in _LEGACY_DROPPED_KEYS:
            kwargs.pop(key, None)

        self.text_config = text_config
        self.vision_config = vision_config
        return kwargs


__all__ = ["Lfm3VlConfig", "Lfm3VlVisionConfig"]
