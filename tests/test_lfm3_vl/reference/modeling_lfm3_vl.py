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
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.configuration_utils import PreTrainedConfig
from transformers.generation import GenerationMixin
from transformers.integrations import use_kernel_forward_from_hub
from transformers.masking_utils import (
    create_causal_mask,
    create_recurrent_attention_mask,
    create_sliding_window_causal_mask,
)
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, BaseModelOutputWithPooling, CausalLMOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import (
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
    torch_compilable_check,
)
from transformers.utils.generic import (
    accepts_precomputed_kwargs,
    is_flash_attention_requested,
    maybe_autocast,
    merge_with_config_defaults,
)
from transformers.utils.output_capturing import OutputRecorder, capture_outputs
from transformers.vision_utils import get_max_seqlen, get_vision_attention_seqlens, get_vision_position_ids

from .configuration_lfm3_vl import Lfm3VlConfig, Lfm3VlVisionConfig
from .modeling_lfm3 import Lfm3Model


@use_kernel_forward_from_hub("RMSNorm")
class Lfm3VlRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        """
        Lfm3VlRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Lfm3VlMLP(nn.Module):
    def __init__(self, config: Lfm3VlConfig):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Lfm3VlVisionRotaryEmbedding(nn.Module):
    """
    Simple axial 2D rope with same freqs used for H and W grids. The freqs are
    pre-computed using `head-dim//4` which is later used to concat H and W positions.
    The final angles rotate over the whole head dim, no partial rotation involved.
    """

    def __init__(self, config: Lfm3VlVisionConfig):
        super().__init__()
        self.config = config

        self.rope_type = self.config.rope_parameters["rope_type"]
        rope_init_fn: Callable = self.compute_axial_rope_parameters
        if self.rope_type != "axial":
            raise ValueError(f"{self.__class__.__name__} supports only axial rope, but requested {self.rope_type}")
        inv_freq, self.attention_scaling = rope_init_fn(self.config)

        self.inv_freq = nn.Buffer(inv_freq, persistent=False)
        self.original_inv_freq = nn.Buffer(inv_freq.clone(), persistent=False)

    @staticmethod
    def compute_axial_rope_parameters(config: Lfm3VlVisionConfig, **kwargs) -> tuple[torch.Tensor, float]:
        """Inverse frequencies of the axial RoPE over half the head dimension per axis, with an attention scaling of 1."""
        base = config.rope_parameters["rope_theta"]
        spatial_dim = config.hidden_size // config.num_attention_heads // 2
        inv_freq = 1.0 / (base ** (torch.arange(0, spatial_dim, 2, dtype=torch.float) / spatial_dim))
        return inv_freq, 1.0

    @torch.no_grad()
    def forward(self, x, position_ids):
        # position_ids: (2, N) — row 0 = h coords, row 1 = w coords
        position_ids_expanded = position_ids[..., None].float()
        device_type = x.device.type if isinstance(x.device.type, str) else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False):
            freqs = position_ids_expanded * self.inv_freq.float()
            cos = freqs.cos() * self.attention_scaling
            sin = freqs.sin() * self.attention_scaling

        cos = self.recomposition_frequencies(cos)
        sin = self.recomposition_frequencies(sin)
        return cos, sin

    def recomposition_frequencies(self, freq):
        """
        Recompose the frequencies into the final spatial layout used per each grid.
        """
        freq_h, freq_w = freq[:, 0], freq[:, 1]
        freq_hw = torch.cat([freq_h, freq_w], dim=-1)
        return torch.cat([freq_hw, freq_hw], dim=-1)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class Lfm3VlVisionAttention(nn.Module):
    """Bidirectional grouped-query attention over the patches of one frame, with RMSNorm on the query and key heads."""

    def __init__(self, config: Lfm3VlVisionConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = 0.0
        self.is_causal = False
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_layernorm = Lfm3VlRMSNorm(self.head_dim, eps=config.norm_eps)
        self.k_layernorm = Lfm3VlRMSNorm(self.head_dim, eps=config.norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        max_seqlen: int | None = None,
        frame_shapes: list[list[int]] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        query_states = self.q_layernorm(self.q_proj(hidden_states).view(seq_length, self.num_heads, self.head_dim))
        key_states = self.k_layernorm(
            self.k_proj(hidden_states).view(seq_length, self.num_key_value_heads, self.head_dim)
        )
        value_states = self.v_proj(hidden_states).view(seq_length, self.num_key_value_heads, self.head_dim)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        if is_flash_attention_requested(self.config):
            max_seqlen = get_max_seqlen(cu_seqlens, self.config, kwargs={"max_seqlen": max_seqlen})
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0,
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                max_length_q=max_seqlen,
                max_length_k=max_seqlen,
                is_causal=False,
                **kwargs,
            )
        else:
            # Other implementations: frames of one shape (every video, equal-size images) attend as one batch, mixed
            # shapes frame by frame. `frame_shapes` gives the lengths without a device sync.
            if frame_shapes is not None:
                lengths = [height * width for height, width in frame_shapes]
            else:
                lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
            if len(set(lengths)) == 1:
                query_states, key_states, value_states = (
                    tensor.reshape(-1, len(lengths), lengths[0], self.head_dim).transpose(0, 1)
                    for tensor in (query_states, key_states, value_states)
                )
                attn_output, attn_weights = attention_interface(
                    self,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask=None,
                    scaling=self.scaling,
                    dropout=0.0,
                    is_causal=False,
                    **kwargs,
                )
            else:
                splits = [torch.split(tensor, lengths, dim=2) for tensor in (query_states, key_states, value_states)]
                attn_outputs, attn_weights = [], []
                for q, k, v in zip(*splits):
                    output, weights = attention_interface(
                        self,
                        q,
                        k,
                        v,
                        attention_mask=None,
                        scaling=self.scaling,
                        dropout=0.0,
                        is_causal=False,
                        **kwargs,
                    )
                    attn_outputs.append(output)
                    attn_weights.append(weights)
                attn_output = torch.cat(attn_outputs, dim=1)
                # frames of different shapes have differently sized maps: one entry per frame, or None under sdpa
                attn_weights = None if attn_weights[0] is None else tuple(attn_weights)

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights


class Lfm3VlVisionConv2d(nn.Module):
    """Gated depthwise 2D convolution over the patch grid of each frame."""

    def __init__(self, config: Lfm3VlVisionConfig):
        super().__init__()
        self.spatial_merge_size = config.spatial_merge_size
        self.in_proj = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.spatial_conv = nn.Conv2d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=config.conv_kernel_size,
            padding=config.conv_kernel_size // 2,
            groups=config.hidden_size,
            bias=False,
        )
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor, frame_shapes: list[list[int]]):
        B, C, x = self.in_proj(hidden_states).chunk(3, dim=-1)
        gated = B * x
        merge = self.spatial_merge_size
        num_frames = len(frame_shapes)
        if len(set(map(tuple, frame_shapes))) == 1:
            # frames of one shape (every video, equal-size images) convolve as one batch: merge-block order -> raster
            # grid, convolve, back
            height, width = frame_shapes[0]
            frames = gated.view(num_frames, height // merge, width // merge, merge, merge, -1)
            frames = frames.permute(0, 5, 1, 3, 2, 4).reshape(num_frames, -1, height, width)
            frames = self.spatial_conv(frames)
            frames = frames.view(num_frames, -1, height // merge, merge, width // merge, merge).permute(
                0, 2, 4, 3, 5, 1
            )
            conv_output = frames.reshape(gated.shape[0], -1)
        else:
            conv_outputs = []
            lengths = [height * width for height, width in frame_shapes]
            for frame, (height, width) in zip(torch.split(gated, lengths, dim=0), frame_shapes):
                frame = frame.view(height // merge, width // merge, merge, merge, -1)
                frame = frame.permute(4, 0, 2, 1, 3).reshape(1, -1, height, width)
                frame = self.spatial_conv(frame)
                frame = frame.view(-1, height // merge, merge, width // merge, merge).permute(1, 3, 2, 4, 0)
                conv_outputs.append(frame.reshape(height * width, -1))
            conv_output = torch.cat(conv_outputs, dim=0)
        return self.out_proj(C * conv_output)


class Lfm3VlVisionLayer(GradientCheckpointingLayer):
    def __init__(self, config: Lfm3VlVisionConfig, layer_idx: int):
        super().__init__()
        self.is_attention_layer = config.layer_types[layer_idx] == "full_attention"
        if self.is_attention_layer:
            self.self_attn = Lfm3VlVisionAttention(config)
        else:
            self.conv = Lfm3VlVisionConv2d(config)
        self.feed_forward = Lfm3VlMLP(config)
        self.operator_norm = Lfm3VlRMSNorm(config.hidden_size, eps=config.norm_eps)
        self.ffn_norm = Lfm3VlRMSNorm(config.hidden_size, eps=config.norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        frame_shapes: list[list[int]],
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if self.is_attention_layer:
            operator_output, _ = self.self_attn(
                self.operator_norm(hidden_states),
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                frame_shapes=frame_shapes,
                **kwargs,
            )
        else:
            operator_output = self.conv(self.operator_norm(hidden_states), frame_shapes)
        hidden_states = hidden_states + operator_output
        hidden_states = hidden_states + self.feed_forward(self.ffn_norm(hidden_states))
        return hidden_states


def create_masks_by_layer_type(
    config: PreTrainedConfig,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor | None,
    past_key_values: Cache | None,
    position_ids: torch.Tensor | None = None,
) -> dict:
    """The cache sizes each mask from the layer it is asked about, and the generic mask utilities ask about the first
    non-sliding layer, which for a leading `conv` layer redirects to the first attention layer of any type: name the
    first layer of each mask's own type instead."""
    config = config.get_text_config()
    mask_kwargs = {
        "config": config,
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "past_key_values": past_key_values,
        "position_ids": position_ids,
    }
    layer_types, causal_mask_mapping = config.layer_types, {}
    if "full_attention" in layer_types:
        layer_idx = layer_types.index("full_attention")
        causal_mask_mapping["full_attention"] = create_causal_mask(**mask_kwargs, layer_idx=layer_idx)
    if "sliding_attention" in layer_types:
        layer_idx = layer_types.index("sliding_attention")
        causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(
            **mask_kwargs, layer_idx=layer_idx
        )
    if "conv" in layer_types:
        causal_mask_mapping["conv"] = create_recurrent_attention_mask(**mask_kwargs)
    return causal_mask_mapping


@auto_docstring
class Lfm3VlPreTrainedModel(PreTrainedModel):
    config: Lfm3VlConfig
    base_model_prefix = "model"
    input_modalities = ("image", "video", "text")
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = ["past_key_values"]

    _supports_flash_attn = True
    _supports_sdpa = True
    # the cached decode step traces without graph breaks, like the text model
    _can_compile_fullgraph = True
    _supports_flex_attn = True
    _supports_attention_backend = True
    _no_split_modules = ["Lfm3VlVisionLayer"]

    @staticmethod
    def create_masks_for_generate(
        config: PreTrainedConfig,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None,
        position_ids: torch.Tensor | None = None,
        **kwargs,
    ) -> dict:
        # the static-cache masks of the text model are sized by layer type, not by the generic first attention layer
        return create_masks_by_layer_type(config, inputs_embeds, attention_mask, past_key_values, position_ids)


def get_vision_frame_shapes(grid_thw: torch.Tensor) -> list[list[int]]:
    return grid_thw[:, 1:].repeat_interleave(grid_thw[:, 0], dim=0).tolist()


@auto_docstring(
    custom_intro="""
    The LFM3-VL vision encoder: a stack of gated convolution and bidirectional attention layers over the patches of
    each image or video frame, embedded by a linear patchifier.
    """
)
class Lfm3VlVisionModel(Lfm3VlPreTrainedModel):
    config: Lfm3VlVisionConfig
    input_modalities = ("image", "video")
    _input_embed_layer = "patchifier"
    _can_record_outputs = {
        "hidden_states": Lfm3VlVisionLayer,
        "attentions": OutputRecorder(Lfm3VlVisionAttention, index=1),
    }

    def __init__(self, config: Lfm3VlVisionConfig):
        super().__init__(config)
        self.spatial_merge_size = config.spatial_merge_size
        patch_dim = config.in_channels * config.temporal_patch_size * config.patch_size**2
        self.patchifier = nn.Linear(patch_dim, config.hidden_size, bias=False)
        self.embedding_norm = Lfm3VlRMSNorm(config.hidden_size, eps=config.norm_eps)
        self.rotary_pos_emb = Lfm3VlVisionRotaryEmbedding(config)
        self.layers = nn.ModuleList(
            [Lfm3VlVisionLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self, pixel_values: torch.Tensor, grid_thw: torch.Tensor, **kwargs: Unpack[TransformersKwargs]
    ) -> BaseModelOutputWithPooling:
        r"""
        pixel_values (`torch.Tensor` of shape `(num_patches, in_channels * temporal_patch_size * patch_size**2)`):
            The flattened patches of every image or video, grouped by merge block.
        grid_thw (`torch.LongTensor` of shape `(num_images_or_videos, 3)`):
            The temporal, height and width dimensions of the patch grid of each image or video.
        """
        position_ids = get_vision_position_ids(grid_thw, self.spatial_merge_size, kwargs=kwargs)
        cu_seqlens, max_seqlen = get_vision_attention_seqlens(grid_thw, self.config, kwargs=kwargs)
        frame_shapes = get_vision_frame_shapes(grid_thw)

        hidden_states = self.patchifier(pixel_values.to(self.patchifier.weight.dtype))
        hidden_states = self.embedding_norm(hidden_states)
        position_embeddings = self.rotary_pos_emb(hidden_states, position_ids)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                cu_seqlens=cu_seqlens,
                frame_shapes=frame_shapes,
                max_seqlen=max_seqlen,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        return BaseModelOutputWithPooling(last_hidden_state=hidden_states)


class Lfm3VlMultiModalProjector(nn.Module):
    """Merges each block of `downsample_factor**2` vision tokens into one language token."""

    def __init__(self, config: Lfm3VlConfig):
        super().__init__()
        self.downsample_factor = config.downsample_factor
        self.layer_norm = nn.LayerNorm(config.vision_config.hidden_size, eps=config.projector_layer_norm_eps)
        self.linear_1 = nn.Linear(
            config.vision_config.hidden_size * config.downsample_factor**2,
            config.projector_hidden_size,
            bias=config.projector_bias,
        )
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(
            config.projector_hidden_size, config.text_config.hidden_size, bias=config.projector_bias
        )

    def forward(self, image_features: torch.Tensor):
        hidden_states = self.layer_norm(image_features).reshape(-1, self.linear_1.in_features)
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


@auto_docstring(
    custom_intro="""
    Base class for Lfm3Vl outputs, with hidden states and attentions.
    """
)
@dataclass
class Lfm3VlModelOutputWithPast(BaseModelOutputWithPast):
    r"""
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    image_hidden_states (`torch.FloatTensor`, *optional*):
        A `torch.FloatTensor` of size `(num_image_tokens, hidden_size)`.
        image_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
    video_hidden_states (`torch.FloatTensor`, *optional*):
        A `torch.FloatTensor` of size `(num_video_tokens, hidden_size)`.
        video_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
    """

    image_hidden_states: torch.FloatTensor | None = None
    video_hidden_states: torch.FloatTensor | None = None


@auto_docstring(
    custom_intro="""
    Base class for Lfm3Vl causal language model (or autoregressive) outputs.
    """
)
@dataclass
class Lfm3VlCausalLMOutputWithPast(CausalLMOutputWithPast):
    r"""
    image_hidden_states (`torch.FloatTensor`, *optional*):
        A `torch.FloatTensor` of size `(num_image_tokens, hidden_size)`.
        image_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
    video_hidden_states (`torch.FloatTensor`, *optional*):
        A `torch.FloatTensor` of size `(num_video_tokens, hidden_size)`.
        video_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
    """

    image_hidden_states: torch.FloatTensor | None = None
    video_hidden_states: torch.FloatTensor | None = None


@auto_docstring(
    custom_intro="""
    The Lfm3Vl model which consists of a vision backbone and a language model, without a language modeling head.
    """
)
class Lfm3VlModel(Lfm3VlPreTrainedModel):
    def __init__(self, config: Lfm3VlConfig):
        super().__init__(config)
        self.vision_tower = Lfm3VlVisionModel(config.vision_config)
        self.multi_modal_projector = Lfm3VlMultiModalProjector(config)
        self.language_model = Lfm3Model(config.text_config)
        self.post_init()

    @accepts_precomputed_kwargs(modality="image")
    @can_return_tuple
    @auto_docstring(
        custom_intro="Obtains image last hidden states from the vision tower and apply multimodal projection."
    )
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        pixel_values (`torch.FloatTensor` of shape `(num_patches, in_channels * temporal_patch_size * patch_size**2)`):
            The flattened patches of every image, grouped by merge block.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`):
            The temporal, height and width dimensions of the patch grid of each image.
        """
        image_outputs = self.vision_tower(pixel_values, grid_thw=image_grid_thw, return_dict=True, **kwargs)
        image_features = self.multi_modal_projector(image_outputs.last_hidden_state)
        split_sizes = (image_grid_thw.prod(-1) // self.vision_tower.spatial_merge_size**2).tolist()
        image_outputs.pooler_output = torch.split(image_features, split_sizes)
        return image_outputs

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: torch.FloatTensor | None = None,
        video_features: torch.FloatTensor | None = None,
    ):
        """Placeholder masks from `input_ids` or `inputs_embeds`, checked against the feature counts."""
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.full((), self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
            special_video_mask = inputs_embeds == self.get_input_embeddings()(
                torch.full((), self.config.video_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_video_mask = special_video_mask.all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id
            special_video_mask = input_ids == self.config.video_token_id

        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).to(inputs_embeds.device)
        if image_features is not None:
            torch_compilable_check(
                n_image_tokens * inputs_embeds.shape[-1] == image_features.numel(),
                f"Image features and image tokens do not match, tokens: {n_image_tokens}, features: {image_features.shape[0]}",
            )

        n_video_tokens = special_video_mask.sum()
        special_video_mask = special_video_mask.unsqueeze(-1).to(inputs_embeds.device)
        if video_features is not None:
            torch_compilable_check(
                n_video_tokens * inputs_embeds.shape[-1] == video_features.numel(),
                f"Video features and video tokens do not match, tokens: {n_video_tokens}, features: {video_features.shape[0]}",
            )
        return special_image_mask, special_video_mask

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        mm_encoder_outputs: dict[str, BaseModelOutputWithPooling] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Lfm3VlModelOutputWithPast:
        r"""
        pixel_values (`torch.FloatTensor` of shape `(num_patches, in_channels * temporal_patch_size * patch_size**2)`, *optional*):
            The flattened patches of every image, grouped by merge block.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width dimensions of the patch grid of each image.
        pixel_values_videos (`torch.FloatTensor` of shape `(num_patches, in_channels * temporal_patch_size * patch_size**2)`, *optional*):
            The flattened patches of every video, grouped by merge block.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width dimensions of the patch grid of each video.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if (pixel_values is not None or pixel_values_videos is not None) and mm_encoder_outputs is not None:
            raise ValueError(
                "You cannot specify both pixel_values/pixel_values_videos and mm_encoder_outputs at the same time"
            )

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        mm_encoder_outputs = mm_encoder_outputs if mm_encoder_outputs is not None else {}
        if (pixel_values is None) != (image_grid_thw is None):
            raise ValueError("pixel_values and image_grid_thw must be supplied together")
        if (pixel_values_videos is None) != (video_grid_thw is None):
            raise ValueError("pixel_values_videos and video_grid_thw must be supplied together")
        if mm_encoder_outputs.get("image") is None and pixel_values is not None:
            mm_encoder_outputs["image"] = self.get_image_features(
                pixel_values, image_grid_thw, return_dict=True, **kwargs
            )
        if mm_encoder_outputs.get("video") is None and pixel_values_videos is not None:
            mm_encoder_outputs["video"] = self.get_video_features(
                pixel_values_videos, video_grid_thw, return_dict=True, **kwargs
            )

        image_features = None
        if mm_encoder_outputs.get("image") is not None:
            image_features = torch.cat(mm_encoder_outputs["image"].pooler_output, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_features
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_features)

        video_features = None
        if mm_encoder_outputs.get("video") is not None:
            video_features = torch.cat(mm_encoder_outputs["video"].pooler_output, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_features
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_features)

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        return Lfm3VlModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features,
            video_hidden_states=video_features,
        )

    @accepts_precomputed_kwargs(modality="video")
    @can_return_tuple
    @auto_docstring(
        custom_intro="Obtains video last hidden states from the vision tower and apply multimodal projection."
    )
    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: torch.LongTensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        pixel_values_videos (`torch.FloatTensor` of shape `(num_patches, in_channels * temporal_patch_size * patch_size**2)`):
            The flattened patches of every video, grouped by merge block.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`):
            The temporal, height and width dimensions of the patch grid of each video.
        """
        return self.get_image_features(pixel_values_videos, video_grid_thw, **kwargs)


def reject_packed_generation(config: PreTrainedConfig, input_ids, generation_config, model_kwargs, is_first_iteration):
    """`generate` decodes from the cache its prefill fills, and no mask it builds isolates packed documents
    (`position_ids` restarting): the prompt is refused before the prefill, one forward or every chunk, writes to it."""
    position_ids, attention_mask = model_kwargs.get("position_ids"), model_kwargs.get("attention_mask")
    cache = model_kwargs.get("past_key_values")
    if cache is None or position_ids is None or position_ids.ndim != 2:
        return
    if attention_mask is not None and attention_mask.ndim != 2:
        return
    has_attention = any(t != "conv" for t in config.get_text_config().layer_types)
    filled = bool(cache.get_seq_length() > 0 if has_attention else cache.has_previous_state())
    inputs_embeds = model_kwargs.get("inputs_embeds") if is_first_iteration else None
    length = input_ids.shape[1] if inputs_embeds is None else inputs_embeds.shape[1]
    if generation_config.prefill_chunk_size is not None:
        # the chunks run from the start of the prompt
        position_ids = position_ids[:, :length]
    else:
        if filled and has_attention:
            # the slice `_prefill` feeds: a prompt that spans the cache is cut to its uncached part
            mask_length = getattr(generation_config, "_mask_length", -1)
            if attention_mask is not None:
                mask_length = attention_mask.shape[1]
            if inputs_embeds is not None or mask_length == length:
                length -= int(cache.get_seq_length())
        position_ids = position_ids[:, position_ids.shape[1] - length :]
    if attention_mask is None:
        attention_mask = torch.ones_like(position_ids)
    if bool(packed_under_padding(position_ids, attention_mask, filled).any()):
        raise ValueError(
            "`generate` cannot decode packed documents (`position_ids` restarting, or starting at 0 on a cache that "
            "holds tokens): attention would span them. Generate each document in its own row of a left-padded batch."
        )


def packed_under_padding(position_ids: torch.Tensor, padding_mask: torch.Tensor, filled) -> torch.Tensor:
    """Padding repeats position 0, so packing is a restart between two kept tokens, or a kept position 0 on a filled
    cache whose row holds a kept token; attention under a padding mask cannot isolate it."""
    restarts = torch.diff(position_ids, prepend=position_ids[:, :1] - 1, dim=-1) != 1
    valid = padding_mask[:, -restarts.shape[-1] :].bool()
    packed = restarts[:, 1:] & valid[:, 1:] & valid[:, :-1]
    if filled:
        # a row whose cached tokens are all padding (chunked prefill of a left-padded batch) has no document to pack
        # against; a mask covering only the chunk keeps the strict rule
        cached = padding_mask[:, : padding_mask.shape[-1] - restarts.shape[-1]]
        cached_valid = cached.bool().any(-1, keepdim=True) if cached.shape[-1] else torch.ones_like(valid[:, :1])
        packed = packed | (valid[:, :1] & (position_ids[:, :1] == 0) & cached_valid)
    return packed


@auto_docstring(
    custom_intro="""
    The LFM3_VL model which consists of a vision backbone and a language model.
    """
)
class Lfm3VlForConditionalGeneration(Lfm3VlPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}

    def __init__(self, config: Lfm3VlConfig):
        super().__init__(config)
        self.model = Lfm3VlModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    @auto_docstring
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        pixel_values (`torch.FloatTensor` of shape `(num_patches, in_channels * temporal_patch_size * patch_size**2)`):
            The flattened patches of every image, grouped by merge block.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`):
            The temporal, height and width dimensions of the patch grid of each image.
        """
        return self.model.get_image_features(pixel_values, image_grid_thw, **kwargs)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        mm_encoder_outputs: dict[str, BaseModelOutputWithPooling] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Lfm3VlCausalLMOutputWithPast:
        r"""
        pixel_values (`torch.FloatTensor` of shape `(num_patches, in_channels * temporal_patch_size * patch_size**2)`, *optional*):
            The flattened patches of every image, grouped by merge block.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width dimensions of the patch grid of each image.
        pixel_values_videos (`torch.FloatTensor` of shape `(num_patches, in_channels * temporal_patch_size * patch_size**2)`, *optional*):
            The flattened patches of every video, grouped by merge block.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width dimensions of the patch grid of each video.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoProcessor, AutoModelForImageTextToText

        >>> model = AutoModelForImageTextToText.from_pretrained("LiquidAI/LFM3-VL-2B")
        >>> processor = AutoProcessor.from_pretrained("LiquidAI/LFM3-VL-2B")

        >>> conversation = [
        ...     {
        ...         "role": "user",
        ...         "content": [
        ...             {"type": "image", "url": "https://www.ilankelman.org/stopsigns/australia.jpg"},
        ...             {"type": "text", "text": "What is in this image?"},
        ...         ],
        ...     },
        ... ]

        >>> inputs = processor.apply_chat_template(
        ...     conversation, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ... )

        >>> outputs = model.generate(**inputs, max_new_tokens=45)
        >>> processor.batch_decode(outputs, skip_special_tokens=True)[0]
        ```"""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            use_cache=use_cache,
            mm_encoder_outputs=mm_encoder_outputs,
            **kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )

        return Lfm3VlCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
            video_hidden_states=outputs.video_hidden_states,
        )

    def _prefill(self, input_ids, generation_config, model_kwargs, is_first_iteration=True):
        # the text model's refusal, before the prefill writes to the cache
        reject_packed_generation(self.config, input_ids, generation_config, model_kwargs, is_first_iteration)
        return super()._prefill(input_ids, generation_config, model_kwargs, is_first_iteration=is_first_iteration)

    @auto_docstring
    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: torch.LongTensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        pixel_values_videos (`torch.FloatTensor` of shape `(num_patches, in_channels * temporal_patch_size * patch_size**2)`):
            The flattened patches of every video, grouped by merge block.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`):
            The temporal, height and width dimensions of the patch grid of each video.
        """
        return self.model.get_video_features(pixel_values_videos, video_grid_thw, **kwargs)


__all__ = ["Lfm3VlForConditionalGeneration", "Lfm3VlModel", "Lfm3VlPreTrainedModel", "Lfm3VlVisionModel"]
