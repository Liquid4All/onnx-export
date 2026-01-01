"""Vision encoder builder for LFM2-VL ONNX export.

This module contains the VisionEmbedBuilder class which creates an ONNX graph
combining the SigLIP2 vision encoder with the MLP projector.
"""

import logging

import numpy as np
import onnx
from onnx import TensorProto, helper

from liquidonnx.builder_base import ONNXBuilderBase
from liquidonnx.lfm2_vl import VISION_MODE_CONV2D, VISION_MODE_TILED
from liquidonnx.lfm2_vl.builder.config import LFM2VLConfig

logger = logging.getLogger(__name__)


class VisionEmbedBuilder(ONNXBuilderBase):
    """
    Fused vision encoder + projector builder for ONNX export.

    Creates an ONNX graph that combines:
    - SigLIP2 vision encoder (patch embedding + transformer layers)
    - MLP projector with pixel unshuffle

    Graph structure:
        pixel_values [B, N, 768] or [B, 3, H, W]
            ↓
        ┌─────────────────────────────────────┐
        │  Patch Embedding                    │
        │  + Position Embedding (bilinear)    │
        └─────────────────────────────────────┘
            ↓
        ┌─────────────────────────────────────┐
        │  N × Transformer Encoder Layers     │
        │  (Self-Attention + MLP)             │
        └─────────────────────────────────────┘
            ↓
        Post LayerNorm
            ↓
        ┌─────────────────────────────────────┐
        │  Projector                          │
        │  Pixel Unshuffle (2x2 -> 4x)        │
        │  + LayerNorm + MLP                  │
        └─────────────────────────────────────┘
            ↓
        image_embeddings [B, N/4, text_hidden]

    Supports two input formats:
    - "tiled": [batch, num_patches, 768] pre-extracted patches (HuggingFace style)
    - "conv2d": [batch, 3, H, W] raw image pixels (llama.cpp style)
    """

    def __init__(
        self,
        config: LFM2VLConfig,
        vision_input_format: str = VISION_MODE_TILED,
    ):
        """
        Args:
            config: Model configuration
            vision_input_format: "tiled" for [B, N, 768] or "conv2d" for [B, 3, H, W]
        """
        super().__init__()
        self.config = config
        self.vision_config = config.vision_config
        self.head_dim = config.vision_config.hidden_size // config.vision_config.num_attention_heads
        self.vision_input_format = vision_input_format

        # Projector dimensions
        self.vision_hidden = config.vision_config.hidden_size
        self.text_hidden = config.text_config.hidden_size
        self.proj_hidden = config.projector_hidden_size
        self.downsample = config.downsample_factor

    def make_vision_layernorm(
        self, input_name: str, weight_name: str, bias_name: str, output_name: str
    ) -> str:
        """Create LayerNormalization node with vision encoder epsilon."""
        return self.make_layernorm(
            input_name,
            weight_name,
            bias_name,
            output_name,
            epsilon=self.vision_config.layer_norm_eps,
        )

    def _build_pos_embed_resize(
        self,
        pos_emb_input: str,
        spatial_h: str,
        spatial_w: str,
        src_h: int,
        src_w: int,
        hidden_size: int,
    ) -> str:
        """Build position embedding resize using ONNX Resize operator.

        For upsampling (which is the common case for position embeddings),
        PyTorch antialias and regular bilinear produce identical results.
        So we use simple ONNX Resize with half_pixel mode.

        Args:
            pos_emb_input: Input tensor name (1, hidden, src_h, src_w)
            spatial_h: Output height (scalar tensor name)
            spatial_w: Output width (scalar tensor name)
            src_h: Source height (constant, e.g., 16) - unused, kept for API compat
            src_w: Source width (constant, e.g., 16) - unused, kept for API compat
            hidden_size: Hidden dimension (e.g., 768)

        Returns:
            Output tensor name (1, tgt_h * tgt_w, hidden)
        """
        prefix = "interp"

        self.add_initializer(f"{prefix}/batch_1", np.array([1], dtype=np.int64))
        self.add_initializer(f"{prefix}/hidden", np.array([hidden_size], dtype=np.int64))
        self.add_initializer(f"{prefix}/unsq_0", np.array([0], dtype=np.int64))

        spatial_h_unsq = self.make_node(
            "Unsqueeze", [spatial_h, f"{prefix}/unsq_0"], [f"{prefix}/h_unsq"]
        )
        spatial_w_unsq = self.make_node(
            "Unsqueeze", [spatial_w, f"{prefix}/unsq_0"], [f"{prefix}/w_unsq"]
        )

        # sizes: [1, hidden_size, spatial_h, spatial_w]
        sizes = self.make_node(
            "Concat",
            [f"{prefix}/batch_1", f"{prefix}/hidden", spatial_h_unsq, spatial_w_unsq],
            [f"{prefix}/sizes"],
            axis=0,
        )

        # Empty ROI required by ONNX Resize
        self.add_initializer(f"{prefix}/empty_roi", np.array([], dtype=np.float32))

        resized = self.make_node(
            "Resize",
            [pos_emb_input, f"{prefix}/empty_roi", "", sizes],  # Empty scales, use sizes
            [f"{prefix}/resized"],
            mode="linear",
            coordinate_transformation_mode="half_pixel",
        )

        # [1, hidden, h, w] → [1, h, w, hidden]
        transposed = self.make_node(
            "Transpose", [resized], [f"{prefix}/transposed"], perm=[0, 2, 3, 1]
        )

        # [1, h, w, hidden] → [1, h*w, hidden]
        self.add_initializer(
            f"{prefix}/reshape_out", np.array([1, -1, hidden_size], dtype=np.int64)
        )
        output = self.make_node("Reshape", [transposed, f"{prefix}/reshape_out"], ["pos_emb/final"])

        return output

    def build_inputs(self):
        """Create model inputs based on vision_input_format."""
        if self.vision_input_format == VISION_MODE_CONV2D:
            # Conv2d mode: raw image input [batch, channels, height, width]
            # Simpler preprocessing (just resize + normalize), like llama.cpp
            self.inputs.append(
                helper.make_tensor_value_info(
                    "pixel_values",
                    TensorProto.FLOAT,
                    ["batch_size", self.vision_config.num_channels, "height", "width"],
                )
            )
            # Spatial dimensions after n_merge (for projector reshape)
            # spatial_h = height / patch_size / n_merge
            # spatial_w = width / patch_size / n_merge
            self.inputs.append(
                helper.make_tensor_value_info(
                    "spatial_h",
                    TensorProto.INT64,
                    [],  # scalar
                )
            )
            self.inputs.append(
                helper.make_tensor_value_info(
                    "spatial_w",
                    TensorProto.INT64,
                    [],  # scalar
                )
            )
        else:
            # Tiled mode: pre-extracted patches [batch, num_patches, patch_dim]
            # Requires complex preprocessing (tiling, patch extraction)
            patch_dim = (
                self.vision_config.num_channels
                * self.vision_config.patch_size
                * self.vision_config.patch_size
            )
            self.inputs.append(
                helper.make_tensor_value_info(
                    "pixel_values", TensorProto.FLOAT, ["batch_size", "num_patches", patch_dim]
                )
            )

            # pixel_attention_mask: 1=valid, 0=padded (matches onnx-community naming)
            self.inputs.append(
                helper.make_tensor_value_info(
                    "pixel_attention_mask", TensorProto.INT64, ["batch_size", "num_patches"]
                )
            )

            # spatial_shapes: [batch_size, 2] with (height, width) in patch units
            # Allows non-square images (matches community ONNX and PyTorch)
            self.inputs.append(
                helper.make_tensor_value_info(
                    "spatial_shapes", TensorProto.INT64, ["batch_size", 2]
                )
            )

    def build_outputs(self):
        """Create model outputs."""
        # Image embeddings in text space: [batch, num_image_tokens, text_hidden_size]
        self.outputs.append(
            helper.make_tensor_value_info(
                "image_embeddings",
                TensorProto.FLOAT,
                ["batch_size", "num_image_tokens", self.text_hidden],
            )
        )

    def build_attention_mask(self):
        """Build attention mask preprocessing for tiled mode.

        Converts pixel_attention_mask to additive attention bias:
            - Input: 1=valid, 0=padded [B, N] int64
            - Output: 0=valid, -inf=masked [B, num_heads, N, N] float32

        Matches the community ONNX approach which uses attention_bias (6th input)
        to MultiHeadAttention rather than key_padding_mask (5th input).
        """
        if self.vision_input_format != VISION_MODE_TILED:
            return

        num_heads = self.vision_config.num_attention_heads

        # Cast to float32
        mask_float = self.make_node(
            "Cast", ["pixel_attention_mask"], ["attn_mask/float"], to=TensorProto.FLOAT
        )

        # Invert: 1.0 - mask (now 0=valid, 1=masked)
        self.add_initializer("attn_mask/one_f", np.array(1.0, dtype=np.float32))
        inverted = self.make_node("Sub", ["attn_mask/one_f", mask_float], ["attn_mask/inverted"])

        # Multiply by -inf to create additive bias (0=valid, -inf=masked)
        self.add_initializer(
            "attn_mask/neg_inf", np.array(-3.4028234663852886e38, dtype=np.float32)
        )
        bias_2d = self.make_node("Mul", [inverted, "attn_mask/neg_inf"], ["attn_mask/bias_2d"])

        # Unsqueeze to [B, 1, 1, N] for broadcasting
        self.add_initializer("attn_mask/axes_unsq", np.array([1, 2], dtype=np.int64))
        bias_4d = self.make_node(
            "Unsqueeze", [bias_2d, "attn_mask/axes_unsq"], ["attn_mask/bias_unsq"]
        )

        shape = self.make_node("Shape", ["pixel_attention_mask"], ["attn_mask/shape"])

        self.add_initializer("attn_mask/idx_0", np.array(0, dtype=np.int64))
        batch_size = self.make_node(
            "Gather", [shape, "attn_mask/idx_0"], ["attn_mask/batch_size"], axis=0
        )
        batch_unsq = self.make_node(
            "Unsqueeze", [batch_size, "attn_mask/idx_0"], ["attn_mask/batch_unsq"]
        )

        self.add_initializer("attn_mask/idx_1", np.array(1, dtype=np.int64))
        seq_len = self.make_node(
            "Gather", [shape, "attn_mask/idx_1"], ["attn_mask/seq_len"], axis=0
        )
        seq_unsq = self.make_node("Unsqueeze", [seq_len, "attn_mask/idx_0"], ["attn_mask/seq_unsq"])

        self.add_initializer("attn_mask/num_heads", np.array([num_heads], dtype=np.int64))
        expand_shape = self.make_node(
            "Concat",
            [batch_unsq, "attn_mask/num_heads", seq_unsq, seq_unsq],
            ["attn_mask/expand_shape"],
            axis=0,
        )

        self.make_node("Expand", [bias_4d, expand_shape], ["attention_bias"])

    def build_patch_embedding(self) -> str:
        """Build patch embedding layer with position embeddings.

        Graph structure:
            pixel_values
                ↓
            ┌─────────────────────────────────────┐
            │  Patch Projection                   │
            │  Conv2d: [B,3,H,W] → Conv2d(16,16)  │
            │  Tiled:  [B,N,768] → Linear         │
            └─────────────────────────────────────┘
                ↓
            patch_embeds [B, N, hidden]
                ↓
            ┌─────────────────────────────────────┐
            │  Position Embeddings                │
            │  Bilinear interpolation 16x16 → HxW │
            │  Tile across batch                  │
            └─────────────────────────────────────┘
                ↓
            pos_embeds [B, N, hidden]
                ↓
            Add (patch_embeds + pos_embeds)
                ↓
            patch_embeddings [B, N, hidden]

        Position embeddings are stored as 16x16 learned embeddings and bilinearly
        interpolated to match the input spatial size (sqrt(N) x sqrt(N) for tiled,
        H/P x W/P for conv2d where P=patch_size).
        """
        prefix = "vision_model.embeddings.patch_embedding"
        H = self.vision_config.hidden_size
        P = self.vision_config.patch_size
        C = self.vision_config.num_channels

        linear_weight = self.weights[f"{prefix}.weight"]
        linear_bias = self.weights[f"{prefix}.bias"]

        if self.vision_input_format == VISION_MODE_CONV2D:
            # === Conv2d mode ===
            # Reshape Linear weights to Conv2d format
            # Linear: [hidden_size, C*P*P] → Conv2d: [hidden_size, C, P, P]
            # Linear weight is [out_features, in_features] = [768, 768]
            # The original model flattens patches as HWC (P*P*C = 16*16*3 = 768)
            # So we first reshape to [H, P, P, C] then transpose to [H, C, P, P]
            # This matches the GGUF converter: view(H, 16, 16, 3).permute(0, 3, 1, 2)
            conv_weight = linear_weight.reshape(H, P, P, C).transpose(0, 3, 1, 2)  # [H, C, P, P]
            self.add_initializer(f"{prefix}.conv_weight", conv_weight)
            self.add_initializer(f"{prefix}.bias", linear_bias)

            # Conv2d: [B, C, H, W] -> [B, hidden, H/P, W/P]
            conv_out = self.make_node(
                "Conv",
                ["pixel_values", f"{prefix}.conv_weight", f"{prefix}.bias"],
                ["patch_embed/conv_out"],
                kernel_shape=[P, P],
                strides=[P, P],
                pads=[0, 0, 0, 0],
            )

            # [B, H, h, w] → [B, h, w, H] → [B, N, H]
            transposed = self.make_node(
                "Transpose", [conv_out], ["patch_embed/transposed"], perm=[0, 2, 3, 1]
            )
            self.add_initializer("patch_embed/reshape_3d", np.array([0, -1, H], dtype=np.int64))
            patch_embeds = self.make_node(
                "Reshape", [transposed, "patch_embed/reshape_3d"], ["patch_embed/out"]
            )
        else:
            # === Tiled mode ===
            # Linear projection (original)
            self.add_initializer(f"{prefix}.weight", linear_weight.T)  # Transpose for MatMul
            self.add_initializer(f"{prefix}.bias", linear_bias)

            # MatMul: [B, N, patch_dim] x [patch_dim, H] -> [B, N, H]
            matmul_out = self.make_node(
                "MatMul", ["pixel_values", f"{prefix}.weight"], ["patch_embed/matmul"]
            )
            patch_embeds = self.make_node(
                "Add", [matmul_out, f"{prefix}.bias"], ["patch_embed/out"]
            )

        # === Position embeddings with bilinear interpolation ===
        pos_emb_prefix = "vision_model.embeddings.position_embedding"
        pos_emb_weight = self.weights[f"{pos_emb_prefix}.weight"]

        # Reshape to (16, 16, 768) then permute to (1, 768, 16, 16) for Resize
        pos_emb_4d = pos_emb_weight.reshape(16, 16, H).transpose(2, 0, 1)  # (768, 16, 16)
        pos_emb_4d = pos_emb_4d[np.newaxis, ...]  # (1, 768, 16, 16)
        self.add_initializer("pos_emb/4d", pos_emb_4d)

        input_shape = self.make_node("Shape", ["pixel_values"], ["pos_emb/input_shape"])
        self.add_initializer("pos_emb/axes_0", np.array([0], dtype=np.int64))

        if self.vision_input_format == VISION_MODE_CONV2D:
            # Conv2d mode: use passed spatial dimensions
            # spatial_h, spatial_w are AFTER n_merge (final projector output size)
            # For position embeddings, we need BEFORE n_merge: spatial * n_merge
            n_merge = self.downsample  # downsample_factor is the n_merge value
            self.add_initializer("pos_emb/n_merge", np.array(n_merge, dtype=np.int64))

            pre_merge_h = self.make_node(
                "Mul", ["spatial_h", "pos_emb/n_merge"], ["pos_emb/pre_merge_h"]
            )
            pre_merge_w = self.make_node(
                "Mul", ["spatial_w", "pos_emb/n_merge"], ["pos_emb/pre_merge_w"]
            )
        else:
            # Tiled mode: extract spatial dimensions from spatial_shapes input
            # spatial_shapes: [batch, 2] with (height, width) in patch units
            # Use first batch item (all have same spatial dims in a batch)
            self.add_initializer("pos_emb/idx_0_batch", np.array(0, dtype=np.int64))
            self.add_initializer("pos_emb/idx_0", np.array(0, dtype=np.int64))
            self.add_initializer("pos_emb/idx_1", np.array(1, dtype=np.int64))

            first_spatial = self.make_node(
                "Gather",
                ["spatial_shapes", "pos_emb/idx_0_batch"],
                ["pos_emb/first_spatial"],
                axis=0,
            )

            spatial_h = self.make_node(
                "Gather", [first_spatial, "pos_emb/idx_0"], ["pos_emb/spatial_h"], axis=0
            )
            spatial_w = self.make_node(
                "Gather", [first_spatial, "pos_emb/idx_1"], ["pos_emb/spatial_w"], axis=0
            )

        # === Bilinear interpolation using ONNX Resize ===
        # For upsampling, PyTorch antialias and regular bilinear are identical
        pos_emb_final = self._build_pos_embed_resize(
            pos_emb_input="pos_emb/4d",
            spatial_h=spatial_h if self.vision_input_format == VISION_MODE_TILED else pre_merge_h,
            spatial_w=spatial_w if self.vision_input_format == VISION_MODE_TILED else pre_merge_w,
            src_h=16,  # Original position embedding grid size
            src_w=16,
            hidden_size=H,
        )

        # Get batch size and num_patches from input shape
        self.add_initializer("pos_emb/idx_0_scalar", np.array(0, dtype=np.int64))
        self.add_initializer("pos_emb/idx_1_scalar", np.array(1, dtype=np.int64))
        batch_size = self.make_node(
            "Gather", [input_shape, "pos_emb/idx_0_scalar"], ["pos_emb/batch_size"], axis=0
        )
        num_patches = self.make_node(
            "Gather", [input_shape, "pos_emb/idx_1_scalar"], ["pos_emb/num_patches"], axis=0
        )

        if self.vision_input_format == VISION_MODE_TILED:
            # Tiled mode: Handle padding (input may have more patches than H*W)
            # Fill padded positions with first token's position embedding
            self.add_initializer("pos_emb/slice_start", np.array([0], dtype=np.int64))
            self.add_initializer("pos_emb/slice_end", np.array([1], dtype=np.int64))
            self.add_initializer("pos_emb/slice_axis", np.array([1], dtype=np.int64))
            first_token = self.make_node(
                "Slice",
                [pos_emb_final, "pos_emb/slice_start", "pos_emb/slice_end", "pos_emb/slice_axis"],
                ["pos_emb/first_token"],
            )

            actual_num_patches = self.make_node(
                "Mul", [spatial_h, spatial_w], ["pos_emb/actual_num_patches"]
            )

            self.add_initializer("pos_emb/zero", np.array(0, dtype=np.int64))
            self.add_initializer("pos_emb/one_step", np.array(1, dtype=np.int64))
            indices = self.make_node(
                "Range", ["pos_emb/zero", num_patches, "pos_emb/one_step"], ["pos_emb/indices"]
            )

            # Valid mask: indices < actual_num_patches
            is_valid = self.make_node("Less", [indices, actual_num_patches], ["pos_emb/is_valid"])
            # Unsqueeze for broadcasting: (num_patches,) -> (1, num_patches, 1)
            self.add_initializer("pos_emb/unsq_axes", np.array([0, 2], dtype=np.int64))
            is_valid_3d = self.make_node(
                "Unsqueeze", [is_valid, "pos_emb/unsq_axes"], ["pos_emb/is_valid_3d"]
            )

            num_patches_unsq = self.make_node(
                "Unsqueeze", [num_patches, "pos_emb/axes_0"], ["pos_emb/num_patches_unsq"]
            )
            self.add_initializer("pos_emb/one_arr", np.array([1], dtype=np.int64))
            self.add_initializer("pos_emb/hidden_arr", np.array([H], dtype=np.int64))
            expand_shape = self.make_node(
                "Concat",
                ["pos_emb/one_arr", num_patches_unsq, "pos_emb/hidden_arr"],
                ["pos_emb/expand_shape"],
                axis=0,
            )
            first_token_expanded = self.make_node(
                "Expand", [first_token, expand_shape], ["pos_emb/first_token_expanded"]
            )

            padding_size = self.make_node(
                "Sub", [num_patches, actual_num_patches], ["pos_emb/padding_size"]
            )
            self.add_initializer("pos_emb/zeros_4", np.array([0, 0, 0, 0], dtype=np.int64))
            self.add_initializer("pos_emb/zero_arr", np.array([0], dtype=np.int64))
            padding_size_unsq = self.make_node(
                "Unsqueeze", [padding_size, "pos_emb/axes_0"], ["pos_emb/padding_size_unsq"]
            )
            pads = self.make_node(
                "Concat",
                ["pos_emb/zeros_4", padding_size_unsq, "pos_emb/zero_arr"],
                ["pos_emb/pads"],
                axis=0,
            )
            pos_emb_padded = self.make_node(
                "Pad", [pos_emb_final, pads], ["pos_emb/padded"], mode="constant"
            )

            pos_emb_with_padding = self.make_node(
                "Where",
                [is_valid_3d, pos_emb_padded, first_token_expanded],
                ["pos_emb/with_padding"],
            )

            batch_unsq = self.make_node(
                "Unsqueeze", [batch_size, "pos_emb/axes_0"], ["pos_emb/batch_unsq"]
            )
            self.add_initializer("pos_emb/ones_2d", np.array([1, 1], dtype=np.int64))
            tile_repeats = self.make_node(
                "Concat", [batch_unsq, "pos_emb/ones_2d"], ["pos_emb/tile_repeats"], axis=0
            )
            pos_emb_tiled = self.make_node(
                "Tile", [pos_emb_with_padding, tile_repeats], ["pos_emb/tiled"]
            )
        else:
            # Conv2d mode: no padding, just tile across batch
            batch_unsq = self.make_node(
                "Unsqueeze", [batch_size, "pos_emb/axes_0"], ["pos_emb/batch_unsq"]
            )
            self.add_initializer("pos_emb/ones_2d", np.array([1, 1], dtype=np.int64))
            tile_repeats = self.make_node(
                "Concat", [batch_unsq, "pos_emb/ones_2d"], ["pos_emb/tile_repeats"], axis=0
            )
            pos_emb_tiled = self.make_node("Tile", [pos_emb_final, tile_repeats], ["pos_emb/tiled"])

        return self.make_node("Add", [patch_embeds, pos_emb_tiled], ["patch_embeddings"])

    def build_encoder_layer(self, layer_idx: int, hidden_state: str) -> str:
        """Build a single SigLIP2 transformer encoder layer.

        Graph structure:
            hidden_state
                ↓
            LayerNorm1
                ↓
            ┌─────────────────────────────────────┐
            │  Self-Attention                     │
            │  Q, K, V projections                │
            │  Reshape [B, N, nh, hd]             │
            │  Transpose [B, nh, N, hd]           │
            │  Scaled Dot-Product Attention       │
            │  Output projection                  │
            └─────────────────────────────────────┘
                ↓
            Add (residual)
                ↓
            LayerNorm2
                ↓
            ┌─────────────────────────────────────┐
            │  MLP (GELU activation)              │
            │  fc1: hidden → intermediate         │
            │  GELU                               │
            │  fc2: intermediate → hidden         │
            └─────────────────────────────────────┘
                ↓
            Add (residual)
                ↓
            output
        """
        prefix = f"vision_model.encoder.layers.{layer_idx}"
        nh = self.vision_config.num_attention_heads
        hd = self.head_dim

        self.add_initializer(
            f"{prefix}.layer_norm1.weight", self.weights[f"{prefix}.layer_norm1.weight"]
        )
        self.add_initializer(
            f"{prefix}.layer_norm1.bias", self.weights[f"{prefix}.layer_norm1.bias"]
        )

        self.add_initializer(
            f"{prefix}.self_attn.q_proj.weight", self.weights[f"{prefix}.self_attn.q_proj.weight"].T
        )
        self.add_initializer(
            f"{prefix}.self_attn.q_proj.bias", self.weights[f"{prefix}.self_attn.q_proj.bias"]
        )
        self.add_initializer(
            f"{prefix}.self_attn.k_proj.weight", self.weights[f"{prefix}.self_attn.k_proj.weight"].T
        )
        self.add_initializer(
            f"{prefix}.self_attn.k_proj.bias", self.weights[f"{prefix}.self_attn.k_proj.bias"]
        )
        self.add_initializer(
            f"{prefix}.self_attn.v_proj.weight", self.weights[f"{prefix}.self_attn.v_proj.weight"].T
        )
        self.add_initializer(
            f"{prefix}.self_attn.v_proj.bias", self.weights[f"{prefix}.self_attn.v_proj.bias"]
        )
        self.add_initializer(
            f"{prefix}.self_attn.out_proj.weight",
            self.weights[f"{prefix}.self_attn.out_proj.weight"].T,
        )
        self.add_initializer(
            f"{prefix}.self_attn.out_proj.bias", self.weights[f"{prefix}.self_attn.out_proj.bias"]
        )

        self.add_initializer(
            f"{prefix}.layer_norm2.weight", self.weights[f"{prefix}.layer_norm2.weight"]
        )
        self.add_initializer(
            f"{prefix}.layer_norm2.bias", self.weights[f"{prefix}.layer_norm2.bias"]
        )

        self.add_initializer(f"{prefix}.mlp.fc1.weight", self.weights[f"{prefix}.mlp.fc1.weight"].T)
        self.add_initializer(f"{prefix}.mlp.fc1.bias", self.weights[f"{prefix}.mlp.fc1.bias"])
        self.add_initializer(f"{prefix}.mlp.fc2.weight", self.weights[f"{prefix}.mlp.fc2.weight"].T)
        self.add_initializer(f"{prefix}.mlp.fc2.bias", self.weights[f"{prefix}.mlp.fc2.bias"])

        residual = hidden_state

        normed = self.make_vision_layernorm(
            hidden_state,
            f"{prefix}.layer_norm1.weight",
            f"{prefix}.layer_norm1.bias",
            f"{prefix}/ln1",
        )

        q = self.make_node(
            "MatMul", [normed, f"{prefix}.self_attn.q_proj.weight"], [f"{prefix}/q_matmul"]
        )
        q = self.make_node("Add", [q, f"{prefix}.self_attn.q_proj.bias"], [f"{prefix}/q"])

        k = self.make_node(
            "MatMul", [normed, f"{prefix}.self_attn.k_proj.weight"], [f"{prefix}/k_matmul"]
        )
        k = self.make_node("Add", [k, f"{prefix}.self_attn.k_proj.bias"], [f"{prefix}/k"])

        v = self.make_node(
            "MatMul", [normed, f"{prefix}.self_attn.v_proj.weight"], [f"{prefix}/v_matmul"]
        )
        v = self.make_node("Add", [v, f"{prefix}.self_attn.v_proj.bias"], [f"{prefix}/v"])

        scale = 1.0 / (hd**0.5)

        # Fused MultiHeadAttention (com.microsoft)
        # Inputs: query, key, value, bias, key_padding_mask, attention_bias, past_key, past_value
        # attention_bias: 4D additive mask [B, num_heads, N, N] with 0=valid, -inf=masked
        attn_bias = "attention_bias" if self.vision_input_format == VISION_MODE_TILED else ""
        attn_out_reshaped = self.make_node(
            "MultiHeadAttention",
            [q, k, v, "", "", attn_bias, "", ""],
            [f"{prefix}/attn_out"],
            domain="com.microsoft",
            num_heads=nh,
            scale=scale,
        )

        out_proj = self.make_node(
            "MatMul",
            [attn_out_reshaped, f"{prefix}.self_attn.out_proj.weight"],
            [f"{prefix}/out_proj_matmul"],
        )
        out_proj = self.make_node(
            "Add", [out_proj, f"{prefix}.self_attn.out_proj.bias"], [f"{prefix}/out_proj"]
        )

        hidden_state = self.make_node("Add", [residual, out_proj], [f"{prefix}/residual1"])

        residual2 = hidden_state
        normed2 = self.make_vision_layernorm(
            hidden_state,
            f"{prefix}.layer_norm2.weight",
            f"{prefix}.layer_norm2.bias",
            f"{prefix}/ln2",
        )

        fc1 = self.make_node(
            "MatMul", [normed2, f"{prefix}.mlp.fc1.weight"], [f"{prefix}/fc1_matmul"]
        )
        fc1 = self.make_node("Add", [fc1, f"{prefix}.mlp.fc1.bias"], [f"{prefix}/fc1"])
        fc1_act = self.make_gelu(fc1, f"{prefix}/fc1_act")

        fc2 = self.make_node(
            "MatMul", [fc1_act, f"{prefix}.mlp.fc2.weight"], [f"{prefix}/fc2_matmul"]
        )
        fc2 = self.make_node("Add", [fc2, f"{prefix}.mlp.fc2.bias"], [f"{prefix}/fc2"])

        return self.make_node("Add", [residual2, fc2], [f"{prefix}/residual2"])

    def build_post_layernorm(self, hidden_state: str) -> str:
        """Build post layer norm."""
        self.add_initializer(
            "vision_model.post_layernorm.weight", self.weights["vision_model.post_layernorm.weight"]
        )
        self.add_initializer(
            "vision_model.post_layernorm.bias", self.weights["vision_model.post_layernorm.bias"]
        )
        return self.make_vision_layernorm(
            hidden_state,
            "vision_model.post_layernorm.weight",
            "vision_model.post_layernorm.bias",
            "vision_embeddings",
        )

    def build_projector(self, vision_embeddings: str) -> str:
        """Build the MLP projector with pixel unshuffle.

        Graph structure:
            vision_embeddings [B, N, C]
                ↓
            ┌─────────────────────────────────────┐
            │  Reshape to 4D                      │
            │  [B, N, C] → [B, H, W, C]           │
            │  (H = W = sqrt(N) for square)       │
            └─────────────────────────────────────┘
                ↓
            ┌─────────────────────────────────────┐
            │  Pixel Unshuffle (2x2 → 4x channel) │
            │  [B, H, W, C] → [B, H/2, W/2, C*4]  │
            │                                     │
            │  Steps:                             │
            │  1. reshape [B,H,W/2,C*2]           │
            │  2. transpose [B,W/2,H,C*2]         │
            │  3. reshape [B,W/2,H/2,C*4]         │
            │  4. transpose [B,H/2,W/2,C*4]       │
            └─────────────────────────────────────┘
                ↓
            Flatten [B, N/4, C*4]
                ↓
            LayerNorm
                ↓
            Linear (C*4 → proj_hidden) + GELU
                ↓
            Linear (proj_hidden → text_hidden)
                ↓
            image_embeddings [B, N/4, text_hidden]

        The pixel unshuffle operation reduces spatial resolution by 2x while
        increasing channel dimension by 4x, matching the PyTorch implementation
        in Lfm2VlMultiModalProjector.pixel_unshuffle().
        """
        ds = self.downsample
        C = self.vision_hidden
        input_dim = C * ds * ds

        use_layernorm = getattr(self.config, "projector_use_layernorm", True)
        if use_layernorm:
            self.add_initializer(
                "multi_modal_projector.layer_norm.weight",
                self.weights["multi_modal_projector.layer_norm.weight"],
            )
            self.add_initializer(
                "multi_modal_projector.layer_norm.bias",
                self.weights["multi_modal_projector.layer_norm.bias"],
            )

        self.add_initializer(
            "multi_modal_projector.linear_1.weight",
            self.weights["multi_modal_projector.linear_1.weight"].T,
        )
        if self.config.projector_bias:
            self.add_initializer(
                "multi_modal_projector.linear_1.bias",
                self.weights["multi_modal_projector.linear_1.bias"],
            )

        self.add_initializer(
            "multi_modal_projector.linear_2.weight",
            self.weights["multi_modal_projector.linear_2.weight"].T,
        )
        if self.config.projector_bias:
            self.add_initializer(
                "multi_modal_projector.linear_2.bias",
                self.weights["multi_modal_projector.linear_2.bias"],
            )

        self.add_initializer("proj/shape_indices_batch", np.array(0, dtype=np.int64))
        batch_size = self.make_node(
            "Gather",
            [
                self.make_node("Shape", [vision_embeddings], ["proj/input_shape"]),
                "proj/shape_indices_batch",
            ],
            ["proj/batch_size"],
            axis=0,
        )

        self.add_initializer("proj/hidden_size", np.array([C], dtype=np.int64))
        self.add_initializer("proj/axes_0", np.array([0], dtype=np.int64))

        if self.vision_input_format == VISION_MODE_CONV2D:
            # Conv2d mode: use passed spatial dimensions
            # spatial_h, spatial_w are AFTER n_merge, so we need to multiply by n_merge
            # to get the pre-merge spatial dimensions for the first reshape
            n_merge = self.downsample
            self.add_initializer("proj/n_merge", np.array(n_merge, dtype=np.int64))

            pre_merge_h = self.make_node("Mul", ["spatial_h", "proj/n_merge"], ["proj/pre_merge_h"])
            pre_merge_w = self.make_node("Mul", ["spatial_w", "proj/n_merge"], ["proj/pre_merge_w"])

            reshape_4d_shape = self.make_node(
                "Concat",
                [
                    self.make_node("Unsqueeze", [batch_size, "proj/axes_0"], ["proj/batch_unsq"]),
                    self.make_node(
                        "Unsqueeze", [pre_merge_h, "proj/axes_0"], ["proj/spatial_h_unsq"]
                    ),
                    self.make_node(
                        "Unsqueeze", [pre_merge_w, "proj/axes_0"], ["proj/spatial_w_unsq"]
                    ),
                    "proj/hidden_size",
                ],
                ["proj/reshape_4d_shape"],
                axis=0,
            )

            spatial_h_name = pre_merge_h
            half_spatial_h_name = "spatial_h"
            half_spatial_w_name = "spatial_w"
        else:
            self.add_initializer("proj/idx_0_batch", np.array(0, dtype=np.int64))
            self.add_initializer("proj/idx_0", np.array(0, dtype=np.int64))
            self.add_initializer("proj/idx_1", np.array(1, dtype=np.int64))

            first_spatial = self.make_node(
                "Gather", ["spatial_shapes", "proj/idx_0_batch"], ["proj/first_spatial"], axis=0
            )

            spatial_h = self.make_node(
                "Gather", [first_spatial, "proj/idx_0"], ["proj/spatial_h"], axis=0
            )
            spatial_w = self.make_node(
                "Gather", [first_spatial, "proj/idx_1"], ["proj/spatial_w"], axis=0
            )

            reshape_4d_shape = self.make_node(
                "Concat",
                [
                    self.make_node("Unsqueeze", [batch_size, "proj/axes_0"], ["proj/batch_unsq"]),
                    self.make_node(
                        "Unsqueeze", [spatial_h, "proj/axes_0"], ["proj/spatial_h_unsq"]
                    ),
                    self.make_node(
                        "Unsqueeze", [spatial_w, "proj/axes_0"], ["proj/spatial_w_unsq"]
                    ),
                    "proj/hidden_size",
                ],
                ["proj/reshape_4d_shape"],
                axis=0,
            )

            self.add_initializer("proj/two_tiled", np.array(2, dtype=np.int64))
            half_spatial_h = self.make_node(
                "Div", [spatial_h, "proj/two_tiled"], ["proj/half_spatial_h_tiled"]
            )
            half_spatial_w = self.make_node(
                "Div", [spatial_w, "proj/two_tiled"], ["proj/half_spatial_w_tiled"]
            )
            spatial_h_name = spatial_h
            half_spatial_h_name = half_spatial_h
            half_spatial_w_name = half_spatial_w

            # Slice out only valid patches before reshaping (input may be padded)
            actual_num_patches = self.make_node(
                "Mul", [spatial_h, spatial_w], ["proj/actual_num_patches"]
            )
            self.add_initializer("proj/slice_start", np.array([0], dtype=np.int64))
            self.add_initializer("proj/slice_axes", np.array([1], dtype=np.int64))
            actual_unsq = self.make_node(
                "Unsqueeze", [actual_num_patches, "proj/axes_0"], ["proj/actual_unsq"]
            )
            vision_embeddings = self.make_node(
                "Slice",
                [vision_embeddings, "proj/slice_start", actual_unsq, "proj/slice_axes"],
                ["proj/valid_embeddings"],
            )

        hidden_4d = self.make_node(
            "Reshape", [vision_embeddings, reshape_4d_shape], ["proj/hidden_4d"]
        )

        # Pixel unshuffle: (B, H, W, C) -> (B, H/2, W/2, C*4)
        self.add_initializer("proj/c_times_2", np.array([C * ds], dtype=np.int64))
        reshape1_shape = self.make_node(
            "Concat",
            [
                self.make_node("Unsqueeze", [batch_size, "proj/axes_0"], ["proj/b1"]),
                self.make_node("Unsqueeze", [spatial_h_name, "proj/axes_0"], ["proj/h1"]),
                self.make_node("Unsqueeze", [half_spatial_w_name, "proj/axes_0"], ["proj/w_half1"]),
                "proj/c_times_2",
            ],
            ["proj/reshape1_shape"],
            axis=0,
        )
        step1 = self.make_node("Reshape", [hidden_4d, reshape1_shape], ["proj/step1"])
        step2 = self.make_node("Transpose", [step1], ["proj/step2"], perm=[0, 2, 1, 3])

        self.add_initializer("proj/c_times_4", np.array([input_dim], dtype=np.int64))
        reshape2_shape = self.make_node(
            "Concat",
            [
                self.make_node("Unsqueeze", [batch_size, "proj/axes_0"], ["proj/b2"]),
                self.make_node("Unsqueeze", [half_spatial_w_name, "proj/axes_0"], ["proj/w_half2"]),
                self.make_node("Unsqueeze", [half_spatial_h_name, "proj/axes_0"], ["proj/h_half2"]),
                "proj/c_times_4",
            ],
            ["proj/reshape2_shape"],
            axis=0,
        )
        step3 = self.make_node("Reshape", [step2, reshape2_shape], ["proj/step3"])
        step4 = self.make_node("Transpose", [step3], ["proj/step4"], perm=[0, 2, 1, 3])

        self.add_initializer("proj/reshape_3d", np.array([0, -1, input_dim], dtype=np.int64))
        unshuffled = self.make_node("Reshape", [step4, "proj/reshape_3d"], ["proj/unshuffled"])

        if use_layernorm:
            normed = self.make_node(
                "LayerNormalization",
                [
                    unshuffled,
                    "multi_modal_projector.layer_norm.weight",
                    "multi_modal_projector.layer_norm.bias",
                ],
                ["proj/normed"],
                epsilon=1e-5,
            )
        else:
            normed = unshuffled

        fc1 = self.make_node(
            "MatMul", [normed, "multi_modal_projector.linear_1.weight"], ["proj/fc1_matmul"]
        )
        if self.config.projector_bias:
            fc1 = self.make_node("Add", [fc1, "multi_modal_projector.linear_1.bias"], ["proj/fc1"])

        fc1_act = self.make_gelu(fc1, "proj/fc1_act", approximate="none")

        fc2 = self.make_node(
            "MatMul", [fc1_act, "multi_modal_projector.linear_2.weight"], ["proj/fc2_matmul"]
        )
        if self.config.projector_bias:
            fc2 = self.make_node(
                "Add", [fc2, "multi_modal_projector.linear_2.bias"], ["image_embeddings"]
            )
        else:
            self.make_node("Identity", [fc2], ["image_embeddings"])

        return "image_embeddings"

    def load_weights(self, weights: dict[str, np.ndarray]):
        for name, weight in weights.items():
            if name.startswith("model.vision_tower.vision_model."):
                new_name = name.replace("model.vision_tower.", "")
                self.weights[new_name] = weight
            elif name.startswith("vision_model."):
                self.weights[name] = weight
            elif name.startswith("model.multi_modal_projector."):
                new_name = name.replace("model.", "")
                self.weights[new_name] = weight
            elif name.startswith("multi_modal_projector."):
                self.weights[name] = weight

        logger.info(f"Loaded {len(self.weights)} vision + projector weights")

    def build(self) -> onnx.ModelProto:
        logger.info("Building fused vision encoder + projector...")

        self.build_inputs()
        self.build_outputs()
        self.build_attention_mask()

        hidden_state = self.build_patch_embedding()

        for layer_idx in range(self.vision_config.num_hidden_layers):
            logger.info(f"Building vision layer {layer_idx}...")
            hidden_state = self.build_encoder_layer(layer_idx, hidden_state)

        vision_embeddings = self.build_post_layernorm(hidden_state)

        logger.info("Building projector...")
        self.build_projector(vision_embeddings)

        model = self.build_graph("embed_images", producer_name="lfm2-vl-builder")
        logger.info(f"Vision + projector model built: {len(self.nodes)} nodes")
        return model
