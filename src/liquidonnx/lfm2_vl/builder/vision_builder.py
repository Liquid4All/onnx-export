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
        self, input_name: str, weight_name: str, bias_name: str, path: str
    ) -> str:
        """Create LayerNormalization node with vision encoder epsilon.

        Args:
            input_name: Input tensor
            weight_name: Scale weight
            bias_name: Bias
            path: Logical path (e.g., "/vision_encoder/layers.0/ln_1")
        """
        # Use "LayerNorm" suffix to match community naming convention
        return self.make_layernorm(
            input_name,
            weight_name,
            bias_name,
            path,
            epsilon=self.vision_config.layer_norm_eps,
            name="LayerNorm",
        )

    def _split_spatial_shapes(self, output_prefix: str) -> tuple[str, str]:
        """Split spatial_shapes into per-batch h and w tensors.

        Args:
            output_prefix: Output node prefix

        Returns:
            Tuple of (h_per_batch, w_per_batch) each of shape [B, 1]
        """
        p = output_prefix
        # Split spatial_shapes [B, 2] into h [B, 1] and w [B, 1]
        self.make_node(
            "Split",
            ["spatial_shapes"],
            [f"{p}/split_shapes/h", f"{p}/split_shapes/w"],
            axis=1,
            num_outputs=2,
        )
        return f"{p}/split_shapes/h", f"{p}/split_shapes/w"

    def _extract_max_spatial_dims(self, output_prefix: str) -> tuple[str, str]:
        """Extract max spatial dimensions from spatial_shapes input.

        Returns the maximum H and W across all images in the batch.
        Used by position embedding which needs max dimensions for Resize.

        Args:
            output_prefix: Output node prefix

        Returns:
            Tuple of (max_h, max_w) as scalar int64 tensors
        """
        p = output_prefix
        # ReduceMax across batch: [B, 2] → [2] to get max(H), max(W)
        max_spatial = self.make_node(
            "ReduceMax",
            ["spatial_shapes", self.get_constant("INT64", [0])],
            [f"{p}/max_spatial/output_0"],
            keepdims=0,
        )

        spatial_h = self.make_node(
            "Gather",
            [max_spatial, self.get_constant("INT64", 0)],
            [f"{p}/spatial_h/output_0"],
            axis=0,
        )
        spatial_w = self.make_node(
            "Gather",
            [max_spatial, self.get_constant("INT64", 1)],
            [f"{p}/spatial_w/output_0"],
            axis=0,
        )

        return spatial_h, spatial_w

    def _compute_aligned_max_dims(
        self, h_per_batch: str, w_per_batch: str, prefix: str
    ) -> tuple[str, str]:
        """Compute aligned max dimensions (rounded up to even for pixel unshuffle).

        Args:
            h_per_batch: Per-batch heights [B, 1]
            w_per_batch: Per-batch widths [B, 1]
            prefix: Output node prefix

        Returns:
            Tuple of (aligned_max_h, aligned_max_w) as scalar int64
        """
        p = prefix

        # Get max across batch: [B, 1] → scalar (ReduceMax over all axes with keepdims=0)
        max_h = self.make_node("ReduceMax", [h_per_batch], [f"{p}/max_h/output_0"], keepdims=0)
        max_w = self.make_node("ReduceMax", [w_per_batch], [f"{p}/max_w/output_0"], keepdims=0)

        # Align to even: ((x + 1) // 2) * 2
        two = self.get_constant("INT64", 2)
        one = self.get_constant("INT64", 1)

        aligned_h = self.make_node(
            "Mul",
            [
                self.make_node(
                    "Div",
                    [self.make_node("Add", [max_h, one], [f"{p}/align_max_h_add/output_0"]), two],
                    [f"{p}/align_max_h_div/output_0"],
                ),
                two,
            ],
            [f"{p}/align_max_h/output_0"],
        )
        aligned_w = self.make_node(
            "Mul",
            [
                self.make_node(
                    "Div",
                    [self.make_node("Add", [max_w, one], [f"{p}/align_max_w_add/output_0"]), two],
                    [f"{p}/align_max_w_div/output_0"],
                ),
                two,
            ],
            [f"{p}/align_max_w/output_0"],
        )

        return aligned_h, aligned_w

    def _build_grid_indices(
        self,
        batch_size: str,
        num_patches: str,
        h_per_batch: str,
        w_per_batch: str,
        prefix: str,
    ) -> str:
        """Build grid indices for ScatterND: (batch_idx, y, x) for each patch.

        For each patch at position [b, i], compute:
            y = i // w[b]
            x = i % w[b]
            indices[b, i] = [b, y, x]

        Args:
            batch_size: Scalar batch size
            num_patches: Scalar number of patches (max across batch)
            h_per_batch: Heights per batch [B, 1]
            w_per_batch: Widths per batch [B, 1]
            prefix: Output node prefix

        Returns:
            Grid indices tensor of shape [B, num_patches, 3]
        """
        p = prefix
        axes_0 = self.get_constant("INT64", [0])
        axes_1 = self.get_constant("INT64", [1])

        # Create patch indices [0, 1, 2, ..., num_patches-1]
        patch_range = self.make_node(
            "Range",
            [self.get_constant("INT64", 0), num_patches, self.get_constant("INT64", 1)],
            [f"{p}/patch_range/output_0"],
        )
        # Unsqueeze to [1, num_patches] for broadcasting
        patch_range_2d = self.make_node(
            "Unsqueeze", [patch_range, axes_0], [f"{p}/patch_range_2d/output_0"]
        )

        # w_per_batch is [B, 1], broadcast to [B, num_patches]
        # y = patch_idx // w
        grid_y = self.make_node("Div", [patch_range_2d, w_per_batch], [f"{p}/grid_y/output_0"])
        # x = patch_idx % w
        grid_x = self.make_node("Mod", [patch_range_2d, w_per_batch], [f"{p}/grid_x/output_0"])

        # Create batch indices [0, 1, ..., B-1] expanded to [B, num_patches]
        batch_range = self.make_node(
            "Range",
            [self.get_constant("INT64", 0), batch_size, self.get_constant("INT64", 1)],
            [f"{p}/batch_range/output_0"],
        )
        batch_range_2d = self.make_node(
            "Unsqueeze", [batch_range, axes_1], [f"{p}/batch_range_2d/output_0"]
        )
        # Broadcast to [B, num_patches] using zeros from patch_range
        zeros = self.make_node(
            "Mul", [patch_range_2d, self.get_constant("INT64", 0)], [f"{p}/zeros/output_0"]
        )
        grid_b = self.make_node("Add", [batch_range_2d, zeros], [f"{p}/grid_b/output_0"])

        # Stack [b, y, x] -> [B, num_patches, 3]
        grid_b_3d = self.make_node(
            "Unsqueeze", [grid_b, self.get_constant("INT64", [-1])], [f"{p}/grid_b_3d/output_0"]
        )
        grid_y_3d = self.make_node(
            "Unsqueeze", [grid_y, self.get_constant("INT64", [-1])], [f"{p}/grid_y_3d/output_0"]
        )
        grid_x_3d = self.make_node(
            "Unsqueeze", [grid_x, self.get_constant("INT64", [-1])], [f"{p}/grid_x_3d/output_0"]
        )

        return self.make_node(
            "Concat",
            [grid_b_3d, grid_y_3d, grid_x_3d],
            [f"{p}/grid_indices/output_0"],
            axis=-1,
        )

    def _build_scatter_canvas(
        self,
        features: str,
        grid_indices: str,
        mask: str,
        batch_size: str,
        aligned_h: str,
        aligned_w: str,
        hidden_dim: int,
        prefix: str,
    ) -> str:
        """Build ScatterND canvas to place features at grid positions.

        Creates a [B, aligned_h, aligned_w, hidden] canvas filled with zeros,
        then uses ScatterND to place valid features at their (b, y, x) positions.

        INVARIANT: The mask (pixel_attention_mask) must be False for all patches
        beyond the valid region (patch_idx >= h[b] * w[b]) to prevent ScatterND
        index collisions. This is guaranteed by the HuggingFace image processor.

        Args:
            features: Input features [B, num_patches, hidden]
            grid_indices: Grid indices [B, num_patches, 3]
            mask: Validity mask [B, num_patches] (bool)
            batch_size: Scalar batch size
            aligned_h: Aligned max height (scalar)
            aligned_w: Aligned max width (scalar)
            hidden_dim: Hidden dimension size
            prefix: Output node prefix

        Returns:
            Filled canvas [B, aligned_h, aligned_w, hidden]
        """
        p = prefix
        axes_0 = self.get_constant("INT64", [0])

        # Flatten features and indices for Compress
        flat_features = self.make_node(
            "Reshape",
            [features, self.get_constant("INT64", [-1, hidden_dim])],
            [f"{p}/flat_features/output_0"],
        )
        flat_indices = self.make_node(
            "Reshape",
            [grid_indices, self.get_constant("INT64", [-1, 3])],
            [f"{p}/flat_indices/output_0"],
        )
        flat_mask = self.make_node(
            "Reshape", [mask, self.get_constant("INT64", [-1])], [f"{p}/flat_mask/output_0"]
        )

        # Compress to get only valid features and indices
        valid_features = self.make_node(
            "Compress", [flat_features, flat_mask], [f"{p}/valid_features/output_0"], axis=0
        )
        valid_indices = self.make_node(
            "Compress", [flat_indices, flat_mask], [f"{p}/valid_indices/output_0"], axis=0
        )

        # Create canvas shape [B, aligned_h, aligned_w, hidden]
        canvas_shape = self.make_node(
            "Concat",
            [
                self.make_node("Unsqueeze", [batch_size, axes_0], [f"{p}/canv_b/output_0"]),
                self.make_node("Unsqueeze", [aligned_h, axes_0], [f"{p}/canv_h/output_0"]),
                self.make_node("Unsqueeze", [aligned_w, axes_0], [f"{p}/canv_w/output_0"]),
                self.get_constant("INT64", [hidden_dim]),
            ],
            [f"{p}/canvas_shape/output_0"],
            axis=0,
        )

        # Create zero-filled canvas
        canvas = self.make_node(
            "ConstantOfShape",
            [canvas_shape],
            [f"{p}/canvas/output_0"],
            value=helper.make_tensor("value", TensorProto.FLOAT, [1], [0.0]),
        )

        # ScatterND: place valid features into canvas
        return self.make_node(
            "ScatterND",
            [canvas, valid_indices, valid_features],
            [f"{p}/filled_canvas/output_0"],
        )

    def _build_output_validity_mask(
        self,
        h_per_batch: str,
        w_per_batch: str,
        aligned_h: str,
        aligned_w: str,
        downsample: int,
        prefix: str,
    ) -> str:
        """Build validity mask for output tokens after pixel unshuffle.

        For each position (b, y, x) in the output grid, it's valid if:
            y < h[b] / downsample AND x < w[b] / downsample

        INVARIANT: Spatial shapes (h_per_batch, w_per_batch) are always even because
        preprocessing pads images to be divisible by patch_size * downsample_factor (32).
        This ensures integer division by downsample (2) produces exact results.

        Args:
            h_per_batch: Heights per batch [B, 1] - always even
            w_per_batch: Widths per batch [B, 1] - always even
            aligned_h: Aligned max height (scalar)
            aligned_w: Aligned max width (scalar)
            downsample: Downsample factor (typically 2)
            prefix: Output node prefix

        Returns:
            Flat validity mask [B * (aligned_h/ds) * (aligned_w/ds)] as bool
        """
        p = prefix
        ds = self.get_constant("INT64", downsample)

        # Output spatial dims after pixel unshuffle
        out_h = self.make_node("Div", [aligned_h, ds], [f"{p}/out_h/output_0"])
        out_w = self.make_node("Div", [aligned_w, ds], [f"{p}/out_w/output_0"])

        # Valid dims per batch (after downsampling)
        valid_h = self.make_node("Div", [h_per_batch, ds], [f"{p}/valid_h/output_0"])
        valid_w = self.make_node("Div", [w_per_batch, ds], [f"{p}/valid_w/output_0"])

        # Create iota ranges for h and w
        iota_h = self.make_node(
            "Range",
            [self.get_constant("INT64", 0), out_h, self.get_constant("INT64", 1)],
            [f"{p}/iota_h/output_0"],
        )
        iota_w = self.make_node(
            "Range",
            [self.get_constant("INT64", 0), out_w, self.get_constant("INT64", 1)],
            [f"{p}/iota_w/output_0"],
        )

        # Expand iota_h to [B, out_h, 1] and iota_w to [B, 1, out_w]
        iota_h_3d = self.make_node(
            "Unsqueeze",
            [iota_h, self.get_constant("INT64", [0, 2])],
            [f"{p}/iota_h_3d/output_0"],
        )
        iota_w_3d = self.make_node(
            "Unsqueeze",
            [iota_w, self.get_constant("INT64", [0, 1])],
            [f"{p}/iota_w_3d/output_0"],
        )

        # Expand valid_h to [B, 1, 1] and valid_w to [B, 1, 1]
        valid_h_3d = self.make_node(
            "Unsqueeze", [valid_h, self.get_constant("INT64", [-1])], [f"{p}/valid_h_3d/output_0"]
        )
        valid_w_3d = self.make_node(
            "Unsqueeze", [valid_w, self.get_constant("INT64", [-1])], [f"{p}/valid_w_3d/output_0"]
        )

        # h_mask: iota_h < valid_h -> [B, out_h, 1]
        h_mask = self.make_node("Less", [iota_h_3d, valid_h_3d], [f"{p}/h_mask/output_0"])
        # w_mask: iota_w < valid_w -> [B, 1, out_w]
        w_mask = self.make_node("Less", [iota_w_3d, valid_w_3d], [f"{p}/w_mask/output_0"])

        # final_mask: h_mask AND w_mask -> [B, out_h, out_w]
        final_mask = self.make_node("And", [h_mask, w_mask], [f"{p}/final_mask/output_0"])

        # Flatten to [B * out_h * out_w]
        return self.make_node(
            "Reshape",
            [final_mask, self.get_constant("INT64", [-1])],
            [f"{p}/flat_output_mask/output_0"],
        )

    def get_constant(self, dtype: str, value) -> str:
        """Get or create a constant with community-style naming via Constant node.

        Constants are named: /model/constants/TYPE/value
        Examples:
            /model/constants/INT64/0
            /model/constants/FLOAT/1.0
            /model/constants/INT64/[1, 2]

        Args:
            dtype: "INT64" or "FLOAT"
            value: The constant value (scalar, list, or numpy array)

        Returns:
            The constant output name
        """
        # Convert value to string representation for naming
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                val_str = str(value.item())
            else:
                val_str = str(value.tolist()).replace(" ", "")
        elif isinstance(value, (list, tuple)):
            val_str = str(list(value)).replace(" ", "")
        else:
            val_str = str(value)
        output_name = f"/model/constants/{dtype}/{val_str}"

        # Only add if not already present (check initializers)
        if not any(init.name == output_name for init in self.initializers):
            if dtype == "INT64":
                arr = np.array(value, dtype=np.int64)
            else:
                arr = np.array(value, dtype=np.float32)
            # Add as initializer (matches community convention)
            self.add_initializer(output_name, arr)

        return output_name

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
        if self.vision_input_format == VISION_MODE_TILED:
            # Tiled mode: 2D output after Compress [total_tokens, hidden]
            # Supports different-sized images in same batch (tokens concatenated)
            self.outputs.append(
                helper.make_tensor_value_info(
                    "image_features",
                    TensorProto.FLOAT,
                    ["num_image_tokens", self.text_hidden],
                )
            )
        else:
            # Conv2d mode: 3D output [batch, num_image_tokens, hidden]
            self.outputs.append(
                helper.make_tensor_value_info(
                    "image_features",
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

        p = "/model/attn_mask_reformat_full"
        num_heads = self.vision_config.num_attention_heads

        # Cast to float32
        mask_float = self.make_node(
            "Cast", ["pixel_attention_mask"], [f"{p}/Cast/output_0"], to=TensorProto.FLOAT
        )

        # Invert: 1.0 - mask (now 0=valid, 1=masked)
        inverted = self.make_node(
            "Sub", [self.get_constant("FLOAT", 1.0), mask_float], [f"{p}/Sub/output_0"]
        )

        # Multiply by -inf to create additive bias (0=valid, -inf=masked)
        bias_2d = self.make_node(
            "Mul",
            [inverted, self.get_constant("FLOAT", -3.4028234663852886e38)],
            [f"{p}/Mul/output_0"],
        )

        # Unsqueeze to [B, 1, 1, N] for broadcasting
        bias_4d = self.make_node(
            "Unsqueeze", [bias_2d, self.get_constant("INT64", [1, 2])], [f"{p}/Unsqueeze/output_0"]
        )

        shape = self.make_node(
            "Shape", ["pixel_attention_mask"], [f"{p}/Shape_for_expand/Shape/output_0"]
        )

        batch_size = self.make_node(
            "Gather",
            [shape, self.get_constant("INT64", 0)],
            [f"{p}/Shape_for_expand/Gather_0/output_0"],
            axis=0,
        )
        batch_unsq = self.make_node(
            "Unsqueeze",
            [batch_size, self.get_constant("INT64", [0])],
            [f"{p}/Unsqueeze_batch/output_0"],
        )

        seq_len = self.make_node(
            "Gather",
            [shape, self.get_constant("INT64", 1)],
            [f"{p}/Shape_for_expand/Gather_1/output_0"],
            axis=0,
        )
        seq_unsq = self.make_node(
            "Unsqueeze",
            [seq_len, self.get_constant("INT64", [0])],
            [f"{p}/Unsqueeze_seqlen/output_0"],
        )

        expand_shape = self.make_node(
            "Concat",
            [batch_unsq, self.get_constant("INT64", [num_heads]), seq_unsq, seq_unsq],
            [f"{p}/Concat_expand_shape/output_0"],
            axis=0,
        )

        self.make_node("Expand", [bias_4d, expand_shape], [f"{p}/Expand/output_0"])

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
        # Community naming: model.embeddings.patch_embedding
        pytorch_prefix = "vision_model.embeddings.patch_embedding"
        H = self.vision_config.hidden_size
        P = self.vision_config.patch_size
        C = self.vision_config.num_channels

        linear_weight = self.weights[f"{pytorch_prefix}.weight"]
        linear_bias = self.weights[f"{pytorch_prefix}.bias"]

        if self.vision_input_format == VISION_MODE_CONV2D:
            # === Conv2d mode ===
            # Reshape Linear weights to Conv2d format
            # Linear: [hidden_size, C*P*P] → Conv2d: [hidden_size, C, P, P]
            # Linear weight is [out_features, in_features] = [768, 768]
            # The original model flattens patches as HWC (P*P*C = 16*16*3 = 768)
            # So we first reshape to [H, P, P, C] then transpose to [H, C, P, P]
            # This matches the GGUF converter: view(H, 16, 16, 3).permute(0, 3, 1, 2)
            conv_weight = linear_weight.reshape(H, P, P, C).transpose(0, 3, 1, 2)  # [H, C, P, P]
            self.add_initializer("model.embeddings.patch_embedding.Conv.weight", conv_weight)
            self.add_initializer("model.embeddings.patch_embedding.Conv.bias", linear_bias)

            # Conv2d: [B, C, H, W] -> [B, hidden, H/P, W/P]
            conv_out = self.make_node(
                "Conv",
                [
                    "pixel_values",
                    "model.embeddings.patch_embedding.Conv.weight",
                    "model.embeddings.patch_embedding.Conv.bias",
                ],
                ["/model/embeddings/patch_embedding/Conv/output_0"],
                kernel_shape=[P, P],
                strides=[P, P],
                pads=[0, 0, 0, 0],
            )

            # [B, H, h, w] → [B, h, w, H] → [B, N, H]
            transposed = self.make_node(
                "Transpose",
                [conv_out],
                ["/model/embeddings/patch_embedding/Transpose/output_0"],
                perm=[0, 2, 3, 1],
            )
            patch_embeds = self.make_node(
                "Reshape",
                [transposed, self.get_constant("INT64", [0, -1, H])],
                ["/model/embeddings/patch_embedding/Reshape/output_0"],
            )
        else:
            # === Tiled mode ===
            # Linear projection (original)
            self.add_initializer(
                "model.embeddings.patch_embedding.MatMul.weight", linear_weight.T
            )  # Transpose for MatMul
            self.add_initializer("model.embeddings.patch_embedding.Add.bias", linear_bias)

            # MatMul: [B, N, patch_dim] x [patch_dim, H] -> [B, N, H]
            matmul_out = self.make_node(
                "MatMul",
                ["pixel_values", "model.embeddings.patch_embedding.MatMul.weight"],
                ["/model/embeddings/patch_embedding/MatMul/output_0"],
            )
            patch_embeds = self.make_node(
                "Add",
                [matmul_out, "model.embeddings.patch_embedding.Add.bias"],
                ["/model/embeddings/patch_embedding/Add/output_0"],
            )

        # === Position embeddings ===
        pe = "/model/embeddings/pos_embed"
        input_shape = self.make_node("Shape", ["pixel_values"], [f"{pe}/input_shape/output_0"])

        if self.vision_input_format == VISION_MODE_CONV2D:
            # Conv2d mode: use Resize (simpler, matches llama.cpp style)
            pos_emb_prefix = "vision_model.embeddings.position_embedding"
            pos_emb_weight = self.weights[f"{pos_emb_prefix}.weight"]
            pos_emb_4d = pos_emb_weight.reshape(16, 16, H).transpose(2, 0, 1)
            pos_emb_4d = pos_emb_4d[np.newaxis, ...]
            self.add_initializer(f"{pe}/base_weight", pos_emb_4d)

            n_merge = self.downsample
            self.add_initializer(f"{pe}/n_merge", np.array(n_merge, dtype=np.int64))

            axes_0 = self.get_constant("INT64", [0])
            pre_merge_h = self.make_node(
                "Mul", ["spatial_h", f"{pe}/n_merge"], [f"{pe}/pre_merge_h/output_0"]
            )
            pre_merge_w = self.make_node(
                "Mul", ["spatial_w", f"{pe}/n_merge"], [f"{pe}/pre_merge_w/output_0"]
            )

            spatial_h_unsq = self.make_node(
                "Unsqueeze", [pre_merge_h, axes_0], [f"{pe}/h_unsq/output_0"]
            )
            spatial_w_unsq = self.make_node(
                "Unsqueeze", [pre_merge_w, axes_0], [f"{pe}/w_unsq/output_0"]
            )

            sizes = self.make_node(
                "Concat",
                [
                    self.get_constant("INT64", [1]),
                    self.get_constant("INT64", [H]),
                    spatial_h_unsq,
                    spatial_w_unsq,
                ],
                [f"{pe}/sizes/output_0"],
                axis=0,
            )

            self.add_initializer(f"{pe}/empty_roi", np.array([], dtype=np.float32))
            resized = self.make_node(
                "Resize",
                [f"{pe}/base_weight", f"{pe}/empty_roi", "", sizes],
                [f"{pe}/resized/output_0"],
                mode="linear",
                coordinate_transformation_mode="half_pixel",
            )

            transposed = self.make_node(
                "Transpose", [resized], [f"{pe}/transposed/output_0"], perm=[0, 2, 3, 1]
            )
            pos_emb_final = self.make_node(
                "Reshape",
                [transposed, self.get_constant("INT64", [1, -1, H])],
                [f"{pe}/final/output_0"],
            )

            # Get batch size for tiling
            batch_size = self.make_node(
                "Gather",
                [input_shape, self.get_constant("INT64", 0)],
                [f"{pe}/batch_size/output_0"],
                axis=0,
            )
            batch_unsq = self.make_node(
                "Unsqueeze", [batch_size, axes_0], [f"{pe}/batch_unsq/output_0"]
            )
            tile_repeats = self.make_node(
                "Concat",
                [batch_unsq, self.get_constant("INT64", [1, 1])],
                [f"{pe}/tile_repeats/output_0"],
                axis=0,
            )
            pos_emb_tiled = self.make_node(
                "Tile", [pos_emb_final, tile_repeats], [f"{pe}/tiled/output_0"]
            )
        else:
            # === Position embedding interpolation (tiled mode) ===
            # Strategy: Size position embeddings for the largest image in the batch,
            # then filter/slice for each image. This trades memory (unused positions
            # for smaller images) for simplicity (single Resize op instead of per-image).
            # The Compress operator later removes padding tokens, so extra positions
            # don't affect the final output.
            pe = "/model/embeddings/pos_embed"
            spatial_h, spatial_w = self._extract_max_spatial_dims(pe)

            # Use Resize for position embedding interpolation
            pos_emb_prefix = "vision_model.embeddings.position_embedding"
            pos_emb_weight = self.weights[f"{pos_emb_prefix}.weight"]
            pos_emb_4d = pos_emb_weight.reshape(16, 16, H).transpose(2, 0, 1)
            pos_emb_4d = pos_emb_4d[np.newaxis, ...]  # [1, H, 16, 16]
            self.add_initializer(f"{pe}/base_weight", pos_emb_4d)

            axes_0 = self.get_constant("INT64", [0])
            spatial_h_unsq = self.make_node(
                "Unsqueeze", [spatial_h, axes_0], [f"{pe}/h_unsq/output_0"]
            )
            spatial_w_unsq = self.make_node(
                "Unsqueeze", [spatial_w, axes_0], [f"{pe}/w_unsq/output_0"]
            )

            sizes = self.make_node(
                "Concat",
                [
                    self.get_constant("INT64", [1]),
                    self.get_constant("INT64", [H]),
                    spatial_h_unsq,
                    spatial_w_unsq,
                ],
                [f"{pe}/sizes/output_0"],
                axis=0,
            )

            self.add_initializer(f"{pe}/empty_roi", np.array([], dtype=np.float32))
            resized = self.make_node(
                "Resize",
                [f"{pe}/base_weight", f"{pe}/empty_roi", "", sizes],
                [f"{pe}/resized/output_0"],
                mode="linear",
                coordinate_transformation_mode="half_pixel",
            )

            transposed = self.make_node(
                "Transpose", [resized], [f"{pe}/transposed/output_0"], perm=[0, 2, 3, 1]
            )
            pos_emb_final = self.make_node(
                "Reshape",
                [transposed, self.get_constant("INT64", [1, -1, H])],
                [f"{pe}/final/output_0"],
            )

            # Get batch size and num_patches from input shape
            batch_size = self.make_node(
                "Gather",
                [input_shape, self.get_constant("INT64", 0)],
                [f"{pe}/batch_size/output_0"],
                axis=0,
            )
            num_patches = self.make_node(
                "Gather",
                [input_shape, self.get_constant("INT64", 1)],
                [f"{pe}/num_patches/output_0"],
                axis=0,
            )

            # Handle padding (input may have more patches than H*W)
            # Fill padded positions with first token's position embedding
            first_token = self.make_node(
                "Slice",
                [
                    pos_emb_final,
                    self.get_constant("INT64", [0]),
                    self.get_constant("INT64", [1]),
                    self.get_constant("INT64", [1]),
                ],
                [f"{pe}/padding/slice_first_token/output_0"],
            )

            actual_num_patches = self.make_node(
                "Mul", [spatial_h, spatial_w], [f"{pe}/padding/actual_num_patches/output_0"]
            )

            indices = self.make_node(
                "Range",
                [
                    self.get_constant("INT64", 0),
                    num_patches,
                    self.get_constant("INT64", 1),
                ],
                [f"{pe}/padding/indices/output_0"],
            )

            # Valid mask: indices < actual_num_patches
            is_valid = self.make_node(
                "Less", [indices, actual_num_patches], [f"{pe}/padding/is_valid_mask/output_0"]
            )
            is_valid_3d = self.make_node(
                "Unsqueeze",
                [is_valid, self.get_constant("INT64", [0, 2])],
                [f"{pe}/padding/unsqueeze_mask/output_0"],
            )

            num_patches_unsq = self.make_node(
                "Unsqueeze", [num_patches, axes_0], [f"{pe}/padding/num_patches_unsq/output_0"]
            )
            expand_shape = self.make_node(
                "Concat",
                [
                    self.get_constant("INT64", [1]),
                    num_patches_unsq,
                    self.get_constant("INT64", [H]),
                ],
                [f"{pe}/padding/expand_shape/output_0"],
                axis=0,
            )
            first_token_expanded = self.make_node(
                "Expand",
                [first_token, expand_shape],
                [f"{pe}/padding/first_token_expanded/output_0"],
            )

            padding_size = self.make_node(
                "Sub", [num_patches, actual_num_patches], [f"{pe}/padding/padding_size/output_0"]
            )
            padding_size_unsq = self.make_node(
                "Unsqueeze", [padding_size, axes_0], [f"{pe}/padding/padding_size_unsq/output_0"]
            )
            pads = self.make_node(
                "Concat",
                [
                    self.get_constant("INT64", [0, 0, 0, 0]),
                    padding_size_unsq,
                    self.get_constant("INT64", [0]),
                ],
                [f"{pe}/padding/pads/output_0"],
                axis=0,
            )
            pos_emb_padded = self.make_node(
                "Pad", [pos_emb_final, pads], [f"{pe}/padding/padded/output_0"], mode="constant"
            )

            pos_emb_with_padding = self.make_node(
                "Where",
                [is_valid_3d, pos_emb_padded, first_token_expanded],
                [f"{pe}/padding/with_padding/output_0"],
            )

            batch_unsq = self.make_node(
                "Unsqueeze", [batch_size, axes_0], [f"{pe}/batch_unsq/output_0"]
            )
            tile_repeats = self.make_node(
                "Concat",
                [batch_unsq, self.get_constant("INT64", [1, 1])],
                [f"{pe}/tile_repeats/output_0"],
                axis=0,
            )
            pos_emb_tiled = self.make_node(
                "Tile", [pos_emb_with_padding, tile_repeats], [f"{pe}/tiled/output_0"]
            )

        return self.make_node(
            "Add", [patch_embeds, pos_emb_tiled], ["/model/embeddings/Add/output_0"]
        )

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
        # PyTorch weight prefix
        pt_prefix = f"vision_model.encoder.layers.{layer_idx}"
        # Community ONNX naming
        layer = f"/model/layers.{layer_idx}"
        w_prefix = f"model.layers.{layer_idx}"

        nh = self.vision_config.num_attention_heads
        hd = self.head_dim

        # LayerNorm1 weights
        self.add_initializer(
            f"{w_prefix}.layer_norm1_layernorm.weight",
            self.weights[f"{pt_prefix}.layer_norm1.weight"],
        )
        self.add_initializer(
            f"{w_prefix}.layer_norm1_layernorm.bias",
            self.weights[f"{pt_prefix}.layer_norm1.bias"],
        )

        # Q/K/V projection weights
        self.add_initializer(
            f"{w_prefix}.attn.q_proj.MatMul.weight",
            self.weights[f"{pt_prefix}.self_attn.q_proj.weight"].T,
        )
        self.add_initializer(
            f"{w_prefix}.attn.q_proj.Add.bias",
            self.weights[f"{pt_prefix}.self_attn.q_proj.bias"],
        )
        self.add_initializer(
            f"{w_prefix}.attn.k_proj.MatMul.weight",
            self.weights[f"{pt_prefix}.self_attn.k_proj.weight"].T,
        )
        self.add_initializer(
            f"{w_prefix}.attn.k_proj.Add.bias",
            self.weights[f"{pt_prefix}.self_attn.k_proj.bias"],
        )
        self.add_initializer(
            f"{w_prefix}.attn.v_proj.MatMul.weight",
            self.weights[f"{pt_prefix}.self_attn.v_proj.weight"].T,
        )
        self.add_initializer(
            f"{w_prefix}.attn.v_proj.Add.bias",
            self.weights[f"{pt_prefix}.self_attn.v_proj.bias"],
        )
        self.add_initializer(
            f"{w_prefix}.attn.out_proj.MatMul.weight",
            self.weights[f"{pt_prefix}.self_attn.out_proj.weight"].T,
        )
        self.add_initializer(
            f"{w_prefix}.attn.out_proj.Add.bias",
            self.weights[f"{pt_prefix}.self_attn.out_proj.bias"],
        )

        # LayerNorm2 weights
        self.add_initializer(
            f"{w_prefix}.layer_norm2_layernorm.weight",
            self.weights[f"{pt_prefix}.layer_norm2.weight"],
        )
        self.add_initializer(
            f"{w_prefix}.layer_norm2_layernorm.bias",
            self.weights[f"{pt_prefix}.layer_norm2.bias"],
        )

        # MLP weights
        self.add_initializer(
            f"{w_prefix}.mlp.fc1.MatMul.weight",
            self.weights[f"{pt_prefix}.mlp.fc1.weight"].T,
        )
        self.add_initializer(
            f"{w_prefix}.mlp.fc1.Add.bias",
            self.weights[f"{pt_prefix}.mlp.fc1.bias"],
        )
        self.add_initializer(
            f"{w_prefix}.mlp.fc2.MatMul.weight",
            self.weights[f"{pt_prefix}.mlp.fc2.weight"].T,
        )
        self.add_initializer(
            f"{w_prefix}.mlp.fc2.Add.bias",
            self.weights[f"{pt_prefix}.mlp.fc2.bias"],
        )

        residual = hidden_state

        normed = self.make_vision_layernorm(
            hidden_state,
            f"{w_prefix}.layer_norm1_layernorm.weight",
            f"{w_prefix}.layer_norm1_layernorm.bias",
            f"{layer}/layer_norm1_layernorm",
        )

        q = self.make_node(
            "MatMul",
            [normed, f"{w_prefix}.attn.q_proj.MatMul.weight"],
            [f"{layer}/attn/q_proj/MatMul/output_0"],
        )
        q = self.make_node(
            "Add",
            [q, f"{w_prefix}.attn.q_proj.Add.bias"],
            [f"{layer}/attn/q_proj/Add/output_0"],
        )

        k = self.make_node(
            "MatMul",
            [normed, f"{w_prefix}.attn.k_proj.MatMul.weight"],
            [f"{layer}/attn/k_proj/MatMul/output_0"],
        )
        k = self.make_node(
            "Add",
            [k, f"{w_prefix}.attn.k_proj.Add.bias"],
            [f"{layer}/attn/k_proj/Add/output_0"],
        )

        v = self.make_node(
            "MatMul",
            [normed, f"{w_prefix}.attn.v_proj.MatMul.weight"],
            [f"{layer}/attn/v_proj/MatMul/output_0"],
        )
        v = self.make_node(
            "Add",
            [v, f"{w_prefix}.attn.v_proj.Add.bias"],
            [f"{layer}/attn/v_proj/Add/output_0"],
        )

        scale = 1.0 / (hd**0.5)

        # Fused MultiHeadAttention (com.microsoft)
        # Inputs: query, key, value, bias, key_padding_mask, attention_bias, past_key, past_value
        # attention_bias: 4D additive mask [B, num_heads, N, N] with 0=valid, -inf=masked
        attn_bias = (
            "/model/attn_mask_reformat_full/Expand/output_0"
            if self.vision_input_format == VISION_MODE_TILED
            else ""
        )
        attn_out_reshaped = self.make_node(
            "MultiHeadAttention",
            [q, k, v, "", "", attn_bias, "", ""],
            [f"{layer}/attn/MultiHeadAttention/output_0"],
            domain="com.microsoft",
            num_heads=nh,
            scale=scale,
        )

        out_proj = self.make_node(
            "MatMul",
            [attn_out_reshaped, f"{w_prefix}.attn.out_proj.MatMul.weight"],
            [f"{layer}/attn/out_proj/MatMul/output_0"],
        )
        out_proj = self.make_node(
            "Add",
            [out_proj, f"{w_prefix}.attn.out_proj.Add.bias"],
            [f"{layer}/attn/out_proj/Add/output_0"],
        )

        # Community naming: Add_1 for attention residual
        hidden_state = self.make_node(
            "Add", [residual, out_proj], [f"{layer}/Add_1/output_0"]
        )

        residual2 = hidden_state
        normed2 = self.make_vision_layernorm(
            hidden_state,
            f"{w_prefix}.layer_norm2_layernorm.weight",
            f"{w_prefix}.layer_norm2_layernorm.bias",
            f"{layer}/layer_norm2_layernorm",
        )

        fc1 = self.make_node(
            "MatMul",
            [normed2, f"{w_prefix}.mlp.fc1.MatMul.weight"],
            [f"{layer}/mlp/fc1/MatMul/output_0"],
        )
        fc1 = self.make_node(
            "Add",
            [fc1, f"{w_prefix}.mlp.fc1.Add.bias"],
            [f"{layer}/mlp/fc1/Add/output_0"],
        )
        fc1_act = self.make_gelu(fc1, f"{layer}/mlp/fc1")

        fc2 = self.make_node(
            "MatMul",
            [fc1_act, f"{w_prefix}.mlp.fc2.MatMul.weight"],
            [f"{layer}/mlp/fc2/MatMul/output_0"],
        )
        fc2 = self.make_node(
            "Add",
            [fc2, f"{w_prefix}.mlp.fc2.Add.bias"],
            [f"{layer}/mlp/fc2/Add/output_0"],
        )

        # Community naming: Add_2 for MLP residual
        return self.make_node("Add", [residual2, fc2], [f"{layer}/Add_2/output_0"])

    def build_post_layernorm(self, hidden_state: str) -> str:
        """Build post layer norm."""
        self.add_initializer(
            "model.post_layernorm_layernorm.weight",
            self.weights["vision_model.post_layernorm.weight"],
        )
        self.add_initializer(
            "model.post_layernorm_layernorm.bias",
            self.weights["vision_model.post_layernorm.bias"],
        )
        return self.make_vision_layernorm(
            hidden_state,
            "model.post_layernorm_layernorm.weight",
            "model.post_layernorm_layernorm.bias",
            "/model/post_layernorm_layernorm",
        )

    def build_projector(self, vision_embeddings: str) -> str:
        """Build the MLP projector with pixel unshuffle.

        For tiled mode with variable spatial shapes, uses ScatterND canvas approach:
            1. Split spatial_shapes into per-batch h and w
            2. Compute aligned max dims (rounded to even for pixel unshuffle)
            3. Build grid indices (batch, y, x) for each patch
            4. Use ScatterND to place features into canvas
            5. Apply pixel unshuffle on canvas
            6. Apply projector MLP
            7. Build per-image validity mask and Compress

        Graph structure:
            vision_embeddings [B, N, C]
                ↓
            ┌─────────────────────────────────────┐
            │  ScatterND Canvas (tiled mode)      │
            │  [B, N, C] → [B, aligned_H, aligned_W, C] │
            └─────────────────────────────────────┘
                ↓
            ┌─────────────────────────────────────┐
            │  Pixel Unshuffle (2x2 → 4x channel) │
            │  [B, H, W, C] → [B, H/2, W/2, C*4]  │
            └─────────────────────────────────────┘
                ↓
            LayerNorm → Linear → GELU → Linear
                ↓
            ┌─────────────────────────────────────┐
            │  Validity Mask + Compress           │
            │  Extract valid tokens per image     │
            └─────────────────────────────────────┘
                ↓
            image_embeddings [total_valid_tokens, text_hidden]
        """
        ds = self.downsample
        C = self.vision_hidden
        input_dim = C * ds * ds

        # Community naming prefixes
        p = "/model/multimodal_projector"
        lp = "/model/layers.projector"

        use_layernorm = getattr(self.config, "projector_use_layernorm", True)
        if use_layernorm:
            self.add_initializer(
                "model.layers.projector.multi_modal_projector_layernorm.weight",
                self.weights["multi_modal_projector.layer_norm.weight"],
            )
            self.add_initializer(
                "model.layers.projector.multi_modal_projector_layernorm.bias",
                self.weights["multi_modal_projector.layer_norm.bias"],
            )

        self.add_initializer(
            "model.multimodal_projector.linear_1.MatMul.weight",
            self.weights["multi_modal_projector.linear_1.weight"].T,
        )
        if self.config.projector_bias:
            self.add_initializer(
                "model.multimodal_projector.linear_1.Add.bias",
                self.weights["multi_modal_projector.linear_1.bias"],
            )

        self.add_initializer(
            "model.multimodal_projector.linear_2.MatMul.weight",
            self.weights["multi_modal_projector.linear_2.weight"].T,
        )
        if self.config.projector_bias:
            self.add_initializer(
                "model.multimodal_projector.linear_2.Add.bias",
                self.weights["multi_modal_projector.linear_2.bias"],
            )

        axes_0 = self.get_constant("INT64", [0])
        input_shape = self.make_node("Shape", [vision_embeddings], [f"{p}/input_shape/output_0"])
        batch_size = self.make_node(
            "Gather",
            [input_shape, self.get_constant("INT64", 0)],
            [f"{p}/batch_size/output_0"],
            axis=0,
        )
        num_patches = self.make_node(
            "Gather",
            [input_shape, self.get_constant("INT64", 1)],
            [f"{p}/num_patches/output_0"],
            axis=0,
        )

        if self.vision_input_format == VISION_MODE_CONV2D:
            # Conv2d mode: use passed spatial dimensions
            n_merge = self.downsample
            self.add_initializer(f"{p}/n_merge", np.array(n_merge, dtype=np.int64))

            pre_merge_h = self.make_node(
                "Mul", ["spatial_h", f"{p}/n_merge"], [f"{p}/pre_merge_h/output_0"]
            )
            pre_merge_w = self.make_node(
                "Mul", ["spatial_w", f"{p}/n_merge"], [f"{p}/pre_merge_w/output_0"]
            )

            reshape_4d_shape = self.make_node(
                "Concat",
                [
                    self.make_node("Unsqueeze", [batch_size, axes_0], [f"{p}/batch_unsq/output_0"]),
                    self.make_node(
                        "Unsqueeze", [pre_merge_h, axes_0], [f"{p}/spatial_h_unsq/output_0"]
                    ),
                    self.make_node(
                        "Unsqueeze", [pre_merge_w, axes_0], [f"{p}/spatial_w_unsq/output_0"]
                    ),
                    self.get_constant("INT64", [C]),
                ],
                [f"{p}/reshape_4d_shape/output_0"],
                axis=0,
            )

            hidden_4d = self.make_node(
                "Reshape", [vision_embeddings, reshape_4d_shape], [f"{p}/hidden_4d/output_0"]
            )

            spatial_h_name = pre_merge_h
            half_spatial_h_name = "spatial_h"
            half_spatial_w_name = "spatial_w"
        else:
            # === Tiled mode: ScatterND canvas approach ===
            # This supports variable spatial shapes per batch element

            # Split spatial_shapes into h and w per batch
            h_per_batch, w_per_batch = self._split_spatial_shapes(p)

            # Compute aligned max dims (rounded to even for pixel unshuffle)
            aligned_h, aligned_w = self._compute_aligned_max_dims(h_per_batch, w_per_batch, p)

            # Build grid indices (batch, y, x) for each patch
            grid_indices = self._build_grid_indices(
                batch_size, num_patches, h_per_batch, w_per_batch, f"{p}/grid"
            )

            # Cast pixel_attention_mask to bool for ScatterND
            mask_bool = self.make_node(
                "Cast", ["pixel_attention_mask"], [f"{p}/mask_bool/output_0"], to=TensorProto.BOOL
            )

            # ScatterND: place features into canvas [B, aligned_h, aligned_w, C]
            hidden_4d = self._build_scatter_canvas(
                vision_embeddings,
                grid_indices,
                mask_bool,
                batch_size,
                aligned_h,
                aligned_w,
                C,
                f"{p}/scatter",
            )

            spatial_h_name = aligned_h
            half_spatial_h_name = self.make_node(
                "Div", [aligned_h, self.get_constant("INT64", 2)], [f"{p}/half_h/output_0"]
            )
            half_spatial_w_name = self.make_node(
                "Div", [aligned_w, self.get_constant("INT64", 2)], [f"{p}/half_w/output_0"]
            )

        # Pixel unshuffle: (B, H, W, C) -> (B, H/2, W/2, C*4)
        reshape1_shape = self.make_node(
            "Concat",
            [
                self.make_node("Unsqueeze", [batch_size, axes_0], [f"{p}/unshuffle/b1/output_0"]),
                self.make_node(
                    "Unsqueeze", [spatial_h_name, axes_0], [f"{p}/unshuffle/h1/output_0"]
                ),
                self.make_node(
                    "Unsqueeze", [half_spatial_w_name, axes_0], [f"{p}/unshuffle/w_half1/output_0"]
                ),
                self.get_constant("INT64", [C * ds]),
            ],
            [f"{p}/unshuffle/reshape1_shape/output_0"],
            axis=0,
        )
        step1 = self.make_node(
            "Reshape", [hidden_4d, reshape1_shape], [f"{p}/unshuffle/step1/output_0"]
        )
        step2 = self.make_node(
            "Transpose", [step1], [f"{p}/unshuffle/step2/output_0"], perm=[0, 2, 1, 3]
        )

        reshape2_shape = self.make_node(
            "Concat",
            [
                self.make_node("Unsqueeze", [batch_size, axes_0], [f"{p}/unshuffle/b2/output_0"]),
                self.make_node(
                    "Unsqueeze", [half_spatial_w_name, axes_0], [f"{p}/unshuffle/w_half2/output_0"]
                ),
                self.make_node(
                    "Unsqueeze", [half_spatial_h_name, axes_0], [f"{p}/unshuffle/h_half2/output_0"]
                ),
                self.get_constant("INT64", [input_dim]),
            ],
            [f"{p}/unshuffle/reshape2_shape/output_0"],
            axis=0,
        )
        step3 = self.make_node(
            "Reshape", [step2, reshape2_shape], [f"{p}/unshuffle/step3/output_0"]
        )
        step4 = self.make_node(
            "Transpose", [step3], [f"{p}/unshuffle/step4/output_0"], perm=[0, 2, 1, 3]
        )

        unshuffled = self.make_node(
            "Reshape",
            [step4, self.get_constant("INT64", [0, -1, input_dim])],
            [f"{p}/unshuffle/output_0"],
        )

        if use_layernorm:
            normed = self.make_node(
                "LayerNormalization",
                [
                    unshuffled,
                    "model.layers.projector.multi_modal_projector_layernorm.weight",
                    "model.layers.projector.multi_modal_projector_layernorm.bias",
                ],
                [f"{lp}/multi_modal_projector_layernorm/output_0"],
                epsilon=1e-5,
            )
        else:
            normed = unshuffled

        fc1 = self.make_node(
            "MatMul",
            [normed, "model.multimodal_projector.linear_1.MatMul.weight"],
            [f"{p}/linear_1/MatMul/output_0"],
        )
        if self.config.projector_bias:
            fc1 = self.make_node(
                "Add",
                [fc1, "model.multimodal_projector.linear_1.Add.bias"],
                [f"{p}/linear_1/Add/output_0"],
            )

        fc1_act = self.make_gelu(fc1, f"{p}/linear_1", approximate="none")

        fc2 = self.make_node(
            "MatMul",
            [fc1_act, "model.multimodal_projector.linear_2.MatMul.weight"],
            [f"{p}/linear_2/MatMul/output_0"],
        )
        if self.config.projector_bias:
            fc2 = self.make_node(
                "Add",
                [fc2, "model.multimodal_projector.linear_2.Add.bias"],
                [f"{p}/linear_2/Add/output_0"],
            )
        else:
            fc2 = self.make_node("Identity", [fc2], [f"{p}/linear_2/output_0"])

        if self.vision_input_format == VISION_MODE_TILED:
            # === Build output validity mask and Compress ===
            # For each position in [B, out_h, out_w], check if y < h[b]/2 AND x < w[b]/2
            output_mask = self._build_output_validity_mask(
                h_per_batch,
                w_per_batch,
                aligned_h,
                aligned_w,
                ds,
                f"{p}/out_mask",
            )

            # Flatten embeddings [B, out_h * out_w, hidden] → [B * out_h * out_w, hidden]
            embeds_flat = self.make_node(
                "Reshape",
                [fc2, self.get_constant("INT64", [-1, self.text_hidden])],
                [f"{p}/compress/embeds_flat/output_0"],
            )

            # Compress: select only valid tokens → [total_valid_tokens, hidden]
            self.make_node("Compress", [embeds_flat, output_mask], ["image_features"], axis=0)
        else:
            # Conv2d mode: single image, keep 3D output
            self.make_node("Identity", [fc2], ["image_features"])

        return "image_features"

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

    def build_value_info(self):
        """Build ValueInfo entries for weights and intermediate tensors."""
        H = self.vision_hidden
        nh = self.vision_config.num_attention_heads
        hd = self.head_dim
        intermediate = self.vision_config.intermediate_size
        num_layers = self.vision_config.num_hidden_layers
        text_hidden = self.text_hidden
        proj_hidden = self.proj_hidden

        # === Weight shapes (from initializers) ===
        for init in self.initializers:
            shape = list(init.dims)
            dtype = init.data_type
            self.add_value_info(init.name, dtype, shape)

        # === Per-layer outputs ===
        for layer_idx in range(num_layers):
            layer = f"/model/layers.{layer_idx}"

            # LayerNorm outputs
            self.add_value_info(
                f"{layer}/layer_norm1_layernorm/LayerNorm/output_0",
                TensorProto.FLOAT,
                ["batch_size", "num_patches", H],
            )
            self.add_value_info(
                f"{layer}/layer_norm2_layernorm/LayerNorm/output_0",
                TensorProto.FLOAT,
                ["batch_size", "num_patches", H],
            )

            # Attention Q/K/V projections
            self.add_value_info(
                f"{layer}/attn/q_proj/MatMul/output_0",
                TensorProto.FLOAT,
                ["batch_size", "num_patches", H],
            )
            self.add_value_info(
                f"{layer}/attn/q_proj/Add/output_0",
                TensorProto.FLOAT,
                ["batch_size", "num_patches", H],
            )
            self.add_value_info(
                f"{layer}/attn/k_proj/MatMul/output_0",
                TensorProto.FLOAT,
                ["batch_size", "num_patches", H],
            )
            self.add_value_info(
                f"{layer}/attn/k_proj/Add/output_0",
                TensorProto.FLOAT,
                ["batch_size", "num_patches", H],
            )
            self.add_value_info(
                f"{layer}/attn/v_proj/MatMul/output_0",
                TensorProto.FLOAT,
                ["batch_size", "num_patches", H],
            )
            self.add_value_info(
                f"{layer}/attn/v_proj/Add/output_0",
                TensorProto.FLOAT,
                ["batch_size", "num_patches", H],
            )

            # MultiHeadAttention output
            self.add_value_info(
                f"{layer}/attn/MultiHeadAttention/output_0",
                TensorProto.FLOAT,
                ["batch_size", "num_patches", H],
            )

            # Output projection
            self.add_value_info(
                f"{layer}/attn/out_proj/MatMul/output_0",
                TensorProto.FLOAT,
                ["batch_size", "num_patches", H],
            )
            self.add_value_info(
                f"{layer}/attn/out_proj/Add/output_0",
                TensorProto.FLOAT,
                ["batch_size", "num_patches", H],
            )

            # Residual adds
            self.add_value_info(
                f"{layer}/Add_1/output_0",
                TensorProto.FLOAT,
                ["batch_size", "num_patches", H],
            )
            self.add_value_info(
                f"{layer}/Add_2/output_0",
                TensorProto.FLOAT,
                ["batch_size", "num_patches", H],
            )

            # MLP
            self.add_value_info(
                f"{layer}/mlp/fc1/MatMul/output_0",
                TensorProto.FLOAT,
                ["batch_size", "num_patches", intermediate],
            )
            self.add_value_info(
                f"{layer}/mlp/fc1/Add/output_0",
                TensorProto.FLOAT,
                ["batch_size", "num_patches", intermediate],
            )
            self.add_value_info(
                f"{layer}/mlp/fc1/Gelu/output_0",
                TensorProto.FLOAT,
                ["batch_size", "num_patches", intermediate],
            )
            self.add_value_info(
                f"{layer}/mlp/fc2/MatMul/output_0",
                TensorProto.FLOAT,
                ["batch_size", "num_patches", H],
            )
            self.add_value_info(
                f"{layer}/mlp/fc2/Add/output_0",
                TensorProto.FLOAT,
                ["batch_size", "num_patches", H],
            )

        # Post LayerNorm
        self.add_value_info(
            "/model/post_layernorm_layernorm/LayerNorm/output_0",
            TensorProto.FLOAT,
            ["batch_size", "num_patches", H],
        )

        # Projector outputs (main ones)
        self.add_value_info(
            "/multi_modal_projector/fc1/MatMul/output_0",
            TensorProto.FLOAT,
            ["batch_size", "num_patches_projected", proj_hidden],
        )
        self.add_value_info(
            "/multi_modal_projector/fc1/Add/output_0",
            TensorProto.FLOAT,
            ["batch_size", "num_patches_projected", proj_hidden],
        )
        self.add_value_info(
            "/multi_modal_projector/fc2/MatMul/output_0",
            TensorProto.FLOAT,
            ["batch_size", "num_patches_projected", text_hidden],
        )
        self.add_value_info(
            "/multi_modal_projector/fc2/Add/output_0",
            TensorProto.FLOAT,
            ["batch_size", "num_patches_projected", text_hidden],
        )

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

        self.build_value_info()

        model = self.build_graph("embed_images")
        logger.info(f"Vision + projector model built: {len(self.nodes)} nodes, {len(self.value_info)} value_info")
        return model
