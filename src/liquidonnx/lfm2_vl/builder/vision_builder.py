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

    def _extract_max_spatial_dims(self, output_prefix: str) -> tuple[str, str]:
        """Extract max spatial dimensions from spatial_shapes input.

        Used by both patch embedding and projector to get consistent max(H), max(W)
        across the batch for tiled mode. Uses ReduceMax to find maximum dimensions
        across all images in the batch.

        Args:
            output_prefix: Output node prefix (e.g., "/model/embeddings/pos_embed",
                          "/model/multimodal_projector")

        Returns:
            Tuple of (spatial_h_node_name, spatial_w_node_name)
        """
        # ReduceMax across batch: [B, 2] → [2] to get max(H), max(W)
        max_spatial = self.make_node(
            "ReduceMax",
            ["spatial_shapes", self.get_constant("INT64", [0])],
            [f"{output_prefix}/max_spatial/output_0"],
            keepdims=0,
        )

        spatial_h = self.make_node(
            "Gather",
            [max_spatial, self.get_constant("INT64", 0)],
            [f"{output_prefix}/spatial_h/output_0"],
            axis=0,
        )
        spatial_w = self.make_node(
            "Gather",
            [max_spatial, self.get_constant("INT64", 1)],
            [f"{output_prefix}/spatial_w/output_0"],
            axis=0,
        )

        return spatial_h, spatial_w

    def _build_mask_downsampling(
        self,
        batch_size: str,
        spatial_h: str,
        spatial_w: str,
        half_spatial_h: str,
        half_spatial_w: str,
        actual_num_patches: str,
    ) -> str:
        """Build mask downsampling for tiled mode Compress operator.

        Downsamples pixel_attention_mask to match projector output (N → N/4).
        Uses ReduceMin (AND logic) over 2x2 blocks: merged token is valid only
        if ALL 4 input patches are valid.

        Shape transformations:
            [B, N_max] → slice → [B, H*W] → reshape → [B, H, W]
            → reshape → [B, H/2, 2, W/2, 2] → ReduceMin → [B, H/2, W/2]
            → flatten → [B * H/2 * W/2]

        Args:
            batch_size: Node name for batch size
            spatial_h: Node name for spatial height
            spatial_w: Node name for spatial width
            half_spatial_h: Node name for H/2
            half_spatial_w: Node name for W/2
            actual_num_patches: Node name for H * W

        Returns:
            Node name for boolean mask ready for Compress operator
        """
        # Use subprefix to avoid name collisions with main projector nodes
        m = "/model/multimodal_projector/mask_ds"
        axes_0 = self.get_constant("INT64", [0])

        # Slice end and axis
        actual_unsq = self.make_node(
            "Unsqueeze", [actual_num_patches, axes_0], [f"{m}/actual_unsq/output_0"]
        )

        # Step 1: Slice and reshape to spatial grid [B, N] → [B, H, W]
        mask_shape_4d = self.make_node(
            "Concat",
            [
                self.make_node("Unsqueeze", [batch_size, axes_0], [f"{m}/batch_unsq/output_0"]),
                self.make_node("Unsqueeze", [spatial_h, axes_0], [f"{m}/spatial_h_unsq/output_0"]),
                self.make_node("Unsqueeze", [spatial_w, axes_0], [f"{m}/spatial_w_unsq/output_0"]),
            ],
            [f"{m}/mask_shape_4d/output_0"],
            axis=0,
        )

        mask_sliced = self.make_node(
            "Slice",
            [
                "pixel_attention_mask",
                self.get_constant("INT64", [0]),
                actual_unsq,
                self.get_constant("INT64", [1]),
            ],
            [f"{m}/mask_sliced/output_0"],
        )
        mask_3d = self.make_node(
            "Reshape", [mask_sliced, mask_shape_4d], [f"{m}/mask_3d/output_0"]
        )

        # Step 2: Reshape for 2x2 pooling [B, H, W] → [B, H/2, 2, W/2, 2]
        two = self.get_constant("INT64", [2])
        pool_shape = self.make_node(
            "Concat",
            [
                self.make_node("Unsqueeze", [batch_size, axes_0], [f"{m}/b_pool/output_0"]),
                self.make_node("Unsqueeze", [half_spatial_h, axes_0], [f"{m}/h_half/output_0"]),
                two,
                self.make_node("Unsqueeze", [half_spatial_w, axes_0], [f"{m}/w_half/output_0"]),
                two,
            ],
            [f"{m}/pool_shape/output_0"],
            axis=0,
        )
        mask_5d = self.make_node("Reshape", [mask_3d, pool_shape], [f"{m}/mask_5d/output_0"])

        # Step 3: ReduceMin over 2x2 groups [B, H/2, 2, W/2, 2] → [B, H/2, W/2]
        mask_pooled = self.make_node(
            "ReduceMin",
            [mask_5d, self.get_constant("INT64", [2, 4])],
            [f"{m}/mask_pooled/output_0"],
            keepdims=0,
        )

        # Step 4: Flatten mask [B, H/2, W/2] → [B * H/2 * W/2]
        mask_flat = self.make_node(
            "Reshape", [mask_pooled, self.get_constant("INT64", [-1])], [f"{m}/flat_mask/output_0"]
        )

        # Cast mask to bool for Compress
        return self.make_node(
            "Cast", [mask_flat], [f"{m}/cast_mask_bool/output_0"], to=TensorProto.BOOL
        )

    def get_constant(self, dtype: str, value) -> str:
        """Get or create a constant with community-style naming.

        Constants are named: /model/constants/TYPE/value
        Examples:
            /model/constants/INT64/0
            /model/constants/FLOAT/1.0
            /model/constants/INT64/[1, 2]

        Args:
            dtype: "INT64" or "FLOAT"
            value: The constant value (scalar, list, or numpy array)

        Returns:
            The constant name
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

        name = f"/model/constants/{dtype}/{val_str}"

        # Only add if not already present
        if not any(init.name == name for init in self.initializers):
            if dtype == "INT64":
                arr = np.array(value, dtype=np.int64)
            else:
                arr = np.array(value, dtype=np.float32)
            self.add_initializer(name, arr)

        return name

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
                    "image_embeddings",
                    TensorProto.FLOAT,
                    ["num_image_tokens", self.text_hidden],
                )
            )
        else:
            # Conv2d mode: 3D output [batch, num_image_tokens, hidden]
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
                "Expand", [first_token, expand_shape], [f"{pe}/padding/first_token_expanded/output_0"]
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
            f"{layer}/layer_norm1_layernorm/output_0",
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

        hidden_state = self.make_node(
            "Add", [residual, out_proj], [f"{layer}/attn_residual/Add/output_0"]
        )

        residual2 = hidden_state
        normed2 = self.make_vision_layernorm(
            hidden_state,
            f"{w_prefix}.layer_norm2_layernorm.weight",
            f"{w_prefix}.layer_norm2_layernorm.bias",
            f"{layer}/layer_norm2_layernorm/output_0",
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
        fc1_act = self.make_gelu(fc1, f"{layer}/mlp/fc1/Gelu/output_0")

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

        return self.make_node(
            "Add", [residual2, fc2], [f"{layer}/mlp_residual/Add/output_0"]
        )

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
            "/model/post_layernorm_layernorm/output_0",
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
        batch_size = self.make_node(
            "Gather",
            [
                self.make_node("Shape", [vision_embeddings], [f"{p}/input_shape/output_0"]),
                self.get_constant("INT64", 0),
            ],
            [f"{p}/batch_size/output_0"],
            axis=0,
        )

        if self.vision_input_format == VISION_MODE_CONV2D:
            # Conv2d mode: use passed spatial dimensions
            # spatial_h, spatial_w are AFTER n_merge, so we need to multiply by n_merge
            # to get the pre-merge spatial dimensions for the first reshape
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

            spatial_h_name = pre_merge_h
            half_spatial_h_name = "spatial_h"
            half_spatial_w_name = "spatial_w"
        else:
            # Tiled mode: use max spatial dimensions across batch
            spatial_h, spatial_w = self._extract_max_spatial_dims(p)

            reshape_4d_shape = self.make_node(
                "Concat",
                [
                    self.make_node("Unsqueeze", [batch_size, axes_0], [f"{p}/batch_unsq/output_0"]),
                    self.make_node(
                        "Unsqueeze", [spatial_h, axes_0], [f"{p}/spatial_h_unsq/output_0"]
                    ),
                    self.make_node(
                        "Unsqueeze", [spatial_w, axes_0], [f"{p}/spatial_w_unsq/output_0"]
                    ),
                    self.get_constant("INT64", [C]),
                ],
                [f"{p}/reshape_4d_shape/output_0"],
                axis=0,
            )

            half_spatial_h = self.make_node(
                "Div",
                [spatial_h, self.get_constant("INT64", 2)],
                [f"{p}/half_spatial_h/output_0"],
            )
            half_spatial_w = self.make_node(
                "Div",
                [spatial_w, self.get_constant("INT64", 2)],
                [f"{p}/half_spatial_w/output_0"],
            )
            spatial_h_name = spatial_h
            half_spatial_h_name = half_spatial_h
            half_spatial_w_name = half_spatial_w

            # Slice out only valid patches before reshaping (input may be padded)
            actual_num_patches = self.make_node(
                "Mul", [spatial_h, spatial_w], [f"{p}/proj_actual_num_patches/output_0"]
            )
            actual_unsq = self.make_node(
                "Unsqueeze", [actual_num_patches, axes_0], [f"{p}/proj_actual_unsq/output_0"]
            )
            vision_embeddings = self.make_node(
                "Slice",
                [
                    vision_embeddings,
                    self.get_constant("INT64", [0]),
                    actual_unsq,
                    self.get_constant("INT64", [1]),
                ],
                [f"{p}/valid_embeddings/output_0"],
            )

        hidden_4d = self.make_node(
            "Reshape", [vision_embeddings, reshape_4d_shape], [f"{p}/hidden_4d/output_0"]
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

        fc1_act = self.make_gelu(fc1, f"{p}/linear_1/Gelu/output_0", approximate="none")

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
            # === Flatten output across batch using Compress ===
            # This allows different-sized images in the same batch
            # Output: [total_valid_tokens, hidden] instead of [B, N/4, hidden]

            # Build downsampled mask for Compress operator
            mask_bool = self._build_mask_downsampling(
                batch_size,
                spatial_h,
                spatial_w,
                half_spatial_h_name,
                half_spatial_w_name,
                actual_num_patches,
            )

            # Flatten embeddings [B, N/4, hidden] → [B * N/4, hidden]
            embeds_flat = self.make_node(
                "Reshape",
                [fc2, self.get_constant("INT64", [-1, self.text_hidden])],
                [f"{p}/compress/embeds_flat/output_0"],
            )

            # Compress: select only valid tokens → [total_valid_tokens, hidden]
            self.make_node(
                "Compress", [embeds_flat, mask_bool], ["image_embeddings"], axis=0
            )
        else:
            # Conv2d mode: single image, keep 3D output
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
