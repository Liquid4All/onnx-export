"""
LFM2-VL Builder for ONNX export.

This builder exports LFM2-VL vision-language models as three ONNX models:
- embed_tokens.onnx: Token embedding lookup (input_ids -> inputs_embeds)
- embed_images.onnx: SigLIP2 vision encoder + MLP projector (fused)
- decoder.onnx: LFM2 language model backbone (takes inputs_embeds, not input_ids)

The separation of embed_tokens allows clean fusion of text and image embeddings:
1. embed_tokens(input_ids) -> text_embeds
2. embed_images(pixel_values) -> image_embeds
3. Concatenate at <image> positions
4. decoder(inputs_embeds) -> logits

Vision Input Formats:
- Tiled (-T): Input [batch, num_patches, 768] with pre-extracted patches
              Requires complex preprocessing (tiling, patch extraction)
- Conv2d (-C): Input [batch, 3, H, W] with raw normalized image
              Simple preprocessing (resize + normalize), like llama.cpp

Usage:
    # Export with tiled input (default, HuggingFace compatible)
    uv run lfm2_vl.py --model LiquidAI/LFM2-VL-1.6B --output LFM2-VL-1.6B-ONNX -T

    # Export with conv2d input (simpler preprocessing, llama.cpp style)
    uv run lfm2_vl.py --model LiquidAI/LFM2-VL-1.6B --output LFM2-VL-1.6B-ONNX -C

    # Available models:
    # - LiquidAI/LFM2-VL-450M  (350M backbone + 86M SigLIP2)
    # - LiquidAI/LFM2-VL-1.6B  (1.2B backbone + 400M SigLIP2)
    # - LiquidAI/LFM2-VL-3B    (2.6B backbone + 400M SigLIP2)
"""

import logging
from dataclasses import dataclass

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from liquidonnx.lfm2.builder import LFM2Builder, LFM2Config
from liquidonnx.lfm2_vl import VISION_MODE_CONV2D, VISION_MODE_TILED

logger = logging.getLogger(__name__)


@dataclass
class SigLIP2Config:
    """Configuration for SigLIP2 vision encoder."""

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
    """Configuration for LFM2-VL model."""

    text_config: LFM2Config
    vision_config: SigLIP2Config
    projector_hidden_size: int
    projector_hidden_act: str = "gelu"
    projector_bias: bool = True
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
            downsample_factor=getattr(config, "downsample_factor", 2),
            image_token_id=getattr(config, "image_token_id", 396),
            tile_size=getattr(config, "tile_size", 512),
            max_tiles=getattr(config, "max_tiles", 10),
        )


class VisionEmbedBuilder:
    """
    Fused vision encoder + projector builder for ONNX export.

    Creates an ONNX graph that combines:
    - SigLIP2 vision encoder (patch embedding + transformer layers)
    - MLP projector with pixel unshuffle

    Output: image embeddings in text embedding space

    Supports two input formats:
    - "tiled": [batch, num_patches, 768] pre-extracted patches (HuggingFace style)
    - "conv2d": [batch, 3, H, W] raw image pixels (llama.cpp style)
    """

    def __init__(self, config: LFM2VLConfig, vision_input_format: str = VISION_MODE_TILED):
        """
        Args:
            config: Model configuration
            vision_input_format: "tiled" for [B, N, 768] or "conv2d" for [B, 3, H, W]
        """
        self.config = config
        self.vision_config = config.vision_config
        self.head_dim = config.vision_config.hidden_size // config.vision_config.num_attention_heads
        self.vision_input_format = vision_input_format

        # Projector dimensions
        self.vision_hidden = config.vision_config.hidden_size
        self.text_hidden = config.text_config.hidden_size
        self.proj_hidden = config.projector_hidden_size
        self.downsample = config.downsample_factor

        # Graph components
        self.nodes: list[onnx.NodeProto] = []
        self.inputs: list[onnx.ValueInfoProto] = []
        self.outputs: list[onnx.ValueInfoProto] = []
        self.initializers: list[onnx.TensorProto] = []

        # Weights storage
        self.weights: dict[str, np.ndarray] = {}

        # Node counter
        self._node_count = 0

    def _unique_name(self, prefix: str) -> str:
        self._node_count += 1
        return f"{prefix}_{self._node_count}"

    def add_initializer(self, name: str, tensor: np.ndarray, dtype=None):
        """Add weight tensor as graph initializer."""
        if dtype is None:
            if tensor.dtype not in [np.int32, np.int64]:
                tensor = tensor.astype(np.float32)
        else:
            tensor = tensor.astype(dtype)
        self.initializers.append(numpy_helper.from_array(tensor, name))

    def make_node(
        self,
        op_type: str,
        inputs: list[str],
        outputs: list[str],
        name: str = None,
        domain: str = "",
        **attrs,
    ) -> str:
        """Create an ONNX node and return the first output name."""
        if name is None:
            name = self._unique_name(op_type)

        node = helper.make_node(op_type, inputs, outputs, name=name, domain=domain, **attrs)
        self.nodes.append(node)
        return outputs[0] if outputs else None

    def make_layernorm(
        self, input_name: str, weight_name: str, bias_name: str, output_name: str
    ) -> str:
        """Create LayerNormalization node."""
        return self.make_node(
            "LayerNormalization",
            inputs=[input_name, weight_name, bias_name],
            outputs=[output_name],
            epsilon=self.vision_config.layer_norm_eps,
        )

    def make_gelu(self, input_name: str, output_name: str) -> str:
        """Create GELU activation (approximate tanh version)."""
        # Use standard ONNX Gelu (opset 20+) which supports approximate attribute
        return self.make_node("Gelu", [input_name], [output_name], approximate="tanh")

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

            # patch_attention_mask only needed for tiled mode
            self.inputs.append(
                helper.make_tensor_value_info(
                    "patch_attention_mask", TensorProto.INT64, ["batch_size", "num_patches"]
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

    def build_patch_embedding(self) -> str:
        """Build patch embedding layer with position embeddings.

        Supports two modes:
        - Tiled: Input [B, N, 768], uses Linear projection
        - Conv2d: Input [B, 3, H, W], uses Conv2d(kernel=16, stride=16)

        Position embeddings are bilinearly interpolated from 16x16 to match input spatial size.

        Output: [batch, num_patches, hidden_size]
        """
        prefix = "vision_model.embeddings.patch_embedding"
        H = self.vision_config.hidden_size
        P = self.vision_config.patch_size  # 16
        C = self.vision_config.num_channels  # 3

        # Load patch embedding weights
        linear_weight = self.weights[f"{prefix}.weight"]  # [hidden_size, patch_dim] = [768, 768]
        linear_bias = self.weights[f"{prefix}.bias"]  # [hidden_size]

        if self.vision_input_format == VISION_MODE_CONV2D:
            # =====================================================================
            # Conv2d mode: reshape Linear weights to Conv2d format
            # Linear: [hidden_size, C*P*P] -> Conv2d: [hidden_size, C, P, P]
            # =====================================================================
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

            # Reshape from [B, H, h, w] to [B, h*w, H]
            # First transpose to [B, h, w, H]
            transposed = self.make_node(
                "Transpose", [conv_out], ["patch_embed/transposed"], perm=[0, 2, 3, 1]
            )
            # Then reshape to [B, N, H]
            self.add_initializer("patch_embed/reshape_3d", np.array([0, -1, H], dtype=np.int64))
            patch_embeds = self.make_node(
                "Reshape", [transposed, "patch_embed/reshape_3d"], ["patch_embed/out"]
            )
        else:
            # =====================================================================
            # Tiled mode: Linear projection (original)
            # =====================================================================
            self.add_initializer(f"{prefix}.weight", linear_weight.T)  # Transpose for MatMul
            self.add_initializer(f"{prefix}.bias", linear_bias)

            # MatMul: [B, N, patch_dim] x [patch_dim, H] -> [B, N, H]
            matmul_out = self.make_node(
                "MatMul", ["pixel_values", f"{prefix}.weight"], ["patch_embed/matmul"]
            )
            patch_embeds = self.make_node(
                "Add", [matmul_out, f"{prefix}.bias"], ["patch_embed/out"]
            )

        # =====================================================================
        # Position embeddings with bilinear interpolation
        # =====================================================================
        # Position embedding: (256, 768) = (16*16, 768)
        pos_emb_prefix = "vision_model.embeddings.position_embedding"
        pos_emb_weight = self.weights[f"{pos_emb_prefix}.weight"]  # (256, 768)

        # Reshape to (16, 16, 768) then permute to (1, 768, 16, 16) for Resize
        pos_emb_4d = pos_emb_weight.reshape(16, 16, H).transpose(2, 0, 1)  # (768, 16, 16)
        pos_emb_4d = pos_emb_4d[np.newaxis, ...]  # (1, 768, 16, 16)
        self.add_initializer("pos_emb/4d", pos_emb_4d)

        # Get target spatial size based on input format
        input_shape = self.make_node("Shape", ["pixel_values"], ["pos_emb/input_shape"])
        self.add_initializer("pos_emb/axes_0", np.array([0], dtype=np.int64))

        if self.vision_input_format == VISION_MODE_CONV2D:
            # Conv2d mode: use passed spatial dimensions
            # spatial_h, spatial_w are AFTER n_merge (final projector output size)
            # For position embeddings, we need BEFORE n_merge: spatial * n_merge
            n_merge = self.downsample  # downsample_factor is the n_merge value
            self.add_initializer("pos_emb/n_merge", np.array(n_merge, dtype=np.int64))

            # Compute pre-merge spatial dimensions: spatial_h * n_merge, spatial_w * n_merge
            pre_merge_h = self.make_node(
                "Mul", ["spatial_h", "pos_emb/n_merge"], ["pos_emb/pre_merge_h"]
            )
            pre_merge_w = self.make_node(
                "Mul", ["spatial_w", "pos_emb/n_merge"], ["pos_emb/pre_merge_w"]
            )

            # Build target size tensor: [1, 768, pre_merge_h, pre_merge_w]
            self.add_initializer("pos_emb/one", np.array([1], dtype=np.int64))
            self.add_initializer("pos_emb/hidden", np.array([H], dtype=np.int64))
            pre_merge_h_unsq = self.make_node(
                "Unsqueeze", [pre_merge_h, "pos_emb/axes_0"], ["pos_emb/pre_merge_h_unsq"]
            )
            pre_merge_w_unsq = self.make_node(
                "Unsqueeze", [pre_merge_w, "pos_emb/axes_0"], ["pos_emb/pre_merge_w_unsq"]
            )

            target_size = self.make_node(
                "Concat",
                ["pos_emb/one", "pos_emb/hidden", pre_merge_h_unsq, pre_merge_w_unsq],
                ["pos_emb/target_size"],
                axis=0,
            )
        else:
            # Tiled mode: input is [B, N, patch_dim], target size is sqrt(N) x sqrt(N)
            self.add_initializer("pos_emb/idx_1", np.array(1, dtype=np.int64))  # scalar
            num_patches = self.make_node(
                "Gather", [input_shape, "pos_emb/idx_1"], ["pos_emb/num_patches"], axis=0
            )

            # sqrt(num_patches) to get spatial size
            num_patches_float = self.make_node(
                "Cast", [num_patches], ["pos_emb/np_float"], to=TensorProto.FLOAT
            )
            spatial_float = self.make_node("Sqrt", [num_patches_float], ["pos_emb/spatial_float"])
            spatial_int = self.make_node(
                "Cast", [spatial_float], ["pos_emb/spatial_int"], to=TensorProto.INT64
            )

            # Build target size tensor: [1, 768, target_h, target_w]
            self.add_initializer("pos_emb/one", np.array([1], dtype=np.int64))
            self.add_initializer("pos_emb/hidden", np.array([H], dtype=np.int64))
            spatial_unsq = self.make_node(
                "Unsqueeze", [spatial_int, "pos_emb/axes_0"], ["pos_emb/spatial_unsq"]
            )

            target_size = self.make_node(
                "Concat",
                ["pos_emb/one", "pos_emb/hidden", spatial_unsq, spatial_unsq],
                ["pos_emb/target_size"],
                axis=0,
            )

        # Use Resize with bilinear interpolation
        # Resize needs: X, roi, scales, sizes
        self.add_initializer("pos_emb/empty_roi", np.array([], dtype=np.float32))
        self.add_initializer("pos_emb/empty_scales", np.array([], dtype=np.float32))

        resized_pos_emb = self.make_node(
            "Resize",
            ["pos_emb/4d", "pos_emb/empty_roi", "pos_emb/empty_scales", target_size],
            ["pos_emb/resized"],
            mode="linear",  # bilinear for 2D
            coordinate_transformation_mode="half_pixel",  # Match tiled model's default
        )

        # Reshape from (1, 768, H, W) back to (1, H*W, 768)
        # First transpose to (1, H, W, 768)
        resized_transposed = self.make_node(
            "Transpose", [resized_pos_emb], ["pos_emb/transposed"], perm=[0, 2, 3, 1]
        )

        # Flatten to (1, H*W, 768)
        self.add_initializer("pos_emb/reshape_3d", np.array([1, -1, H], dtype=np.int64))
        pos_emb_final = self.make_node(
            "Reshape", [resized_transposed, "pos_emb/reshape_3d"], ["pos_emb/final"]
        )

        # Broadcast position embeddings to batch size
        # Get batch size (use scalar index)
        self.add_initializer("pos_emb/idx_0", np.array(0, dtype=np.int64))  # scalar
        batch_size = self.make_node(
            "Gather", [input_shape, "pos_emb/idx_0"], ["pos_emb/batch_size"], axis=0
        )

        # Tile position embeddings across batch: (1, N, H) -> (B, N, H)
        batch_unsq = self.make_node(
            "Unsqueeze", [batch_size, "pos_emb/axes_0"], ["pos_emb/batch_unsq"]
        )
        self.add_initializer("pos_emb/ones_2d", np.array([1, 1], dtype=np.int64))
        tile_repeats = self.make_node(
            "Concat", [batch_unsq, "pos_emb/ones_2d"], ["pos_emb/tile_repeats"], axis=0
        )
        pos_emb_tiled = self.make_node("Tile", [pos_emb_final, tile_repeats], ["pos_emb/tiled"])

        # =====================================================================
        # Add patch embeddings + position embeddings
        # =====================================================================
        return self.make_node("Add", [patch_embeds, pos_emb_tiled], ["patch_embeddings"])

    def build_encoder_layer(self, layer_idx: int, hidden_state: str) -> str:
        """Build a single transformer encoder layer."""
        prefix = f"vision_model.encoder.layers.{layer_idx}"
        H = self.vision_config.hidden_size
        nh = self.vision_config.num_attention_heads
        hd = self.head_dim

        # Load weights
        # Layer norm 1
        self.add_initializer(
            f"{prefix}.layer_norm1.weight", self.weights[f"{prefix}.layer_norm1.weight"]
        )
        self.add_initializer(
            f"{prefix}.layer_norm1.bias", self.weights[f"{prefix}.layer_norm1.bias"]
        )

        # Self attention
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

        # Layer norm 2
        self.add_initializer(
            f"{prefix}.layer_norm2.weight", self.weights[f"{prefix}.layer_norm2.weight"]
        )
        self.add_initializer(
            f"{prefix}.layer_norm2.bias", self.weights[f"{prefix}.layer_norm2.bias"]
        )

        # MLP
        self.add_initializer(f"{prefix}.mlp.fc1.weight", self.weights[f"{prefix}.mlp.fc1.weight"].T)
        self.add_initializer(f"{prefix}.mlp.fc1.bias", self.weights[f"{prefix}.mlp.fc1.bias"])
        self.add_initializer(f"{prefix}.mlp.fc2.weight", self.weights[f"{prefix}.mlp.fc2.weight"].T)
        self.add_initializer(f"{prefix}.mlp.fc2.bias", self.weights[f"{prefix}.mlp.fc2.bias"])

        residual = hidden_state

        # Layer norm 1
        normed = self.make_layernorm(
            hidden_state,
            f"{prefix}.layer_norm1.weight",
            f"{prefix}.layer_norm1.bias",
            f"{prefix}/ln1",
        )

        # Self attention
        # Q, K, V projections
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

        # Reshape to [B, N, nh, hd] then transpose to [B, nh, N, hd]
        self.add_initializer(f"{prefix}/reshape_qkv", np.array([0, -1, nh, hd], dtype=np.int64))
        q_4d = self.make_node("Reshape", [q, f"{prefix}/reshape_qkv"], [f"{prefix}/q_4d"])
        k_4d = self.make_node("Reshape", [k, f"{prefix}/reshape_qkv"], [f"{prefix}/k_4d"])
        v_4d = self.make_node("Reshape", [v, f"{prefix}/reshape_qkv"], [f"{prefix}/v_4d"])

        q_t = self.make_node("Transpose", [q_4d], [f"{prefix}/q_t"], perm=[0, 2, 1, 3])
        k_t = self.make_node("Transpose", [k_4d], [f"{prefix}/k_t"], perm=[0, 2, 1, 3])
        v_t = self.make_node("Transpose", [v_4d], [f"{prefix}/v_t"], perm=[0, 2, 1, 3])

        # Scaled dot-product attention
        scale = 1.0 / (hd**0.5)
        self.add_initializer(f"{prefix}/scale", np.array(scale, dtype=np.float32))

        # Q @ K^T
        k_t_transposed = self.make_node("Transpose", [k_t], [f"{prefix}/k_t_t"], perm=[0, 1, 3, 2])
        scores = self.make_node("MatMul", [q_t, k_t_transposed], [f"{prefix}/scores"])
        scores_scaled = self.make_node(
            "Mul", [scores, f"{prefix}/scale"], [f"{prefix}/scores_scaled"]
        )

        # Softmax
        attn_weights = self.make_node(
            "Softmax", [scores_scaled], [f"{prefix}/attn_weights"], axis=-1
        )

        # Attention output
        attn_out = self.make_node("MatMul", [attn_weights, v_t], [f"{prefix}/attn_out"])

        # Transpose back and reshape
        attn_out_t = self.make_node(
            "Transpose", [attn_out], [f"{prefix}/attn_out_t"], perm=[0, 2, 1, 3]
        )
        self.add_initializer(f"{prefix}/reshape_out", np.array([0, -1, H], dtype=np.int64))
        attn_out_reshaped = self.make_node(
            "Reshape", [attn_out_t, f"{prefix}/reshape_out"], [f"{prefix}/attn_out_reshaped"]
        )

        # Output projection
        out_proj = self.make_node(
            "MatMul",
            [attn_out_reshaped, f"{prefix}.self_attn.out_proj.weight"],
            [f"{prefix}/out_proj_matmul"],
        )
        out_proj = self.make_node(
            "Add", [out_proj, f"{prefix}.self_attn.out_proj.bias"], [f"{prefix}/out_proj"]
        )

        # Residual 1
        hidden_state = self.make_node("Add", [residual, out_proj], [f"{prefix}/residual1"])

        # Layer norm 2
        residual2 = hidden_state
        normed2 = self.make_layernorm(
            hidden_state,
            f"{prefix}.layer_norm2.weight",
            f"{prefix}.layer_norm2.bias",
            f"{prefix}/ln2",
        )

        # MLP
        fc1 = self.make_node(
            "MatMul", [normed2, f"{prefix}.mlp.fc1.weight"], [f"{prefix}/fc1_matmul"]
        )
        fc1 = self.make_node("Add", [fc1, f"{prefix}.mlp.fc1.bias"], [f"{prefix}/fc1"])
        fc1_act = self.make_gelu(fc1, f"{prefix}/fc1_act")

        fc2 = self.make_node(
            "MatMul", [fc1_act, f"{prefix}.mlp.fc2.weight"], [f"{prefix}/fc2_matmul"]
        )
        fc2 = self.make_node("Add", [fc2, f"{prefix}.mlp.fc2.bias"], [f"{prefix}/fc2"])

        # Residual 2
        return self.make_node("Add", [residual2, fc2], [f"{prefix}/residual2"])

    def build_post_layernorm(self, hidden_state: str) -> str:
        """Build post layer norm."""
        self.add_initializer(
            "vision_model.post_layernorm.weight", self.weights["vision_model.post_layernorm.weight"]
        )
        self.add_initializer(
            "vision_model.post_layernorm.bias", self.weights["vision_model.post_layernorm.bias"]
        )
        return self.make_layernorm(
            hidden_state,
            "vision_model.post_layernorm.weight",
            "vision_model.post_layernorm.bias",
            "vision_embeddings",
        )

    def build_projector(self, vision_embeddings: str) -> str:
        """Build the MLP projector with pixel unshuffle.

        PyTorch projector (modeling_lfm2_vl.py Lfm2VlMultiModalProjector):
        1. pixel_unshuffle: (B, H, W, C) -> (B, H/2, W/2, C*4)
           Note: PyTorch code uses confusing variable names (width=H, height=W)
           but the actual operations match standard row-major convention.
        2. layer_norm
        3. linear_1 + gelu + linear_2

        Our vision_embeddings is (B, N, C) where N = H*W (row-major order).
        We reshape to 4D, apply pixel_unshuffle ops, then flatten back.
        """
        ds = self.downsample
        C = self.vision_hidden  # 768
        input_dim = C * ds * ds  # 3072 after pixel unshuffle

        # Load weights
        # Layer norm
        self.add_initializer(
            "multi_modal_projector.layer_norm.weight",
            self.weights["multi_modal_projector.layer_norm.weight"],
        )
        self.add_initializer(
            "multi_modal_projector.layer_norm.bias",
            self.weights["multi_modal_projector.layer_norm.bias"],
        )

        # Linear layers
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

        # Step 1: Reshape from (B, N, C) to (B, H, W, C)
        # For 1024 patches: (B, 1024, 768) -> (B, 32, 32, 768)
        # Tokens are in row-major order, so we reshape to (B, H, W, C)

        # First get batch size dynamically (use scalar indices)
        self.add_initializer("proj/shape_indices_batch", np.array(0, dtype=np.int64))  # scalar
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
            n_merge = self.downsample  # downsample_factor is the n_merge value
            self.add_initializer("proj/n_merge", np.array(n_merge, dtype=np.int64))

            # Pre-merge dimensions: spatial_h * n_merge, spatial_w * n_merge
            pre_merge_h = self.make_node("Mul", ["spatial_h", "proj/n_merge"], ["proj/pre_merge_h"])
            pre_merge_w = self.make_node("Mul", ["spatial_w", "proj/n_merge"], ["proj/pre_merge_w"])

            # Build reshape target: [batch, pre_merge_h, pre_merge_w, C]
            # Match position embedding order: row-major (H first, then W)
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

            # Store for use in pixel_unshuffle (now H comes first in the 4D tensor)
            spatial_h_name = pre_merge_h
            half_spatial_h_name = "spatial_h"  # Already the post-merge size
            half_spatial_w_name = "spatial_w"  # Already the post-merge size
        else:
            # Tiled mode: compute spatial size from sqrt(seq_len), assumes square
            self.add_initializer("proj/shape_indices_seq", np.array(1, dtype=np.int64))  # scalar
            seq_len = self.make_node(
                "Gather",
                [
                    self.make_node("Shape", [vision_embeddings], ["proj/input_shape2"]),
                    "proj/shape_indices_seq",
                ],
                ["proj/seq_len"],
                axis=0,
            )
            seq_len_float = self.make_node(
                "Cast", [seq_len], ["proj/seq_len_float"], to=TensorProto.FLOAT
            )
            spatial_float = self.make_node("Sqrt", [seq_len_float], ["proj/spatial_float"])
            spatial_size = self.make_node(
                "Cast", [spatial_float], ["proj/spatial_size"], to=TensorProto.INT64
            )

            # Build reshape target: [batch, spatial, spatial, C] (square)
            reshape_4d_shape = self.make_node(
                "Concat",
                [
                    self.make_node("Unsqueeze", [batch_size, "proj/axes_0"], ["proj/batch_unsq"]),
                    self.make_node(
                        "Unsqueeze", [spatial_size, "proj/axes_0"], ["proj/spatial1_unsq"]
                    ),
                    self.make_node(
                        "Unsqueeze", [spatial_size, "proj/axes_0"], ["proj/spatial2_unsq"]
                    ),
                    "proj/hidden_size",
                ],
                ["proj/reshape_4d_shape"],
                axis=0,
            )

            # For tiled mode, compute half_spatial
            self.add_initializer("proj/two_tiled", np.array(2, dtype=np.int64))
            half_spatial = self.make_node(
                "Div", [spatial_size, "proj/two_tiled"], ["proj/half_spatial_tiled"]
            )
            spatial_h_name = spatial_size
            half_spatial_w_name = half_spatial
            half_spatial_h_name = half_spatial

        # Reshape to 4D: (B, N, C) -> (B, H, W, C)
        hidden_4d = self.make_node(
            "Reshape", [vision_embeddings, reshape_4d_shape], ["proj/hidden_4d"]
        )

        # Step 2: Pixel unshuffle (matches PyTorch Lfm2VlMultiModalProjector.pixel_unshuffle)
        # Input: (B, H, W, C)
        # Operations:
        #   reshape:   (B, H, W, C)     -> (B, H, W/2, C*2)
        #   transpose: (B, H, W/2, C*2) -> (B, W/2, H, C*2)
        #   reshape:   (B, W/2, H, C*2) -> (B, W/2, H/2, C*4)
        #   transpose: (B, W/2, H/2, C*4) -> (B, H/2, W/2, C*4)
        # Output: (B, H/2, W/2, C*4)

        # First reshape: (B, H, W, C) -> (B, H, W/2, C*2)
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

        # First transpose: (B, H, W/2, C*2) -> (B, W/2, H, C*2)
        step2 = self.make_node("Transpose", [step1], ["proj/step2"], perm=[0, 2, 1, 3])

        # Second reshape: (B, W/2, H, C*2) -> (B, W/2, H/2, C*4)
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

        # Second transpose: (B, W/2, H/2, C*4) -> (B, H/2, W/2, C*4)
        step4 = self.make_node("Transpose", [step3], ["proj/step4"], perm=[0, 2, 1, 3])

        # Flatten to 3D: (B, H/2, W/2, C*4) -> (B, N/4, C*4)
        self.add_initializer("proj/reshape_3d", np.array([0, -1, input_dim], dtype=np.int64))
        unshuffled = self.make_node("Reshape", [step4, "proj/reshape_3d"], ["proj/unshuffled"])

        # Step 3: Layer norm
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

        # Step 4: Linear 1
        fc1 = self.make_node(
            "MatMul", [normed, "multi_modal_projector.linear_1.weight"], ["proj/fc1_matmul"]
        )
        if self.config.projector_bias:
            fc1 = self.make_node("Add", [fc1, "multi_modal_projector.linear_1.bias"], ["proj/fc1"])

        # GELU
        fc1_act = self.make_gelu(fc1, "proj/fc1_act")

        # Step 5: Linear 2
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
        """Load weights from dict."""
        # Filter vision model and projector weights
        # Handle different prefixes: model.vision_tower.vision_model.* -> vision_model.*
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
        """Build the fused vision encoder + projector ONNX model."""
        logger.info("Building fused vision encoder + projector...")

        # Build graph structure
        self.build_inputs()
        self.build_outputs()

        # Patch embedding
        hidden_state = self.build_patch_embedding()

        # Encoder layers
        for layer_idx in range(self.vision_config.num_hidden_layers):
            logger.info(f"Building vision layer {layer_idx}...")
            hidden_state = self.build_encoder_layer(layer_idx, hidden_state)

        # Post layer norm
        vision_embeddings = self.build_post_layernorm(hidden_state)

        # Projector (fused)
        logger.info("Building projector...")
        self.build_projector(vision_embeddings)

        # Create graph
        graph = helper.make_graph(
            self.nodes,
            "embed_images",
            self.inputs,
            self.outputs,
            self.initializers,
        )

        # Create model
        model = helper.make_model(
            graph,
            opset_imports=[
                helper.make_opsetid("", 21),
                helper.make_opsetid("com.microsoft", 1),
            ],
            ir_version=9,
        )
        model.producer_name = "lfm2-vl-builder"

        logger.info(f"Vision + projector model built: {len(self.nodes)} nodes")
        return model


class EmbedTokensBuilder:
    """
    Simple token embedding builder for ONNX export.

    Creates an ONNX graph that maps input_ids to embeddings via Gather.
    This allows the decoder to take inputs_embeds, enabling clean
    text/image embedding fusion.
    """

    def __init__(self, config: LFM2VLConfig):
        self.config = config
        self.hidden_size = config.text_config.hidden_size
        self.vocab_size = config.text_config.vocab_size

        # Graph components
        self.nodes: list[onnx.NodeProto] = []
        self.inputs: list[onnx.ValueInfoProto] = []
        self.outputs: list[onnx.ValueInfoProto] = []
        self.initializers: list[onnx.TensorProto] = []

        # Weights
        self.embed_weight: np.ndarray | None = None

    def load_weights(self, weights: dict[str, np.ndarray]):
        """Load embedding weights."""
        # Try different possible prefixes
        for prefix in [
            "model.language_model.embed_tokens.weight",
            "language_model.embed_tokens.weight",
            "model.embed_tokens.weight",
        ]:
            if prefix in weights:
                self.embed_weight = weights[prefix].astype(np.float32)
                logger.info(f"Loaded embed_tokens weight: {self.embed_weight.shape}")
                return

        raise ValueError("Could not find embed_tokens weight in model")

    def build(self) -> onnx.ModelProto:
        """Build the embed_tokens ONNX model."""
        logger.info("Building embed_tokens...")

        # Input: input_ids [batch_size, sequence_length]
        self.inputs.append(
            helper.make_tensor_value_info(
                "input_ids", TensorProto.INT64, ["batch_size", "sequence_length"]
            )
        )

        # Output: inputs_embeds [batch_size, sequence_length, hidden_size]
        self.outputs.append(
            helper.make_tensor_value_info(
                "inputs_embeds",
                TensorProto.FLOAT,
                ["batch_size", "sequence_length", self.hidden_size],
            )
        )

        # Add embedding weight as initializer
        self.initializers.append(numpy_helper.from_array(self.embed_weight, "weight"))

        # Single Gather node: inputs_embeds = Gather(weight, input_ids, axis=0)
        node = helper.make_node(
            "Gather",
            inputs=["weight", "input_ids"],
            outputs=["inputs_embeds"],
            name="embed_tokens",
            axis=0,
        )
        self.nodes.append(node)

        # Create graph
        graph = helper.make_graph(
            self.nodes,
            "embed_tokens",
            self.inputs,
            self.outputs,
            self.initializers,
        )

        # Create model
        model = helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", 21)],
            ir_version=9,
        )
        model.producer_name = "lfm2-vl-builder"

        logger.info(
            f"embed_tokens built: {len(self.nodes)} nodes, "
            f"vocab_size={self.vocab_size}, hidden_size={self.hidden_size}"
        )
        return model


def export_vl_model(model_path: str, output_dir: str, vision_input_format: str = VISION_MODE_TILED):
    """Export LFM2-VL model to ONNX (embed_tokens + embed_images + decoder).

    Args:
        model_path: HuggingFace model path
        output_dir: Output directory for ONNX files
        vision_input_format: "tiled" for [B, N, 768] or "conv2d" for [B, 3, H, W]
    """
    import gc
    import json
    import pathlib

    import torch
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

    output_dir = pathlib.Path(output_dir)
    logger.info(f"Vision input format: {vision_input_format}")

    # Load config
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    vl_config = LFM2VLConfig.from_hf_config(config)

    # Load model weights
    logger.info(f"Loading weights from {model_path}...")
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, torch_dtype=torch.float32, trust_remote_code=True
    )

    weights = {}
    for name, param in model.named_parameters():
        weights[name] = param.detach().numpy()
        logger.debug(f"Loaded: {name} {param.shape}")

    logger.info(f"Loaded {len(weights)} total weights")

    del model
    gc.collect()

    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_dir = output_dir / "onnx"
    onnx_dir.mkdir(exist_ok=True)

    # =========================================================================
    # 1. Export embed_tokens (token embedding lookup)
    # =========================================================================
    logger.info("Exporting embed_tokens...")
    embed_tokens_builder = EmbedTokensBuilder(vl_config)
    embed_tokens_builder.load_weights(weights)
    embed_tokens_model = embed_tokens_builder.build()

    embed_tokens_path = onnx_dir / "embed_tokens.onnx"
    # embed_tokens is small enough to not need external data
    onnx.save_model(embed_tokens_model, str(embed_tokens_path))
    logger.info(f"embed_tokens saved to {embed_tokens_path}")
    del embed_tokens_model
    del embed_tokens_builder

    # =========================================================================
    # 2. Export embed_images (vision encoder + projector)
    # =========================================================================
    logger.info(
        f"Exporting embed_images (vision encoder + projector) [{vision_input_format} mode]..."
    )
    vision_builder = VisionEmbedBuilder(vl_config, vision_input_format=vision_input_format)
    vision_builder.load_weights(weights)
    vision_model = vision_builder.build()

    vision_path = onnx_dir / "embed_images.onnx"
    vision_data_path = onnx_dir / "embed_images.onnx_data"
    if vision_data_path.exists():
        vision_data_path.unlink()
    onnx.save_model(
        vision_model,
        str(vision_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="embed_images.onnx_data",
        size_threshold=1024,
    )
    logger.info(f"embed_images saved to {vision_path}")

    # Free memory before decoder export
    del vision_model
    del vision_builder
    gc.collect()

    # =========================================================================
    # 3. Export decoder (takes inputs_embeds, not input_ids)
    # =========================================================================
    logger.info("Exporting decoder (with inputs_embeds input)...")

    # Use LFM2Builder but modify inputs to use inputs_embeds
    text_builder = LFM2Builder(vl_config.text_config)

    # Filter text model weights (they have "model.language_model." prefix in VL model)
    for name, weight in weights.items():
        if name.startswith("model.language_model."):
            new_name = name.replace("model.language_model.", "model.")
            text_builder.weights[new_name] = weight
        elif name.startswith("language_model."):
            new_name = name.replace("language_model.", "model.")
            text_builder.weights[new_name] = weight

    # Clear original weights to free memory
    weights.clear()
    gc.collect()

    # Build custom inputs: inputs_embeds instead of input_ids
    H = vl_config.text_config.hidden_size

    # inputs_embeds: pre-computed embeddings (text + image fused)
    text_builder.inputs.append(
        helper.make_tensor_value_info(
            "inputs_embeds", TensorProto.FLOAT, ["batch_size", "sequence_length", H]
        )
    )

    # attention_mask
    text_builder.inputs.append(
        helper.make_tensor_value_info(
            "attention_mask", TensorProto.INT64, ["batch_size", "total_sequence_length"]
        )
    )

    # position_ids
    text_builder.inputs.append(
        helper.make_tensor_value_info(
            "position_ids", TensorProto.INT64, ["batch_size", "sequence_length"]
        )
    )

    # Conv caches
    for idx in text_builder.conv_indices:
        text_builder.inputs.append(
            helper.make_tensor_value_info(
                f"past_conv.{idx}",
                TensorProto.FLOAT,
                ["batch_size", H, vl_config.text_config.conv_L_cache],
            )
        )

    # KV caches
    for idx in text_builder.attn_indices:
        text_builder.inputs.append(
            helper.make_tensor_value_info(
                f"past_key_values.{idx}.key",
                TensorProto.FLOAT,
                [
                    "batch_size",
                    vl_config.text_config.num_key_value_heads,
                    "past_sequence_length",
                    text_builder.head_dim,
                ],
            )
        )
        text_builder.inputs.append(
            helper.make_tensor_value_info(
                f"past_key_values.{idx}.value",
                TensorProto.FLOAT,
                [
                    "batch_size",
                    vl_config.text_config.num_key_value_heads,
                    "past_sequence_length",
                    text_builder.head_dim,
                ],
            )
        )

    # Build outputs
    text_builder.build_outputs()

    # Build RoPE and attention mask preprocessing
    text_builder.build_rope_cache()
    text_builder.build_attention_mask_subgraph()

    # Skip build_embedding() - use inputs_embeds directly as hidden_state
    # But we still need embed_tokens weight for lm_head (tied weights)
    text_builder.add_initializer(
        "model.embed_tokens.weight", text_builder.weights["model.embed_tokens.weight"]
    )
    hidden_state = "inputs_embeds"

    # Build all layers
    for layer_idx in range(vl_config.text_config.num_hidden_layers):
        layer_type = vl_config.text_config.layer_types[layer_idx]
        logger.info(f"Building text layer {layer_idx} ({layer_type})...")

        if layer_type == "conv":
            hidden_state = text_builder.build_conv_layer(layer_idx, hidden_state)
        else:
            hidden_state = text_builder.build_attention_layer(layer_idx, hidden_state)

    text_builder.build_lm_head(hidden_state)

    # Clear weights dict to free memory before graph creation
    text_builder.weights.clear()
    gc.collect()

    logger.info("Building decoder graph...")
    text_graph = helper.make_graph(
        text_builder.nodes,
        "decoder",
        text_builder.inputs,
        text_builder.outputs,
        text_builder.initializers,
    )

    text_model = helper.make_model(
        text_graph,
        opset_imports=[
            helper.make_opsetid("", 21),
            helper.make_opsetid("com.microsoft", 1),
        ],
        ir_version=9,
    )
    text_model.producer_name = "lfm2-vl-builder"

    # Clear references to free memory
    text_builder.nodes.clear()
    text_builder.initializers.clear()
    gc.collect()

    decoder_path = onnx_dir / "decoder.onnx"
    decoder_data_path = onnx_dir / "decoder.onnx_data"
    if decoder_data_path.exists():
        decoder_data_path.unlink()

    logger.info("Saving decoder (this may take a while for large models)...")
    onnx.save_model(
        text_model,
        str(decoder_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="decoder.onnx_data",
        size_threshold=1024,
    )  # Move tensors > 1KB to external file
    logger.info(f"decoder saved to {decoder_path}")

    del text_model
    gc.collect()

    # Copy tokenizer and config
    try:
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        processor.save_pretrained(output_dir)
    except Exception as e:
        logger.warning(f"Could not save processor: {e}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer.save_pretrained(output_dir)

    config.save_pretrained(output_dir)

    # Create generation_config.json
    gen_config = {
        "_from_model_config": True,
        "bos_token_id": config.text_config.bos_token_id
        if hasattr(config.text_config, "bos_token_id")
        else 1,
        "eos_token_id": config.text_config.eos_token_id
        if hasattr(config.text_config, "eos_token_id")
        else 7,
        "pad_token_id": 0,
        "transformers_version": "4.57.0",
    }
    gen_config_path = output_dir / "generation_config.json"
    gen_config_path.write_text(json.dumps(gen_config, indent=2))

    # Print summary
    total_size = 0
    for fpath in onnx_dir.iterdir():
        if fpath.is_file():
            size = fpath.stat().st_size
            total_size += size
            logger.info(f"  {fpath.name}: {size / 1e6:.1f} MB")

    logger.info(f"Total ONNX size: {total_size / 1e9:.2f} GB")
    logger.info(f"Output directory: {output_dir}")

    return output_dir
