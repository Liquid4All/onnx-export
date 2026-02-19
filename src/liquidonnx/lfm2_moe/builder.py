"""
LFM2-MoE Builder for ONNX export.

Extends the LFM2 builder with Mixture of Experts (MoE) support using
the com.microsoft:MoE or QMoE (quantized) fused operator.

Architecture Overview:
    input_ids → Embedding → [Conv/Attention Layers] → LayerNorm → LM Head → logits

Layer Types:
    - Conv layers: Gated short convolution with depthwise conv1d
    - Attention layers: Grouped Query Attention with RoPE and Q/K normalization
    - MLP: Dense MLP for first `num_dense_layers`, Sparse MoE for rest

MoE Block:
    hidden_state
        ↓
    LayerNorm (ffn_norm)
        ↓
    Router: hidden_state @ gate.weight → sigmoid → + expert_bias → TopK
        ↓
    MoE/QMoE operator (com.microsoft) with:
        - gate_up_proj: [num_experts, 2*moe_intermediate_size, hidden_size]
        - down_proj: [num_experts, hidden_size, moe_intermediate_size]
        - SwiGLU activation, normalized routing weights
        ↓
    Add (residual)

QMoE (Quantized MoE):
    Uses INT4 block quantization for expert weights:
    - Weights packed as UINT8 (2 int4 values per byte)
    - Per-block scales (FP32) and zero points (UINT8)
    - block_size=32 (matching community approach)
"""

import logging
from dataclasses import dataclass

import numpy as np
import onnx
from onnx import TensorProto, helper

from liquidonnx.builder_base import ONNXBuilderBase

logger = logging.getLogger(__name__)

# === Constants ===
INT4_BITS = 4
INT4_MAX = (1 << INT4_BITS) - 1  # 15, max value for unsigned 4-bit
DEFAULT_BLOCK_SIZE = 32  # Default block size for quantization
SCALE_EPS = 1e-10  # Threshold for clamping small quantization scales
# Use FP32 min for masking in FP32 models (matches community); FP16 conversion will handle FP16 models
MASK_VALUE = float(np.finfo(np.float32).min)  # -3.4028234663852886e+38


@dataclass
class LFM2MoEConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int
    layer_types: list[str]
    intermediate_size: int
    moe_intermediate_size: int
    num_dense_layers: int
    num_experts: int
    num_experts_per_tok: int
    use_expert_bias: bool = True
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 1.0
    conv_L_cache: int = 3
    max_position_embeddings: int = 128000
    norm_eps: float = 1e-5
    rope_theta: float = 1000000.0

    @classmethod
    def from_hf_config(cls, config) -> "LFM2MoEConfig":
        rope_theta = 1000000.0
        if hasattr(config, "rope_parameters") and config.rope_parameters:
            rope_theta = config.rope_parameters.get("rope_theta", rope_theta)
        elif hasattr(config, "rope_theta"):
            rope_theta = config.rope_theta

        return cls(
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            vocab_size=config.vocab_size,
            layer_types=config.layer_types,
            intermediate_size=config.intermediate_size,
            moe_intermediate_size=config.moe_intermediate_size,
            num_dense_layers=config.num_dense_layers,
            num_experts=config.num_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            use_expert_bias=getattr(config, "use_expert_bias", True),
            norm_topk_prob=getattr(config, "norm_topk_prob", True),
            routed_scaling_factor=getattr(config, "routed_scaling_factor", 1.0),
            conv_L_cache=getattr(config, "conv_L_cache", 3),
            max_position_embeddings=config.max_position_embeddings,
            norm_eps=getattr(config, "norm_eps", 1e-5),
            rope_theta=rope_theta,
        )


# === INT4 Block Quantization for QMoE ===


def quantize_int4_block(
    weight: np.ndarray, block_size: int = DEFAULT_BLOCK_SIZE
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantize weight tensor to INT4 with block-wise scales and zero points.

    Args:
        weight: FP32 weight tensor of shape [..., K] where K is quantized dimension
        block_size: Number of elements per quantization block

    Returns:
        quant: UINT8 tensor with packed INT4 values (2 per byte)
        scales: FP32 scales, one per block
        zero_points: UINT8 packed zero points (2 per byte)
    """
    *batch_dims, K = weight.shape
    n_blocks = (K + block_size - 1) // block_size

    # Pad K to multiple of block_size
    pad_K = n_blocks * block_size
    if pad_K != K:
        pad_shape = list(weight.shape)
        pad_shape[-1] = pad_K - K
        weight = np.concatenate([weight, np.zeros(pad_shape, dtype=weight.dtype)], axis=-1)

    # Reshape to [..., n_blocks, block_size]
    weight_blocked = weight.reshape(*batch_dims, n_blocks, block_size)

    # Compute min/max per block
    w_min = weight_blocked.min(axis=-1, keepdims=True)
    w_max = weight_blocked.max(axis=-1, keepdims=True)

    # Compute scale and zero point for unsigned 4-bit (0 to INT4_MAX)
    scale = (w_max - w_min) / float(INT4_MAX)
    # Clamp small scales to 1.0 to match community behavior
    # This handles near-constant blocks (e.g., zero-padding) consistently
    scale = np.where(scale < SCALE_EPS, 1.0, scale)
    zero_point = np.round(-w_min / scale).clip(0, INT4_MAX).astype(np.uint8)

    # Quantize using ORT formula: q = round(w/s + zp) to match community
    # This differs from round((w - w_min)/s) due to rounding order
    quant = np.round(weight_blocked / scale + zero_point).clip(0, INT4_MAX).astype(np.uint8)

    # Pack two INT4 values into one UINT8 (low nibble first)
    # Shape: [..., n_blocks, block_size] -> [..., n_blocks, block_size//2]
    quant_packed = quant[..., 0::2] | (quant[..., 1::2] << 4)

    # Reshape scales: [..., n_blocks, 1] -> [..., n_blocks]
    scales = scale.squeeze(-1).astype(np.float32)

    # Pack zero points: [..., n_blocks] -> [..., n_blocks//2]
    zero_point = zero_point.squeeze(-1)
    if n_blocks % 2 == 1:
        # Pad zero_point for packing
        zp_shape = list(zero_point.shape)
        zp_shape[-1] = 1
        zero_point = np.concatenate([zero_point, np.zeros(zp_shape, dtype=np.uint8)], axis=-1)
    zp_packed = zero_point[..., 0::2] | (zero_point[..., 1::2] << 4)

    # Reshape quant_packed: [..., n_blocks, block_size//2] -> [..., n_blocks * block_size // 2]
    quant_final = quant_packed.reshape(*batch_dims, -1)

    return quant_final, scales, zp_packed


class LFM2MoEBuilder(ONNXBuilderBase):
    """
    Builds an optimized ONNX graph for LFM2-MoE with:
    - Conv/SSM layers with gating
    - Full attention layers with GQA
    - Dense MLP for shallow layers, Sparse MoE for deeper layers
    - Fused Microsoft operators

    Q4 Mode (use_q4=True):
        Matches onnx-community Q4 structure with:
        - GatherBlockQuantized for embeddings
        - MatMulNBits INT4 for linear layers (attention, MLP, conv)
        - FP32 MatMul for MoE router (preserves routing quality, WebGPU compatible)
        - QMoE INT4 for MoE expert weights
    """

    def __init__(
        self,
        config: LFM2MoEConfig,
        use_integrated_rope: bool = False,
        use_qmoe: bool = False,
        qmoe_block_size: int = DEFAULT_BLOCK_SIZE,
        use_q4: bool = False,
    ):
        super().__init__()
        self.config = config
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.use_integrated_rope = use_integrated_rope
        self.use_qmoe = use_qmoe or use_q4  # Q4 mode implies QMoE for MoE layers
        self.use_q4 = use_q4  # Full Q4 quantization (all layers)
        self.qmoe_block_size = qmoe_block_size

        self.conv_indices = [i for i, t in enumerate(config.layer_types) if t == "conv"]
        self.attn_indices = [i for i, t in enumerate(config.layer_types) if t == "full_attention"]

    # === Q4 Quantization Methods ===

    def _quantize_for_matmul_nbits(self, weight: np.ndarray, name: str) -> tuple[str, str, str]:
        """Quantize weight for MatMulNBits operator.

        Args:
            weight: FP32 weight tensor of shape [K, N] (already transposed for MatMul)
            name: Base name for initializers

        Returns:
            Tuple of (quant_name, scales_name, zp_name, K, N) for initializer references

        Uses community 3D layout for quant weights: (N, n_blocks, block_size//2)
        """
        K, N = weight.shape
        block_size = self.qmoe_block_size

        # Transpose to [N, K] for block-wise quantization along K
        weight_t = weight.T  # [N, K]

        quant, scales, zp = quantize_int4_block(weight_t, block_size)

        # Reshape quant from 2D (N, K//2) to 3D (N, n_blocks, block_size//2)
        # to match community layout
        n_blocks = (K + block_size - 1) // block_size
        quant_3d = quant.reshape(N, n_blocks, block_size // 2)

        quant_name = f"{name}_quant"
        scales_name = f"{name}_scales"
        zp_name = f"{name}_zp"

        self.add_initializer(quant_name, quant_3d, dtype=np.uint8)
        self.add_initializer(scales_name, scales)
        self.add_initializer(zp_name, zp, dtype=np.uint8)

        return quant_name, scales_name, zp_name, K, N

    def make_matmul_nbits(
        self, input_name: str, weight: np.ndarray, name: str, output_name: str
    ) -> str:
        """Create MatMulNBits node for INT4 quantized linear layer.

        Args:
            input_name: Input tensor name
            weight: Weight matrix [K, N] (already transposed for MatMul)
            name: Base name for the operation
            output_name: Output tensor name

        Returns:
            Output tensor name
        """
        quant_name, scales_name, zp_name, K, N = self._quantize_for_matmul_nbits(weight, name)

        return self.make_node(
            "MatMulNBits",
            [input_name, quant_name, scales_name, zp_name],
            [output_name],
            domain="com.microsoft",
            K=K,
            N=N,
            bits=4,
            block_size=self.qmoe_block_size,
        )

    def make_gather_block_quantized(
        self, weight: np.ndarray, indices_name: str, name: str, output_name: str
    ) -> str:
        """Create GatherBlockQuantized node for INT4 quantized embedding lookup.

        Args:
            weight: Embedding weight [vocab_size, hidden_size]
            indices_name: Input token IDs tensor name
            name: Base name for initializers
            output_name: Output tensor name

        Returns:
            Output tensor name
        """
        vocab_size, hidden_size = weight.shape
        block_size = self.qmoe_block_size

        # Quantize along hidden_size (axis=1)
        quant, scales, zp = quantize_int4_block(weight, block_size)

        quant_name = f"{name}_quant"
        scales_name = f"{name}_scales"
        zp_name = f"{name}_zp"

        self.add_initializer(quant_name, quant, dtype=np.uint8)
        self.add_initializer(scales_name, scales)
        self.add_initializer(zp_name, zp, dtype=np.uint8)

        return self.make_node(
            "GatherBlockQuantized",
            [quant_name, indices_name, scales_name, zp_name],
            [output_name],
            domain="com.microsoft",
            bits=4,
            block_size=block_size,
            gather_axis=0,
            quantize_axis=1,
        )

    def is_moe_layer(self, layer_idx: int) -> bool:
        return layer_idx >= self.config.num_dense_layers

    def _interleave_gate_up(self, gate_up: np.ndarray) -> np.ndarray:
        """Interleave gate and up projections for MoE operator.

        PyTorch format: [num_experts, 2*intermediate, hidden]
            - First half: gate projections [g0, g1, ..., g_n]
            - Second half: up projections [u0, u1, ..., u_n]

        MoE operator format: [num_experts, 2*intermediate, hidden]
            - Interleaved: [g0, u0, g1, u1, ..., g_n, u_n]
        """
        n_experts, total_intermediate, hidden = gate_up.shape
        half = total_intermediate // 2

        gate = gate_up[:, :half, :]  # [n_experts, intermediate, hidden]
        up = gate_up[:, half:, :]  # [n_experts, intermediate, hidden]

        interleaved = np.empty_like(gate_up)
        interleaved[:, 0::2, :] = gate  # Even indices get gate
        interleaved[:, 1::2, :] = up  # Odd indices get up

        return interleaved

    def _prepare_qmoe_weights(self, prefix: str, gate_up: np.ndarray, down: np.ndarray):
        """Quantize MoE expert weights for QMoE operator.

        Args:
            prefix: Layer prefix (e.g., "model.layers.2")
            gate_up: Interleaved gate_up_proj [n_experts, 2*intermediate, hidden]
            down: down_proj [n_experts, hidden, intermediate]

        Uses community naming convention with underscores:
        model_layers_X_moe_experts_gate_up_proj_weight_quant
        """
        block_size = self.qmoe_block_size
        # Convert prefix to underscore format for community naming
        prefix_underscore = prefix.replace(".", "_")

        # Quantize gate_up_proj along last dimension (hidden_size)
        gate_up_quant, gate_up_scales, gate_up_zp = quantize_int4_block(gate_up, block_size)
        self.add_initializer(
            f"{prefix_underscore}_moe_experts_gate_up_proj_weight_quant",
            gate_up_quant,
            dtype=np.uint8,
        )
        self.add_initializer(
            f"{prefix_underscore}_moe_experts_gate_up_proj_weight_scales", gate_up_scales
        )
        self.add_initializer(
            f"{prefix_underscore}_moe_experts_gate_up_proj_weight_zp", gate_up_zp, dtype=np.uint8
        )

        # Quantize down_proj along last dimension (intermediate_size)
        down_quant, down_scales, down_zp = quantize_int4_block(down, block_size)
        self.add_initializer(
            f"{prefix_underscore}_moe_experts_down_proj_weight_quant", down_quant, dtype=np.uint8
        )
        self.add_initializer(
            f"{prefix_underscore}_moe_experts_down_proj_weight_scales", down_scales
        )
        self.add_initializer(
            f"{prefix_underscore}_moe_experts_down_proj_weight_zp", down_zp, dtype=np.uint8
        )

    def make_simple_layernorm(
        self, input_name: str, weight_name: str, path: str, name: str = None
    ) -> str:
        """Create SimplifiedLayerNormalization node.

        Args:
            input_name: Input tensor
            weight_name: Scale weight
            path: Logical path (e.g., "/model/layers.0/input_layernorm")
            name: Override name in path (default: "SimplifiedLayerNormalization").
                  Use "LayerNorm" for community convention.
        """
        return self.make_layernorm(
            input_name, weight_name, None, path, epsilon=self.config.norm_eps, name=name
        )

    def make_skip_layernorm(
        self, input_name: str, skip_name: str, weight_name: str, output_name: str
    ) -> str:
        return self.make_node(
            "SkipSimplifiedLayerNormalization",
            inputs=[input_name, skip_name, weight_name],
            outputs=[output_name],
            domain="com.microsoft",
            epsilon=self.config.norm_eps,
        )

    def prepare_layer_weights(self, layer_idx: int, layer_type: str):
        """Prepare layer weights as ONNX initializers.

        In Q4 mode, linear weights are quantized on-the-fly by make_matmul_nbits.
        Only LayerNorm weights and conv kernels are added here.

        Uses community naming convention:
        - ffn_norm -> ffn_layernorm
        - operator_norm -> operator_layernorm
        - conv.weight -> conv.conv.weight
        - self_attn.q_layernorm -> attn.q_norm.layernorm
        """
        prefix = f"model.layers.{layer_idx}"

        # LayerNorm weights are always FP32 (community naming)
        self.add_initializer(
            f"{prefix}.operator_layernorm.weight", self.weights[f"{prefix}.operator_norm.weight"]
        )
        self.add_initializer(
            f"{prefix}.ffn_layernorm.weight", self.weights[f"{prefix}.ffn_norm.weight"]
        )

        if self.is_moe_layer(layer_idx):
            # === MoE weights ===
            # Router kept as FP32 MatMul even in Q4 mode (WebGPU compatible,
            # preserves routing quality; q4f16 conversion handles FP32→FP16)
            self.add_initializer(
                f"{prefix}.moe.router.MatMul.weight",
                self.weights[f"{prefix}.feed_forward.gate.weight"].T,
            )

            if self.config.use_expert_bias:
                self.add_initializer(
                    f"{prefix}.moe.expert_bias",
                    self.weights[f"{prefix}.feed_forward.expert_bias"],
                )

            # Expert weights: gate_up_proj and down_proj
            gate_up = self.weights[f"{prefix}.feed_forward.experts.gate_up_proj"]
            gate_up_interleaved = self._interleave_gate_up(gate_up)
            down = self.weights[f"{prefix}.feed_forward.experts.down_proj"]

            if self.use_qmoe:
                # Quantize weights for QMoE
                self._prepare_qmoe_weights(prefix, gate_up_interleaved, down)
            else:
                self.add_initializer(
                    f"{prefix}.moe.experts.gate_up_proj.weight", gate_up_interleaved
                )
                self.add_initializer(f"{prefix}.moe.experts.down_proj.weight", down)
        else:
            # === Dense MLP weights ===
            if not self.use_q4:
                # Only add FP32 weights in non-Q4 mode (Q4 quantizes on-the-fly)
                self.add_initializer(
                    f"{prefix}.mlp.gate_proj.MatMul.weight",
                    self.weights[f"{prefix}.feed_forward.w1.weight"].T,
                )
                self.add_initializer(
                    f"{prefix}.mlp.up_proj.MatMul.weight",
                    self.weights[f"{prefix}.feed_forward.w3.weight"].T,
                )
                self.add_initializer(
                    f"{prefix}.mlp.down_proj.MatMul.weight",
                    self.weights[f"{prefix}.feed_forward.w2.weight"].T,
                )

        if layer_type == "conv":
            # Conv kernel is not quantized (kept FP32) - community naming: conv.conv.weight
            self.add_initializer(
                f"{prefix}.conv.conv.weight", self.weights[f"{prefix}.conv.conv.weight"]
            )
            if not self.use_q4:
                # Conv projections only in non-Q4 mode (Q4 quantizes on-the-fly)
                self.add_initializer(
                    f"{prefix}.conv.in_proj.MatMul.weight",
                    self.weights[f"{prefix}.conv.in_proj.weight"].T,
                )
                self.add_initializer(
                    f"{prefix}.conv.out_proj.MatMul.weight",
                    self.weights[f"{prefix}.conv.out_proj.weight"].T,
                )
        else:
            # === Attention weights ===
            # LayerNorm weights (community naming: attn.q_norm.layernorm)
            self.add_initializer(
                f"{prefix}.attn.q_norm.layernorm.weight",
                self.weights[f"{prefix}.self_attn.q_layernorm.weight"],
            )
            self.add_initializer(
                f"{prefix}.attn.k_norm.layernorm.weight",
                self.weights[f"{prefix}.self_attn.k_layernorm.weight"],
            )
            if not self.use_q4:
                # Attention projections only in non-Q4 mode (Q4 quantizes on-the-fly)
                # Community naming: attn.*_proj.MatMul.weight
                self.add_initializer(
                    f"{prefix}.attn.q_proj.MatMul.weight",
                    self.weights[f"{prefix}.self_attn.q_proj.weight"].T,
                )
                self.add_initializer(
                    f"{prefix}.attn.k_proj.MatMul.weight",
                    self.weights[f"{prefix}.self_attn.k_proj.weight"].T,
                )
                self.add_initializer(
                    f"{prefix}.attn.v_proj.MatMul.weight",
                    self.weights[f"{prefix}.self_attn.v_proj.weight"].T,
                )
                self.add_initializer(
                    f"{prefix}.attn.o_proj.MatMul.weight",
                    self.weights[f"{prefix}.self_attn.out_proj.weight"].T,
                )

    def build_inputs(self):
        self.inputs.append(
            helper.make_tensor_value_info(
                "input_ids", TensorProto.INT64, ["batch_size", "sequence_length"]
            )
        )
        self.inputs.append(
            helper.make_tensor_value_info(
                "attention_mask", TensorProto.INT64, ["batch_size", "total_sequence_length"]
            )
        )
        # position_ids only needed when using separate RotaryEmbedding nodes
        if not self.use_integrated_rope:
            self.inputs.append(
                helper.make_tensor_value_info(
                    "position_ids", TensorProto.INT64, ["batch_size", "sequence_length"]
                )
            )

        # Interleave past states by layer index (matching community convention)
        for idx in range(self.config.num_hidden_layers):
            layer_type = self.config.layer_types[idx]
            if layer_type == "conv":
                self.inputs.append(
                    helper.make_tensor_value_info(
                        f"past_conv.{idx}",
                        TensorProto.FLOAT,
                        ["batch_size", self.config.hidden_size, self.config.conv_L_cache],
                    )
                )
            else:
                self.inputs.append(
                    helper.make_tensor_value_info(
                        f"past_key_values.{idx}.key",
                        TensorProto.FLOAT,
                        [
                            "batch_size",
                            self.config.num_key_value_heads,
                            "past_sequence_length",
                            self.head_dim,
                        ],
                    )
                )
                self.inputs.append(
                    helper.make_tensor_value_info(
                        f"past_key_values.{idx}.value",
                        TensorProto.FLOAT,
                        [
                            "batch_size",
                            self.config.num_key_value_heads,
                            "past_sequence_length",
                            self.head_dim,
                        ],
                    )
                )

    def build_outputs(self):
        self.outputs.append(
            helper.make_tensor_value_info(
                "logits",
                TensorProto.FLOAT,
                ["batch_size", "sequence_length", self.config.vocab_size],
            )
        )

        # Interleave present states by layer index (matching community convention)
        for idx in range(self.config.num_hidden_layers):
            layer_type = self.config.layer_types[idx]
            if layer_type == "conv":
                self.outputs.append(
                    helper.make_tensor_value_info(
                        f"present_conv.{idx}",
                        TensorProto.FLOAT,
                        ["batch_size", self.config.hidden_size, self.config.conv_L_cache],
                    )
                )
            else:
                self.outputs.append(
                    helper.make_tensor_value_info(
                        f"present.{idx}.key",
                        TensorProto.FLOAT,
                        [
                            "batch_size",
                            self.config.num_key_value_heads,
                            "total_sequence_length",
                            self.head_dim,
                        ],
                    )
                )
                self.outputs.append(
                    helper.make_tensor_value_info(
                        f"present.{idx}.value",
                        TensorProto.FLOAT,
                        [
                            "batch_size",
                            self.config.num_key_value_heads,
                            "total_sequence_length",
                            self.head_dim,
                        ],
                    )
                )

    def build_embedding(self) -> str:
        embed_weight = self.weights["model.embed_tokens.weight"]

        if self.use_q4:
            # Q4: Use GatherBlockQuantized for INT4 quantized embeddings
            return self.make_gather_block_quantized(
                embed_weight,
                "input_ids",
                "model_embed_tokens_weight",
                "/model/embed_tokens/GatherBlockQuantized/output_0",
            )
        else:
            self.add_initializer("model.embed_tokens.weight", embed_weight)
            return self.make_node(
                "Gather",
                ["model.embed_tokens.weight", "input_ids"],
                ["/model/embed_tokens/Gather/output_0"],
                axis=0,
            )

    def build_rope_cache(self):
        import torch

        head_dim = self.head_dim
        max_seq = self.config.max_position_embeddings
        theta = self.config.rope_theta

        # Use torch for RoPE computation to match community model precision
        dim_idx = torch.arange(0, head_dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (theta ** (dim_idx / head_dim))
        t = torch.arange(max_seq, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)

        self.add_initializer("cos_cache", torch.cos(freqs).numpy())
        self.add_initializer("sin_cache", torch.sin(freqs).numpy())

    def build_attention_mask_subgraph(self):
        # Use community constant naming and path convention
        const_1_arr = self.get_constant([1])  # /model/constants/INT64/[1]
        const_1_scalar = self.get_constant(1)  # /model/constants/INT64/1

        self.make_node(
            "ReduceSum",
            ["attention_mask", const_1_arr],
            ["/model/attn_mask_reformat/attn_mask_subgraph/ReduceSum/output_0"],
            keepdims=0,
        )
        self.make_node(
            "Sub",
            ["/model/attn_mask_reformat/attn_mask_subgraph/ReduceSum/output_0", const_1_arr],
            ["/model/attn_mask_reformat/attn_mask_subgraph/Sub/output_0"],
        )
        self.make_node(
            "Cast",
            ["/model/attn_mask_reformat/attn_mask_subgraph/Sub/output_0"],
            ["/model/attn_mask_reformat/attn_mask_subgraph/Sub/Cast/output_0"],
            to=TensorProto.INT32,
        )

        self.make_node(
            "Shape",
            ["attention_mask"],
            ["/model/attn_mask_reformat/attn_mask_subgraph/Shape/output_0"],
        )
        self.make_node(
            "Gather",
            ["/model/attn_mask_reformat/attn_mask_subgraph/Shape/output_0", const_1_scalar],
            ["/model/attn_mask_reformat/attn_mask_subgraph/Gather_1/output_0"],
            axis=0,
        )
        self.make_node(
            "Cast",
            ["/model/attn_mask_reformat/attn_mask_subgraph/Gather_1/output_0"],
            ["/model/attn_mask_reformat/attn_mask_subgraph/Gather/Cast/output_0"],
            to=TensorProto.INT32,
        )

    def build_conv_layer(self, layer_idx: int, hidden_state: str) -> str:
        prefix = f"/model/layers.{layer_idx}"
        weight_prefix = f"model.layers.{layer_idx}"
        L = self.config.conv_L_cache
        H = self.config.hidden_size
        residual = hidden_state

        normed = self.make_simple_layernorm(
            hidden_state,
            f"{weight_prefix}.operator_layernorm.weight",
            f"{prefix}/operator_layernorm",
            name="LayerNorm",
        )

        if self.use_q4:
            in_proj = self.make_matmul_nbits(
                normed,
                self.weights[f"{weight_prefix}.conv.in_proj.weight"].T,
                f"model_layers_{layer_idx}_conv_in_proj_MatMul_weight",
                f"{prefix}/conv/in_proj/MatMul/output_0",
            )
        else:
            in_proj = self.make_matmul(
                normed,
                f"{weight_prefix}.conv.in_proj.MatMul.weight",
                f"{prefix}/conv/in_proj/MatMul/output_0",
            )
        in_proj_t = self.make_node(
            "Transpose", [in_proj], [f"{prefix}/conv/Transpose_1/output_0"], perm=[0, 2, 1]
        )
        self.make_node(
            "Split",
            [in_proj_t, self.get_constant([H, H, H])],
            [
                f"{prefix}/conv/Split/output_0",
                f"{prefix}/conv/Split/output_1",
                f"{prefix}/conv/Split/output_2",
            ],
            axis=1,
        )

        Bx = self.make_mul(
            f"{prefix}/conv/Split/output_0",
            f"{prefix}/conv/Split/output_2",
            f"{prefix}/conv/Mul_1/output_0",
        )
        conv_input = self.make_node(
            "Concat", [f"past_conv.{layer_idx}", Bx], [f"{prefix}/conv/Conv_Input/output_0"], axis=2
        )
        conv_out_full = self.make_node(
            "Conv",
            [conv_input, f"{weight_prefix}.conv.conv.weight"],
            [f"{prefix}/conv/Conv/output_0"],
            kernel_shape=[L],
            group=H,
        )

        # Extract seq_len from LayerNorm output (shape [B, S, H]) at axis 1
        self.make_node("Shape", [normed], [f"{prefix}/conv/Shape/output_0"])
        self.make_node(
            "Gather",
            [f"{prefix}/conv/Shape/output_0", self.get_constant(1)],
            [f"{prefix}/conv/Gather_1/output_0"],
            axis=0,
        )

        # Slice last seq_len elements (community naming: Neg_Seq_Len, Unsqueeze_starts)
        neg_seq = self.make_mul(
            f"{prefix}/conv/Gather_1/output_0",
            self.get_constant(-1),
            f"{prefix}/conv/Neg_Seq_Len/output_0",
        )
        slice_start = self.make_unsqueeze(
            neg_seq, self.get_constant([0]), f"{prefix}/conv/Unsqueeze_starts/output_0"
        )
        self.make_slice(
            conv_out_full,
            slice_start,
            self.get_constant([np.iinfo(np.int64).max]),
            self.get_constant([2]),
            f"{prefix}/conv/Slice_Conv_Output/output_0",
        )

        # Cache update (Mul_2 before Slice_Cache in community order)
        y = self.make_mul(
            f"{prefix}/conv/Split/output_1",
            f"{prefix}/conv/Slice_Conv_Output/output_0",
            f"{prefix}/conv/Mul_2/output_0",
        )

        self.make_node(
            "Slice",
            [
                conv_input,
                self.get_constant([-L]),
                self.get_constant([np.iinfo(np.int64).max]),
                self.get_constant([2]),
            ],
            [f"present_conv.{layer_idx}"],
            name=f"{prefix}/conv/Slice_Cache",
        )

        y_t = self.make_node(
            "Transpose", [y], [f"{prefix}/conv/Transpose_2/output_0"], perm=[0, 2, 1]
        )

        if self.use_q4:
            out_proj = self.make_matmul_nbits(
                y_t,
                self.weights[f"{weight_prefix}.conv.out_proj.weight"].T,
                f"model_layers_{layer_idx}_conv_out_proj_MatMul_weight",
                f"{prefix}/conv/out_proj/MatMul/output_0",
            )
        else:
            out_proj = self.make_matmul(
                y_t,
                f"{weight_prefix}.conv.out_proj.MatMul.weight",
                f"{prefix}/conv/out_proj/MatMul/output_0",
            )

        hidden_state = self.make_add(residual, out_proj, f"{prefix}/Add_1/output_0")
        return self.build_ffn(layer_idx, hidden_state)

    def build_attention_layer(self, layer_idx: int, hidden_state: str) -> str:
        prefix = f"/model/layers.{layer_idx}"
        weight_prefix = f"model.layers.{layer_idx}"
        H = self.config.hidden_size
        nh = self.config.num_attention_heads
        nkv = self.config.num_key_value_heads
        hd = self.head_dim
        kv_hidden = nkv * hd
        residual = hidden_state

        normed = self.make_simple_layernorm(
            hidden_state,
            f"{weight_prefix}.operator_layernorm.weight",
            f"{prefix}/operator_layernorm",
            name="LayerNorm",
        )

        if self.use_q4:
            q = self.make_matmul_nbits(
                normed,
                self.weights[f"{weight_prefix}.self_attn.q_proj.weight"].T,
                f"model_layers_{layer_idx}_attn_q_proj_MatMul_weight",
                f"{prefix}/attn/q_proj/MatMul/output_0",
            )
            k = self.make_matmul_nbits(
                normed,
                self.weights[f"{weight_prefix}.self_attn.k_proj.weight"].T,
                f"model_layers_{layer_idx}_attn_k_proj_MatMul_weight",
                f"{prefix}/attn/k_proj/MatMul/output_0",
            )
            v = self.make_matmul_nbits(
                normed,
                self.weights[f"{weight_prefix}.self_attn.v_proj.weight"].T,
                f"model_layers_{layer_idx}_attn_v_proj_MatMul_weight",
                f"{prefix}/attn/v_proj/MatMul/output_0",
            )
        else:
            q = self.make_matmul(
                normed,
                f"{weight_prefix}.attn.q_proj.MatMul.weight",
                f"{prefix}/attn/q_proj/MatMul/output_0",
            )
            k = self.make_matmul(
                normed,
                f"{weight_prefix}.attn.k_proj.MatMul.weight",
                f"{prefix}/attn/k_proj/MatMul/output_0",
            )
            v = self.make_matmul(
                normed,
                f"{weight_prefix}.attn.v_proj.MatMul.weight",
                f"{prefix}/attn/v_proj/MatMul/output_0",
            )

        reshape_for_norm = self.get_constant([0, -1, hd])
        q_reshape_back = self.get_constant([0, -1, H])
        k_reshape_back = self.get_constant([0, -1, kv_hidden])

        q_for_norm = self.make_node(
            "Reshape", [q, reshape_for_norm], [f"{prefix}/attn/q_norm/Reshape_1/output_0"]
        )
        q_normed = self.make_simple_layernorm(
            q_for_norm,
            f"{weight_prefix}.attn.q_norm.layernorm.weight",
            f"{prefix}/attn/q_norm",
        )
        q_3d = self.make_node(
            "Reshape", [q_normed, q_reshape_back], [f"{prefix}/attn/q_norm/Reshape_2/output_0"]
        )

        k_for_norm = self.make_node(
            "Reshape", [k, reshape_for_norm], [f"{prefix}/attn/k_norm/Reshape_1/output_0"]
        )
        k_normed = self.make_simple_layernorm(
            k_for_norm,
            f"{weight_prefix}.attn.k_norm.layernorm.weight",
            f"{prefix}/attn/k_norm",
        )
        k_3d = self.make_node(
            "Reshape", [k_normed, k_reshape_back], [f"{prefix}/attn/k_norm/Reshape_2/output_0"]
        )

        scale = 1.0 / (hd**0.5)

        # Use community attn mask output names
        seqlens_k = "/model/attn_mask_reformat/attn_mask_subgraph/Sub/Cast/output_0"
        total_seq = "/model/attn_mask_reformat/attn_mask_subgraph/Gather/Cast/output_0"

        if self.use_integrated_rope:
            self.make_node(
                "GroupQueryAttention",
                [
                    q_3d,
                    k_3d,
                    v,
                    f"past_key_values.{layer_idx}.key",
                    f"past_key_values.{layer_idx}.value",
                    seqlens_k,
                    total_seq,
                    "cos_cache",
                    "sin_cache",
                ],
                [
                    f"{prefix}/attn/GroupQueryAttention/output_0",
                    f"present.{layer_idx}.key",
                    f"present.{layer_idx}.value",
                ],
                domain="com.microsoft",
                num_heads=nh,
                kv_num_heads=nkv,
                scale=scale,
                local_window_size=-1,
                softcap=0.0,
                do_rotary=1,
                rotary_interleaved=0,
            )
        else:
            rope_attrs = {
                "domain": "com.microsoft",
                "interleaved": 0,
                "num_heads": 0,
                "rotary_embedding_dim": 0,
            }
            q_rope = self.make_node(
                "RotaryEmbedding",
                [q_3d, "position_ids", "cos_cache", "sin_cache"],
                [f"{prefix}/attn/RotaryEmbedding/output_0"],
                **rope_attrs,
            )
            k_rope = self.make_node(
                "RotaryEmbedding",
                [k_3d, "position_ids", "cos_cache", "sin_cache"],
                [f"{prefix}/attn/RotaryEmbedding_1/output_0"],
                **rope_attrs,
            )

            self.make_node(
                "GroupQueryAttention",
                [
                    q_rope,
                    k_rope,
                    v,
                    f"past_key_values.{layer_idx}.key",
                    f"past_key_values.{layer_idx}.value",
                    seqlens_k,
                    total_seq,
                    "",
                    "",
                ],
                [
                    f"{prefix}/attn/GroupQueryAttention/output_0",
                    f"present.{layer_idx}.key",
                    f"present.{layer_idx}.value",
                ],
                domain="com.microsoft",
                num_heads=nh,
                kv_num_heads=nkv,
                scale=scale,
                local_window_size=-1,
                softcap=0.0,
                do_rotary=0,
                rotary_interleaved=0,
            )

        if self.use_q4:
            o_proj = self.make_matmul_nbits(
                f"{prefix}/attn/GroupQueryAttention/output_0",
                self.weights[f"{weight_prefix}.self_attn.out_proj.weight"].T,
                f"model_layers_{layer_idx}_attn_o_proj_MatMul_weight",
                f"{prefix}/attn/o_proj/MatMul/output_0",
            )
        else:
            o_proj = self.make_matmul(
                f"{prefix}/attn/GroupQueryAttention/output_0",
                f"{weight_prefix}.attn.o_proj.MatMul.weight",
                f"{prefix}/attn/o_proj/MatMul/output_0",
            )
        hidden_state = self.make_add(residual, o_proj, f"{prefix}/Add_1/output_0")
        return self.build_ffn(layer_idx, hidden_state)

    def build_ffn(self, layer_idx: int, hidden_state: str) -> str:
        """Build feed-forward block (dense MLP or sparse MoE/QMoE)."""
        if self.is_moe_layer(layer_idx):
            if self.use_qmoe:
                return self.build_qmoe(layer_idx, hidden_state)
            return self.build_moe(layer_idx, hidden_state)
        return self.build_dense_mlp(layer_idx, hidden_state)

    def build_dense_mlp(self, layer_idx: int, hidden_state: str) -> str:
        """Build dense MLP block (SwiGLU activation)."""
        prefix = f"/model/layers.{layer_idx}"
        weight_prefix = f"model.layers.{layer_idx}"
        residual = hidden_state

        normed = self.make_simple_layernorm(
            hidden_state,
            f"{weight_prefix}.ffn_layernorm.weight",
            f"{prefix}/ffn_layernorm",
            name="LayerNorm",
        )

        if self.use_q4:
            gate = self.make_matmul_nbits(
                normed,
                self.weights[f"{weight_prefix}.feed_forward.w1.weight"].T,
                f"model_layers_{layer_idx}_mlp_gate_proj_MatMul_weight",
                f"{prefix}/mlp/gate_proj/MatMul/output_0",
            )
            up = self.make_matmul_nbits(
                normed,
                self.weights[f"{weight_prefix}.feed_forward.w3.weight"].T,
                f"model_layers_{layer_idx}_mlp_up_proj_MatMul_weight",
                f"{prefix}/mlp/up_proj/MatMul/output_0",
            )
        else:
            gate = self.make_matmul(
                normed,
                f"{weight_prefix}.mlp.gate_proj.MatMul.weight",
                f"{prefix}/mlp/gate_proj/MatMul/output_0",
            )
            up = self.make_matmul(
                normed,
                f"{weight_prefix}.mlp.up_proj.MatMul.weight",
                f"{prefix}/mlp/up_proj/MatMul/output_0",
            )

        gate_silu = self.make_silu(gate, f"{prefix}/mlp/act_fn")
        gated = self.make_mul(gate_silu, up, f"{prefix}/mlp/Mul/output_0")

        if self.use_q4:
            down = self.make_matmul_nbits(
                gated,
                self.weights[f"{weight_prefix}.feed_forward.w2.weight"].T,
                f"model_layers_{layer_idx}_mlp_down_proj_MatMul_weight",
                f"{prefix}/mlp/down_proj/MatMul/output_0",
            )
        else:
            down = self.make_matmul(
                gated,
                f"{weight_prefix}.mlp.down_proj.MatMul.weight",
                f"{prefix}/mlp/down_proj/MatMul/output_0",
            )

        return self.make_add(residual, down, f"{prefix}/Add_2/output_0")

    def build_moe(self, layer_idx: int, hidden_state: str) -> str:
        """Build Sparse MoE block using com.microsoft:MoE operator.

        Router subgraph (following onnx-community pattern):
            hidden_state @ gate.weight → router_logits
                ↓
            sigmoid → routing_weights
                ↓
            routing_weights + expert_bias → scores_for_routing
                ↓
            TopK(k=4) → (values, indices)
                ↓
            GatherElements(routing_weights, indices) → selected_weights
                ↓
            Log(selected_weights) → log_weights
                ↓
            ScatterElements(neg_inf_matrix, indices, log_weights) → router_probs
                ↓
            Reshape to [-1, num_experts]

        MoE operator:
            - Input: hidden_state, router_probs, gate_up_proj, _, down_proj, _, _, _
            - Uses SwiGLU activation with fusion
            - Normalizes routing weights
        """
        prefix = f"/model/layers.{layer_idx}"
        weight_prefix = f"model.layers.{layer_idx}"
        num_experts = self.config.num_experts
        k = self.config.num_experts_per_tok
        residual = hidden_state

        # === FFN LayerNorm ===
        normed = self.make_simple_layernorm(
            hidden_state,
            f"{weight_prefix}.ffn_layernorm.weight",
            f"{prefix}/ffn_layernorm",
            name="LayerNorm",
        )

        # === Router subgraph ===
        router_logits = self.make_matmul(
            normed,
            f"{weight_prefix}.moe.router.MatMul.weight",
            f"{prefix}/moe/router/MatMul/output_0",
        )
        routing_weights = self.make_sigmoid(router_logits, f"{prefix}/moe/router/Sigmoid/output_0")

        if self.config.use_expert_bias:
            scores_for_routing = self.make_add(
                routing_weights,
                f"{weight_prefix}.moe.expert_bias",
                f"{prefix}/moe/router/Add/output_0",
            )
        else:
            scores_for_routing = routing_weights

        self.add_initializer(f"/model/constants/INT64/[{k}]", np.array([k], dtype=np.int64))
        self.make_node(
            "TopK",
            [scores_for_routing, f"/model/constants/INT64/[{k}]"],
            [f"{prefix}/moe/router/TopK/output_0", f"{prefix}/moe/router/TopK/output_1"],
        )
        self.make_node(
            "GatherElements",
            [routing_weights, f"{prefix}/moe/router/TopK/output_1"],
            [f"{prefix}/moe/router/Gather/output_0"],
            axis=-1,
        )
        self.make_node(
            "Log",
            [f"{prefix}/moe/router/Gather/output_0"],
            [f"{prefix}/moe/router/Log/output_0"],
        )

        # Negative infinity matrix for scatter (masks non-selected experts)
        mask_const_name = f"/model/constants/FLOAT/[{MASK_VALUE}]"
        self.add_initializer(mask_const_name, np.array([MASK_VALUE], dtype=np.float32))
        self.make_node(
            "Shape",
            [router_logits],
            [f"{prefix}/moe/router/Shape/output_0"],
        )
        self.make_node(
            "Expand",
            [mask_const_name, f"{prefix}/moe/router/Shape/output_0"],
            [f"{prefix}/moe/router/Expand/output_0"],
        )
        self.make_node(
            "ScatterElements",
            [
                f"{prefix}/moe/router/Expand/output_0",
                f"{prefix}/moe/router/TopK/output_1",
                f"{prefix}/moe/router/Log/output_0",
            ],
            [f"{prefix}/moe/router/Scatter/output_0"],
            axis=-1,
        )

        # Reshape to [batch * seq_len, num_experts] for MoE operator
        self.add_initializer(
            f"/model/constants/INT64/[-1, {num_experts}]",
            np.array([-1, num_experts], dtype=np.int64),
        )
        router_probs = self.make_node(
            "Reshape",
            [
                f"{prefix}/moe/router/Scatter/output_0",
                f"/model/constants/INT64/[-1, {num_experts}]",
            ],
            [f"{prefix}/moe/router/Reshape/output_0"],
        )

        # === MoE operator ===
        moe_out = self.make_node(
            "MoE",
            [
                normed,
                router_probs,
                f"{weight_prefix}.moe.experts.gate_up_proj.weight",
                "",
                f"{weight_prefix}.moe.experts.down_proj.weight",
                "",
                "",
                "",
            ],
            [f"{prefix}/moe/MoE/output_0"],
            domain="com.microsoft",
            activation_type="swiglu",
            k=k,
            normalize_routing_weights=1 if self.config.norm_topk_prob else 0,
            activation_alpha=1.0,
            activation_beta=0.0,
            swiglu_fusion=1,
            use_sparse_mixer=0,
        )

        return self.make_add(residual, moe_out, f"{prefix}/Add_2/output_0")

    def build_qmoe(self, layer_idx: int, hidden_state: str) -> str:
        """Build Sparse MoE block using com.microsoft:QMoE operator (quantized).

        Uses INT4 block quantization for expert weights with the same router
        subgraph as build_moe.

        QMoE Inputs (14 total):
            [0] hidden_state (after layernorm)
            [1] router_probs (after reshape)
            [2] gate_up_proj_weight_quant: UINT8 packed INT4
            [3] gate_up_proj_weight_scales: FP32
            [4] gate_up_proj_bias (empty)
            [5] down_proj_weight_quant: UINT8 packed INT4
            [6] down_proj_weight_scales: FP32
            [7] down_proj_bias (empty)
            [8] fc1_experts_bias (empty)
            [9] fc2_experts_bias (empty)
            [10] (empty)
            [11] gate_up_proj_weight_zp: UINT8 packed
            [12] down_proj_weight_zp: UINT8 packed
            [13] (empty)
        """
        prefix = f"/model/layers.{layer_idx}"
        weight_prefix = f"model.layers.{layer_idx}"
        num_experts = self.config.num_experts
        k = self.config.num_experts_per_tok
        residual = hidden_state

        # === FFN LayerNorm ===
        normed = self.make_simple_layernorm(
            hidden_state,
            f"{weight_prefix}.ffn_layernorm.weight",
            f"{prefix}/ffn_layernorm",
            name="LayerNorm",
        )

        # === Router subgraph ===
        # Router kept as FP32 MatMul (not quantized) for WebGPU compatibility
        # and to preserve routing quality; q4f16 conversion handles FP32→FP16
        router_logits = self.make_matmul(
            normed,
            f"{weight_prefix}.moe.router.MatMul.weight",
            f"{prefix}/moe/router/MatMul/output_0",
        )

        routing_weights = self.make_sigmoid(router_logits, f"{prefix}/moe/router/Sigmoid/output_0")

        if self.config.use_expert_bias:
            scores_for_routing = self.make_add(
                routing_weights,
                f"{weight_prefix}.moe.expert_bias",
                f"{prefix}/moe/router/Add/output_0",
            )
        else:
            scores_for_routing = routing_weights

        self.add_initializer(f"/model/constants/INT64/[{k}]", np.array([k], dtype=np.int64))
        self.make_node(
            "TopK",
            [scores_for_routing, f"/model/constants/INT64/[{k}]"],
            [f"{prefix}/moe/router/TopK/output_0", f"{prefix}/moe/router/TopK/output_1"],
        )

        self.make_node(
            "GatherElements",
            [routing_weights, f"{prefix}/moe/router/TopK/output_1"],
            [f"{prefix}/moe/router/Gather/output_0"],
            axis=-1,
        )

        self.make_node(
            "Log",
            [f"{prefix}/moe/router/Gather/output_0"],
            [f"{prefix}/moe/router/Log/output_0"],
        )

        mask_const_name = f"/model/constants/FLOAT/[{MASK_VALUE}]"
        self.add_initializer(mask_const_name, np.array([MASK_VALUE], dtype=np.float32))
        self.make_node(
            "Shape",
            [router_logits],
            [f"{prefix}/moe/router/Shape/output_0"],
        )
        self.make_node(
            "Expand",
            [mask_const_name, f"{prefix}/moe/router/Shape/output_0"],
            [f"{prefix}/moe/router/Expand/output_0"],
        )

        self.make_node(
            "ScatterElements",
            [
                f"{prefix}/moe/router/Expand/output_0",
                f"{prefix}/moe/router/TopK/output_1",
                f"{prefix}/moe/router/Log/output_0",
            ],
            [f"{prefix}/moe/router/Scatter/output_0"],
            axis=-1,
        )

        self.add_initializer(
            f"/model/constants/INT64/[-1, {num_experts}]",
            np.array([-1, num_experts], dtype=np.int64),
        )
        router_probs = self.make_node(
            "Reshape",
            [
                f"{prefix}/moe/router/Scatter/output_0",
                f"/model/constants/INT64/[-1, {num_experts}]",
            ],
            [f"{prefix}/moe/router/Reshape/output_0"],
        )

        # === QMoE operator ===
        prefix_underscore = weight_prefix.replace(".", "_")
        qmoe_out = self.make_node(
            "QMoE",
            [
                normed,  # [0] hidden_state
                router_probs,  # [1] router_probs
                f"{prefix_underscore}_moe_experts_gate_up_proj_weight_quant",  # [2] gate_up quant
                f"{prefix_underscore}_moe_experts_gate_up_proj_weight_scales",  # [3] gate_up scales
                "",  # [4] gate_up bias (empty)
                f"{prefix_underscore}_moe_experts_down_proj_weight_quant",  # [5] down quant
                f"{prefix_underscore}_moe_experts_down_proj_weight_scales",  # [6] down scales
                "",  # [7] down bias (empty)
                "",  # [8] fc1_experts_bias (empty)
                "",  # [9] fc2_experts_bias (empty)
                "",  # [10] empty
                f"{prefix_underscore}_moe_experts_gate_up_proj_weight_zp",  # [11] gate_up zp
                f"{prefix_underscore}_moe_experts_down_proj_weight_zp",  # [12] down zp
                "",  # [13] empty
            ],
            [f"{prefix}/moe/QMoE/output_0"],
            domain="com.microsoft",
            activation_type="swiglu",
            block_size=self.qmoe_block_size,
            expert_weight_bits=INT4_BITS,
            k=k,
            normalize_routing_weights=1 if self.config.norm_topk_prob else 0,
            swiglu_fusion=1,
        )

        return self.make_add(residual, qmoe_out, f"{prefix}/Add_2/output_0")

    def build_lm_head(self, hidden_state: str) -> str:
        # Community naming: model.layers.{num_layers}.final_norm_layernorm.weight
        num_layers = self.config.num_hidden_layers
        final_norm_weight = f"model.layers.{num_layers}.final_norm_layernorm.weight"
        final_norm_output = (
            f"/model/layers.{num_layers}/final_norm_layernorm/SkipLayerNorm/output_0"
        )

        self.add_initializer(final_norm_weight, self.weights["model.embedding_norm.weight"])
        normed = self.make_skip_layernorm(
            hidden_state,
            hidden_state,
            final_norm_weight,
            final_norm_output,
        )

        if self.use_q4:
            # Q4: Use MatMulNBits for lm_head with shared embedding weights
            # Community approach: reuse embedding quant (reshaped) and share scales/zp
            # This saves ~19MB by not duplicating scales and zero points

            # Get the embedding quant tensor and reshape for MatMulNBits
            # Embedding quant shape: [vocab_size, K/2] -> [vocab_size, n_blocks, block_size/2]
            embed_quant_name = "model_embed_tokens_weight_quant"
            embed_quant = None
            for init in self.initializers:
                if init.name == embed_quant_name:
                    embed_quant = onnx.numpy_helper.to_array(init)
                    break

            if embed_quant is None:
                raise ValueError("Embedding quant not found - build_embedding must be called first")

            vocab_size = embed_quant.shape[0]
            K = self.config.hidden_size
            n_blocks = K // self.qmoe_block_size

            # Reshape to 3D format for MatMulNBits: [N, n_blocks, block_size/2]
            embed_quant_matmul = embed_quant.reshape(
                vocab_size, n_blocks, self.qmoe_block_size // 2
            )
            self.add_initializer(
                "model_embed_tokens_weight_quant_matmul", embed_quant_matmul, dtype=np.uint8
            )

            # Reuse existing scales and zero points from embedding
            return self.make_node(
                "MatMulNBits",
                [
                    normed,
                    "model_embed_tokens_weight_quant_matmul",
                    "model_embed_tokens_weight_scales",
                    "model_embed_tokens_weight_zp",
                ],
                ["logits"],
                domain="com.microsoft",
                K=K,
                N=vocab_size,
                bits=4,
                block_size=self.qmoe_block_size,
            )
        else:
            self.make_node(
                "Transpose",
                ["model.embed_tokens.weight"],
                ["/lm_head/Transpose/output_0"],
                perm=[1, 0],
            )
            return self.make_node(
                "MatMul",
                [normed, "/lm_head/Transpose/output_0"],
                ["logits"],
                name="/lm_head/MatMul",
            )

    def load_weights(self, model_path: str):
        import torch
        from transformers import AutoModelForCausalLM

        logger.info(f"Loading weights from {model_path}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float32, trust_remote_code=True
        )

        for name, param in model.named_parameters():
            self.weights[name] = param.detach().numpy()
            logger.debug(f"Loaded: {name} {param.shape}")

        # Also load buffers (like expert_bias)
        for name, buf in model.named_buffers():
            self.weights[name] = buf.detach().numpy()
            logger.debug(f"Loaded buffer: {name} {buf.shape}")

        del model
        logger.info(f"Loaded {len(self.weights)} weights")

    def build(self, model_path: str) -> onnx.ModelProto:
        logger.info("Building LFM2-MoE ONNX model...")

        self.load_weights(model_path)

        self.build_inputs()
        self.build_outputs()
        self.build_rope_cache()
        self.build_attention_mask_subgraph()

        hidden_state = self.build_embedding()

        for layer_idx in range(self.config.num_hidden_layers):
            layer_type = self.config.layer_types[layer_idx]
            is_moe = self.is_moe_layer(layer_idx)
            layer_desc = f"{layer_type}" + (" + MoE" if is_moe else " + MLP")
            logger.info(f"Building layer {layer_idx} ({layer_desc})...")

            self.prepare_layer_weights(layer_idx, layer_type)

            if layer_type == "conv":
                hidden_state = self.build_conv_layer(layer_idx, hidden_state)
            else:
                hidden_state = self.build_attention_layer(layer_idx, hidden_state)

        self.build_lm_head(hidden_state)

        self.build_value_info()

        model = self.build_graph("lfm2_moe")
        logger.info(f"Model built: {len(self.nodes)} nodes, {len(self.value_info)} value_info")
        return model

    def build_value_info(self):
        """Build ValueInfo entries for weights and intermediate tensors."""
        H = self.config.hidden_size
        nkv = self.config.num_key_value_heads
        hd = self.head_dim
        kv_hidden = nkv * hd
        intermediate = self.config.intermediate_size
        num_layers = self.config.num_hidden_layers
        mask_prefix = "/model/attn_mask_reformat/attn_mask_subgraph"

        # === Weight shapes (from initializers) ===
        for init in self.initializers:
            if init.name.startswith("/model/constants/"):
                continue
            shape = list(init.dims)
            dtype = init.data_type
            self.add_value_info(init.name, dtype, shape)

        # === Attention mask subgraph outputs ===
        self.add_value_info(f"{mask_prefix}/ReduceSum/output_0", TensorProto.INT64, ["batch_size"])
        self.add_value_info(f"{mask_prefix}/Sub/output_0", TensorProto.INT64, ["batch_size"])
        self.add_value_info(f"{mask_prefix}/Sub/Cast/output_0", TensorProto.INT32, ["batch_size"])
        self.add_value_info(f"{mask_prefix}/Shape/output_0", TensorProto.INT64, [2])
        self.add_value_info(f"{mask_prefix}/Gather/output_0", TensorProto.INT64, [])
        self.add_value_info(f"{mask_prefix}/Gather/Cast/output_0", TensorProto.INT32, [])

        # === Embedding output ===
        self.add_value_info(
            "/model/embed_tokens/Gather/output_0",
            TensorProto.FLOAT,
            ["batch_size", "sequence_length", H],
        )

        # === Per-layer outputs ===
        for layer_idx in range(num_layers):
            prefix = f"/model/layers.{layer_idx}"
            layer_type = self.config.layer_types[layer_idx]
            is_moe = self.is_moe_layer(layer_idx)

            # Operator layernorm output
            self.add_value_info(
                f"{prefix}/operator_layernorm/LayerNorm/output_0",
                TensorProto.FLOAT,
                ["batch_size", "sequence_length", H],
            )

            if layer_type == "conv":
                # Conv layer outputs
                self.add_value_info(
                    f"{prefix}/conv/in_proj/MatMul/output_0",
                    TensorProto.FLOAT,
                    ["batch_size", "sequence_length", 3 * H],
                )
                self.add_value_info(
                    f"{prefix}/conv/Transpose_1/output_0",
                    TensorProto.FLOAT,
                    ["batch_size", 3 * H, "sequence_length"],
                )
                self.add_value_info(
                    f"{prefix}/conv/Split/output_0",
                    TensorProto.FLOAT,
                    ["batch_size", H, "sequence_length"],
                )
                self.add_value_info(
                    f"{prefix}/conv/Split/output_1",
                    TensorProto.FLOAT,
                    ["batch_size", H, "sequence_length"],
                )
                self.add_value_info(
                    f"{prefix}/conv/Split/output_2",
                    TensorProto.FLOAT,
                    ["batch_size", H, "sequence_length"],
                )
                self.add_value_info(
                    f"{prefix}/conv/Mul_1/output_0",
                    TensorProto.FLOAT,
                    ["batch_size", H, "sequence_length"],
                )
                self.add_value_info(
                    f"{prefix}/conv/Conv_Input/output_0",
                    TensorProto.FLOAT,
                    ["batch_size", H, "sequence_length_plus_cache"],
                )
                self.add_value_info(f"{prefix}/conv/Shape/output_0", TensorProto.INT64, [3])
                self.add_value_info(f"{prefix}/conv/Gather_1/output_0", TensorProto.INT64, [])
                self.add_value_info(f"{prefix}/conv/Neg_Seq_Len/output_0", TensorProto.INT64, [])
                self.add_value_info(
                    f"{prefix}/conv/Unsqueeze_starts/output_0", TensorProto.INT64, [1]
                )
                self.add_value_info(
                    f"{prefix}/conv/Mul_2/output_0",
                    TensorProto.FLOAT,
                    ["batch_size", H, "sequence_length"],
                )
                self.add_value_info(
                    f"{prefix}/conv/Transpose_2/output_0",
                    TensorProto.FLOAT,
                    ["batch_size", "sequence_length", H],
                )
                self.add_value_info(
                    f"{prefix}/conv/out_proj/MatMul/output_0",
                    TensorProto.FLOAT,
                    ["batch_size", "sequence_length", H],
                )
            else:
                # Attention layer outputs
                self.add_value_info(
                    f"{prefix}/attn/q_proj/MatMul/output_0",
                    TensorProto.FLOAT,
                    ["batch_size", "sequence_length", H],
                )
                self.add_value_info(
                    f"{prefix}/attn/k_proj/MatMul/output_0",
                    TensorProto.FLOAT,
                    ["batch_size", "sequence_length", kv_hidden],
                )
                self.add_value_info(
                    f"{prefix}/attn/v_proj/MatMul/output_0",
                    TensorProto.FLOAT,
                    ["batch_size", "sequence_length", kv_hidden],
                )
                self.add_value_info(
                    f"{prefix}/attn/out_proj/MatMul/output_0",
                    TensorProto.FLOAT,
                    ["batch_size", "sequence_length", H],
                )

            # Residual add after conv/attention
            self.add_value_info(
                f"{prefix}/Add_1/output_0",
                TensorProto.FLOAT,
                ["batch_size", "sequence_length", H],
            )

            # FFN layernorm
            self.add_value_info(
                f"{prefix}/ffn_layernorm/LayerNorm/output_0",
                TensorProto.FLOAT,
                ["batch_size", "sequence_length", H],
            )

            if is_moe:
                # MoE outputs
                self.add_value_info(
                    f"{prefix}/moe/MoE/output_0",
                    TensorProto.FLOAT,
                    ["batch_size", "sequence_length", H],
                )
            else:
                # Dense MLP outputs
                mlp_inter = intermediate
                self.add_value_info(
                    f"{prefix}/mlp/gate_proj/MatMul/output_0",
                    TensorProto.FLOAT,
                    ["batch_size", "sequence_length", mlp_inter],
                )
                self.add_value_info(
                    f"{prefix}/mlp/up_proj/MatMul/output_0",
                    TensorProto.FLOAT,
                    ["batch_size", "sequence_length", mlp_inter],
                )
                self.add_value_info(
                    f"{prefix}/mlp/down_proj/MatMul/output_0",
                    TensorProto.FLOAT,
                    ["batch_size", "sequence_length", H],
                )

            # Residual add after FFN/MoE
            self.add_value_info(
                f"{prefix}/Add_2/output_0",
                TensorProto.FLOAT,
                ["batch_size", "sequence_length", H],
            )

        # === Final norm and LM head ===
        self.add_value_info(
            f"/model/layers.{num_layers}/final_norm_layernorm/LayerNorm/output_0",
            TensorProto.FLOAT,
            ["batch_size", "sequence_length", H],
        )
