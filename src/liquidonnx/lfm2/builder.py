"""
LFM2 Builder for ONNX export.

This builder follows the onnxruntime-genai builder pattern but uses
the stable onnx.helper API. It can be ported to onnx_ir when that API stabilizes.

The builder creates an optimized ONNX graph with fused operators:
- SimplifiedLayerNormalization (com.microsoft)
- RotaryEmbedding (com.microsoft)
- GroupQueryAttention (com.microsoft)

Architecture Overview:
    input_ids → Embedding → [Conv/Attention Layers] → LayerNorm → LM Head → logits

Layer Types:
    - Conv layers: Gated short convolution with depthwise conv1d
    - Attention layers: Grouped Query Attention with RoPE and Q/K normalization

"""

import logging
from dataclasses import dataclass

import numpy as np
import onnx
from onnx import TensorProto, helper

from liquidonnx.builder_base import ONNXBuilderBase

logger = logging.getLogger(__name__)

# === INT4 Block Quantization ===

INT4_BITS = 4
INT4_MAX = (1 << INT4_BITS) - 1  # 15, max value for unsigned 4-bit
DEFAULT_BLOCK_SIZE = 32
SCALE_EPS = 1e-10


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

    pad_K = n_blocks * block_size
    if pad_K != K:
        pad_shape = list(weight.shape)
        pad_shape[-1] = pad_K - K
        weight = np.concatenate([weight, np.zeros(pad_shape, dtype=weight.dtype)], axis=-1)

    weight_blocked = weight.reshape(*batch_dims, n_blocks, block_size)

    w_min = weight_blocked.min(axis=-1, keepdims=True)
    w_max = weight_blocked.max(axis=-1, keepdims=True)

    scale = (w_max - w_min) / float(INT4_MAX)
    scale = np.where(scale < SCALE_EPS, 1.0, scale)
    zero_point = np.round(-w_min / scale).clip(0, INT4_MAX).astype(np.uint8)

    # q = round(w/s + zp) to match community
    quant = np.round(weight_blocked / scale + zero_point).clip(0, INT4_MAX).astype(np.uint8)

    # Pack two INT4 values into one UINT8 (low nibble first)
    quant_packed = quant[..., 0::2] | (quant[..., 1::2] << 4)

    scales = scale.squeeze(-1).astype(np.float32)

    # Pack zero points
    zero_point = zero_point.squeeze(-1)
    if n_blocks % 2 == 1:
        zp_shape = list(zero_point.shape)
        zp_shape[-1] = 1
        zero_point = np.concatenate([zero_point, np.zeros(zp_shape, dtype=np.uint8)], axis=-1)
    zp_packed = zero_point[..., 0::2] | (zero_point[..., 1::2] << 4)

    quant_final = quant_packed.reshape(*batch_dims, -1)

    return quant_final, scales, zp_packed


@dataclass
class LFM2Config:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int
    layer_types: list[str]
    intermediate_size: int | None = None  # MLP intermediate size, defaults to H * 9 // 2
    conv_L_cache: int = 3
    max_position_embeddings: int = 128000
    norm_eps: float = 1e-5
    rope_theta: float = 1000000.0

    def __post_init__(self):
        if self.intermediate_size is None:
            self.intermediate_size = self.hidden_size * 9 // 2

    @classmethod
    def from_hf_config(cls, config) -> "LFM2Config":
        # Compute intermediate_size using same logic as PyTorch model
        intermediate_size = getattr(config, "intermediate_size", None)
        if intermediate_size is not None and getattr(config, "block_auto_adjust_ff_dim", False):
            intermediate_size = int(2 * intermediate_size / 3)
            multiplier = getattr(config, "block_ffn_dim_multiplier", None)
            if multiplier is not None:
                intermediate_size = int(multiplier * intermediate_size)
                multiple_of = getattr(config, "block_multiple_of", 256)
                intermediate_size = multiple_of * (
                    (intermediate_size + multiple_of - 1) // multiple_of
                )

        return cls(
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            vocab_size=config.vocab_size,
            layer_types=config.layer_types,
            intermediate_size=intermediate_size,
            conv_L_cache=getattr(config, "conv_L_cache", 3),
            max_position_embeddings=config.max_position_embeddings,
            norm_eps=getattr(config, "norm_eps", 1e-5),
            rope_theta=getattr(config, "rope_theta", 1000000.0),
        )


class LFM2Builder(ONNXBuilderBase):
    """
    Builds an optimized ONNX graph with:
    - Conv/SSM layers with gating
    - Full attention layers with GQA
    - Fused Microsoft operators (SimplifiedLayerNormalization, RotaryEmbedding, GroupQueryAttention)
    """

    def __init__(
        self,
        config: LFM2Config,
        use_integrated_rope: bool = False,
        vl_naming: bool = False,
        use_q4: bool = False,
        q4_block_size: int = DEFAULT_BLOCK_SIZE,
    ):
        """
        Args:
            config: Model configuration
            use_integrated_rope: Use RoPE integrated in GroupQueryAttention (do_rotary=1)
                instead of separate RotaryEmbedding ops. May improve numerical precision.
            vl_naming: Use VL-style node naming (Shape, Gather_1) instead of
                LFM2-style (Shape_for_slice, Gather_for_slice). Community VL and LFM2
                models use different conventions.
            use_q4: Use INT4 quantized embedding (GatherBlockQuantized) and lm_head
                (MatMulNBits). Other MatMul layers are left as FP32 for post-export
                quantization.
            q4_block_size: Block size for INT4 quantization (default: 32).
        """
        super().__init__()
        self.config = config
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.use_integrated_rope = use_integrated_rope
        self.vl_naming = vl_naming
        self.use_q4 = use_q4
        self.q4_block_size = q4_block_size

        # Categorize layers
        self.conv_indices = [i for i, t in enumerate(config.layer_types) if t == "conv"]
        self.attn_indices = [i for i, t in enumerate(config.layer_types) if t == "full_attention"]

    # === Q4 Quantization Methods ===

    def _quantize_for_matmul_nbits(
        self, weight: np.ndarray, name: str
    ) -> tuple[str, str, str, int, int]:
        """Quantize weight for MatMulNBits operator.

        Args:
            weight: FP32 weight tensor of shape [K, N] (already transposed for MatMul)
            name: Base name for initializers

        Returns:
            Tuple of (quant_name, scales_name, zp_name, K, N)
        """
        K, N = weight.shape
        block_size = self.q4_block_size

        weight_t = weight.T  # [N, K]
        quant, scales, zp = quantize_int4_block(weight_t, block_size)

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
            block_size=self.q4_block_size,
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
        """
        block_size = self.q4_block_size

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

    def make_simple_layernorm(
        self, input_name: str, weight_name: str, path: str, name: str = None
    ) -> str:
        """Create SimplifiedLayerNormalization node (no bias).

        Args:
            input_name: Input tensor
            weight_name: Scale weight
            path: Logical path (e.g., "/model/layers.0/input_layernorm")
            name: Override name in path (default: "SimplifiedLayerNormalization")
        """
        return self.make_layernorm(
            input_name, weight_name, None, path, epsilon=self.config.norm_eps, name=name
        )

    def make_skip_layernorm(
        self, input_name: str, skip_name: str, weight_name: str, output_name: str, name: str = None
    ) -> str:
        """Create SkipSimplifiedLayerNormalization node (fused skip + layernorm)."""
        return self.make_node(
            "SkipSimplifiedLayerNormalization",
            inputs=[input_name, skip_name, weight_name],
            outputs=[output_name],
            name=name,
            domain="com.microsoft",
            epsilon=self.config.norm_eps,
        )

    def prepare_layer_weights(self, layer_idx: int, layer_type: str):
        """Prepare and register weights for a layer.

        Handles weight transposition and naming for MatMul operations.
        Uses community naming conventions:
        - operator_norm -> operator_layernorm
        - ffn_norm -> ffn_layernorm
        - feed_forward.w1/w2/w3 -> mlp.gate_proj/down_proj/up_proj.MatMul
        - conv.weight -> conv.conv.weight
        - self_attn -> attn, with .MatMul suffix
        """
        prefix = f"model.layers.{layer_idx}"

        # LayerNorm weights (community naming)
        self.add_initializer(
            f"{prefix}.operator_layernorm.weight", self.weights[f"{prefix}.operator_norm.weight"]
        )
        self.add_initializer(
            f"{prefix}.ffn_layernorm.weight", self.weights[f"{prefix}.ffn_norm.weight"]
        )

        # MLP weights (transposed for MatMul, community naming)
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
            self.add_initializer(
                f"{prefix}.conv.in_proj.MatMul.weight",
                self.weights[f"{prefix}.conv.in_proj.weight"].T,
            )
            self.add_initializer(
                f"{prefix}.conv.conv.weight", self.weights[f"{prefix}.conv.conv.weight"]
            )
            self.add_initializer(
                f"{prefix}.conv.out_proj.MatMul.weight",
                self.weights[f"{prefix}.conv.out_proj.weight"].T,
            )
        else:
            # Attention weights (transposed for MatMul, community naming)
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
                f"{prefix}.attn.q_norm.layernorm.weight",
                self.weights[f"{prefix}.self_attn.q_layernorm.weight"],
            )
            self.add_initializer(
                f"{prefix}.attn.k_norm.layernorm.weight",
                self.weights[f"{prefix}.self_attn.k_layernorm.weight"],
            )
            self.add_initializer(
                f"{prefix}.attn.o_proj.MatMul.weight",
                self.weights[f"{prefix}.self_attn.out_proj.weight"].T,
            )

    def build_inputs(self):
        # input_ids
        self.inputs.append(
            helper.make_tensor_value_info(
                "input_ids", TensorProto.INT64, ["batch_size", "sequence_length"]
            )
        )

        # attention_mask
        self.inputs.append(
            helper.make_tensor_value_info(
                "attention_mask", TensorProto.INT64, ["batch_size", "total_sequence_length"]
            )
        )

        # position_ids
        self.inputs.append(
            helper.make_tensor_value_info(
                "position_ids", TensorProto.INT64, ["batch_size", "sequence_length"]
            )
        )

        # Interleave cache inputs by layer index (community convention)
        conv_set = set(self.conv_indices)
        attn_set = set(self.attn_indices)
        for idx in range(self.config.num_hidden_layers):
            if idx in conv_set:
                self.inputs.append(
                    helper.make_tensor_value_info(
                        f"past_conv.{idx}",
                        TensorProto.FLOAT,
                        ["batch_size", self.config.hidden_size, self.config.conv_L_cache],
                    )
                )
            elif idx in attn_set:
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
        # Logits
        self.outputs.append(
            helper.make_tensor_value_info(
                "logits",
                TensorProto.FLOAT,
                ["batch_size", "sequence_length", self.config.vocab_size],
            )
        )

        # Interleave cache outputs by layer index (community convention)
        conv_set = set(self.conv_indices)
        attn_set = set(self.attn_indices)
        for idx in range(self.config.num_hidden_layers):
            if idx in conv_set:
                self.outputs.append(
                    helper.make_tensor_value_info(
                        f"present_conv.{idx}",
                        TensorProto.FLOAT,
                        ["batch_size", self.config.hidden_size, self.config.conv_L_cache],
                    )
                )
            elif idx in attn_set:
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
            return self.make_gather_block_quantized(
                embed_weight,
                "input_ids",
                "model_embed_tokens_weight",
                "/model/embed_tokens/GatherBlockQuantized/output_0",
            )

        self.add_initializer("model.embed_tokens.weight", embed_weight)
        return self.make_node(
            "Gather",
            ["model.embed_tokens.weight", "input_ids"],
            ["/model/embed_tokens/Gather/output_0"],
            axis=0,
        )

    def build_rope_cache(self):
        head_dim = self.head_dim
        max_seq = self.config.max_position_embeddings
        theta = self.config.rope_theta

        inv_freq = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
        t = np.arange(max_seq, dtype=np.float32)
        freqs = np.outer(t, inv_freq)  # [max_seq, head_dim/2]

        # Note: Do NOT concatenate - RotaryEmbedding expects [max_seq, head_dim/2]
        self.add_initializer("cos_cache", np.cos(freqs).astype(np.float32))
        self.add_initializer("sin_cache", np.sin(freqs).astype(np.float32))

    def build_attention_mask_subgraph(self):
        """Build attention mask preprocessing for GroupQueryAttention.

        The -1 offset in seqlens_k is intentional for KV cache semantics:
        - seqlens_k represents the number of past tokens already in the KV cache
        - During generation, attention_mask is all-ones with length cur_len
        - sum(attention_mask) = cur_len (total tokens including current)
        - seqlens_k = cur_len - 1 (past tokens, excluding current being processed)

        Example for a 10-token prompt generating 2 tokens:
        - Prefill:   cur_len=10, seqlens_k=9  (processing 10 tokens, 0 cached before)
        - Decode 1:  cur_len=11, seqlens_k=10 (processing 1 token, 10 cached)
        - Decode 2:  cur_len=12, seqlens_k=11 (processing 1 token, 11 cached)

        Note: This assumes attention_mask is always all-ones (no padding).
        If padding were introduced, this calculation would be incorrect.
        """
        # Community path prefix for attn_mask preprocessing
        mask_prefix = "/model/attn_mask_reformat/attn_mask_subgraph"

        # Use community constant naming via get_constant
        const_1_arr = self.get_constant([1])  # /model/constants/INT64/[1]/output_0
        const_1_scalar = self.get_constant(1)  # /model/constants/INT64/1/output_0

        # seqlens_k = sum of attention_mask per batch - 1 (see docstring for rationale)
        self.make_node(
            "ReduceSum",
            ["attention_mask", const_1_arr],
            [f"{mask_prefix}/ReduceSum/output_0"],
            keepdims=0,
        )
        self.make_node(
            "Sub",
            [f"{mask_prefix}/ReduceSum/output_0", const_1_arr],
            [f"{mask_prefix}/Sub/output_0"],
        )
        # Community naming: Sub/Cast instead of seqlens_k
        self.make_node(
            "Cast",
            [f"{mask_prefix}/Sub/output_0"],
            [f"{mask_prefix}/Sub/Cast/output_0"],
            to=TensorProto.INT32,
        )

        # total_seq_len = shape[1] of attention_mask
        gather_name = "Gather_1" if self.vl_naming else "Gather"
        self.make_node("Shape", ["attention_mask"], [f"{mask_prefix}/Shape/output_0"])
        self.make_node(
            "Gather",
            [f"{mask_prefix}/Shape/output_0", const_1_scalar],
            [f"{mask_prefix}/{gather_name}/output_0"],
            axis=0,
        )
        self.make_node(
            "Cast",
            [f"{mask_prefix}/{gather_name}/output_0"],
            [f"{mask_prefix}/{gather_name}/Cast/output_0"],
            to=TensorProto.INT32,
        )

    def build_conv_layer(self, layer_idx: int, hidden_state: str) -> str:
        """Build a conv/SSM layer.

        Graph structure (matches PyTorch Lfm2ShortConv):
            hidden_state
                ↓
            LayerNorm (operator_layernorm)
                ↓
            Linear (in_proj) → [B, S, 3H]
                ↓
            Transpose → [B, 3H, S]
                ↓
            Split → B[B,H,S], C[B,H,S], x[B,H,S]
                ↓
            Bx = B * x  (gating, no sigmoid)
                ↓
            Concat(past_cache, Bx) → [B, H, L+S]
                ↓
            Conv1D (depthwise, kernel=3, groups=H)
                ↓
            Slice (last S elements) → conv_out
                ↓
            y = C * conv_out  (output gating)
                ↓
            Transpose → [B, S, H]
                ↓
            Linear (out_proj)
                ↓
            Add (residual)
                ↓
            MLP block

        Cache: last L elements of concat input → present_conv.{layer_idx}
        """
        prefix = f"/model/layers.{layer_idx}"
        weight_prefix = f"model.layers.{layer_idx}"
        L = self.config.conv_L_cache
        H = self.config.hidden_size
        residual = hidden_state

        # === LayerNorm (community naming: LayerNorm suffix) ===
        normed = self.make_simple_layernorm(
            hidden_state,
            f"{weight_prefix}.operator_layernorm.weight",
            f"{prefix}/operator_layernorm",
            name="LayerNorm",
        )

        # === In projection + Split ===
        # in_proj: [B, S, H] → [B, S, 3H]
        in_proj = self.make_matmul(
            normed,
            f"{weight_prefix}.conv.in_proj.MatMul.weight",
            f"{prefix}/conv/in_proj/MatMul/output_0",
        )
        # Transpose: [B, S, 3H] → [B, 3H, S] (community: Transpose_1)
        in_proj_t = self.make_node(
            "Transpose", [in_proj], [f"{prefix}/conv/Transpose_1/output_0"], perm=[0, 2, 1]
        )
        # Split into B, C, x (each [B, H, S]) using shared constant
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

        # === Gated convolution (community naming: Mul_1) ===
        # Bx = B * x (input gating)
        Bx = self.make_mul(
            f"{prefix}/conv/Split/output_0",
            f"{prefix}/conv/Split/output_2",
            f"{prefix}/conv/Mul_1/output_0",
        )
        # Concat with cache: [B, H, L] + [B, H, S] → [B, H, L+S] (community: Conv_Input)
        conv_input = self.make_node(
            "Concat", [f"past_conv.{layer_idx}", Bx], [f"{prefix}/conv/Conv_Input/output_0"], axis=2
        )
        # Depthwise Conv1D (kernel=3, community naming)
        conv_out_full = self.make_node(
            "Conv",
            [conv_input, f"{weight_prefix}.conv.conv.weight"],
            [f"{prefix}/conv/Conv/output_0"],
            kernel_shape=[L],
            group=H,
        )

        # === Dynamic slice ===
        # Get sequence length from LayerNorm output shape (axis 1)
        # VL vs LFM2 community models use different naming
        shape_name = "Shape" if self.vl_naming else "Shape_for_slice"
        gather_name = "Gather_1" if self.vl_naming else "Gather_for_slice"
        self.make_node("Shape", [normed], [f"{prefix}/conv/{shape_name}/output_0"])
        seq_len = self.make_node(
            "Gather",
            [f"{prefix}/conv/{shape_name}/output_0", self.get_constant(1)],
            [f"{prefix}/conv/{gather_name}/output_0"],
            axis=0,
        )
        # Negate seq_len for slice start (community: Neg_Seq_Len)
        neg_seq = self.make_mul(
            seq_len, self.get_constant(-1), f"{prefix}/conv/Neg_Seq_Len/output_0"
        )
        # Unsqueeze for slice (community: Unsqueeze_starts)
        slice_start = self.make_unsqueeze(
            neg_seq, self.get_constant([0]), f"{prefix}/conv/Unsqueeze_starts/output_0"
        )
        # Slice last S elements (community: Slice_Conv_Output)
        conv_sliced = self.make_slice(
            conv_out_full,
            slice_start,
            self.get_constant([np.iinfo(np.int64).max]),
            self.get_constant([2]),
            f"{prefix}/conv/Slice_Conv_Output/output_0",
        )

        # === Cache update (community: Slice_Cache) ===
        # Extract last L elements for next iteration using shared constants
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

        # === Output gating + projection (community: Mul_2) ===
        # y = C * conv_out
        y = self.make_mul(
            f"{prefix}/conv/Split/output_1",
            conv_sliced,
            f"{prefix}/conv/Mul_2/output_0",
        )
        # Transpose: [B, H, S] → [B, S, H] (community: Transpose_2)
        y_t = self.make_node(
            "Transpose", [y], [f"{prefix}/conv/Transpose_2/output_0"], perm=[0, 2, 1]
        )
        # out_proj: [B, S, H] → [B, S, H] (community naming)
        out_proj = self.make_matmul(
            y_t,
            f"{weight_prefix}.conv.out_proj.MatMul.weight",
            f"{prefix}/conv/out_proj/MatMul/output_0",
        )

        # === Residual + MLP (community naming: Add_1) ===
        hidden_state = self.make_add(residual, out_proj, f"{prefix}/Add_1/output_0")
        return self.build_mlp(layer_idx, hidden_state)

    def build_attention_layer(self, layer_idx: int, hidden_state: str) -> str:
        """Build an attention layer with Grouped Query Attention.

        Graph structure (matches PyTorch Lfm2Attention):
            hidden_state
                ↓
            LayerNorm (operator_layernorm)
                ↓
            ┌─────────────────────────────────────┐
            │  Q/K/V Projections                  │
            │  Q: [B,S,H] → [B,S,nh*hd]           │
            │  K: [B,S,H] → [B,S,nkv*hd]          │
            │  V: [B,S,H] → [B,S,nkv*hd]          │
            └─────────────────────────────────────┘
                ↓
            ┌─────────────────────────────────────┐
            │  Q/K LayerNorm (per-head)           │
            │  Reshape → [B,-1,hd] → Norm →       │
            │  Reshape → [B,-1,proj_dim]          │
            └─────────────────────────────────────┘
                ↓
            RotaryEmbedding (Q, K only)
                ↓
            GroupQueryAttention
              - Inputs: Q, K, V, past_key, past_value, seqlens_k, total_seq
              - Outputs: attn_out, present_key, present_value
                ↓
            Linear (out_proj)
                ↓
            Add (residual)
                ↓
            MLP block

        Uses Microsoft fused operators:
            - RotaryEmbedding: Applies RoPE to Q/K
            - GroupQueryAttention: Fused attention with KV cache
        """
        prefix = f"/model/layers.{layer_idx}"
        weight_prefix = f"model.layers.{layer_idx}"
        H = self.config.hidden_size
        nh = self.config.num_attention_heads
        nkv = self.config.num_key_value_heads
        hd = self.head_dim
        kv_hidden = nkv * hd
        residual = hidden_state

        # === LayerNorm (community naming: LayerNorm suffix) ===
        normed = self.make_simple_layernorm(
            hidden_state,
            f"{weight_prefix}.operator_layernorm.weight",
            f"{prefix}/operator_layernorm",
            name="LayerNorm",
        )

        # === Q/K/V Projections (community naming) ===
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

        # === Q/K LayerNorm (per-head) ===
        # Reshape to [B, -1, head_dim] for per-head norm using shared constants
        reshape_for_norm = self.get_constant([0, -1, hd])
        q_reshape_back = self.get_constant([0, -1, H])
        k_reshape_back = self.get_constant([0, -1, kv_hidden])

        # Q norm (community naming: Reshape_1, Reshape_2)
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

        # K norm (community naming: Reshape_1, Reshape_2)
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

        # === RoPE + GroupQueryAttention ===
        scale = 1.0 / (hd**0.5)
        mask_prefix = "/model/attn_mask_reformat/attn_mask_subgraph"
        gather_name = "Gather_1" if self.vl_naming else "Gather"

        if self.use_integrated_rope:
            # Integrated RoPE: pass un-rotated Q/K, let GQA apply RoPE internally
            self.make_node(
                "GroupQueryAttention",
                [
                    q_3d,
                    k_3d,
                    v,
                    f"past_key_values.{layer_idx}.key",
                    f"past_key_values.{layer_idx}.value",
                    f"{mask_prefix}/Sub/Cast/output_0",
                    f"{mask_prefix}/{gather_name}/Cast/output_0",
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
            # Separate RoPE: apply RotaryEmbedding first, then GQA (community naming)
            rope_attrs = {
                "domain": "com.microsoft",
                "interleaved": 0,
                "num_heads": 0,
                "rotary_embedding_dim": 0,
            }
            q_rope = self.make_node(
                "RotaryEmbedding",
                [q_3d, "position_ids", "cos_cache", "sin_cache"],
                [f"{prefix}/attn/q_rotary/RotaryEmbedding/output_0"],
                **rope_attrs,
            )
            k_rope = self.make_node(
                "RotaryEmbedding",
                [k_3d, "position_ids", "cos_cache", "sin_cache"],
                [f"{prefix}/attn/k_rotary/RotaryEmbedding/output_0"],
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
                    f"{mask_prefix}/Sub/Cast/output_0",
                    f"{mask_prefix}/{gather_name}/Cast/output_0",
                    "",  # cos_cache (unused, RoPE applied above)
                    "",  # sin_cache (unused, RoPE applied above)
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

        # === Output projection + Residual + MLP (community naming) ===
        o_proj = self.make_matmul(
            f"{prefix}/attn/GroupQueryAttention/output_0",
            f"{weight_prefix}.attn.o_proj.MatMul.weight",
            f"{prefix}/attn/o_proj/MatMul/output_0",
        )
        hidden_state = self.make_add(residual, o_proj, f"{prefix}/Add_1/output_0")
        return self.build_mlp(layer_idx, hidden_state)

    def build_mlp(self, layer_idx: int, hidden_state: str) -> str:
        """Build MLP block (SwiGLU activation).

        Graph structure (matches PyTorch Lfm2MLP):
            hidden_state
                ↓
            LayerNorm (ffn_layernorm)
                ↓
            ┌───────────┬───────────┐
            │ gate_proj │ up_proj   │
            │ (gate)    │ (up)      │
            └─────┬─────┴─────┬─────┘
                  ↓           ↓
                SiLU         │
                  ↓           │
                  └─────*─────┘
                        ↓
                    down_proj (down)
                        ↓
                    Add (residual)
        """
        prefix = f"/model/layers.{layer_idx}"
        weight_prefix = f"model.layers.{layer_idx}"

        residual = hidden_state

        # FFN LayerNorm (community naming: LayerNorm suffix)
        normed = self.make_simple_layernorm(
            hidden_state,
            f"{weight_prefix}.ffn_layernorm.weight",
            f"{prefix}/ffn_layernorm",
            name="LayerNorm",
        )

        # Gate and Up projections (community naming)
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

        # SiLU on gate (community naming: act_fn)
        gate_silu = self.make_silu(gate, f"{prefix}/mlp/act_fn")

        # gate * up
        gated = self.make_mul(gate_silu, up, f"{prefix}/mlp/Mul/output_0")

        # Down projection (community naming)
        down = self.make_matmul(
            gated,
            f"{weight_prefix}.mlp.down_proj.MatMul.weight",
            f"{prefix}/mlp/down_proj/MatMul/output_0",
        )

        # Residual (community naming: Add_2)
        return self.make_add(residual, down, f"{prefix}/Add_2/output_0")

    def build_lm_head(self, hidden_state: str) -> str:
        # Final LayerNorm using SkipSimplifiedLayerNormalization (fused op)
        # Community naming: model.layers.{num_layers}.final_norm_layernorm.weight
        num_layers = self.config.num_hidden_layers
        final_norm_weight = f"model.layers.{num_layers}.final_norm_layernorm.weight"
        # Community uses shorter output path (no op type suffix)
        final_norm_output = f"/model/layers.{num_layers}/final_norm_layernorm/output_0"

        self.add_initializer(final_norm_weight, self.weights["model.embedding_norm.weight"])
        # Community uses SkipLayerNorm as node name suffix
        normed = self.make_skip_layernorm(
            hidden_state,
            hidden_state,
            final_norm_weight,
            final_norm_output,
            name=f"/model/layers.{num_layers}/final_norm_layernorm/SkipLayerNorm",
        )

        if self.use_q4:
            # Q4: Use MatMulNBits for lm_head with shared embedding weights
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
            n_blocks = (K + self.q4_block_size - 1) // self.q4_block_size

            # Reshape to 3D for MatMulNBits: [N, n_blocks, block_size/2]
            embed_quant_matmul = embed_quant.reshape(vocab_size, n_blocks, self.q4_block_size // 2)
            self.add_initializer(
                "model_embed_tokens_weight_quant_matmul", embed_quant_matmul, dtype=np.uint8
            )

            # Reuse scales and zero points from embedding
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
                block_size=self.q4_block_size,
            )

        # FP32: Transpose embed_tokens at runtime instead of storing a copy
        # embed_tokens.weight [vocab, hidden] → [hidden, vocab]
        lm_head_weight = self.make_node(
            "Transpose",
            ["model.embed_tokens.weight"],
            ["/lm_head/Transpose/output_0"],
            perm=[1, 0],
        )

        # Output name must match graph output declaration
        return self.make_matmul(normed, lm_head_weight, "logits")

    def build_value_info(self):
        """Build ValueInfo entries for weights and intermediate tensors.

        Adds shape annotations for all tensors in the graph, matching
        the community model format.
        """
        H = self.config.hidden_size
        nkv = self.config.num_key_value_heads
        hd = self.head_dim
        kv_hidden = nkv * hd
        intermediate = self.config.intermediate_size
        num_layers = self.config.num_hidden_layers
        mask_prefix = "/model/attn_mask_reformat/attn_mask_subgraph"

        # === Weight shapes (concrete) ===
        for init in self.initializers:
            # Skip constants (they have /model/constants/ prefix)
            if init.name.startswith("/model/constants/"):
                continue
            shape = list(init.dims)
            dtype = init.data_type
            self.add_value_info(init.name, dtype, shape)

        # === Attention mask subgraph outputs ===
        gather_name = "Gather_1" if self.vl_naming else "Gather"
        self.add_value_info(f"{mask_prefix}/ReduceSum/output_0", TensorProto.INT64, ["batch_size"])
        self.add_value_info(f"{mask_prefix}/Sub/output_0", TensorProto.INT64, ["batch_size"])
        self.add_value_info(f"{mask_prefix}/Sub/Cast/output_0", TensorProto.INT32, ["batch_size"])
        self.add_value_info(f"{mask_prefix}/Shape/output_0", TensorProto.INT64, [2])
        self.add_value_info(f"{mask_prefix}/{gather_name}/output_0", TensorProto.INT64, [])
        self.add_value_info(f"{mask_prefix}/{gather_name}/Cast/output_0", TensorProto.INT32, [])

        # === Embedding output ===
        if self.use_q4:
            embed_output = "/model/embed_tokens/GatherBlockQuantized/output_0"
        else:
            embed_output = "/model/embed_tokens/Gather/output_0"
        self.add_value_info(embed_output, TensorProto.FLOAT, ["batch_size", "sequence_length", H])

        # === Per-layer outputs ===
        for layer_idx in range(num_layers):
            prefix = f"/model/layers.{layer_idx}"
            layer_type = self.config.layer_types[layer_idx]

            # Operator layernorm output (community uses simplified path)
            self.add_value_info(
                f"{prefix}/operator_layernorm/output_0",
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
                # Community has split_sizes ValueInfo instead of Conv/Slice outputs
                shape_name = "Shape" if self.vl_naming else "Shape_for_slice"
                conv_gather_name = "Gather_1" if self.vl_naming else "Gather_for_slice"
                self.add_value_info(f"{prefix}/conv/split_sizes", TensorProto.INT64, [3])
                self.add_value_info(f"{prefix}/conv/{shape_name}/output_0", TensorProto.INT64, [3])
                self.add_value_info(
                    f"{prefix}/conv/{conv_gather_name}/output_0", TensorProto.INT64, []
                )
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
                # Q/K norm reshapes
                self.add_value_info(
                    f"{prefix}/attn/q_norm/Reshape_1/output_0",
                    TensorProto.FLOAT,
                    ["batch_size", "q_seq_heads", hd],
                )
                self.add_value_info(
                    f"{prefix}/attn/q_norm/SimplifiedLayerNormalization/output_0",
                    TensorProto.FLOAT,
                    ["batch_size", "q_seq_heads", hd],
                )
                self.add_value_info(
                    f"{prefix}/attn/q_norm/Reshape_2/output_0",
                    TensorProto.FLOAT,
                    ["batch_size", "sequence_length", H],
                )
                self.add_value_info(
                    f"{prefix}/attn/k_norm/Reshape_1/output_0",
                    TensorProto.FLOAT,
                    ["batch_size", "kv_seq_heads", hd],
                )
                self.add_value_info(
                    f"{prefix}/attn/k_norm/SimplifiedLayerNormalization/output_0",
                    TensorProto.FLOAT,
                    ["batch_size", "kv_seq_heads", hd],
                )
                self.add_value_info(
                    f"{prefix}/attn/k_norm/Reshape_2/output_0",
                    TensorProto.FLOAT,
                    ["batch_size", "sequence_length", kv_hidden],
                )
                # RoPE outputs
                self.add_value_info(
                    f"{prefix}/attn/q_rotary/RotaryEmbedding/output_0",
                    TensorProto.FLOAT,
                    ["batch_size", "sequence_length", H],
                )
                self.add_value_info(
                    f"{prefix}/attn/k_rotary/RotaryEmbedding/output_0",
                    TensorProto.FLOAT,
                    ["batch_size", "sequence_length", kv_hidden],
                )
                # GQA output
                self.add_value_info(
                    f"{prefix}/attn/GroupQueryAttention/output_0",
                    TensorProto.FLOAT,
                    ["batch_size", "sequence_length", H],
                )
                # Output projection
                self.add_value_info(
                    f"{prefix}/attn/o_proj/MatMul/output_0",
                    TensorProto.FLOAT,
                    ["batch_size", "sequence_length", H],
                )

            # Residual Add_1
            self.add_value_info(
                f"{prefix}/Add_1/output_0", TensorProto.FLOAT, ["batch_size", "sequence_length", H]
            )

            # FFN layernorm
            self.add_value_info(
                f"{prefix}/ffn_layernorm/output_0",
                TensorProto.FLOAT,
                ["batch_size", "sequence_length", H],
            )

            # MLP outputs
            self.add_value_info(
                f"{prefix}/mlp/gate_proj/MatMul/output_0",
                TensorProto.FLOAT,
                ["batch_size", "sequence_length", intermediate],
            )
            self.add_value_info(
                f"{prefix}/mlp/up_proj/MatMul/output_0",
                TensorProto.FLOAT,
                ["batch_size", "sequence_length", intermediate],
            )
            self.add_value_info(
                f"{prefix}/mlp/act_fn/Sigmoid/output_0",
                TensorProto.FLOAT,
                ["batch_size", "sequence_length", intermediate],
            )
            self.add_value_info(
                f"{prefix}/mlp/act_fn/Mul/output_0",
                TensorProto.FLOAT,
                ["batch_size", "sequence_length", intermediate],
            )
            self.add_value_info(
                f"{prefix}/mlp/Mul/output_0",
                TensorProto.FLOAT,
                ["batch_size", "sequence_length", intermediate],
            )
            self.add_value_info(
                f"{prefix}/mlp/down_proj/MatMul/output_0",
                TensorProto.FLOAT,
                ["batch_size", "sequence_length", H],
            )
            self.add_value_info(
                f"{prefix}/Add_2/output_0", TensorProto.FLOAT, ["batch_size", "sequence_length", H]
            )

        # === Final norm and LM head ===
        self.add_value_info(
            f"/model/layers.{num_layers}/final_norm_layernorm/output_0",
            TensorProto.FLOAT,
            ["batch_size", "sequence_length", H],
        )
        if not self.use_q4:
            self.add_value_info(
                "/lm_head/Transpose/output_0", TensorProto.FLOAT, [H, self.config.vocab_size]
            )

    def load_weights(self, model_path: str):
        """Load weights from HuggingFace model."""
        import torch
        from transformers import AutoModelForCausalLM

        logger.info(f"Loading weights from {model_path}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float32, trust_remote_code=True
        )

        for name, param in model.named_parameters():
            self.weights[name] = param.detach().numpy()
            logger.debug(f"Loaded: {name} {param.shape}")

        del model
        logger.info(f"Loaded {len(self.weights)} weights")

    def build(self, model_path: str) -> onnx.ModelProto:
        """Build the complete ONNX model.

        Build phases:
            1. Load weights from HuggingFace model
            2. Create graph inputs/outputs
            3. Build RoPE cache and attention mask preprocessing
            4. Build embedding layer
            5. For each layer: prepare weights, then build graph
            6. Build LM head with tied weights
            7. Create final ONNX model
        """
        logger.info("Building LFM2 ONNX model...")

        # Phase 1: Load weights
        self.load_weights(model_path)

        # Phase 2-3: Build graph structure
        self.build_inputs()
        self.build_outputs()
        self.build_rope_cache()
        self.build_attention_mask_subgraph()

        # Phase 4: Embedding
        hidden_state = self.build_embedding()

        # Phase 5: Layers (prepare weights, then build graph)
        for layer_idx in range(self.config.num_hidden_layers):
            layer_type = self.config.layer_types[layer_idx]
            logger.info(f"Building layer {layer_idx} ({layer_type})...")

            # Prepare weights for this layer (handles transposition)
            self.prepare_layer_weights(layer_idx, layer_type)

            # Build layer graph
            if layer_type == "conv":
                hidden_state = self.build_conv_layer(layer_idx, hidden_state)
            else:
                hidden_state = self.build_attention_layer(layer_idx, hidden_state)

        # LM head
        self.build_lm_head(hidden_state)

        # Build ValueInfo for shape annotations
        self.build_value_info()

        model = self.build_graph("lfm2")
        logger.info(f"Model built: {len(self.nodes)} nodes, {len(self.value_info)} value_info")
        return model
