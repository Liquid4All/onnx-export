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

from liquidonnx.builder_base import SLICE_END, ONNXBuilderBase

logger = logging.getLogger(__name__)


@dataclass
class LFM2Config:
    """Configuration for LFM2 model."""

    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int
    layer_types: list[str]
    conv_L_cache: int = 3
    max_position_embeddings: int = 128000
    norm_eps: float = 1e-5
    rope_theta: float = 1000000.0

    @classmethod
    def from_hf_config(cls, config) -> "LFM2Config":
        return cls(
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            vocab_size=config.vocab_size,
            layer_types=config.layer_types,
            conv_L_cache=getattr(config, "conv_L_cache", 3),
            max_position_embeddings=config.max_position_embeddings,
            norm_eps=getattr(config, "norm_eps", 1e-5),
            rope_theta=getattr(config, "rope_theta", 1000000.0),
        )


class LFM2Builder(ONNXBuilderBase):
    """
    LFM2 model builder for ONNX export.

    Creates an optimized ONNX graph with:
    - Conv/SSM layers with gating
    - Full attention layers with GQA
    - Fused operators for better performance
    """

    def __init__(self, config: LFM2Config, use_integrated_rope: bool = False):
        """
        Args:
            config: Model configuration
            use_integrated_rope: Use RoPE integrated in GroupQueryAttention (do_rotary=1)
                instead of separate RotaryEmbedding ops. May improve numerical precision.
        """
        super().__init__()
        self.config = config
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.use_integrated_rope = use_integrated_rope

        # Categorize layers
        self.conv_indices = [i for i, t in enumerate(config.layer_types) if t == "conv"]
        self.attn_indices = [i for i, t in enumerate(config.layer_types) if t == "full_attention"]

    def make_simple_layernorm(self, input_name: str, weight_name: str, output_name: str) -> str:
        """Create SimplifiedLayerNormalization node (no bias)."""
        return self.make_layernorm(
            input_name, weight_name, None, output_name, epsilon=self.config.norm_eps
        )

    def make_skip_layernorm(
        self, input_name: str, skip_name: str, weight_name: str, output_name: str
    ) -> str:
        """Create SkipSimplifiedLayerNormalization node (fused skip + layernorm)."""
        return self.make_node(
            "SkipSimplifiedLayerNormalization",
            inputs=[input_name, skip_name, weight_name],
            outputs=[output_name],
            domain="com.microsoft",
            epsilon=self.config.norm_eps,
        )

    def prepare_layer_weights(self, layer_idx: int, layer_type: str):
        """Prepare and register weights for a layer.

        Handles weight transposition and naming for MatMul operations.
        Call this before building the layer graph.
        """
        prefix = f"model.layers.{layer_idx}"

        # Common weights
        self.add_initializer(
            f"{prefix}.operator_norm.weight", self.weights[f"{prefix}.operator_norm.weight"]
        )
        self.add_initializer(f"{prefix}.ffn_norm.weight", self.weights[f"{prefix}.ffn_norm.weight"])

        # MLP weights (transposed for MatMul)
        self.add_initializer(
            f"{prefix}.feed_forward.w1.weight", self.weights[f"{prefix}.feed_forward.w1.weight"].T
        )
        self.add_initializer(
            f"{prefix}.feed_forward.w3.weight", self.weights[f"{prefix}.feed_forward.w3.weight"].T
        )
        self.add_initializer(
            f"{prefix}.feed_forward.w2.weight", self.weights[f"{prefix}.feed_forward.w2.weight"].T
        )

        if layer_type == "conv":
            self.add_initializer(
                f"{prefix}.conv.in_proj.weight", self.weights[f"{prefix}.conv.in_proj.weight"].T
            )
            self.add_initializer(
                f"{prefix}.conv.weight", self.weights[f"{prefix}.conv.conv.weight"]
            )
            self.add_initializer(
                f"{prefix}.conv.out_proj.weight", self.weights[f"{prefix}.conv.out_proj.weight"].T
            )
        else:
            # Attention weights (transposed for MatMul)
            self.add_initializer(
                f"{prefix}.self_attn.q_proj.weight",
                self.weights[f"{prefix}.self_attn.q_proj.weight"].T,
            )
            self.add_initializer(
                f"{prefix}.self_attn.k_proj.weight",
                self.weights[f"{prefix}.self_attn.k_proj.weight"].T,
            )
            self.add_initializer(
                f"{prefix}.self_attn.v_proj.weight",
                self.weights[f"{prefix}.self_attn.v_proj.weight"].T,
            )
            self.add_initializer(
                f"{prefix}.self_attn.q_layernorm.weight",
                self.weights[f"{prefix}.self_attn.q_layernorm.weight"],
            )
            self.add_initializer(
                f"{prefix}.self_attn.k_layernorm.weight",
                self.weights[f"{prefix}.self_attn.k_layernorm.weight"],
            )
            self.add_initializer(
                f"{prefix}.self_attn.out_proj.weight",
                self.weights[f"{prefix}.self_attn.out_proj.weight"].T,
            )

    def build_inputs(self):
        """Create model inputs."""
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

        # Conv caches
        for idx in self.conv_indices:
            self.inputs.append(
                helper.make_tensor_value_info(
                    f"past_conv.{idx}",
                    TensorProto.FLOAT,
                    ["batch_size", self.config.hidden_size, self.config.conv_L_cache],
                )
            )

        # KV caches
        for idx in self.attn_indices:
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
        """Create model outputs."""
        # Logits
        self.outputs.append(
            helper.make_tensor_value_info(
                "logits",
                TensorProto.FLOAT,
                ["batch_size", "sequence_length", self.config.vocab_size],
            )
        )

        # Conv cache outputs
        for idx in self.conv_indices:
            self.outputs.append(
                helper.make_tensor_value_info(
                    f"present_conv.{idx}",
                    TensorProto.FLOAT,
                    ["batch_size", self.config.hidden_size, self.config.conv_L_cache],
                )
            )

        # KV cache outputs
        for idx in self.attn_indices:
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
        """Build embedding layer, return output name."""
        self.add_initializer("model.embed_tokens.weight", self.weights["model.embed_tokens.weight"])
        return self.make_node(
            "Gather", ["model.embed_tokens.weight", "input_ids"], ["embed_output"], axis=0
        )

    def build_rope_cache(self):
        """Build RoPE cos/sin caches."""
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
        """Build attention mask preprocessing for GroupQueryAttention."""
        # Compute seqlens_k and total_seq_len from attention_mask
        self.add_initializer("const_1", np.array([1], dtype=np.int64))

        # seqlens_k = sum of attention_mask per batch - 1
        self.make_node(
            "ReduceSum", ["attention_mask", "const_1"], ["/attn_mask/reduce_sum"], keepdims=0
        )
        self.make_node("Sub", ["/attn_mask/reduce_sum", "const_1"], ["/attn_mask/seqlens_k_i64"])
        self.make_node(
            "Cast", ["/attn_mask/seqlens_k_i64"], ["/attn_mask/seqlens_k"], to=TensorProto.INT32
        )

        # total_seq_len = shape[1] of attention_mask
        self.add_initializer("const_1_scalar", np.array(1, dtype=np.int64))
        self.make_node("Shape", ["attention_mask"], ["/attn_mask/shape"])
        self.make_node(
            "Gather", ["/attn_mask/shape", "const_1_scalar"], ["/attn_mask/total_seq_i64"], axis=0
        )
        self.make_node(
            "Cast", ["/attn_mask/total_seq_i64"], ["/attn_mask/total_seq"], to=TensorProto.INT32
        )

    def build_conv_layer(self, layer_idx: int, hidden_state: str) -> str:
        """Build a conv/SSM layer.

        Graph structure (matches PyTorch Lfm2ShortConv):
            hidden_state
                ↓
            LayerNorm (operator_norm)
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
        prefix = f"model.layers.{layer_idx}"
        L = self.config.conv_L_cache
        H = self.config.hidden_size
        residual = hidden_state

        # === Operator LayerNorm ===
        normed = self.make_simple_layernorm(
            hidden_state, f"{prefix}.operator_norm.weight", f"{prefix}/op_norm"
        )

        # === In projection + Split ===
        # in_proj: [B, S, H] → [B, S, 3H]
        in_proj = self.make_matmul(normed, f"{prefix}.conv.in_proj.weight", f"{prefix}/in_proj")
        # Transpose: [B, S, 3H] → [B, 3H, S]
        in_proj_t = self.make_node("Transpose", [in_proj], [f"{prefix}/in_proj_t"], perm=[0, 2, 1])
        # Split into B, C, x (each [B, H, S])
        self.add_initializer(f"{prefix}/split_sizes", np.array([H, H, H], dtype=np.int64))
        self.make_node(
            "Split",
            [in_proj_t, f"{prefix}/split_sizes"],
            [f"{prefix}/B", f"{prefix}/C", f"{prefix}/x"],
            axis=1,
        )

        # === Gated convolution ===
        # Bx = B * x (input gating)
        Bx = self.make_mul(f"{prefix}/B", f"{prefix}/x", f"{prefix}/Bx")
        # Concat with cache: [B, H, L] + [B, H, S] → [B, H, L+S]
        conv_input = self.make_node(
            "Concat", [f"past_conv.{layer_idx}", Bx], [f"{prefix}/conv_input"], axis=2
        )
        # Depthwise Conv1D (kernel=3)
        conv_out_full = self.make_node(
            "Conv",
            [conv_input, f"{prefix}.conv.weight"],
            [f"{prefix}/conv_out_full"],
            kernel_shape=[L],
            group=H,
        )

        # === Dynamic slice for conv output ===
        # Get sequence length from Bx shape
        self.make_node("Shape", [Bx], [f"{prefix}/bx_shape"])
        self.add_initializer(f"{prefix}/axis2_idx", np.array(2, dtype=np.int64))
        self.make_node(
            "Gather", [f"{prefix}/bx_shape", f"{prefix}/axis2_idx"], [f"{prefix}/seq_len"], axis=0
        )
        # Slice last S elements
        self.make_slice_last_n(conv_out_full, f"{prefix}/seq_len", f"{prefix}/conv_out", axis=2)

        # === Cache update ===
        # Extract last L elements for next iteration
        self.add_initializer(f"{prefix}/cache_start", np.array([-L], dtype=np.int64))
        self.add_initializer(f"{prefix}/cache_end", SLICE_END.copy())
        self.add_initializer(f"{prefix}/cache_axis", np.array([2], dtype=np.int64))
        self.make_node(
            "Slice",
            [conv_input, f"{prefix}/cache_start", f"{prefix}/cache_end", f"{prefix}/cache_axis"],
            [f"present_conv.{layer_idx}"],
        )

        # === Output gating and projection ===
        # y = C * conv_out
        y = self.make_mul(f"{prefix}/C", f"{prefix}/conv_out", f"{prefix}/y")
        # Transpose: [B, H, S] → [B, S, H]
        y_t = self.make_node("Transpose", [y], [f"{prefix}/y_t"], perm=[0, 2, 1])
        # out_proj: [B, S, H] → [B, S, H]
        out_proj = self.make_matmul(y_t, f"{prefix}.conv.out_proj.weight", f"{prefix}/out_proj")

        # === Residual + MLP ===
        hidden_state = self.make_add(residual, out_proj, f"{prefix}/residual1")
        return self.build_mlp(layer_idx, hidden_state)

    def build_attention_layer(self, layer_idx: int, hidden_state: str) -> str:
        """Build an attention layer with Grouped Query Attention.

        Graph structure (matches PyTorch Lfm2Attention):
            hidden_state
                ↓
            LayerNorm (operator_norm)
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
        prefix = f"model.layers.{layer_idx}"
        H = self.config.hidden_size
        nh = self.config.num_attention_heads
        nkv = self.config.num_key_value_heads
        hd = self.head_dim
        kv_hidden = nkv * hd
        residual = hidden_state

        # === Operator LayerNorm ===
        normed = self.make_simple_layernorm(
            hidden_state, f"{prefix}.operator_norm.weight", f"{prefix}/op_norm"
        )

        # === Q/K/V Projections ===
        q = self.make_matmul(normed, f"{prefix}.self_attn.q_proj.weight", f"{prefix}/q")
        k = self.make_matmul(normed, f"{prefix}.self_attn.k_proj.weight", f"{prefix}/k")
        v = self.make_matmul(normed, f"{prefix}.self_attn.v_proj.weight", f"{prefix}/v")

        # === Q/K LayerNorm (per-head normalization) ===
        # Reshape to [B, -1, head_dim] for per-head norm
        self.add_initializer(f"{prefix}/reshape_for_norm", np.array([0, -1, hd], dtype=np.int64))
        self.add_initializer(f"{prefix}/q_reshape_back", np.array([0, -1, H], dtype=np.int64))
        self.add_initializer(
            f"{prefix}/k_reshape_back", np.array([0, -1, kv_hidden], dtype=np.int64)
        )

        # Q norm
        q_for_norm = self.make_node(
            "Reshape", [q, f"{prefix}/reshape_for_norm"], [f"{prefix}/q_for_norm"]
        )
        q_normed = self.make_simple_layernorm(
            q_for_norm, f"{prefix}.self_attn.q_layernorm.weight", f"{prefix}/q_normed"
        )
        q_3d = self.make_node("Reshape", [q_normed, f"{prefix}/q_reshape_back"], [f"{prefix}/q_3d"])

        # K norm
        k_for_norm = self.make_node(
            "Reshape", [k, f"{prefix}/reshape_for_norm"], [f"{prefix}/k_for_norm"]
        )
        k_normed = self.make_simple_layernorm(
            k_for_norm, f"{prefix}.self_attn.k_layernorm.weight", f"{prefix}/k_normed"
        )
        k_3d = self.make_node("Reshape", [k_normed, f"{prefix}/k_reshape_back"], [f"{prefix}/k_3d"])

        # === RoPE + GroupQueryAttention ===
        scale = 1.0 / (hd**0.5)

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
                    "/attn_mask/seqlens_k",
                    "/attn_mask/total_seq",
                    "cos_cache",
                    "sin_cache",
                ],
                [f"{prefix}/attn_out", f"present.{layer_idx}.key", f"present.{layer_idx}.value"],
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
            # Separate RoPE: apply RotaryEmbedding first, then GQA
            rope_attrs = {
                "domain": "com.microsoft",
                "interleaved": 0,
                "num_heads": 0,
                "rotary_embedding_dim": 0,
            }
            q_rope = self.make_node(
                "RotaryEmbedding",
                [q_3d, "position_ids", "cos_cache", "sin_cache"],
                [f"{prefix}/q_rope"],
                **rope_attrs,
            )
            k_rope = self.make_node(
                "RotaryEmbedding",
                [k_3d, "position_ids", "cos_cache", "sin_cache"],
                [f"{prefix}/k_rope"],
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
                    "/attn_mask/seqlens_k",
                    "/attn_mask/total_seq",
                    "",  # cos_cache (unused, RoPE applied above)
                    "",  # sin_cache (unused, RoPE applied above)
                ],
                [f"{prefix}/attn_out", f"present.{layer_idx}.key", f"present.{layer_idx}.value"],
                domain="com.microsoft",
                num_heads=nh,
                kv_num_heads=nkv,
                scale=scale,
                local_window_size=-1,
                softcap=0.0,
                do_rotary=0,
                rotary_interleaved=0,
            )

        # === Output projection + Residual + MLP ===
        o_proj = self.make_matmul(
            f"{prefix}/attn_out", f"{prefix}.self_attn.out_proj.weight", f"{prefix}/o_proj"
        )
        hidden_state = self.make_add(residual, o_proj, f"{prefix}/residual1")
        return self.build_mlp(layer_idx, hidden_state)

    def build_mlp(self, layer_idx: int, hidden_state: str) -> str:
        """Build MLP block (SwiGLU activation).

        Graph structure (matches PyTorch Lfm2MLP):
            hidden_state
                ↓
            LayerNorm (ffn_norm)
                ↓
            ┌───────────┬───────────┐
            │ Linear w1 │ Linear w3 │
            │ (gate)    │ (up)      │
            └─────┬─────┴─────┬─────┘
                  ↓           ↓
                SiLU         │
                  ↓           │
                  └─────*─────┘
                        ↓
                    Linear w2 (down)
                        ↓
                    Add (residual)
        """
        prefix = f"model.layers.{layer_idx}"

        residual = hidden_state

        # FFN LayerNorm
        normed = self.make_simple_layernorm(
            hidden_state, f"{prefix}.ffn_norm.weight", f"{prefix}/ffn_norm"
        )

        # Gate (w1) and Up (w3)
        gate = self.make_matmul(normed, f"{prefix}.feed_forward.w1.weight", f"{prefix}/mlp_gate")
        up = self.make_matmul(normed, f"{prefix}.feed_forward.w3.weight", f"{prefix}/mlp_up")

        # SiLU on gate
        gate_silu = self.make_silu(gate, f"{prefix}/mlp_gate_silu")

        # gate * up
        gated = self.make_mul(gate_silu, up, f"{prefix}/mlp_gated")

        # Down projection (w2)
        down = self.make_matmul(gated, f"{prefix}.feed_forward.w2.weight", f"{prefix}/mlp_down")

        # Residual
        return self.make_add(residual, down, f"{prefix}/residual2")

    def build_lm_head(self, hidden_state: str) -> str:
        """Build LM head."""
        # Final LayerNorm using SkipSimplifiedLayerNormalization (fused op)
        # Pass hidden_state as both input and skip for better numerical stability
        self.add_initializer(
            "model.embedding_norm.weight", self.weights["model.embedding_norm.weight"]
        )
        normed = self.make_skip_layernorm(
            hidden_state, hidden_state, "model.embedding_norm.weight", "final_norm"
        )

        # LM head with tied embeddings - use Transpose to share weights
        # This reuses model.embed_tokens.weight instead of storing a separate copy
        self.make_node(
            "Transpose", ["model.embed_tokens.weight"], ["lm_head.weight_transposed"], perm=[1, 0]
        )

        return self.make_matmul(normed, "lm_head.weight_transposed", "logits")

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

        model = self.build_graph("lfm2", producer_name="lfm2-builder")
        logger.info(f"Model built: {len(self.nodes)} nodes")
        return model
