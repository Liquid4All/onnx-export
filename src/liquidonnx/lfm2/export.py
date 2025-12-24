"""
LFM2 Builder for ONNX export.

This builder follows the onnxruntime-genai builder pattern but uses
the stable onnx.helper API. It can be ported to onnx_ir when that API stabilizes.

The builder creates an optimized ONNX graph with fused operators:
- SimplifiedLayerNormalization (com.microsoft)
- RotaryEmbedding (com.microsoft)
- GroupQueryAttention (com.microsoft)
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

logger = logging.getLogger(__name__)


@dataclass
class LFM2Config:
    """Configuration for LFM2 model."""
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int
    layer_types: List[str]
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


class LFM2Builder:
    """
    LFM2 model builder for ONNX export.

    Creates an optimized ONNX graph with:
    - Conv/SSM layers with gating
    - Full attention layers with GQA
    - Fused operators for better performance
    """

    def __init__(self, config: LFM2Config):
        self.config = config
        self.head_dim = config.hidden_size // config.num_attention_heads

        # Categorize layers
        self.conv_indices = [i for i, t in enumerate(config.layer_types) if t == "conv"]
        self.attn_indices = [i for i, t in enumerate(config.layer_types) if t == "full_attention"]

        # Graph components
        self.nodes: List[onnx.NodeProto] = []
        self.inputs: List[onnx.ValueInfoProto] = []
        self.outputs: List[onnx.ValueInfoProto] = []
        self.initializers: List[onnx.TensorProto] = []

        # Weights storage
        self.weights: Dict[str, np.ndarray] = {}

        # Node counter for unique names
        self._node_count = 0

    def _unique_name(self, prefix: str) -> str:
        self._node_count += 1
        return f"{prefix}_{self._node_count}"

    def add_initializer(self, name: str, tensor: np.ndarray, dtype=None):
        """Add weight tensor as graph initializer."""
        if dtype is None:
            # Default to float32 for weights, preserve dtype for constants
            if tensor.dtype in [np.int32, np.int64]:
                pass  # Keep int types as-is
            else:
                tensor = tensor.astype(np.float32)
        else:
            tensor = tensor.astype(dtype)
        self.initializers.append(numpy_helper.from_array(tensor, name))

    def make_node(self, op_type: str, inputs: List[str], outputs: List[str],
                  name: str = None, domain: str = "", **attrs) -> str:
        """Create an ONNX node and return the first output name."""
        if name is None:
            name = self._unique_name(op_type)

        node = helper.make_node(op_type, inputs, outputs, name=name, domain=domain, **attrs)
        self.nodes.append(node)
        return outputs[0] if outputs else None

    def make_layernorm(self, input_name: str, weight_name: str, output_name: str) -> str:
        """Create SimplifiedLayerNormalization node."""
        return self.make_node(
            "SimplifiedLayerNormalization",
            inputs=[input_name, weight_name],
            outputs=[output_name],
            epsilon=self.config.norm_eps,
        )

    def make_skip_layernorm(self, input_name: str, skip_name: str, weight_name: str, output_name: str) -> str:
        """Create SkipSimplifiedLayerNormalization node (fused skip + layernorm)."""
        return self.make_node(
            "SkipSimplifiedLayerNormalization",
            inputs=[input_name, skip_name, weight_name],
            outputs=[output_name],
            domain="com.microsoft",
            epsilon=self.config.norm_eps,
        )

    def make_matmul(self, input_name: str, weight_name: str, output_name: str) -> str:
        """Create MatMul node."""
        return self.make_node("MatMul", [input_name, weight_name], [output_name])

    def make_add(self, a: str, b: str, output_name: str) -> str:
        """Create Add node."""
        return self.make_node("Add", [a, b], [output_name])

    def make_mul(self, a: str, b: str, output_name: str) -> str:
        """Create Mul node."""
        return self.make_node("Mul", [a, b], [output_name])

    def make_sigmoid(self, input_name: str, output_name: str) -> str:
        """Create Sigmoid node."""
        return self.make_node("Sigmoid", [input_name], [output_name])

    def make_silu(self, input_name: str, output_name: str) -> str:
        """Create SiLU activation (x * sigmoid(x))."""
        sigmoid_out = self.make_sigmoid(input_name, f"{output_name}_sigmoid")
        return self.make_mul(input_name, sigmoid_out, output_name)

    def build_inputs(self):
        """Create model inputs."""
        # input_ids
        self.inputs.append(helper.make_tensor_value_info(
            "input_ids", TensorProto.INT64, ["batch_size", "sequence_length"]
        ))

        # attention_mask
        self.inputs.append(helper.make_tensor_value_info(
            "attention_mask", TensorProto.INT64, ["batch_size", "total_sequence_length"]
        ))

        # position_ids
        self.inputs.append(helper.make_tensor_value_info(
            "position_ids", TensorProto.INT64, ["batch_size", "sequence_length"]
        ))

        # Conv caches
        for idx in self.conv_indices:
            self.inputs.append(helper.make_tensor_value_info(
                f"past_conv.{idx}", TensorProto.FLOAT,
                ["batch_size", self.config.hidden_size, self.config.conv_L_cache]
            ))

        # KV caches
        for idx in self.attn_indices:
            self.inputs.append(helper.make_tensor_value_info(
                f"past_key_values.{idx}.key", TensorProto.FLOAT,
                ["batch_size", self.config.num_key_value_heads, "past_sequence_length", self.head_dim]
            ))
            self.inputs.append(helper.make_tensor_value_info(
                f"past_key_values.{idx}.value", TensorProto.FLOAT,
                ["batch_size", self.config.num_key_value_heads, "past_sequence_length", self.head_dim]
            ))

    def build_outputs(self):
        """Create model outputs."""
        # Logits
        self.outputs.append(helper.make_tensor_value_info(
            "logits", TensorProto.FLOAT, ["batch_size", "sequence_length", self.config.vocab_size]
        ))

        # Conv cache outputs
        for idx in self.conv_indices:
            self.outputs.append(helper.make_tensor_value_info(
                f"present_conv.{idx}", TensorProto.FLOAT,
                ["batch_size", self.config.hidden_size, self.config.conv_L_cache]
            ))

        # KV cache outputs
        for idx in self.attn_indices:
            self.outputs.append(helper.make_tensor_value_info(
                f"present.{idx}.key", TensorProto.FLOAT,
                ["batch_size", self.config.num_key_value_heads, "total_sequence_length", self.head_dim]
            ))
            self.outputs.append(helper.make_tensor_value_info(
                f"present.{idx}.value", TensorProto.FLOAT,
                ["batch_size", self.config.num_key_value_heads, "total_sequence_length", self.head_dim]
            ))

    def build_embedding(self) -> str:
        """Build embedding layer, return output name."""
        self.add_initializer("model.embed_tokens.weight", self.weights["model.embed_tokens.weight"])
        return self.make_node("Gather", ["model.embed_tokens.weight", "input_ids"], ["embed_output"], axis=0)

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
        self.make_node("ReduceSum", ["attention_mask", "const_1"],
                       ["/attn_mask/reduce_sum"], keepdims=0)
        self.make_node("Sub", ["/attn_mask/reduce_sum", "const_1"],
                       ["/attn_mask/seqlens_k_i64"])
        self.make_node("Cast", ["/attn_mask/seqlens_k_i64"],
                       ["/attn_mask/seqlens_k"], to=TensorProto.INT32)

        # total_seq_len = shape[1] of attention_mask
        self.add_initializer("const_1_scalar", np.array(1, dtype=np.int64))
        self.make_node("Shape", ["attention_mask"], ["/attn_mask/shape"])
        self.make_node("Gather", ["/attn_mask/shape", "const_1_scalar"],
                       ["/attn_mask/total_seq_i64"], axis=0)
        self.make_node("Cast", ["/attn_mask/total_seq_i64"],
                       ["/attn_mask/total_seq"], to=TensorProto.INT32)

    def build_conv_layer(self, layer_idx: int, hidden_state: str) -> str:
        """Build a conv/SSM layer."""
        prefix = f"model.layers.{layer_idx}"
        L = self.config.conv_L_cache

        # Load weights (using actual HF weight names)
        self.add_initializer(f"{prefix}.operator_norm.weight",
                             self.weights[f"{prefix}.operator_norm.weight"])
        self.add_initializer(f"{prefix}.conv.in_proj.weight",
                             self.weights[f"{prefix}.conv.in_proj.weight"].T)
        self.add_initializer(f"{prefix}.conv.weight",
                             self.weights[f"{prefix}.conv.conv.weight"])
        self.add_initializer(f"{prefix}.conv.out_proj.weight",
                             self.weights[f"{prefix}.conv.out_proj.weight"].T)
        self.add_initializer(f"{prefix}.ffn_norm.weight",
                             self.weights[f"{prefix}.ffn_norm.weight"])
        self.add_initializer(f"{prefix}.feed_forward.w1.weight",
                             self.weights[f"{prefix}.feed_forward.w1.weight"].T)
        self.add_initializer(f"{prefix}.feed_forward.w3.weight",
                             self.weights[f"{prefix}.feed_forward.w3.weight"].T)
        self.add_initializer(f"{prefix}.feed_forward.w2.weight",
                             self.weights[f"{prefix}.feed_forward.w2.weight"].T)

        residual = hidden_state
        H = self.config.hidden_size

        # Operator LayerNorm
        normed = self.make_layernorm(hidden_state, f"{prefix}.operator_norm.weight",
                                      f"{prefix}/op_norm")

        # In projection: [B, S, H] -> [B, S, 3H]
        in_proj = self.make_matmul(normed, f"{prefix}.conv.in_proj.weight", f"{prefix}/in_proj")

        # Transpose for Split: [B, S, 3H] -> [B, 3H, S]
        in_proj_t = self.make_node("Transpose", [in_proj], [f"{prefix}/in_proj_t"], perm=[0, 2, 1])

        # Split into B, C, x (each [B, H, S])
        self.add_initializer(f"{prefix}/split_sizes", np.array([H, H, H], dtype=np.int64))
        self.make_node("Split", [in_proj_t, f"{prefix}/split_sizes"],
                       [f"{prefix}/B", f"{prefix}/C", f"{prefix}/x"], axis=1)

        # Bx = B * x (no sigmoid, just multiply)
        Bx = self.make_mul(f"{prefix}/B", f"{prefix}/x", f"{prefix}/Bx")

        # Concat with past conv cache: [B, H, L] + [B, H, S] -> [B, H, L+S]
        conv_input = self.make_node("Concat", [f"past_conv.{layer_idx}", Bx],
                                    [f"{prefix}/conv_input"], axis=2)

        # Conv1D (depthwise): kernel_shape = 3 (matches weight shape [H, 1, 3])
        conv_out_full = self.make_node("Conv", [conv_input, f"{prefix}.conv.weight"],
                                       [f"{prefix}/conv_out_full"],
                                       kernel_shape=[3], group=H)

        # Slice conv output to match sequence length (take last S elements)
        # Get shape of Bx to determine sequence length dynamically
        self.make_node("Shape", [Bx], [f"{prefix}/bx_shape"])
        self.add_initializer(f"{prefix}/const_2_scalar", np.array(2, dtype=np.int64))
        self.make_node("Gather", [f"{prefix}/bx_shape", f"{prefix}/const_2_scalar"],
                       [f"{prefix}/seq_len"], axis=0)
        self.add_initializer(f"{prefix}/const_neg1", np.array(-1, dtype=np.int64))
        self.make_node("Mul", [f"{prefix}/seq_len", f"{prefix}/const_neg1"],
                       [f"{prefix}/neg_seq_len"])
        self.add_initializer(f"{prefix}/const_0_1d", np.array([0], dtype=np.int64))
        self.make_node("Unsqueeze", [f"{prefix}/neg_seq_len", f"{prefix}/const_0_1d"],
                       [f"{prefix}/slice_start"])
        self.add_initializer(f"{prefix}/slice_end_max", np.array([9223372036854775807], dtype=np.int64))
        self.add_initializer(f"{prefix}/slice_axis_2", np.array([2], dtype=np.int64))
        self.make_node("Slice", [conv_out_full, f"{prefix}/slice_start", f"{prefix}/slice_end_max", f"{prefix}/slice_axis_2"],
                       [f"{prefix}/conv_out"])

        # Extract new cache (last L elements of conv_input)
        self.add_initializer(f"{prefix}/cache_slice_starts", np.array([-L], dtype=np.int64))
        self.add_initializer(f"{prefix}/cache_slice_ends", np.array([2147483647], dtype=np.int64))
        self.add_initializer(f"{prefix}/cache_slice_axes", np.array([2], dtype=np.int64))
        self.make_node("Slice", [conv_input, f"{prefix}/cache_slice_starts", f"{prefix}/cache_slice_ends", f"{prefix}/cache_slice_axes"],
                       [f"present_conv.{layer_idx}"])

        # y = C * conv_out (element-wise multiply in [B, H, S] space)
        y = self.make_mul(f"{prefix}/C", f"{prefix}/conv_out", f"{prefix}/y")

        # Transpose back: [B, H, S] -> [B, S, H]
        y_t = self.make_node("Transpose", [y], [f"{prefix}/y_t"], perm=[0, 2, 1])

        # Out projection
        out_proj = self.make_matmul(y_t, f"{prefix}.conv.out_proj.weight", f"{prefix}/out_proj")

        # Residual
        hidden_state = self.make_add(residual, out_proj, f"{prefix}/residual1")

        # MLP
        hidden_state = self.build_mlp(layer_idx, hidden_state)

        return hidden_state

    def build_attention_layer(self, layer_idx: int, hidden_state: str) -> str:
        """Build an attention layer with GQA."""
        prefix = f"model.layers.{layer_idx}"
        H = self.config.hidden_size
        nh = self.config.num_attention_heads
        nkv = self.config.num_key_value_heads
        hd = self.head_dim

        # Load weights (using actual HF weight names)
        self.add_initializer(f"{prefix}.operator_norm.weight",
                             self.weights[f"{prefix}.operator_norm.weight"])
        self.add_initializer(f"{prefix}.self_attn.q_proj.weight",
                             self.weights[f"{prefix}.self_attn.q_proj.weight"].T)
        self.add_initializer(f"{prefix}.self_attn.k_proj.weight",
                             self.weights[f"{prefix}.self_attn.k_proj.weight"].T)
        self.add_initializer(f"{prefix}.self_attn.v_proj.weight",
                             self.weights[f"{prefix}.self_attn.v_proj.weight"].T)
        self.add_initializer(f"{prefix}.self_attn.q_layernorm.weight",
                             self.weights[f"{prefix}.self_attn.q_layernorm.weight"])
        self.add_initializer(f"{prefix}.self_attn.k_layernorm.weight",
                             self.weights[f"{prefix}.self_attn.k_layernorm.weight"])
        self.add_initializer(f"{prefix}.self_attn.out_proj.weight",
                             self.weights[f"{prefix}.self_attn.out_proj.weight"].T)
        self.add_initializer(f"{prefix}.ffn_norm.weight",
                             self.weights[f"{prefix}.ffn_norm.weight"])
        self.add_initializer(f"{prefix}.feed_forward.w1.weight",
                             self.weights[f"{prefix}.feed_forward.w1.weight"].T)
        self.add_initializer(f"{prefix}.feed_forward.w3.weight",
                             self.weights[f"{prefix}.feed_forward.w3.weight"].T)
        self.add_initializer(f"{prefix}.feed_forward.w2.weight",
                             self.weights[f"{prefix}.feed_forward.w2.weight"].T)

        residual = hidden_state

        # Operator LayerNorm
        normed = self.make_layernorm(hidden_state, f"{prefix}.operator_norm.weight",
                                      f"{prefix}/op_norm")

        # Q, K, V projections
        # Q: [B, S, H] -> [B, S, nh*hd] = [B, S, 2048]
        # K: [B, S, H] -> [B, S, nkv*hd] = [B, S, 512]
        # V: [B, S, H] -> [B, S, nkv*hd] = [B, S, 512]
        q = self.make_matmul(normed, f"{prefix}.self_attn.q_proj.weight", f"{prefix}/q")
        k = self.make_matmul(normed, f"{prefix}.self_attn.k_proj.weight", f"{prefix}/k")
        v = self.make_matmul(normed, f"{prefix}.self_attn.v_proj.weight", f"{prefix}/v")

        # Q norm: Reshape to [B, -1, head_dim] for per-head norm, then back to [B, -1, hidden_size]
        # This flattens batch*seq*heads into the middle dimension
        self.add_initializer(f"{prefix}/q_reshape_for_norm", np.array([0, -1, hd], dtype=np.int64))
        q_for_norm = self.make_node("Reshape", [q, f"{prefix}/q_reshape_for_norm"], [f"{prefix}/q_for_norm"])
        q_normed = self.make_layernorm(q_for_norm, f"{prefix}.self_attn.q_layernorm.weight", f"{prefix}/q_normed")
        self.add_initializer(f"{prefix}/q_reshape_back", np.array([0, -1, H], dtype=np.int64))
        q_3d = self.make_node("Reshape", [q_normed, f"{prefix}/q_reshape_back"], [f"{prefix}/q_3d"])

        # K norm: Reshape to [B, -1, head_dim] for per-head norm, then back to [B, -1, kv_hidden_size]
        kv_hidden = nkv * hd
        self.add_initializer(f"{prefix}/k_reshape_for_norm", np.array([0, -1, hd], dtype=np.int64))
        k_for_norm = self.make_node("Reshape", [k, f"{prefix}/k_reshape_for_norm"], [f"{prefix}/k_for_norm"])
        k_normed = self.make_layernorm(k_for_norm, f"{prefix}.self_attn.k_layernorm.weight", f"{prefix}/k_normed")
        self.add_initializer(f"{prefix}/k_reshape_back", np.array([0, -1, kv_hidden], dtype=np.int64))
        k_3d = self.make_node("Reshape", [k_normed, f"{prefix}/k_reshape_back"], [f"{prefix}/k_3d"])

        # RoPE - use num_heads=0 and rotary_embedding_dim=0 to let operator infer
        q_rope = self.make_node("RotaryEmbedding",
                                [q_3d, "position_ids", "cos_cache", "sin_cache"],
                                [f"{prefix}/q_rope"],
                                domain="com.microsoft", interleaved=0, num_heads=0, rotary_embedding_dim=0)
        k_rope = self.make_node("RotaryEmbedding",
                                [k_3d, "position_ids", "cos_cache", "sin_cache"],
                                [f"{prefix}/k_rope"],
                                domain="com.microsoft", interleaved=0, num_heads=0, rotary_embedding_dim=0)

        # GroupQueryAttention (all inputs are 3D: [B, S, heads*hd])
        scale = 1.0 / (hd ** 0.5)  # 0.125 for head_dim=64
        self.make_node("GroupQueryAttention",
                       [q_rope, k_rope, v,
                        f"past_key_values.{layer_idx}.key", f"past_key_values.{layer_idx}.value",
                        "/attn_mask/seqlens_k", "/attn_mask/total_seq",
                        "", ""],  # cos/sin cache (empty, we apply rotary before)
                       [f"{prefix}/attn_out", f"present.{layer_idx}.key", f"present.{layer_idx}.value"],
                       domain="com.microsoft",
                       num_heads=nh, kv_num_heads=nkv, scale=scale,
                       local_window_size=-1, softcap=0.0, do_rotary=0, rotary_interleaved=0)

        # Output projection (attn_out is [B, S, H] from GQA)
        o_proj = self.make_matmul(f"{prefix}/attn_out", f"{prefix}.self_attn.out_proj.weight", f"{prefix}/o_proj")

        # Residual
        hidden_state = self.make_add(residual, o_proj, f"{prefix}/residual1")

        # MLP
        hidden_state = self.build_mlp(layer_idx, hidden_state)

        return hidden_state

    def build_mlp(self, layer_idx: int, hidden_state: str) -> str:
        """Build MLP block."""
        prefix = f"model.layers.{layer_idx}"

        residual = hidden_state

        # FFN LayerNorm
        normed = self.make_layernorm(hidden_state, f"{prefix}.ffn_norm.weight",
                                      f"{prefix}/ffn_norm")

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
        self.add_initializer("model.embedding_norm.weight",
                             self.weights["model.embedding_norm.weight"])
        normed = self.make_skip_layernorm(hidden_state, hidden_state,
                                          "model.embedding_norm.weight", "final_norm")

        # LM head with tied embeddings - use Transpose to share weights
        # This reuses model.embed_tokens.weight instead of storing a separate copy
        self.make_node("Transpose", ["model.embed_tokens.weight"], ["lm_head.weight_transposed"],
                       perm=[1, 0])

        return self.make_matmul(normed, "lm_head.weight_transposed", "logits")

    def load_weights(self, model_path: str):
        """Load weights from HuggingFace model."""
        from transformers import AutoModelForCausalLM
        import torch

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
        """Build the complete ONNX model."""
        logger.info("Building LFM2 ONNX model...")

        # Load weights
        self.load_weights(model_path)

        # Build graph structure
        self.build_inputs()
        self.build_outputs()
        self.build_rope_cache()
        self.build_attention_mask_subgraph()

        # Embedding
        hidden_state = self.build_embedding()

        # Layers
        for layer_idx in range(self.config.num_hidden_layers):
            layer_type = self.config.layer_types[layer_idx]
            logger.info(f"Building layer {layer_idx} ({layer_type})...")

            if layer_type == "conv":
                hidden_state = self.build_conv_layer(layer_idx, hidden_state)
            else:
                hidden_state = self.build_attention_layer(layer_idx, hidden_state)

        # LM head
        self.build_lm_head(hidden_state)

        # Create graph
        graph = helper.make_graph(
            self.nodes,
            "lfm2",
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
        model.producer_name = "lfm2-builder"

        logger.info(f"Model built: {len(self.nodes)} nodes")
        return model


def export_model(model_path: str, output_dir: str):
    """Export LFM2 model to ONNX."""
    import os
    from transformers import AutoConfig, AutoTokenizer

    # Load config
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    lfm2_config = LFM2Config.from_hf_config(config)

    # Build model
    builder = LFM2Builder(lfm2_config)
    model = builder.build(model_path)

    # Save model
    os.makedirs(output_dir, exist_ok=True)
    onnx_dir = os.path.join(output_dir, "onnx")
    os.makedirs(onnx_dir, exist_ok=True)

    output_path = os.path.join(onnx_dir, "decoder_fp32.onnx")

    # Remove existing external data file to avoid appending
    external_data_path = os.path.join(onnx_dir, "decoder_fp32.onnx_data")
    if os.path.exists(external_data_path):
        os.remove(external_data_path)

    onnx.save_model(model, output_path, save_as_external_data=True,
                    all_tensors_to_one_file=True, location="decoder_fp32.onnx_data")

    logger.info(f"Model saved to {output_path}")

    # Copy tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)
    config.save_pretrained(output_dir)

    # Create generation_config.json (required by Transformers.js)
    import json
    gen_config = {
        "_from_model_config": True,
        "bos_token_id": config.bos_token_id,
        "eos_token_id": config.eos_token_id,
        "pad_token_id": getattr(config, 'pad_token_id', 0),
        "transformers_version": "4.54.0"
    }
    gen_config_path = os.path.join(output_dir, "generation_config.json")
    with open(gen_config_path, "w") as f:
        json.dump(gen_config, f, indent=2)

    # Add transformers.js_config to config.json (for external data support)
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "r") as f:
        cfg = json.load(f)
    cfg["transformers.js_config"] = {
        "kv_cache_dtype": {"fp32": "float32"},
        "use_external_data_format": True
    }
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)

    # Copy chat_template from tokenizer if available, and save to both
    # tokenizer_config.json and chat_template.jinja
    tokenizer_config_path = os.path.join(output_dir, "tokenizer_config.json")
    chat_template_path = os.path.join(output_dir, "chat_template.jinja")

    if tokenizer.chat_template:
        # Save chat_template.jinja
        with open(chat_template_path, "w") as f:
            f.write(tokenizer.chat_template)

        # Ensure it's also in tokenizer_config.json
        if os.path.exists(tokenizer_config_path):
            with open(tokenizer_config_path, "r") as f:
                tok_cfg = json.load(f)
            if "chat_template" not in tok_cfg:
                tok_cfg["chat_template"] = tokenizer.chat_template
                with open(tokenizer_config_path, "w") as f:
                    json.dump(tok_cfg, f, indent=2)

    # Print summary
    size_mb = os.path.getsize(output_path) / 1e6
    data_path = os.path.join(onnx_dir, "decoder_fp32.onnx_data")
    data_size_gb = os.path.getsize(data_path) / 1e9 if os.path.exists(data_path) else 0
    logger.info(f"Model size: {size_mb:.2f} MB + {data_size_gb:.2f} GB data")

    return output_path


def main():
    """Entry point for lfm2-export command."""
    import argparse

    parser = argparse.ArgumentParser(description="Export LFM2 to ONNX")
    parser.add_argument("--model", type=str, required=True, help="Model path")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--verify", action="store_true", help="Verify against PyTorch after export")
    parser.add_argument("--community", type=str, help="Community ONNX path for comparison")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    export_model(args.model, args.output)

    if args.verify:
        from verify import NumericalVerifier
        verifier = NumericalVerifier(args.model)
        verifier.verify_against_pytorch(args.output)
        if args.community:
            verifier.verify_against_community(args.output, args.community)
        verifier.test_generation(args.output)
        verifier.print_report()


if __name__ == "__main__":
    main()
