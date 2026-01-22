"""
Audio Detokenizer ONNX builder.

Converts audio codes (from depthformer) to STFT features for waveform synthesis.
"""

import logging
import pathlib

import numpy as np
import onnx
from onnx import TensorProto, helper

from liquidonnx.builder_base import ONNXBuilderBase

logger = logging.getLogger(__name__)


class AudioDetokenizerBuilder(ONNXBuilderBase):
    """Builder for audio detokenizer ONNX export.

    The audio detokenizer has the following architecture:
    1. FusedEmbedding: 8 codebooks (2048 vocab each) → [B, T, 512]
    2. LFM (8 layers): Mix of conv and sliding_attention layers
    3. Linear: [B, T, 512] → [B, T, 1282] (STFT space)

    Layer types: ["conv", "conv", "sliding_attention", "conv",
                  "sliding_attention", "conv", "sliding_attention", "conv"]
    """

    def __init__(self, config: dict, weights: dict[str, np.ndarray]):
        super().__init__()
        self.config = config
        self.weights = weights

        # Model configuration
        self.hidden_size = config.get("hidden_size", 512)
        self.num_attention_heads = config.get("num_attention_heads", 16)
        self.num_key_value_heads = config.get("num_key_value_heads", 8)
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.intermediate_size = config.get("intermediate_size") or (self.hidden_size * 9 // 2)
        self.output_size = config.get("output_size", 1282)
        self.norm_eps = config.get("norm_eps", 1e-5)
        self.num_codebooks = 8
        self.codebook_vocab = 2048
        self.conv_L_cache = 3

        # Layer types: 4 conv, 4 sliding_attention
        self.layer_types = config.get(
            "layer_types",
            [
                "conv",
                "conv",
                "sliding_attention",
                "conv",
                "sliding_attention",
                "conv",
                "sliding_attention",
                "conv",
            ],
        )
        self.num_layers = len(self.layer_types)
        self.sliding_window = config.get("sliding_window", 30)

    def build_inputs(self):
        """Build graph inputs."""
        self.inputs.append(
            helper.make_tensor_value_info(
                "audio_codes", TensorProto.INT64, ["batch_size", self.num_codebooks, "time"]
            )
        )

    def build_outputs(self):
        """Build graph outputs."""
        self.outputs.append(
            helper.make_tensor_value_info(
                "stft_features",
                TensorProto.FLOAT,
                ["batch_size", "time", self.output_size],
            )
        )

    def build_embedding(self) -> str:
        """Build fused codebook embedding.

        Input: audio_codes [B, 8, T]
        Output: embedded [B, T, 512]
        """
        # Embedding weight: [16384, 512] (8 codebooks * 2048)
        emb_weight = self.weights["emb.emb.weight"].astype(np.float32)
        self.add_initializer("emb.weight", emb_weight)

        # Codebook offsets: [0, 2048, 4096, ...]
        offsets = np.array(
            [i * self.codebook_vocab for i in range(self.num_codebooks)], dtype=np.int64
        ).reshape(1, self.num_codebooks, 1)
        self.add_initializer("codebook_offsets", offsets)

        # Add offsets to codes: [B, 8, T]
        self.make_node("Add", ["audio_codes", "codebook_offsets"], ["/emb/offset_codes/output_0"])

        # Transpose: [B, 8, T] -> [B, T, 8]
        self.make_node(
            "Transpose",
            ["/emb/offset_codes/output_0"],
            ["/emb/transposed/output_0"],
            perm=[0, 2, 1],
        )

        # Get shape for reshape back
        self.make_node("Shape", ["/emb/transposed/output_0"], ["/emb/shape/output_0"])

        # Flatten for gather: [B, T, 8] -> [B*T*8]
        self.make_reshape("/emb/transposed/output_0", self.get_constant([-1]), "/emb/flat/output_0")

        # Gather embeddings: [B*T*8, 512]
        self.make_gather("emb.weight", "/emb/flat/output_0", "/emb/gathered/output_0")

        # Get batch and time dimensions
        batch_dim = self.make_slice(
            "/emb/shape/output_0",
            self.get_constant([0]),
            self.get_constant([1]),
            self.get_constant([0]),
            "/emb/batch_dim/output_0",
        )
        time_dim = self.make_slice(
            "/emb/shape/output_0",
            self.get_constant([1]),
            self.get_constant([2]),
            self.get_constant([0]),
            "/emb/time_dim/output_0",
        )

        # Build reshape shape [B, T, 8, 512]
        reshape_shape = self.make_concat(
            [batch_dim, time_dim, self.get_constant([8]), self.get_constant([self.hidden_size])],
            "/emb/reshape_shape/output_0",
            axis=0,
        )

        # Reshape: [B*T*8, 512] -> [B, T, 8, 512]
        self.make_reshape("/emb/gathered/output_0", reshape_shape, "/emb/reshaped/output_0")

        # Mean across codebooks: [B, T, 8, 512] -> [B, T, 512]
        self.make_node(
            "ReduceMean",
            ["/emb/reshaped/output_0", self.get_constant([2])],
            ["/emb/summed/output_0"],
            keepdims=0,
        )

        emb_output = "/emb/summed/output_0"

        # === 6x Upsampling ===
        # [B, T, H] → transpose → [B, H, T] → resize 6x → [B, H, 6T] → transpose → [B, 6T, H]
        self.make_transpose(emb_output, "/emb/pre_upsample_t/output_0", perm=[0, 2, 1])

        # Resize: [B, H, T] → [B, H, 6*T]
        self.add_initializer("upsample_scales", np.array([1.0, 1.0, 6.0], dtype=np.float32))
        self.add_initializer("empty_roi", np.array([], dtype=np.float32))

        node = helper.make_node(
            "Resize",
            ["/emb/pre_upsample_t/output_0", "empty_roi", "upsample_scales"],
            ["/emb/upsampled/output_0"],
            name="/emb/upsample",
            mode="nearest",
            coordinate_transformation_mode="asymmetric",
            nearest_mode="floor",
        )
        self.nodes.append(node)

        # Transpose back: [B, H, 6T] → [B, 6T, H]
        return self.make_transpose(
            "/emb/upsampled/output_0", "/emb/post_upsample_t/output_0", perm=[0, 2, 1]
        )

    def build_mlp(self, layer_idx: int, hidden_state: str) -> str:
        """Build MLP block (SwiGLU activation) using shared helper."""
        prefix = f"/lfm/layers.{layer_idx}"
        weight_prefix = f"lfm.layers.{layer_idx}"

        residual = hidden_state

        # FFN LayerNorm
        self.add_initializer(
            f"{weight_prefix}.ffn_norm.weight",
            self.weights[f"{weight_prefix}.ffn_norm.weight"].astype(np.float32),
        )
        normed = self.make_layernorm(
            hidden_state,
            f"{weight_prefix}.ffn_norm.weight",
            None,
            f"{prefix}/ffn_norm",
            epsilon=self.norm_eps,
        )

        # Prepare weights (transposed for MatMul)
        w1_name = f"{weight_prefix}.w1.weight"
        w2_name = f"{weight_prefix}.w2.weight"
        w3_name = f"{weight_prefix}.w3.weight"

        self.add_initializer(
            w1_name, self.weights[f"{weight_prefix}.feed_forward.w1.weight"].astype(np.float32).T
        )
        self.add_initializer(
            w2_name, self.weights[f"{weight_prefix}.feed_forward.w2.weight"].astype(np.float32).T
        )
        self.add_initializer(
            w3_name, self.weights[f"{weight_prefix}.feed_forward.w3.weight"].astype(np.float32).T
        )

        # Use shared SwiGLU helper
        return self.build_swiglu_ffn(
            normed, w1_name, w2_name, w3_name, f"{prefix}/mlp", residual=residual
        )

    def build_conv_layer(self, layer_idx: int, hidden_state: str) -> str:
        """Build a conv layer (short convolution with gating)."""
        prefix = f"/lfm/layers.{layer_idx}"
        weight_prefix = f"lfm.layers.{layer_idx}"
        H = self.hidden_size
        L = self.conv_L_cache  # kernel size

        residual = hidden_state

        # Operator LayerNorm
        self.add_initializer(
            f"{weight_prefix}.operator_norm.weight",
            self.weights[f"{weight_prefix}.operator_norm.weight"].astype(np.float32),
        )
        normed = self.make_layernorm(
            hidden_state,
            f"{weight_prefix}.operator_norm.weight",
            None,
            f"{prefix}/operator_norm",
            epsilon=self.norm_eps,
        )

        # In projection: [B, T, H] -> [B, T, 3H]
        in_proj_w = self.weights[f"{weight_prefix}.conv.in_proj.weight"].astype(np.float32).T
        self.add_initializer(f"{weight_prefix}.in_proj.weight", in_proj_w)
        in_proj = self.make_matmul(
            normed, f"{weight_prefix}.in_proj.weight", f"{prefix}/conv/in_proj/output_0"
        )

        # Transpose: [B, T, 3H] -> [B, 3H, T]
        in_proj_t = self.make_transpose(
            in_proj, f"{prefix}/conv/transpose1/output_0", perm=[0, 2, 1]
        )

        # Split into B, C, x (each [B, H, T])
        node = helper.make_node(
            "Split",
            [in_proj_t, self.get_constant([H, H, H])],
            [
                f"{prefix}/conv/B/output_0",
                f"{prefix}/conv/C/output_0",
                f"{prefix}/conv/x/output_0",
            ],
            name=f"{prefix}/conv/split",
            axis=1,
        )
        self.nodes.append(node)

        # Bx = B * x (input gating, no sigmoid)
        Bx = self.make_mul(
            f"{prefix}/conv/B/output_0", f"{prefix}/conv/x/output_0", f"{prefix}/conv/Bx/output_0"
        )

        # Pad Bx for causal convolution: [B, H, T] -> [B, H, L-1 + T]
        self.add_initializer("conv_pads", np.array([0, 0, L - 1, 0, 0, 0], dtype=np.int64))
        Bx_padded = self.make_node(
            "Pad", [Bx, "conv_pads"], [f"{prefix}/conv/padded/output_0"], mode="constant"
        )

        # Depthwise Conv1D (kernel=L, groups=H)
        conv_w = self.weights[f"{weight_prefix}.conv.conv.weight"].astype(np.float32)
        self.add_initializer(f"{weight_prefix}.conv.weight", conv_w)
        conv_out = self.make_node(
            "Conv",
            [Bx_padded, f"{weight_prefix}.conv.weight"],
            [f"{prefix}/conv/conv1d/output_0"],
            kernel_shape=[L],
            group=H,
        )

        # Output gating: y = C * conv_out
        y = self.make_mul(f"{prefix}/conv/C/output_0", conv_out, f"{prefix}/conv/y/output_0")

        # Transpose: [B, H, T] -> [B, T, H]
        y_t = self.make_transpose(y, f"{prefix}/conv/transpose2/output_0", perm=[0, 2, 1])

        # Out projection
        out_proj_w = self.weights[f"{weight_prefix}.conv.out_proj.weight"].astype(np.float32).T
        self.add_initializer(f"{weight_prefix}.out_proj.weight", out_proj_w)
        out_proj = self.make_matmul(
            y_t, f"{weight_prefix}.out_proj.weight", f"{prefix}/conv/out_proj/output_0"
        )

        # Residual
        hidden_state = self.make_add(residual, out_proj, f"{prefix}/conv/residual/output_0")

        # MLP
        return self.build_mlp(layer_idx, hidden_state)

    def build_rope(self, q_4d: str, k_4d: str, prefix: str) -> tuple[str, str]:
        """Apply Rotary Position Embedding (RoPE) to Q and K.

        Input shapes: [B, T, nh, hd] or [B, T, nkv, hd]
        Output shapes: same as input

        PyTorch's rotate_half splits first/second half:
        - rotate_half([x0, ..., x15, x16, ..., x31]) = [-x16, ..., -x31, x0, ..., x15]
        - cos/sin are concatenated: [c0, ..., c15, c0, ..., c15]
        """
        hd = self.head_dim
        rope_theta = self.config.get("rope_theta", 1000000.0)

        # Precompute inverse frequencies
        inv_freq = self.compute_rope_inv_freq(hd, rope_theta)
        self.add_initializer(f"{prefix}/rope_inv_freq", inv_freq)

        # Get sequence length from Q shape [B, T, nh, hd]
        self.make_node("Shape", [q_4d], [f"{prefix}/q_shape/output_0"])
        seq_len = self.make_gather(
            f"{prefix}/q_shape/output_0", self.get_constant(1), f"{prefix}/seq_len/output_0"
        )

        # Create position indices: [0, 1, ..., T-1]
        self.add_initializer("range_start", np.array(0, dtype=np.int64))
        self.add_initializer("range_step", np.array(1, dtype=np.int64))
        positions = self.make_node(
            "Range", ["range_start", seq_len, "range_step"], [f"{prefix}/positions/output_0"]
        )

        # Cast positions to float and reshape to [T, 1]
        positions_f = self.make_node(
            "Cast", [positions], [f"{prefix}/positions_f/output_0"], to=TensorProto.FLOAT
        )
        positions_r = self.make_reshape(
            positions_f, self.get_constant([-1, 1]), f"{prefix}/positions_r/output_0"
        )

        # Compute position * inv_freq: [T, 1] * [hd//2] -> [T, hd//2]
        freqs = self.make_mul(positions_r, f"{prefix}/rope_inv_freq", f"{prefix}/freqs/output_0")

        # Compute cos and sin: [T, hd//2]
        cos_half = self.make_node("Cos", [freqs], [f"{prefix}/cos_half/output_0"])
        sin_half = self.make_node("Sin", [freqs], [f"{prefix}/sin_half/output_0"])

        # Concatenate to get [T, hd]: [c0, c1, ..., c15, c0, c1, ..., c15]
        cos_hd = self.make_concat([cos_half, cos_half], f"{prefix}/cos_hd/output_0", axis=-1)
        sin_hd = self.make_concat([sin_half, sin_half], f"{prefix}/sin_hd/output_0", axis=-1)

        # Reshape for broadcast: [T, hd] -> [1, T, 1, hd] for [B, T, nh, hd]
        cos_bc = self.make_reshape(
            cos_hd, self.get_constant([1, -1, 1, hd]), f"{prefix}/cos_bc/output_0"
        )
        sin_bc = self.make_reshape(
            sin_hd, self.get_constant([1, -1, 1, hd]), f"{prefix}/sin_bc/output_0"
        )

        # === rotate_half for Q ===
        half_hd = hd // 2
        q_first = f"{prefix}/q_first/output_0"
        q_second = f"{prefix}/q_second/output_0"
        node = helper.make_node(
            "Split",
            [q_4d, self.get_constant([half_hd, half_hd])],
            [q_first, q_second],
            name=f"{prefix}/q_split",
            axis=-1,
        )
        self.nodes.append(node)

        # rotate_half: [-second, first]
        q_second_neg = self.make_node("Neg", [q_second], [f"{prefix}/q_second_neg/output_0"])
        q_rot_half = self.make_concat(
            [q_second_neg, q_first], f"{prefix}/q_rot_half/output_0", axis=-1
        )

        # Apply: q_rot = q * cos + rotate_half(q) * sin
        q_cos = self.make_mul(q_4d, cos_bc, f"{prefix}/q_cos/output_0")
        q_sin = self.make_mul(q_rot_half, sin_bc, f"{prefix}/q_sin/output_0")
        q_rope = self.make_add(q_cos, q_sin, f"{prefix}/q_rope/output_0")

        # === rotate_half for K ===
        k_first = f"{prefix}/k_first/output_0"
        k_second = f"{prefix}/k_second/output_0"
        node = helper.make_node(
            "Split",
            [k_4d, self.get_constant([half_hd, half_hd])],
            [k_first, k_second],
            name=f"{prefix}/k_split",
            axis=-1,
        )
        self.nodes.append(node)

        k_second_neg = self.make_node("Neg", [k_second], [f"{prefix}/k_second_neg/output_0"])
        k_rot_half = self.make_concat(
            [k_second_neg, k_first], f"{prefix}/k_rot_half/output_0", axis=-1
        )

        k_cos = self.make_mul(k_4d, cos_bc, f"{prefix}/k_cos/output_0")
        k_sin = self.make_mul(k_rot_half, sin_bc, f"{prefix}/k_sin/output_0")
        k_rope = self.make_add(k_cos, k_sin, f"{prefix}/k_rope/output_0")

        return q_rope, k_rope

    def build_attention_layer(self, layer_idx: int, hidden_state: str) -> str:
        """Build a sliding attention layer with causal sliding window mask and RoPE."""
        prefix = f"/lfm/layers.{layer_idx}"
        weight_prefix = f"lfm.layers.{layer_idx}"
        H = self.hidden_size
        nh = self.num_attention_heads
        nkv = self.num_key_value_heads
        hd = self.head_dim

        residual = hidden_state

        # Operator LayerNorm
        self.add_initializer(
            f"{weight_prefix}.operator_norm.weight",
            self.weights[f"{weight_prefix}.operator_norm.weight"].astype(np.float32),
        )
        normed = self.make_layernorm(
            hidden_state,
            f"{weight_prefix}.operator_norm.weight",
            None,
            f"{prefix}/operator_norm",
            epsilon=self.norm_eps,
        )

        # Q/K/V projections
        q_w = self.weights[f"{weight_prefix}.self_attn.q_proj.weight"].astype(np.float32).T
        k_w = self.weights[f"{weight_prefix}.self_attn.k_proj.weight"].astype(np.float32).T
        v_w = self.weights[f"{weight_prefix}.self_attn.v_proj.weight"].astype(np.float32).T

        self.add_initializer(f"{weight_prefix}.q.weight", q_w)
        self.add_initializer(f"{weight_prefix}.k.weight", k_w)
        self.add_initializer(f"{weight_prefix}.v.weight", v_w)

        q = self.make_matmul(normed, f"{weight_prefix}.q.weight", f"{prefix}/attn/q/output_0")
        k = self.make_matmul(normed, f"{weight_prefix}.k.weight", f"{prefix}/attn/k/output_0")
        v = self.make_matmul(normed, f"{weight_prefix}.v.weight", f"{prefix}/attn/v/output_0")

        # Q/K LayerNorm (per-head)
        # Note: Cannot use make_per_head_layernorm here because it flattens to [-1, hd]
        # which loses batch dimension info needed for output shape [0, -1, nh, hd].
        q_ln_w = self.weights[f"{weight_prefix}.self_attn.q_layernorm.weight"].astype(np.float32)
        k_ln_w = self.weights[f"{weight_prefix}.self_attn.k_layernorm.weight"].astype(np.float32)
        self.add_initializer(f"{weight_prefix}.q_ln.weight", q_ln_w)
        self.add_initializer(f"{weight_prefix}.k_ln.weight", k_ln_w)

        # Reshape Q for per-head norm: [B, T, H] -> [B, T*nh, hd]
        q_reshaped = self.make_reshape(
            q, self.get_constant([0, -1, hd]), f"{prefix}/attn/q_reshape1/output_0"
        )
        q_normed = self.make_layernorm(
            q_reshaped,
            f"{weight_prefix}.q_ln.weight",
            None,
            f"{prefix}/attn/q_norm",
            epsilon=self.norm_eps,
        )
        q_4d = self.make_reshape(
            q_normed, self.get_constant([0, -1, nh, hd]), f"{prefix}/attn/q_4d/output_0"
        )

        k_reshaped = self.make_reshape(
            k, self.get_constant([0, -1, hd]), f"{prefix}/attn/k_reshape1/output_0"
        )
        k_normed = self.make_layernorm(
            k_reshaped,
            f"{weight_prefix}.k_ln.weight",
            None,
            f"{prefix}/attn/k_norm",
            epsilon=self.norm_eps,
        )
        k_4d = self.make_reshape(
            k_normed, self.get_constant([0, -1, nkv, hd]), f"{prefix}/attn/k_4d/output_0"
        )

        # Apply RoPE to Q and K (before transpose)
        q_rope, k_rope = self.build_rope(q_4d, k_4d, f"{prefix}/rope")

        # Transpose: [B, T, nh, hd] -> [B, nh, T, hd]
        q_4d_t = self.make_transpose(q_rope, f"{prefix}/attn/q_4d_t/output_0", perm=[0, 2, 1, 3])
        k_4d_t = self.make_transpose(k_rope, f"{prefix}/attn/k_4d_t/output_0", perm=[0, 2, 1, 3])

        v_4d = self.make_reshape(
            v, self.get_constant([0, -1, nkv, hd]), f"{prefix}/attn/v_4d/output_0"
        )
        v_4d_t = self.make_transpose(v_4d, f"{prefix}/attn/v_4d_t/output_0", perm=[0, 2, 1, 3])

        # Scaled dot product attention
        scale = 1.0 / np.sqrt(hd)

        # GQA: expand KV heads to match Q heads
        k_gqa, v_gqa = self.expand_kv_for_gqa(k_4d_t, v_4d_t, nh, nkv, hd, f"{prefix}/attn")

        # K transpose for Q @ K^T: [B, nh, T, hd] -> [B, nh, hd, T]
        k_gqa_t = self.make_transpose(k_gqa, f"{prefix}/attn/k_gqa_t/output_0", perm=[0, 1, 3, 2])

        # Attention scores: Q @ K^T [B, nh, T, T]
        scores = self.make_matmul(q_4d_t, k_gqa_t, f"{prefix}/attn/scores/output_0")
        scores_scaled = self.make_mul(
            scores,
            self.get_constant(scale, dtype=np.float32),
            f"{prefix}/attn/scores_scaled/output_0",
        )

        # === Causal Sliding Window Mask ===
        self.make_node("Shape", [scores], [f"{prefix}/attn/scores_shape/output_0"])
        seq_len = self.make_gather(
            f"{prefix}/attn/scores_shape/output_0",
            self.get_constant(2),
            f"{prefix}/attn/seq_len/output_0",
        )

        # Create position indices: [0, 1, 2, ..., T-1]
        indices = self.make_node(
            "Range",
            ["range_start", seq_len, "range_step"],
            [f"{prefix}/attn/indices/output_0"],
        )

        # Create row indices [T, 1] and col indices [1, T]
        row_idx = self.make_unsqueeze(
            indices, self.get_constant([1]), f"{prefix}/attn/row_idx/output_0"
        )
        col_idx = self.make_unsqueeze(
            indices, self.get_constant([0]), f"{prefix}/attn/col_idx/output_0"
        )

        # Distance matrix: d_idx = col_idx - row_idx [T, T]
        d_idx = self.make_node("Sub", [col_idx, row_idx], [f"{prefix}/attn/d_idx/output_0"])

        # Mask conditions
        cond1 = self.make_node(
            "LessOrEqual", [d_idx, self.get_constant(0)], [f"{prefix}/attn/cond1/output_0"]
        )
        sw_neg = -self.sliding_window
        cond2 = self.make_node(
            "Greater", [d_idx, self.get_constant(sw_neg)], [f"{prefix}/attn/cond2/output_0"]
        )

        # Combined mask
        valid_mask = self.make_node("And", [cond1, cond2], [f"{prefix}/attn/valid_mask/output_0"])
        invalid_mask = self.make_node("Not", [valid_mask], [f"{prefix}/attn/invalid_mask/output_0"])
        invalid_mask_f = self.make_node(
            "Cast", [invalid_mask], [f"{prefix}/attn/invalid_mask_f/output_0"], to=TensorProto.FLOAT
        )
        mask_bias = self.make_mul(
            invalid_mask_f,
            self.get_constant(-1e9, dtype=np.float32),
            f"{prefix}/attn/mask_bias/output_0",
        )

        # Add mask bias to scores
        scores_masked = self.make_add(
            scores_scaled, mask_bias, f"{prefix}/attn/scores_masked/output_0"
        )

        # Softmax on masked scores
        attn_weights = self.make_node(
            "Softmax", [scores_masked], [f"{prefix}/attn/softmax/output_0"], axis=-1
        )

        # Attention output: [B, nh, T, hd]
        attn_out = self.make_matmul(attn_weights, v_gqa, f"{prefix}/attn/attn_out/output_0")

        # Reshape back: [B, nh, T, hd] -> [B, T, H]
        attn_out_t = self.make_transpose(
            attn_out, f"{prefix}/attn/attn_out_t/output_0", perm=[0, 2, 1, 3]
        )
        attn_out_3d = self.make_reshape(
            attn_out_t, self.get_constant([0, -1, H]), f"{prefix}/attn/attn_out_3d/output_0"
        )

        # Output projection
        o_w = self.weights[f"{weight_prefix}.self_attn.out_proj.weight"].astype(np.float32).T
        self.add_initializer(f"{weight_prefix}.o.weight", o_w)
        o_proj = self.make_matmul(
            attn_out_3d, f"{weight_prefix}.o.weight", f"{prefix}/attn/o_proj/output_0"
        )

        # Residual
        hidden_state = self.make_add(residual, o_proj, f"{prefix}/attn/residual/output_0")

        # MLP
        return self.build_mlp(layer_idx, hidden_state)

    def build_output_linear(self, hidden_state: str) -> str:
        """Build final linear projection to STFT space."""
        # Final embedding norm
        if "lfm.embedding_norm.weight" in self.weights:
            self.add_initializer(
                "lfm.embedding_norm.weight",
                self.weights["lfm.embedding_norm.weight"].astype(np.float32),
            )
            hidden_state = self.make_layernorm(
                hidden_state,
                "lfm.embedding_norm.weight",
                None,
                "/lfm/final_norm",
                epsilon=self.norm_eps,
            )

        # Linear projection: [B, T, H] -> [B, T, output_size]
        lin_w = self.weights["lin.weight"].astype(np.float32).T
        lin_b = self.weights.get("lin.bias", np.zeros(self.output_size)).astype(np.float32)
        self.add_initializer("lin.weight", lin_w)
        self.add_initializer("lin.bias", lin_b)

        lin_out = self.make_matmul(hidden_state, "lin.weight", "/lin/matmul/output_0")
        return self.make_add(lin_out, "lin.bias", "stft_features")

    def build(self) -> onnx.ModelProto:
        """Build the complete audio detokenizer ONNX model."""
        # Build inputs/outputs
        self.build_inputs()
        self.build_outputs()

        # Build embedding
        hidden_state = self.build_embedding()

        # Build LFM layers
        for layer_idx in range(self.num_layers):
            layer_type = self.layer_types[layer_idx]
            if layer_type == "sliding_attention":
                logger.info(
                    f"Building detokenizer layer {layer_idx} (attention with sliding window)..."
                )
                hidden_state = self.build_attention_layer(layer_idx, hidden_state)
            else:
                logger.info(f"Building detokenizer layer {layer_idx} (conv)...")
                hidden_state = self.build_conv_layer(layer_idx, hidden_state)

        # Build output linear
        self.build_output_linear(hidden_state)

        # Use inherited build_graph method
        return self.build_graph("audio_detokenizer", ms_domain=False)


def export_audio_detokenizer_builder(model_path: str, onnx_dir: pathlib.Path) -> pathlib.Path:
    """Export audio detokenizer using ONNX builder with full LFM layers.

    The audio detokenizer converts audio codes to waveform:
    1. Embedding: [B, 8, T] -> [B, T, 512] (fused codebook embedding)
    2. LFM: [B, T, 512] -> [B, T, 512] (8-layer transformer with conv and attention)
    3. Linear: [B, T, 512] -> [B, T, 1282] (STFT space projection)
    4. ISTFT: [B, T, 1282] -> waveform (done in numpy/scipy)

    This exports steps 1-3 to ONNX. Step 4 is done in numpy.
    """
    logger.info("Exporting audio_detokenizer.onnx (full LFM builder version)...")

    import json as json_module

    from liquid_audio.utils import get_model_dir
    from safetensors.torch import load_file

    cache_dir = get_model_dir(model_path)
    config_path = cache_dir / "audio_detokenizer" / "config.json"

    if not config_path.exists():
        logger.warning("Audio detokenizer not found in model, skipping export")
        return None

    with open(config_path) as f:
        detok_config = json_module.load(f)

    logger.info(f"Audio detokenizer config: {detok_config}")

    # Load weights directly from checkpoint
    weights_path = cache_dir / "audio_detokenizer" / "model.safetensors"
    checkpoint_weights = load_file(str(weights_path))

    # Convert to numpy
    detok_weights = {}
    for name, param in checkpoint_weights.items():
        detok_weights[name] = param.float().cpu().numpy()

    logger.info(f"Loaded {len(detok_weights)} audio detokenizer weights from checkpoint")

    # Add ISTFT window (needed for inference)
    import torch

    istft_window = torch.hann_window(1280).numpy()
    detok_weights["istft.window"] = istft_window

    # Build the model using AudioDetokenizerBuilder
    builder = AudioDetokenizerBuilder(detok_config, detok_weights)
    model = builder.build()

    # Save the model
    output_path = onnx_dir / "audio_detokenizer.onnx"
    onnx.save_model(model, str(output_path))
    logger.info(f"audio_detokenizer saved to {output_path}")

    # Note: ISTFT window is generated at runtime via np.hanning() in infer.py
    # This makes inference compatible with transformers.js which cannot load numpy files

    return output_path
