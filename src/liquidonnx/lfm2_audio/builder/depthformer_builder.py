"""
Depthformer ONNX builders for autoregressive audio codebook prediction.

The depthformer predicts 8 audio codebook tokens autoregressively:
1. depth_linear: [B, 2048] → [B, 8, 1024] (called 1× per frame)
2. depthformer_unified: Transformer with KV cache (called 8× per frame)

Architecture:
    decoder hidden_states → depth_linear → 8 slices
                                            ↓
                          depthformer_unified (8 iterations) → 8 tokens
"""

import logging
import pathlib

import numpy as np
import onnx
from onnx import TensorProto, helper

from liquidonnx.builder_base import ONNXBuilderBase

logger = logging.getLogger(__name__)


class DepthLinearBuilder(ONNXBuilderBase):
    """Builder for depth_linear.onnx: [B, 2048] → [B, 8, 1024].

    Projects decoder hidden states to 8 depth slices for depthformer input.
    Simple MatMul + Add + Reshape operation.
    """

    def __init__(
        self, input_hidden_size: int = 2048, num_codebooks: int = 8, depth_dim: int = 1024
    ):
        super().__init__()
        self.input_hidden_size = input_hidden_size
        self.num_codebooks = num_codebooks
        self.depth_dim = depth_dim
        self.output_size = num_codebooks * depth_dim  # 8192

    def build_inputs(self):
        """Build graph input for decoder hidden states."""
        self.inputs.append(
            helper.make_tensor_value_info(
                "hidden_states",
                TensorProto.FLOAT,
                ["batch", self.input_hidden_size],
            )
        )

    def build_outputs(self):
        """Build graph output for depth slices."""
        self.outputs.append(
            helper.make_tensor_value_info(
                "depth_hidden",
                TensorProto.FLOAT,
                ["batch", self.num_codebooks, self.depth_dim],
            )
        )

    def build_projection(self) -> str:
        """Build depth_linear projection: [B, 2048] → [B, 8, 1024].

        Operations:
            1. MatMul: [B, 2048] × [2048, 8192] → [B, 8192]
            2. Add: [B, 8192] + [8192] → [B, 8192]
            3. Reshape: [B, 8192] → [B, 8, 1024]
        """
        prefix = "/depth_linear"

        # MatMul: hidden_states @ weight
        matmul_out = self.make_matmul(
            "hidden_states",
            "depth_linear.weight",
            f"{prefix}/MatMul/output_0",
        )

        # Add: + bias
        add_out = self.make_add(matmul_out, "depth_linear.bias", f"{prefix}/Add/output_0")

        # Reshape: [B, 8192] → [B, 8, 1024]
        # Dynamic batch dim with -1
        reshape_shape = self.get_constant([-1, self.num_codebooks, self.depth_dim])
        return self.make_reshape(add_out, reshape_shape, "depth_hidden")

    def load_weights(self, model_path: str):
        """Load depth_linear weights from HuggingFace model.

        Args:
            model_path: HuggingFace model ID or local path
        """
        from huggingface_hub import hf_hub_download
        from safetensors import safe_open

        logger.info(f"Loading depth_linear weights from {model_path}...")

        safetensors_path = hf_hub_download(model_path, "model.safetensors")

        with safe_open(safetensors_path, framework="np", device="cpu") as f:
            for key in f.keys():
                if key.startswith("depth_linear."):
                    self.weights[key] = f.get_tensor(key)

        logger.info(f"Loaded {len(self.weights)} weights")

    def prepare_weights(self):
        """Register weights as initializers (transposed for MatMul)."""
        # Transpose weight from [out, in] to [in, out] for MatMul
        weight = self.weights["depth_linear.weight"].astype(np.float32).T
        bias = self.weights["depth_linear.bias"].astype(np.float32)

        self.add_initializer("depth_linear.weight", weight)
        self.add_initializer("depth_linear.bias", bias)

    def build(self, model_path: str) -> onnx.ModelProto:
        """Build the complete ONNX model for depth_linear.

        Args:
            model_path: HuggingFace model ID or local path

        Returns:
            ONNX ModelProto
        """
        logger.info("Building depth_linear ONNX model...")

        # Load weights
        self.load_weights(model_path)

        # Build graph structure
        self.build_inputs()
        self.build_outputs()

        # Prepare weights as initializers
        self.prepare_weights()

        # Build the projection graph
        self.build_projection()

        model = self.build_graph("depth_linear")
        logger.info(f"Model built: {len(self.nodes)} nodes")
        return model


def export_depth_linear_builder(model_path: str, onnx_dir: pathlib.Path) -> pathlib.Path:
    """Export depth_linear.onnx using ONNX builder (no torch.onnx.export).

    Args:
        model_path: HuggingFace model ID or local path
        onnx_dir: Output directory for ONNX models

    Returns:
        Path to exported vocoder_projection.onnx
    """
    builder = DepthLinearBuilder()
    model = builder.build(model_path)

    output_path = onnx_dir / "vocoder_projection.onnx"
    onnx.save(model, str(output_path))

    logger.info(f"vocoder_projection saved to {output_path}")
    return output_path


class DepthformerUnifiedBuilder(ONNXBuilderBase):
    """Builder for depthformer_unified.onnx: autoregressive transformer with KV cache.

    Consolidates transformer step, 8 embedding tables, and 8 logits projections
    into a single ONNX model. Uses step_idx input to select appropriate weights.

    Architecture (per step):
        1. Gather current depth slice by step_idx
        2. Lookup prev_token embedding (zero for step 0)
        3. Add slice + embedding → transformer input [B, 1, 1024]
        4. 6 transformer layers with KV cache (GQA attention + SwiGLU FFN)
        5. Step-indexed RMSNorm and logits projection

    Inputs:
        depth_slices: [B, 8, 1024] - All 8 slices from depth_linear
        step_idx: scalar int64 - Which codebook step (0-7)
        prev_token: [B] int64 - Previous step's sampled token
        past_keys: [6, B, past_len, 8, 32] - KV cache keys
        past_values: [6, B, past_len, 8, 32] - KV cache values

    Outputs:
        logits: [B, 2049] - Codebook logits
        token_embed: [B, 1024] - Placeholder (unused)
        new_keys: [6, B, new_len, 8, 32] - Updated KV cache keys
        new_values: [6, B, new_len, 8, 32] - Updated KV cache values
    """

    def __init__(self):
        super().__init__()
        # Architecture constants
        self.dim = 1024
        self.num_layers = 6
        self.num_heads = 32  # Q heads
        self.num_kv_heads = 8  # KV heads (GQA)
        self.head_dim = 32
        self.intermediate_size = 2816
        self.vocab_size = 2049
        self.num_codebooks = 8
        self.norm_eps = 1e-5
        self.rope_theta = 10000.0
        self.max_seq_len = 16  # Max positions for RoPE (8 steps + cache)

    def build_inputs(self):
        """Build graph inputs."""
        # depth_slices: [B, 8, 1024]
        self.inputs.append(
            helper.make_tensor_value_info("depth_slices", TensorProto.FLOAT, ["batch", 8, self.dim])
        )
        # step_idx: scalar
        self.inputs.append(helper.make_tensor_value_info("step_idx", TensorProto.INT64, []))
        # prev_token: [B]
        self.inputs.append(
            helper.make_tensor_value_info("prev_token", TensorProto.INT64, ["batch"])
        )
        # past_keys: [6, B, past_len, 8, 32]
        self.inputs.append(
            helper.make_tensor_value_info(
                "past_keys",
                TensorProto.FLOAT,
                [self.num_layers, "batch", "past_len", self.num_kv_heads, self.head_dim],
            )
        )
        # past_values: [6, B, past_len, 8, 32]
        self.inputs.append(
            helper.make_tensor_value_info(
                "past_values",
                TensorProto.FLOAT,
                [self.num_layers, "batch", "past_len", self.num_kv_heads, self.head_dim],
            )
        )

    def build_outputs(self):
        """Build graph outputs."""
        # logits: [B, 2049]
        self.outputs.append(
            helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", self.vocab_size])
        )
        # token_embed: [B, 1024] - placeholder
        self.outputs.append(
            helper.make_tensor_value_info("token_embed", TensorProto.FLOAT, ["batch", self.dim])
        )
        # new_keys: [6, B, new_len, 8, 32]
        self.outputs.append(
            helper.make_tensor_value_info(
                "new_keys",
                TensorProto.FLOAT,
                [self.num_layers, "batch", "new_len", self.num_kv_heads, self.head_dim],
            )
        )
        # new_values: [6, B, new_len, 8, 32]
        self.outputs.append(
            helper.make_tensor_value_info(
                "new_values",
                TensorProto.FLOAT,
                [self.num_layers, "batch", "new_len", self.num_kv_heads, self.head_dim],
            )
        )

    def compute_rope_freqs(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute rotary position embedding frequencies.

        Returns:
            freqs_cos: [max_seq_len, head_dim//2]
            freqs_sin: [max_seq_len, head_dim//2]
        """
        dim = self.head_dim
        freqs = 1.0 / (self.rope_theta ** (np.arange(0, dim, 2) / dim))
        t = np.arange(self.max_seq_len)
        freqs = np.outer(t, freqs)  # [max_seq_len, head_dim//2]
        freqs_cos = np.cos(freqs).astype(np.float32)
        freqs_sin = np.sin(freqs).astype(np.float32)
        return freqs_cos, freqs_sin

    def build_get_current_slice(self) -> str:
        """Build gather operation to get current depth slice.

        depth_slices: [B, 8, 1024], step_idx: scalar → [B, 1024]
        """
        prefix = "/get_slice"

        # Get batch size from depth_slices shape
        self.make_node("Shape", ["depth_slices"], [f"{prefix}/shape/output_0"])
        batch_size = self.make_gather(
            f"{prefix}/shape/output_0",
            self.get_constant(0),
            f"{prefix}/batch/output_0",
        )

        # Expand step_idx for gather: scalar → [B, 1, 1024]
        step_unsq1 = self.make_unsqueeze(
            "step_idx", self.get_constant([0]), f"{prefix}/step_unsq1/output_0"
        )
        step_unsq2 = self.make_unsqueeze(
            step_unsq1, self.get_constant([0]), f"{prefix}/step_unsq2/output_0"
        )
        step_unsq3 = self.make_unsqueeze(
            step_unsq2, self.get_constant([0]), f"{prefix}/step_unsq3/output_0"
        )

        # Expand to [B, 1, 1024]
        batch_unsq = self.make_unsqueeze(
            batch_size, self.get_constant([0]), f"{prefix}/batch_unsq/output_0"
        )
        expand_shape = self.make_concat(
            [batch_unsq, self.get_constant([1]), self.get_constant([self.dim])],
            f"{prefix}/expand_shape/output_0",
            axis=0,
        )
        step_expanded = self.make_node(
            "Expand", [step_unsq3, expand_shape], [f"{prefix}/step_exp/output_0"]
        )

        # Gather from depth_slices along axis=1
        gathered = self.make_node(
            "GatherElements",
            ["depth_slices", step_expanded],
            [f"{prefix}/gather/output_0"],
            axis=1,
        )

        # Squeeze dim 1: [B, 1, 1024] → [B, 1024]
        return self.make_node(
            "Squeeze",
            [gathered, self.get_constant([1])],
            [f"{prefix}/squeeze/output_0"],
        )

    def build_prev_embedding(self) -> str:
        """Build previous token embedding lookup with step 0 handling.

        For step 0: returns zeros
        For steps 1-7: looks up prev_token in embed_weights[step_idx-1]
        """
        prefix = "/prev_embed"

        # Clamp step_idx-1 to [0, 7]
        step_minus_1 = self.make_add(
            "step_idx", self.get_constant(-1), f"{prefix}/step_m1/output_0"
        )
        prev_step = self.make_node(
            "Clip",
            [step_minus_1, self.get_constant(0), self.get_constant(7)],
            [f"{prefix}/prev_step/output_0"],
        )

        # Get embedding table for prev_step: stacked_embeds[prev_step] → [2049, 1024]
        # stacked_embeds: [8, 2049, 1024]
        prev_embed_table = self.make_gather(
            "stacked_embed_weights", prev_step, f"{prefix}/table/output_0", axis=0
        )

        # Look up prev_token: [B] → [B, 1024]
        prev_embed_raw = self.make_node(
            "Gather",
            [prev_embed_table, "prev_token"],
            [f"{prefix}/lookup/output_0"],
            axis=0,
        )

        # Zero out for step 0: mask = (step_idx == 0) ? 0 : 1
        is_zero = self.make_node(
            "Equal", ["step_idx", self.get_constant(0)], [f"{prefix}/is_zero/output_0"]
        )
        is_zero_float = self.make_node(
            "Cast", [is_zero], [f"{prefix}/is_zero_f/output_0"], to=TensorProto.FLOAT
        )
        # mask = 1 - is_zero_float
        one_minus = self.make_add(
            self.get_constant(1.0, dtype=np.float32),
            self.make_node("Neg", [is_zero_float], [f"{prefix}/neg_is_zero/output_0"]),
            f"{prefix}/mask/output_0",
        )
        # Unsqueeze mask for broadcast: scalar → [1, 1]
        mask_unsq = self.make_unsqueeze(
            one_minus, self.get_constant([0]), f"{prefix}/mask_unsq/output_0"
        )

        return self.make_mul(prev_embed_raw, mask_unsq, f"{prefix}/masked/output_0")

    def build_rotary_embedding(
        self, q: str, k: str, layer_idx: int, past_len_name: str
    ) -> tuple[str, str]:
        """Build rotary position embedding for Q and K.

        Args:
            q: Query tensor name [B, 1, num_heads, head_dim]
            k: Key tensor name [B, 1, num_kv_heads, head_dim]
            layer_idx: Layer index
            past_len_name: Name of past_len tensor

        Returns:
            (rotated_q, rotated_k) tensor names
        """
        prefix = f"/layers.{layer_idx}/rope"
        hd = self.head_dim

        # Slice freqs for current position: freqs[past_len:past_len+1]
        pos_start = self.make_unsqueeze(
            past_len_name, self.get_constant([0]), f"{prefix}/pos_start/output_0"
        )
        pos_end = self.make_add(pos_start, self.get_constant([1]), f"{prefix}/pos_end/output_0")
        freqs_cos = self.make_slice(
            "rope_freqs_cos",
            pos_start,
            pos_end,
            self.get_constant([0]),
            f"{prefix}/cos_slice/output_0",
        )  # [1, hd//2]
        freqs_sin = self.make_slice(
            "rope_freqs_sin",
            pos_start,
            pos_end,
            self.get_constant([0]),
            f"{prefix}/sin_slice/output_0",
        )  # [1, hd//2]

        # Reshape Q/K for real/imag split: [B, 1, H, D] → [B, 1, H, D//2, 2]
        q_reshaped = self.make_reshape(
            q,
            self.get_constant([0, 1, -1, hd // 2, 2]),
            f"{prefix}/q_reshape/output_0",
        )
        k_reshaped = self.make_reshape(
            k,
            self.get_constant([0, 1, -1, hd // 2, 2]),
            f"{prefix}/k_reshape/output_0",
        )

        # Split real/imag
        self.make_node(
            "Split",
            [q_reshaped, self.get_constant([1, 1])],
            [f"{prefix}/q_real/output_0", f"{prefix}/q_imag/output_0"],
            axis=-1,
        )
        self.make_node(
            "Split",
            [k_reshaped, self.get_constant([1, 1])],
            [f"{prefix}/k_real/output_0", f"{prefix}/k_imag/output_0"],
            axis=-1,
        )

        # Squeeze last dim: [B, 1, H, D//2, 1] → [B, 1, H, D//2]
        q_real = self.make_node(
            "Squeeze",
            [f"{prefix}/q_real/output_0", self.get_constant([-1])],
            [f"{prefix}/q_real_sq/output_0"],
        )
        q_imag = self.make_node(
            "Squeeze",
            [f"{prefix}/q_imag/output_0", self.get_constant([-1])],
            [f"{prefix}/q_imag_sq/output_0"],
        )
        k_real = self.make_node(
            "Squeeze",
            [f"{prefix}/k_real/output_0", self.get_constant([-1])],
            [f"{prefix}/k_real_sq/output_0"],
        )
        k_imag = self.make_node(
            "Squeeze",
            [f"{prefix}/k_imag/output_0", self.get_constant([-1])],
            [f"{prefix}/k_imag_sq/output_0"],
        )

        # Broadcast freqs: [1, D//2] → [1, 1, 1, D//2]
        cos = self.make_unsqueeze(
            freqs_cos, self.get_constant([0, 2]), f"{prefix}/cos_bcast/output_0"
        )
        sin = self.make_unsqueeze(
            freqs_sin, self.get_constant([0, 2]), f"{prefix}/sin_bcast/output_0"
        )

        # Apply rotation: (a*cos - b*sin) + i*(a*sin + b*cos)
        # Q
        q_out_real_1 = self.make_mul(q_real, cos, f"{prefix}/q_rc/output_0")
        q_out_real_2 = self.make_mul(q_imag, sin, f"{prefix}/q_is/output_0")
        q_out_real = self.make_add(
            q_out_real_1,
            self.make_node("Neg", [q_out_real_2], [f"{prefix}/q_is_neg/output_0"]),
            f"{prefix}/q_out_real/output_0",
        )
        q_out_imag_1 = self.make_mul(q_real, sin, f"{prefix}/q_rs/output_0")
        q_out_imag_2 = self.make_mul(q_imag, cos, f"{prefix}/q_ic/output_0")
        q_out_imag = self.make_add(q_out_imag_1, q_out_imag_2, f"{prefix}/q_out_imag/output_0")

        # K
        k_out_real_1 = self.make_mul(k_real, cos, f"{prefix}/k_rc/output_0")
        k_out_real_2 = self.make_mul(k_imag, sin, f"{prefix}/k_is/output_0")
        k_out_real = self.make_add(
            k_out_real_1,
            self.make_node("Neg", [k_out_real_2], [f"{prefix}/k_is_neg/output_0"]),
            f"{prefix}/k_out_real/output_0",
        )
        k_out_imag_1 = self.make_mul(k_real, sin, f"{prefix}/k_rs/output_0")
        k_out_imag_2 = self.make_mul(k_imag, cos, f"{prefix}/k_ic/output_0")
        k_out_imag = self.make_add(k_out_imag_1, k_out_imag_2, f"{prefix}/k_out_imag/output_0")

        # Stack and flatten: [B, 1, H, D//2] × 2 → [B, 1, H, D//2, 2] → [B, 1, H, D]
        q_stacked = self.make_node(
            "Concat",
            [
                self.make_unsqueeze(
                    q_out_real, self.get_constant([-1]), f"{prefix}/q_real_unsq/output_0"
                ),
                self.make_unsqueeze(
                    q_out_imag, self.get_constant([-1]), f"{prefix}/q_imag_unsq/output_0"
                ),
            ],
            [f"{prefix}/q_stack/output_0"],
            axis=-1,
        )
        k_stacked = self.make_node(
            "Concat",
            [
                self.make_unsqueeze(
                    k_out_real, self.get_constant([-1]), f"{prefix}/k_real_unsq/output_0"
                ),
                self.make_unsqueeze(
                    k_out_imag, self.get_constant([-1]), f"{prefix}/k_imag_unsq/output_0"
                ),
            ],
            [f"{prefix}/k_stack/output_0"],
            axis=-1,
        )

        q_out = self.make_reshape(
            q_stacked,
            self.get_constant([0, 1, -1, hd]),
            f"{prefix}/q_out/output_0",
        )
        k_out = self.make_reshape(
            k_stacked,
            self.get_constant([0, 1, -1, hd]),
            f"{prefix}/k_out/output_0",
        )

        return q_out, k_out

    def build_transformer_layer(
        self, x: str, layer_idx: int, past_k: str, past_v: str, past_len_name: str
    ) -> tuple[str, str, str]:
        """Build a single transformer layer.

        Args:
            x: Input tensor [B, 1, 1024]
            layer_idx: Layer index
            past_k: Past keys [B, past_len, num_kv_heads, head_dim]
            past_v: Past values [B, past_len, num_kv_heads, head_dim]
            past_len_name: Name of past_len scalar tensor

        Returns:
            (output, new_k, new_v) tensor names
        """
        prefix = f"/layers.{layer_idx}"
        nh = self.num_heads
        nkv = self.num_kv_heads
        hd = self.head_dim

        residual = x

        # === LayerNorm (SimplifiedLayerNormalization = RMSNorm) ===
        normed = self.make_layernorm(
            x, f"layer.{layer_idx}.operator_norm.weight", None, f"{prefix}/op_norm"
        )

        # === QKV Projection ===
        # qkv_proj: [1024] → [1536] (Q=1024, K=256, V=256)
        qkv = self.make_linear(
            normed,
            self.weights[f"depthformer.layers.{layer_idx}.operator.qkv_proj.weight"],
            f"layer.{layer_idx}.qkv.weight",
            f"{prefix}/qkv",
        )

        # Split QKV
        q_dim = nh * hd  # 1024
        kv_dim = nkv * hd  # 256
        self.make_node(
            "Split",
            [qkv, self.get_constant([q_dim, kv_dim, kv_dim])],
            [f"{prefix}/q/output_0", f"{prefix}/k/output_0", f"{prefix}/v/output_0"],
            axis=-1,
        )

        # Reshape to [B, 1, H, D]
        q_4d = self.make_reshape(
            f"{prefix}/q/output_0",
            self.get_constant([0, 1, nh, hd]),
            f"{prefix}/q_4d/output_0",
        )
        k_4d = self.make_reshape(
            f"{prefix}/k/output_0",
            self.get_constant([0, 1, nkv, hd]),
            f"{prefix}/k_4d/output_0",
        )
        v_4d = self.make_reshape(
            f"{prefix}/v/output_0",
            self.get_constant([0, 1, nkv, hd]),
            f"{prefix}/v_4d/output_0",
        )

        # === Q/K LayerNorm (per-head) ===
        # Reshape to [B*1*H, D] for layernorm, then back
        q_flat = self.make_reshape(q_4d, self.get_constant([-1, hd]), f"{prefix}/q_flat/output_0")
        k_flat = self.make_reshape(k_4d, self.get_constant([-1, hd]), f"{prefix}/k_flat/output_0")

        q_normed = self.make_layernorm(
            q_flat, f"layer.{layer_idx}.q_ln.weight", None, f"{prefix}/q_ln"
        )
        k_normed = self.make_layernorm(
            k_flat, f"layer.{layer_idx}.k_ln.weight", None, f"{prefix}/k_ln"
        )

        # Use -1 for batch dimension since 0 would pick up wrong dim after flatten
        q_4d = self.make_reshape(
            q_normed, self.get_constant([-1, 1, nh, hd]), f"{prefix}/q_4d_ln/output_0"
        )
        k_4d = self.make_reshape(
            k_normed, self.get_constant([-1, 1, nkv, hd]), f"{prefix}/k_4d_ln/output_0"
        )

        # === Rotary Embeddings ===
        q_rope, k_rope = self.build_rotary_embedding(q_4d, k_4d, layer_idx, past_len_name)

        # === KV Cache Concat ===
        # Concat along seq_len (dim 1): [B, past_len, H, D] + [B, 1, H, D]
        new_k = self.make_concat([past_k, k_rope], f"{prefix}/new_k/output_0", axis=1)
        new_v = self.make_concat([past_v, v_4d], f"{prefix}/new_v/output_0", axis=1)

        # === Attention ===
        # Transpose for attention: [B, S, H, D] → [B, H, S, D]
        q_t = self.make_transpose(q_rope, f"{prefix}/q_t/output_0", perm=[0, 2, 1, 3])
        k_t = self.make_transpose(new_k, f"{prefix}/k_t/output_0", perm=[0, 2, 1, 3])
        v_t = self.make_transpose(new_v, f"{prefix}/v_t/output_0", perm=[0, 2, 1, 3])

        # GQA: expand KV heads to match Q heads
        # k_t, v_t: [B, nkv, S, D] → [B, nh, S, D]
        num_groups = nh // nkv  # 4
        # Unsqueeze: [B, nkv, S, D] → [B, nkv, 1, S, D]
        k_exp = self.make_unsqueeze(k_t, self.get_constant([2]), f"{prefix}/k_exp/output_0")
        v_exp = self.make_unsqueeze(v_t, self.get_constant([2]), f"{prefix}/v_exp/output_0")
        # Expand: [B, nkv, 1, S, D] → [B, nkv, num_groups, S, D]
        k_expanded = self.make_node(
            "Expand",
            [k_exp, self.get_constant([1, 1, num_groups, 1, 1])],
            [f"{prefix}/k_expanded/output_0"],
        )
        v_expanded = self.make_node(
            "Expand",
            [v_exp, self.get_constant([1, 1, num_groups, 1, 1])],
            [f"{prefix}/v_expanded/output_0"],
        )
        # Reshape: [B, nkv, num_groups, S, D] → [B, nh, S, D]
        k_gqa = self.make_reshape(
            k_expanded, self.get_constant([0, nh, -1, hd]), f"{prefix}/k_gqa/output_0"
        )
        v_gqa = self.make_reshape(
            v_expanded, self.get_constant([0, nh, -1, hd]), f"{prefix}/v_gqa/output_0"
        )

        # Scaled dot-product attention
        # Q @ K^T: [B, nh, 1, D] @ [B, nh, D, S] → [B, nh, 1, S]
        k_gqa_t = self.make_transpose(k_gqa, f"{prefix}/k_gqa_t/output_0", perm=[0, 1, 3, 2])
        scores = self.make_matmul(q_t, k_gqa_t, f"{prefix}/scores/output_0")

        # Scale
        scale = 1.0 / np.sqrt(hd)
        scaled = self.make_mul(
            scores, self.get_constant(scale, dtype=np.float32), f"{prefix}/scaled/output_0"
        )

        # Softmax
        attn_weights = self.make_node("Softmax", [scaled], [f"{prefix}/attn_w/output_0"], axis=-1)

        # Attention output: [B, nh, 1, S] @ [B, nh, S, D] → [B, nh, 1, D]
        attn_out = self.make_matmul(attn_weights, v_gqa, f"{prefix}/attn_out/output_0")

        # Transpose back: [B, nh, 1, D] → [B, 1, nh, D] → [B, 1, 1024]
        attn_t = self.make_transpose(attn_out, f"{prefix}/attn_t/output_0", perm=[0, 2, 1, 3])
        attn_flat = self.make_reshape(
            attn_t, self.get_constant([0, 1, -1]), f"{prefix}/attn_flat/output_0"
        )

        # === Output Projection + Residual ===
        out_proj = self.make_linear(
            attn_flat,
            self.weights[f"depthformer.layers.{layer_idx}.operator.out_proj.weight"],
            f"layer.{layer_idx}.out.weight",
            f"{prefix}/out_proj",
        )
        h = self.make_add(out_proj, residual, f"{prefix}/res1/output_0")

        # === FFN (SwiGLU) ===
        ffn_normed = self.make_layernorm(
            h, f"layer.{layer_idx}.ffn_norm.weight", None, f"{prefix}/ffn_norm"
        )

        # w1 and w3 in parallel, then SiLU(w1) * w3, then w2
        w1_out = self.make_linear(
            ffn_normed,
            self.weights[f"depthformer.layers.{layer_idx}.feed_forward.w1.weight"],
            f"layer.{layer_idx}.w1.weight",
            f"{prefix}/w1",
        )
        w3_out = self.make_linear(
            ffn_normed,
            self.weights[f"depthformer.layers.{layer_idx}.feed_forward.w3.weight"],
            f"layer.{layer_idx}.w3.weight",
            f"{prefix}/w3",
        )

        # SiLU(w1) * w3
        silu = self.make_silu(w1_out, f"{prefix}/silu")
        gate = self.make_mul(silu, w3_out, f"{prefix}/gate/output_0")

        # w2
        ffn_out = self.make_linear(
            gate,
            self.weights[f"depthformer.layers.{layer_idx}.feed_forward.w2.weight"],
            f"layer.{layer_idx}.w2.weight",
            f"{prefix}/w2",
        )

        # Residual
        output = self.make_add(ffn_out, h, f"{prefix}/res2/output_0")

        return output, new_k, new_v

    def build_step_logits(self, x: str) -> str:
        """Build step-indexed logits projection (no norm - to_logits is just linear).

        The PyTorch model's depth_embeddings[i].to_logits is a plain Linear layer
        without any preceding normalization. The embedding_norm is only used
        when embedding tokens, not before computing logits.

        Args:
            x: Transformer output [B, 1024]

        Returns:
            logits tensor name [B, 2049]
        """
        prefix = "/logits"

        # Get step-specific logits weight: stacked_logits_weights[step_idx] → [2049, 1024]
        logits_weight = self.make_gather(
            "stacked_logits_weights", "step_idx", f"{prefix}/logits_w/output_0", axis=0
        )

        # Linear: [B, 1024] @ [1024, 2049] → [B, 2049]
        logits_w_t = self.make_transpose(
            logits_weight, f"{prefix}/logits_w_t/output_0", perm=[1, 0]
        )
        return self.make_matmul(x, logits_w_t, "logits")

    def load_weights(self, model_path: str):
        """Load all depthformer weights from HuggingFace model."""
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        logger.info(f"Loading depthformer weights from {model_path}...")
        safetensors_path = hf_hub_download(model_path, "model.safetensors")

        # Use torch to handle bfloat16
        weights_torch = load_file(safetensors_path)

        for key, tensor in weights_torch.items():
            if key.startswith("depthformer.") or key.startswith("depth_embeddings."):
                self.weights[key] = tensor.float().numpy()

        logger.info(f"Loaded {len(self.weights)} weights")

    def prepare_weights(self):
        """Register all weights as initializers."""
        # === Stacked embeddings: [8, 2049, 1024] ===
        embed_list = []
        logits_list = []
        for i in range(self.num_codebooks):
            embed_list.append(self.weights[f"depth_embeddings.{i}.embedding.weight"])
            logits_list.append(self.weights[f"depth_embeddings.{i}.to_logits.weight"])

        self.add_initializer("stacked_embed_weights", np.stack(embed_list, axis=0))
        self.add_initializer("stacked_logits_weights", np.stack(logits_list, axis=0))

        # === RoPE frequencies ===
        freqs_cos, freqs_sin = self.compute_rope_freqs()
        self.add_initializer("rope_freqs_cos", freqs_cos)
        self.add_initializer("rope_freqs_sin", freqs_sin)

        # === Per-layer weights ===
        for i in range(self.num_layers):
            prefix = f"depthformer.layers.{i}"

            # operator_norm (RMSNorm)
            self.add_initializer(
                f"layer.{i}.operator_norm.weight",
                self.weights[f"{prefix}.operator_norm.weight"],
            )

            # Q/K layernorm
            self.add_initializer(
                f"layer.{i}.q_ln.weight",
                self.weights[f"{prefix}.operator.bounded_attention.q_layernorm.weight"],
            )
            self.add_initializer(
                f"layer.{i}.k_ln.weight",
                self.weights[f"{prefix}.operator.bounded_attention.k_layernorm.weight"],
            )

            # FFN norm
            self.add_initializer(
                f"layer.{i}.ffn_norm.weight", self.weights[f"{prefix}.ffn_norm.weight"]
            )

    def build(self, model_path: str) -> onnx.ModelProto:
        """Build the complete ONNX model for depthformer_unified."""
        logger.info("Building depthformer_unified ONNX model...")

        # Load weights
        self.load_weights(model_path)

        # Build graph structure
        self.build_inputs()
        self.build_outputs()

        # Prepare initializers
        self.prepare_weights()

        # === Build computation graph ===

        # 1. Get current depth slice
        current_slice = self.build_get_current_slice()

        # 2. Get previous token embedding
        prev_embed = self.build_prev_embedding()

        # 3. Combine: (slice + embed) → [B, 1, 1024]
        combined = self.make_add(current_slice, prev_embed, "/input/combined/output_0")
        x = self.make_unsqueeze(combined, self.get_constant([1]), "/input/unsqueeze/output_0")

        # 4. Get past_len from past_keys shape
        self.make_node("Shape", ["past_keys"], ["/past_len/shape/output_0"])
        past_len = self.make_gather(
            "/past_len/shape/output_0",
            self.get_constant(2),
            "/past_len/output_0",
        )

        # 5. Transformer layers
        new_keys_list = []
        new_values_list = []

        for i in range(self.num_layers):
            # Slice past KV for this layer
            layer_past_k = self.make_gather(
                "past_keys", self.get_constant(i), f"/layer_past_k_{i}/output_0", axis=0
            )
            layer_past_v = self.make_gather(
                "past_values", self.get_constant(i), f"/layer_past_v_{i}/output_0", axis=0
            )

            x, new_k, new_v = self.build_transformer_layer(
                x, i, layer_past_k, layer_past_v, past_len
            )

            # Unsqueeze for stacking: [B, S, H, D] → [1, B, S, H, D]
            new_k_unsq = self.make_unsqueeze(
                new_k, self.get_constant([0]), f"/new_k_{i}_unsq/output_0"
            )
            new_v_unsq = self.make_unsqueeze(
                new_v, self.get_constant([0]), f"/new_v_{i}_unsq/output_0"
            )
            new_keys_list.append(new_k_unsq)
            new_values_list.append(new_v_unsq)

        # Stack new KV caches: [6, B, S, H, D]
        self.make_concat(new_keys_list, "new_keys", axis=0)
        self.make_concat(new_values_list, "new_values", axis=0)

        # 6. Squeeze to [B, 1024] and build logits
        output = self.make_node(
            "Squeeze", [x, self.get_constant([1])], ["/output/squeeze/output_0"]
        )
        self.build_step_logits(output)

        # 7. Token embed placeholder (zeros)
        # Get batch size
        self.make_node("Shape", ["depth_slices"], ["/batch_shape/output_0"])
        batch = self.make_gather(
            "/batch_shape/output_0", self.get_constant(0), "/batch_dim/output_0"
        )
        batch_unsq = self.make_unsqueeze(batch, self.get_constant([0]), "/batch_unsq/output_0")
        zeros_shape = self.make_concat(
            [batch_unsq, self.get_constant([self.dim])], "/zeros_shape/output_0", axis=0
        )
        self.make_node(
            "ConstantOfShape",
            [zeros_shape],
            ["token_embed"],
            value=helper.make_tensor("value", TensorProto.FLOAT, [1], [0.0]),
        )

        model = self.build_graph("depthformer_unified", opset_version=21)
        logger.info(f"Model built: {len(self.nodes)} nodes")
        return model


def export_depthformer_unified_builder(model_path: str, onnx_dir: pathlib.Path) -> pathlib.Path:
    """Export depthformer_unified.onnx using ONNX builder (no torch.onnx.export).

    Args:
        model_path: HuggingFace model ID or local path
        onnx_dir: Output directory for ONNX models

    Returns:
        Path to exported vocoder_depthformer.onnx
    """
    builder = DepthformerUnifiedBuilder()
    model = builder.build(model_path)

    output_path = onnx_dir / "vocoder_depthformer.onnx"
    onnx.save(model, str(output_path))

    logger.info(f"vocoder_depthformer saved to {output_path}")
    return output_path
