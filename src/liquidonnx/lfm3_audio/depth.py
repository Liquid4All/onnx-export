"""
LFM3-Audio depth decoder: the codebook logits of one speech frame from the language model's hidden
state, the codes revealed so far and the mask of the hidden ones (Lfm3AudioDepthDecoder.forward).
The session calls it once per MaskGIT pass, 3 times per frame (maskgit_schedule [4, 4, 4]).

depth_decoder.onnx

    hidden_states [B, 2048]   float32  decoder.onnx's hidden_states at the position before the frame
    codes         [B, 12]     int64    revealed codes in [0, 2017); masked entries are ignored (0)
    masked        [B, 12]     bool     True: codebook still hidden, predict it
    logits        [B, 12, 2017] float32

      hidden_states → depth_linear MatMul + bias [B, 12288] → Reshape     [B, 12, 1024]
      codes + 2017 * [0..11] → Gather(depth table [12 * 2017, 1024])      [B, 12, 1024]
      → Where(masked[..., None], mask_embedding, code embeddings) → Add   [B, 12, 1024]
      → 3 LFM3 attention layers (config.depth), BIDIRECTIONAL over the 12 codebook slots
        (plain MatMul/Softmax attention: GroupQueryAttention is causal only), 32/8 heads, hd 32,
        q/k RMSNorm, full RoPE theta 1e6 at positions 0..11, headwise sigmoid gate, no sandwich
        norms, SwiGLU 2816
      → final RMSNorm (model.embedding_norm)                              [B, 12, 1024]
      → per-codebook RMSNorm_k (depth_embeddings.k.embedding_norm, eps config.depth.norm_eps)
      → per-codebook head: logits[:, k] = normed[:, k] @ depth_embeddings.k.embedding.weightᵀ
        (one batched MatMul [B, 12, 1, 1024] x [12, 1024, 2017], named HEAD_NODE)

No KV cache: each call sees all 12 slots.

Weights: load_weights(checkpoint, "depth", config): depth_linear.{weight, bias}, mask_embedding,
model.layers.N.*, model.embedding_norm.weight, depth_embeddings.K.{embedding, embedding_norm}.weight.

The trunk is LFM3Builder's attention layer with its GroupQueryAttention swapped for plain
MatMul/Softmax attention (make_node), so every other node and initializer keeps LFM3Builder's name.

Precision: fp32 and fp16; q8 quantizes only the MatMuls with constant weights except HEAD_NODE
(the code tables and heads stay float, as LFM2.5-Audio keeps them for CPU speed). Never int4.
With no GroupQueryAttention the fp32 graph runs on CUDA too, and it is the only one that matches
the reference's greedy codes exactly: fp16 and q8 flip a few percent of the argmaxes at near-ties
(tests/test_lfm3_audio/test_depth.py).
"""

import pathlib

import numpy as np
import onnx
from onnx import TensorProto, helper

from liquidonnx.lfm3.builder import LFM3Builder
from liquidonnx.lfm3_audio.config import LFM3AudioConfig
from liquidonnx.lfm3_audio.weights import load_weights

INPUTS = ("hidden_states", "codes", "masked")
OUTPUT = "logits"
# The tied per-codebook head, excluded from int8 quantization
HEAD_NODE = "/depth/head/MatMul"


class LFM3AudioDepthBuilder(LFM3Builder):
    def __init__(self, config: LFM3AudioConfig):
        """Builds from config.depth, config.language.hidden_size, config.num_codebooks and
        config.codebook_size."""
        super().__init__(config.depth)
        self.audio_config = config

    def load_weights(self, model_path: str | pathlib.Path):
        self.weights = load_weights(model_path, "depth", self.audio_config)

    def make_node(self, op_type, inputs, outputs, name=None, domain="", **attrs) -> str:
        # LFM3Builder.build_attention_layer runs causal GroupQueryAttention over a KV cache; the
        # depth decoder attends over all slots both ways, without a cache
        if op_type == "GroupQueryAttention":
            return self.make_bidirectional_attention(*inputs[:3], outputs[0])
        return super().make_node(op_type, inputs, outputs, name=name, domain=domain, **attrs)

    def make_bidirectional_attention(self, q: str, k: str, v: str, output: str) -> str:
        """softmax(q kᵀ / √hd) v per head, query head h reading key/value head h // rep as
        repeat_kv does; q [B, S, nh*hd], k and v [B, S, nkv*hd] → output [B, S, nh*hd]."""
        cfg = self.config
        nh, nkv, hd = cfg.num_attention_heads, cfg.num_key_value_heads, self.head_dim
        rep = nh // nkv
        path = output.rsplit("/", 2)[0]
        # [B, S, nh*hd] → [B, nkv, rep, S, hd]
        q = self.make_reshape(
            q, self.get_constant([0, 0, nkv, rep, hd]), f"{path}/q_heads/Reshape/output_0"
        )
        q = self.make_transpose(q, f"{path}/q_heads/Transpose/output_0", [0, 2, 3, 1, 4])
        q = self.make_mul(
            q, self.get_constant(hd**-0.5, np.float32), f"{path}/q_heads/Mul/output_0"
        )
        # [B, S, nkv*hd] → kᵀ [B, nkv, 1, hd, S], v [B, nkv, 1, S, hd]
        kv_heads = self.get_constant([0, 0, nkv, 1, hd])
        k = self.make_reshape(k, kv_heads, f"{path}/k_heads/Reshape/output_0")
        k = self.make_transpose(k, f"{path}/k_heads/Transpose/output_0", [0, 2, 3, 4, 1])
        v = self.make_reshape(v, kv_heads, f"{path}/v_heads/Reshape/output_0")
        v = self.make_transpose(v, f"{path}/v_heads/Transpose/output_0", [0, 2, 3, 1, 4])
        # [B, nkv, rep, S, S] → [B, nkv, rep, S, hd]
        scores = self.make_matmul(q, k, f"{path}/scores/MatMul/output_0")
        probs = self.make_node("Softmax", [scores], [f"{path}/Softmax/output_0"], axis=-1)
        out = self.make_matmul(probs, v, f"{path}/context/MatMul/output_0")
        # → [B, S, nkv, rep, hd] → [B, S, nh*hd]
        out = self.make_transpose(out, f"{path}/context/Transpose/output_0", [0, 3, 1, 2, 4])
        return self.make_reshape(out, self.get_constant([0, 0, nh * hd]), output)

    def build_depth_inputs(self) -> str:
        """depth_linear(hidden_states) plus each slot's code or mask embedding [B, K, D]."""
        K, V = self.audio_config.num_codebooks, self.audio_config.codebook_size
        D = self.config.hidden_size
        H = self.audio_config.language.hidden_size
        for name, dtype, shape in (
            ("hidden_states", TensorProto.FLOAT, ["batch_size", H]),
            ("codes", TensorProto.INT64, ["batch_size", K]),
            ("masked", TensorProto.BOOL, ["batch_size", K]),
        ):
            self.inputs.append(helper.make_tensor_value_info(name, dtype, shape))
        self.outputs.append(
            helper.make_tensor_value_info(OUTPUT, TensorProto.FLOAT, ["batch_size", K, V])
        )

        # === 1. depth_linear: [B, H] → [B, K*D] → [B, K, D] ===
        self.add_initializer("depth_linear.MatMul.weight", self.weights["depth_linear.weight"].T)
        self.add_initializer("depth_linear.bias", self.weights["depth_linear.bias"])
        x = self.make_matmul(
            "hidden_states", "depth_linear.MatMul.weight", "/depth/depth_linear/MatMul/output_0"
        )
        x = self.make_add(x, "depth_linear.bias", "/depth/depth_linear/Add/output_0")
        x = self.make_reshape(
            x, self.get_constant([0, K, D]), "/depth/depth_linear/Reshape/output_0"
        )

        # === 2. Code embeddings from one table [K*V, D]: row code + V*k ===
        table = np.concatenate(
            [self.weights[f"depth_embeddings.{k}.embedding.weight"] for k in range(K)]
        )
        self.add_initializer("depth_embeddings.weight", table)
        rows = self.make_add(
            "codes", self.get_constant(np.arange(K) * V), "/depth/embed_codes/Add/output_0"
        )
        codes = self.make_gather(
            "depth_embeddings.weight", rows, "/depth/embed_codes/Gather/output_0"
        )

        # === 3. Masked slots take mask_embedding instead ===
        self.add_initializer("mask_embedding", self.weights["mask_embedding"])
        masked = self.make_unsqueeze(
            "masked", self.get_constant([2]), "/depth/mask/Unsqueeze/output_0"
        )
        slots = self.make_node(
            "Where", [masked, "mask_embedding", codes], ["/depth/mask/Where/output_0"]
        )
        return self.make_add(x, slots, "/depth/inputs/Add/output_0")

    def build_head(self, hidden_state: str) -> str:
        """Final norm, per-codebook RMSNorm and tied head: [B, K, D] → logits [B, K, V]."""
        K, V = self.audio_config.num_codebooks, self.audio_config.codebook_size
        D = self.config.hidden_size
        eps = self.config.norm_eps
        self.add_initializer(
            "model.embedding_norm.weight", self.weights["model.embedding_norm.weight"]
        )
        x = self.make_layernorm(
            hidden_state, "model.embedding_norm.weight", None, "/depth/final_norm", eps
        )
        # RMSNorm_k: a plain RMSNorm, then each codebook's scale [K, D]
        self.add_initializer("depth_embeddings.norm_ones", np.ones(D, np.float32))
        x = self.make_layernorm(x, "depth_embeddings.norm_ones", None, "/depth/head_norm", eps)
        scales = np.stack(
            [self.weights[f"depth_embeddings.{k}.embedding_norm.weight"] for k in range(K)]
        )
        self.add_initializer("depth_embeddings.norm.weight", scales)
        x = self.make_mul(x, "depth_embeddings.norm.weight", "/depth/head_norm/Mul/output_0")

        # [B, K, 1, D] @ table viewed as [K, D, V] (a Transpose onnxruntime folds at load)
        x = self.make_unsqueeze(x, self.get_constant([2]), "/depth/head/Unsqueeze/output_0")
        head = self.make_reshape(
            "depth_embeddings.weight",
            self.get_constant([K, V, D]),
            "/depth/head/weight/Reshape/output_0",
        )
        head = self.make_transpose(head, "/depth/head/weight/Transpose/output_0", [0, 2, 1])
        logits = self.make_matmul(x, head, f"{HEAD_NODE}/output_0")
        return self.make_node("Squeeze", [logits, self.get_constant([2])], [OUTPUT])

    def build(self, model_path: str | pathlib.Path) -> onnx.ModelProto:
        """depth_decoder.onnx of the checkpoint directory (see the module docstring)."""
        self.load_weights(model_path)
        K = self.audio_config.num_codebooks
        hidden_state = self.build_depth_inputs()

        # RotaryEmbedding reads positions 0..K-1 of every row as "position_ids"
        self.build_rope_cache()
        self.make_node(
            "Expand",
            [
                self.get_constant([list(range(K))]),
                self.make_node("Shape", ["codes"], ["/depth/Shape/output_0"]),
            ],
            ["position_ids"],
        )
        for layer_idx in range(self.config.num_hidden_layers):
            self.prepare_layer_weights(layer_idx, "full_attention")
            hidden_state = self.build_attention_layer(layer_idx, hidden_state, is_swa=False)
        self.build_head(hidden_state)
        return self.build_graph("lfm3_audio_depth")
