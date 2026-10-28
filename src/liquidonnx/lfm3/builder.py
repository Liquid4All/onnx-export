"""
LFM3 dense text model ONNX export.

Builds an optimized graph for the LFM3 hybrid architecture:
    - Conv layers: gated short conv with depthwise Conv1d (kernel = conv_L_cache)
    - Full-attention layers: GQA with partial RoPE, headwise sigmoid-gated output,
      and (when use_peri_ln) post_operator_norm + post_ffn_norm
    - Sliding-window-attention layers: same as full-attention but with
      local_window_size on GQA and a second RoPE table whose theta comes from
      rope_parameters["sliding_attention"] (swa_rope_parameters in older checkpoints)

Uses Microsoft contrib operators: SimplifiedLayerNormalization, RotaryEmbedding,
and GroupQueryAttention.
"""

import logging
import pathlib
from dataclasses import dataclass

import numpy as np
import onnx
from onnx import TensorProto, helper

from liquidonnx.builder_base import ONNXBuilderBase

logger = logging.getLogger(__name__)

INT64_MAX = 2**63 - 1

# Older checkpoints name the sliding-window layers "sliding_window_attention".
LAYER_TYPE_ALIASES = {"sliding_window_attention": "sliding_attention"}
SUPPORTED_LAYER_TYPES = {"conv", "full_attention", "sliding_attention"}


@dataclass
class LFM3Config:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int
    layer_types: list[str]
    head_dim: int
    intermediate_size: int
    conv_L_cache: int = 3
    max_position_embeddings: int = 128000
    norm_eps: float = 1e-5
    # RoPE
    rope_theta: float = 1_000_000.0
    partial_rotary_factor: float = 1.0
    swa_rope_theta: float | None = None
    swa_partial_rotary_factor: float = 1.0
    # LFM3-VL computes inv_freq as theta ** -x, the text models as 1 / theta ** x; the two
    # differ by an ulp, ~2e-3 rad at position 32K.
    rope_negative_power: bool = False
    # Sliding-window attention
    sliding_window: int | None = None
    # Post-operator and post-FFN norms (use_sandwich_norm, use_peri_ln in older checkpoints)
    use_peri_ln: bool = False
    use_gated_attention: bool = False
    gated_attention_style: str = "elementwise"  # "elementwise" | "headwise"

    @classmethod
    def from_hf_config(cls, cfg: dict) -> "LFM3Config":
        """Parse a raw config.json dict of either checkpoint format.

        The dict is read without AutoConfig: transformers 5.15-5.17 rename "conv" layers of
        remote-code configs to "linear_attention", and 5.3+ rejects "sliding_window_attention".

        Older format: flat rope_parameters for full attention, swa_rope_parameters for sliding
        layers, use_peri_ln, use_gated_attention + gated_attention_style.
        Newer format (LFM3-1.7B's): rope_parameters keyed by layer type, use_sandwich_norm, and
        an always-on headwise attention gate.
        """
        if cfg.get("model_type") == "lfm3_vl":
            raise ValueError("an LFM3-VL checkpoint: export it with lfm3-vl-export")
        # === Reject knobs we don't wire into the ONNX graph ===
        # conv_bias would require loading model.layers.*.conv.{in,out}_proj.bias and conv.bias
        # and adding Add nodes / Conv bias input — none of which the builder does today.
        if cfg.get("conv_bias", False):
            raise NotImplementedError(
                "LFM3 export does not support conv_bias=True (conv biases are not wired "
                "into the ONNX graph)."
            )
        # SWA per-layer-type overrides would require diverging the gated-attention and
        # GQA wiring on SWA vs. full-attention layers. Today both reuse global settings.
        for key in (
            "swa_num_key_value_heads",
            "swa_use_gated_attention",
            "swa_gated_attention_style",
        ):
            if cfg.get(key) is not None:
                raise NotImplementedError(
                    f"LFM3 export does not support per-layer-type override {key}={cfg[key]!r}; "
                    "SWA layers must inherit the global attention settings."
                )
        if not cfg.get("tie_embedding", cfg.get("tie_word_embeddings", True)):
            raise NotImplementedError("LFM3 export ties lm_head to the embedding table")

        # === Layer types ===
        num_layers = cfg["num_hidden_layers"]
        layer_types = cfg.get("layer_types")
        if layer_types is None:
            full_attn_idxs = cfg.get("full_attn_idxs")
            if full_attn_idxs is None:
                full_attn_idxs = range(num_layers)
            layer_types = [
                "full_attention" if i in full_attn_idxs else "conv" for i in range(num_layers)
            ]
        layer_types = [LAYER_TYPE_ALIASES.get(t, t) for t in layer_types]
        unknown = sorted({t for t in layer_types if t not in SUPPORTED_LAYER_TYPES})
        if unknown:
            raise NotImplementedError(
                f"LFM3 export does not support layer_types={unknown}; "
                f"supported types are {sorted(SUPPORTED_LAYER_TYPES)}."
            )
        has_swa = "sliding_attention" in layer_types
        if has_swa and cfg.get("sliding_window") is None:
            raise ValueError("sliding_attention layers need sliding_window")

        # === RoPE and attention options ===
        rope = cfg.get("rope_parameters") or cfg.get("rope_scaling") or {}
        if rope and set(rope) <= {"full_attention", "sliding_attention"}:
            full, swa = rope.get("full_attention"), rope.get("sliding_attention")
            if "full_attention" in layer_types and full is None:
                raise ValueError("rope_parameters has no full_attention entry")
            if has_swa and swa is None:
                raise ValueError("rope_parameters has no sliding_attention entry")
            entries = [e for e in (full, swa) if e]
            full = full or {}
            swa_rope_theta = float(swa["rope_theta"]) if swa else None
            swa_partial = float(swa.get("partial_rotary_factor", 1.0)) if swa else 1.0
            # The newer modeling code applies a headwise gate unconditionally.
            use_gated, gate_style = True, "headwise"
        else:
            full = rope
            swa_rope = cfg.get("swa_rope_parameters") or {}
            entries = [e for e in (full, swa_rope) if e]
            # Without swa_rope_parameters the sliding layers reuse the full-attention table.
            swa_rope_theta = float(swa_rope["rope_theta"]) if swa_rope else None
            swa_partial = 1.0
            use_gated = bool(cfg.get("use_gated_attention", False))
            gate_style = cfg.get("gated_attention_style") or "elementwise"
        for entry in entries:
            if entry.get("rope_type", "default") != "default":
                raise NotImplementedError(
                    f"LFM3 export supports only default RoPE, got {entry['rope_type']!r}"
                )

        return cls(
            hidden_size=cfg["hidden_size"],
            num_hidden_layers=num_layers,
            num_attention_heads=cfg["num_attention_heads"],
            num_key_value_heads=cfg["num_key_value_heads"],
            vocab_size=cfg["vocab_size"],
            layer_types=layer_types,
            head_dim=cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"],
            intermediate_size=cfg["intermediate_size"],
            conv_L_cache=cfg.get("conv_L_cache", 3),
            max_position_embeddings=cfg.get("max_position_embeddings", 128000),
            norm_eps=cfg.get("norm_eps", 1e-5),
            rope_theta=float(full.get("rope_theta", 1_000_000.0)),
            partial_rotary_factor=float(full.get("partial_rotary_factor", 1.0)),
            swa_rope_theta=swa_rope_theta,
            swa_partial_rotary_factor=swa_partial,
            sliding_window=cfg.get("sliding_window"),
            use_peri_ln=bool(cfg.get("use_sandwich_norm", cfg.get("use_peri_ln", False))),
            use_gated_attention=use_gated,
            gated_attention_style=gate_style,
        )


class LFM3Builder(ONNXBuilderBase):
    """
    Builds an optimized ONNX graph with:
    - Conv layers (gated short conv, depthwise conv1d)
    - Full-attention layers (GQA + partial RoPE + gated output + peri-LN)
    - Sliding-window-attention layers (GQA + local_window_size + local RoPE table)
    - Fused Microsoft operators (SimplifiedLayerNormalization, RotaryEmbedding, GroupQueryAttention)
    """

    def __init__(self, config: LFM3Config):
        super().__init__()
        self.config = config
        self.head_dim = config.head_dim

        # Categorize layers
        self.conv_indices = [i for i, t in enumerate(config.layer_types) if t == "conv"]
        self.full_attn_indices = [
            i for i, t in enumerate(config.layer_types) if t == "full_attention"
        ]
        self.swa_indices = [i for i, t in enumerate(config.layer_types) if t == "sliding_attention"]
        # All attention-like layers (both share KV cache layout)
        self.attn_indices = sorted(self.full_attn_indices + self.swa_indices)
        # Whether a separate local RoPE table is needed
        self.has_local_rope = bool(self.swa_indices) and config.swa_rope_theta is not None

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

    def prepare_layer_weights(self, layer_idx: int, layer_type: str):
        """Prepare and register weights for a layer.

        Handles weight transposition and naming for MatMul operations.
        Uses these naming conventions:
        - operator_norm -> operator_layernorm (and post_operator_norm -> post_operator_layernorm)
        - ffn_norm -> ffn_layernorm (and post_ffn_norm -> post_ffn_layernorm)
        - feed_forward.w1/w2/w3 -> mlp.gate_proj/down_proj/up_proj.MatMul
        - conv.weight -> conv.conv.weight
        - self_attn -> attn, with .MatMul suffix
        """
        prefix = f"model.layers.{layer_idx}"

        self.add_initializer(
            f"{prefix}.operator_layernorm.weight", self.weights[f"{prefix}.operator_norm.weight"]
        )
        self.add_initializer(
            f"{prefix}.ffn_layernorm.weight", self.weights[f"{prefix}.ffn_norm.weight"]
        )
        if f"{prefix}.post_operator_norm.weight" in self.weights and not self.config.use_peri_ln:
            raise ValueError(
                f"{prefix} has post_operator_norm but the config turns the sandwich norms off"
            )
        if self.config.use_peri_ln:
            self.add_initializer(
                f"{prefix}.post_operator_layernorm.weight",
                self.weights[f"{prefix}.post_operator_norm.weight"],
            )
            self.add_initializer(
                f"{prefix}.post_ffn_layernorm.weight",
                self.weights[f"{prefix}.post_ffn_norm.weight"],
            )

        # MLP weights (transposed for MatMul)
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
            # Attention weights (transposed for MatMul)
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
            # Newer checkpoints name these q_norm / k_norm / o_proj.
            self.add_initializer(
                f"{prefix}.attn.q_norm.layernorm.weight",
                self._weight(f"{prefix}.self_attn.q_norm", f"{prefix}.self_attn.q_layernorm"),
            )
            self.add_initializer(
                f"{prefix}.attn.k_norm.layernorm.weight",
                self._weight(f"{prefix}.self_attn.k_norm", f"{prefix}.self_attn.k_layernorm"),
            )
            self.add_initializer(
                f"{prefix}.attn.o_proj.MatMul.weight",
                self._weight(f"{prefix}.self_attn.o_proj", f"{prefix}.self_attn.out_proj").T,
            )
            gate_name = f"{prefix}.self_attn.gate_proj.weight"
            if self.config.use_gated_attention:
                gate = self.weights[gate_name]
                nh, hd = self.config.num_attention_heads, self.head_dim
                rows = nh if self.config.gated_attention_style == "headwise" else nh * hd
                if gate.shape != (rows, self.config.hidden_size):
                    raise ValueError(
                        f"{gate_name} has shape {gate.shape}, expected "
                        f"{(rows, self.config.hidden_size)} for a "
                        f"{self.config.gated_attention_style} gate"
                    )
                self.add_initializer(f"{prefix}.attn.gate_proj.MatMul.weight", gate.T)
            elif gate_name in self.weights:
                raise ValueError(f"{gate_name} exists but the config turns the gate off")

    def _weight(self, *names: str) -> np.ndarray:
        """The first of names (without the .weight suffix) the checkpoint has."""
        for name in names:
            if f"{name}.weight" in self.weights:
                return self.weights[f"{name}.weight"]
        raise KeyError(f"checkpoint has none of {[f'{n}.weight' for n in names]}")

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

        # logits_to_keep: logits of the last N positions only (0: all)
        self.inputs.append(helper.make_tensor_value_info("logits_to_keep", TensorProto.INT64, [1]))

        # Interleave cache inputs by layer index
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
                ["batch_size", "num_logits", self.config.vocab_size],
            )
        )

        # Interleave cache outputs by layer index
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
        self.add_initializer("model.embed_tokens.weight", self.weights["model.embed_tokens.weight"])
        return self.make_node(
            "Gather",
            ["model.embed_tokens.weight", "input_ids"],
            ["/model/embed_tokens/Gather/output_0"],
            axis=0,
        )

    def _compute_rope_cache(
        self, theta: float, partial_rotary_factor: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute cos/sin RoPE tables matching modeling_lfm3.compute_default_rope_parameters.

        Partial RoPE (Gemma-4 style): the first `rope_angles = (partial * head_dim) // 2`
        inv_freq pairs receive the standard `theta^(2k/dim)` rotation; the remaining
        `dim//2 - rope_angles` pairs are zero. Since `cos(0)=1, sin(0)=0`, those
        head-dim positions pass through RotaryEmbedding as identity, recovering
        full RoPE bit-exactly when `partial_rotary_factor == 1.0`.
        """
        head_dim = self.head_dim
        max_seq = self.config.max_position_embeddings
        rope_angles = int(partial_rotary_factor * head_dim) // 2
        assert 0 < rope_angles <= head_dim // 2, (
            f"partial_rotary_factor={partial_rotary_factor} with head_dim={head_dim} "
            f"produced rope_angles={rope_angles}; must be in (0, head_dim//2]."
        )
        # Lfm3RotaryEmbedding computes inv_freq and the angles position * inv_freq in float32,
        # and the model was trained on those rounded angles: at position 32K the float32 angle
        # is ~2e-3 rad off the exact one. inv_freq comes from torch for bit-identical pow, the
        # product is rounded to float32 as in PyTorch, and only cos/sin run in float64.
        import torch

        exponents = torch.arange(0, 2 * rope_angles, 2, dtype=torch.int64).float() / head_dim
        if self.config.rope_negative_power:
            inv_freq = (theta**-exponents).numpy()
        else:
            inv_freq = (1.0 / (theta**exponents)).numpy()
        nope_angles = head_dim // 2 - rope_angles
        inv_freq = np.concatenate([inv_freq, np.zeros(nope_angles, dtype=np.float32)])
        positions = np.arange(max_seq, dtype=np.float32)
        freqs = (positions[:, None] * inv_freq[None, :]).astype(np.float64)  # [max_seq, head_dim/2]
        return np.cos(freqs).astype(np.float32), np.sin(freqs).astype(np.float32)

    def build_rope_cache(self):
        cos, sin = self._compute_rope_cache(
            self.config.rope_theta, self.config.partial_rotary_factor
        )
        # RotaryEmbedding expects [max_seq, head_dim/2] — do NOT concatenate
        self.add_initializer("cos_cache", cos)
        self.add_initializer("sin_cache", sin)
        # Second RoPE table for SWA layers (own theta) when present
        if self.has_local_rope:
            cos_local, sin_local = self._compute_rope_cache(
                self.config.swa_rope_theta, self.config.swa_partial_rotary_factor
            )
            self.add_initializer("cos_cache_local", cos_local)
            self.add_initializer("sin_cache_local", sin_local)

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
        gather_name = "Gather"
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
        """Build a gated short-conv layer (matches PyTorch Lfm3ShortConv).

        Graph structure:
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
            Conv1D (depthwise, kernel=L_cache, groups=H)
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

        # === LayerNorm ===
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
        # Transpose: [B, S, 3H] → [B, 3H, S]
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

        # === Gated convolution ===
        # Bx = B * x (input gating)
        Bx = self.make_mul(
            f"{prefix}/conv/Split/output_0",
            f"{prefix}/conv/Split/output_2",
            f"{prefix}/conv/Mul_1/output_0",
        )
        # Concat with cache: [B, H, L] + [B, H, S] → [B, H, L+S]
        conv_input = self.make_node(
            "Concat", [f"past_conv.{layer_idx}", Bx], [f"{prefix}/conv/Conv_Input/output_0"], axis=2
        )
        # Depthwise Conv1D (kernel=L_cache)
        conv_out_full = self.make_node(
            "Conv",
            [conv_input, f"{weight_prefix}.conv.conv.weight"],
            [f"{prefix}/conv/Conv/output_0"],
            kernel_shape=[L],
            group=H,
        )

        # === Dynamic slice ===
        # Get sequence length from LayerNorm output shape (axis 1)
        self.make_node("Shape", [normed], [f"{prefix}/conv/Shape_for_slice/output_0"])
        seq_len = self.make_node(
            "Gather",
            [f"{prefix}/conv/Shape_for_slice/output_0", self.get_constant(1)],
            [f"{prefix}/conv/Gather_for_slice/output_0"],
            axis=0,
        )
        # Negate seq_len for slice start
        neg_seq = self.make_mul(
            seq_len, self.get_constant(-1), f"{prefix}/conv/Neg_Seq_Len/output_0"
        )
        # Unsqueeze for slice
        slice_start = self.make_unsqueeze(
            neg_seq, self.get_constant([0]), f"{prefix}/conv/Unsqueeze_starts/output_0"
        )
        # Slice last S elements
        conv_sliced = self.make_slice(
            conv_out_full,
            slice_start,
            self.get_constant([np.iinfo(np.int64).max]),
            self.get_constant([2]),
            f"{prefix}/conv/Slice_Conv_Output/output_0",
        )

        # === Cache update ===
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

        # === Output gating + projection ===
        # y = C * conv_out
        y = self.make_mul(
            f"{prefix}/conv/Split/output_1",
            conv_sliced,
            f"{prefix}/conv/Mul_2/output_0",
        )
        # Transpose: [B, H, S] → [B, S, H]
        y_t = self.make_node(
            "Transpose", [y], [f"{prefix}/conv/Transpose_2/output_0"], perm=[0, 2, 1]
        )
        # out_proj: [B, S, H] → [B, S, H]
        out_proj = self.make_matmul(
            y_t,
            f"{weight_prefix}.conv.out_proj.MatMul.weight",
            f"{prefix}/conv/out_proj/MatMul/output_0",
        )

        # === Peri-LN (post_operator_norm) and Residual + MLP ===
        operator_out = self._maybe_post_operator_norm(layer_idx, out_proj)
        hidden_state = self.make_add(residual, operator_out, f"{prefix}/Add_1/output_0")
        return self.build_mlp(layer_idx, hidden_state)

    def _maybe_post_operator_norm(self, layer_idx: int, operator_out: str) -> str:
        """Apply post_operator_layernorm to the operator output when peri-LN is enabled."""
        if not self.config.use_peri_ln:
            return operator_out
        prefix = f"/model/layers.{layer_idx}"
        weight_prefix = f"model.layers.{layer_idx}"
        return self.make_simple_layernorm(
            operator_out,
            f"{weight_prefix}.post_operator_layernorm.weight",
            f"{prefix}/post_operator_layernorm",
            name="LayerNorm",
        )

    def _maybe_post_ffn_norm(self, layer_idx: int, ffn_out: str) -> str:
        """Apply post_ffn_layernorm to the FFN output when peri-LN is enabled."""
        if not self.config.use_peri_ln:
            return ffn_out
        prefix = f"/model/layers.{layer_idx}"
        weight_prefix = f"model.layers.{layer_idx}"
        return self.make_simple_layernorm(
            ffn_out,
            f"{weight_prefix}.post_ffn_layernorm.weight",
            f"{prefix}/post_ffn_layernorm",
            name="LayerNorm",
        )

    def build_attention_layer(self, layer_idx: int, hidden_state: str, is_swa: bool) -> str:
        """Build an attention layer with Grouped Query Attention.

        Graph structure (matches PyTorch Lfm3Attention):
            hidden_state
                ↓
            LayerNorm (operator_layernorm)  → "normed"
                ↓
            Q/K/V Projections (Q: H→nh*hd, K/V: H→nkv*hd)
                ↓
            Per-head Q/K LayerNorm (Reshape [B,-1,hd] → Norm → Reshape back)
                ↓
            RotaryEmbedding (Q, K only; uses local RoPE cache when is_swa+has_local_rope)
                ↓
            GroupQueryAttention (local_window_size = sliding_window if is_swa else -1)
                ↓
            (optional) Gated output: sigmoid(gate_proj(normed)) * attn_out
                ↓
            Linear (o_proj)
                ↓
            (optional) post_operator_norm  (sandwich / peri-LN)
                ↓
            Add (residual) → MLP block
        """
        prefix = f"/model/layers.{layer_idx}"
        weight_prefix = f"model.layers.{layer_idx}"
        nh = self.config.num_attention_heads
        nkv = self.config.num_key_value_heads
        hd = self.head_dim
        q_hidden = nh * hd
        kv_hidden = nkv * hd
        residual = hidden_state

        # === LayerNorm ===
        normed = self.make_simple_layernorm(
            hidden_state,
            f"{weight_prefix}.operator_layernorm.weight",
            f"{prefix}/operator_layernorm",
            name="LayerNorm",
        )

        # === Q/K/V Projections ===
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
        q_reshape_back = self.get_constant([0, -1, q_hidden])
        k_reshape_back = self.get_constant([0, -1, kv_hidden])

        # Q norm
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

        # K norm
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
        gather_name = "Gather"

        # SWA layers use the local RoPE table (own theta and partial factor) when configured.
        if is_swa and self.has_local_rope:
            cos_name, sin_name = "cos_cache_local", "sin_cache_local"
        else:
            cos_name, sin_name = "cos_cache", "sin_cache"
        # SWA layers bound attention to the configured window; full-attention is unbounded.
        local_window_size = self.config.sliding_window if is_swa else -1

        # Apply RotaryEmbedding to Q and K, then run GroupQueryAttention.
        rope_attrs = {
            "domain": "com.microsoft",
            "interleaved": 0,
            "num_heads": 0,
            "rotary_embedding_dim": 0,
        }
        q_rope = self.make_node(
            "RotaryEmbedding",
            [q_3d, "position_ids", cos_name, sin_name],
            [f"{prefix}/attn/q_rotary/RotaryEmbedding/output_0"],
            **rope_attrs,
        )
        k_rope = self.make_node(
            "RotaryEmbedding",
            [k_3d, "position_ids", cos_name, sin_name],
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
            local_window_size=local_window_size,
            softcap=0.0,
            do_rotary=0,
            rotary_interleaved=0,
        )

        # === Optional sigmoid-gated output ===
        # Gate comes from the post-norm activations ("normed"), NOT the residual stream.
        attn_out = f"{prefix}/attn/GroupQueryAttention/output_0"
        if self.config.use_gated_attention:
            gate_pre = self.make_matmul(
                normed,
                f"{weight_prefix}.attn.gate_proj.MatMul.weight",
                f"{prefix}/attn/gate_proj/MatMul/output_0",
            )
            gate = self.make_node(
                "Sigmoid",
                [gate_pre],
                [f"{prefix}/attn/gate_proj/Sigmoid/output_0"],
            )
            if self.config.gated_attention_style == "headwise":
                # gate: [B,S,nh] → [B,S,nh,1]; attn_out: [B,S,nh*hd] → [B,S,nh,hd]
                gate_4d = self.make_node(
                    "Reshape",
                    [gate, self.get_constant([0, 0, nh, 1])],
                    [f"{prefix}/attn/gate_proj/Reshape/output_0"],
                )
                attn_4d = self.make_node(
                    "Reshape",
                    [attn_out, self.get_constant([0, 0, nh, hd])],
                    [f"{prefix}/attn/GroupQueryAttention/Reshape/output_0"],
                )
                gated_4d = self.make_mul(attn_4d, gate_4d, f"{prefix}/attn/gate_proj/Mul/output_0")
                attn_out = self.make_node(
                    "Reshape",
                    [gated_4d, self.get_constant([0, 0, q_hidden])],
                    [f"{prefix}/attn/gate_proj/Reshape_back/output_0"],
                )
            else:  # "elementwise"
                # gate: [B,S,nh*hd]; attn_out: [B,S,nh*hd]
                attn_out = self.make_mul(attn_out, gate, f"{prefix}/attn/gate_proj/Mul/output_0")

        # === Output projection + (optional) post-norm + Residual + MLP ===
        o_proj = self.make_matmul(
            attn_out,
            f"{weight_prefix}.attn.o_proj.MatMul.weight",
            f"{prefix}/attn/o_proj/MatMul/output_0",
        )
        operator_out = self._maybe_post_operator_norm(layer_idx, o_proj)
        hidden_state = self.make_add(residual, operator_out, f"{prefix}/Add_1/output_0")
        return self.build_mlp(layer_idx, hidden_state)

    def build_mlp(self, layer_idx: int, hidden_state: str) -> str:
        """Build SwiGLU MLP block (matches PyTorch Lfm3MLP).

        ffn_layernorm → (gate_proj | up_proj) → SiLU(gate) * up → down_proj
        → (optional) post_ffn_layernorm → residual Add.
        """
        prefix = f"/model/layers.{layer_idx}"
        weight_prefix = f"model.layers.{layer_idx}"

        residual = hidden_state

        # FFN LayerNorm
        normed = self.make_simple_layernorm(
            hidden_state,
            f"{weight_prefix}.ffn_layernorm.weight",
            f"{prefix}/ffn_layernorm",
            name="LayerNorm",
        )

        # Gate and Up projections
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

        # SiLU on gate
        gate_silu = self.make_silu(gate, f"{prefix}/mlp/act_fn")

        # gate * up
        gated = self.make_mul(gate_silu, up, f"{prefix}/mlp/Mul/output_0")

        # Down projection
        down = self.make_matmul(
            gated,
            f"{weight_prefix}.mlp.down_proj.MatMul.weight",
            f"{prefix}/mlp/down_proj/MatMul/output_0",
        )

        # (optional) Peri-LN on the FFN output, then Residual
        ffn_out = self._maybe_post_ffn_norm(layer_idx, down)
        return self.make_add(residual, ffn_out, f"{prefix}/Add_2/output_0")

    def build_lm_head(self, hidden_state: str) -> str:
        # A prefill otherwise returns [S, vocab] logits: 15 GB in float32 for a 30K-token prompt.
        # logits_to_keep = 0 keeps every position (Slice from -0).
        start = self.make_node("Neg", ["logits_to_keep"], ["/model/logits_to_keep/Neg/output_0"])
        hidden_state = self.make_node(
            "Slice",
            [hidden_state, start, self.get_constant([INT64_MAX]), self.get_constant([1])],
            ["/model/logits_to_keep/Slice/output_0"],
        )
        # Community naming: model.layers.{num_layers}.final_norm_layernorm.weight
        num_layers = self.config.num_hidden_layers
        final_norm_weight = f"model.layers.{num_layers}.final_norm_layernorm.weight"
        # Community uses shorter output path (no op type suffix)
        final_norm_output = f"/model/layers.{num_layers}/final_norm_layernorm/output_0"

        self.add_initializer(final_norm_weight, self.weights["model.embedding_norm.weight"])
        # A plain RMSNorm: SkipSimplifiedLayerNormalization(h, skip=h) normalizes 2h, which is
        # RMSNorm(h) with eps/4 and moved the fp32 logits ~4e-4 off PyTorch's.
        normed = self.make_node(
            "SimplifiedLayerNormalization",
            inputs=[hidden_state, final_norm_weight],
            outputs=[final_norm_output],
            name=f"/model/layers.{num_layers}/final_norm_layernorm/LayerNorm",
            epsilon=self.config.norm_eps,
        )

        # Transpose embed_tokens at runtime instead of storing a copy
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

        Adds shape annotations for all tensors in the graph,
        """
        H = self.config.hidden_size
        nh = self.config.num_attention_heads
        nkv = self.config.num_key_value_heads
        hd = self.head_dim
        q_hidden = nh * hd
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
        gather_name = "Gather"
        self.add_value_info(f"{mask_prefix}/ReduceSum/output_0", TensorProto.INT64, ["batch_size"])
        self.add_value_info(f"{mask_prefix}/Sub/output_0", TensorProto.INT64, ["batch_size"])
        self.add_value_info(f"{mask_prefix}/Sub/Cast/output_0", TensorProto.INT32, ["batch_size"])
        self.add_value_info(f"{mask_prefix}/Shape/output_0", TensorProto.INT64, [2])
        self.add_value_info(f"{mask_prefix}/{gather_name}/output_0", TensorProto.INT64, [])
        self.add_value_info(f"{mask_prefix}/{gather_name}/Cast/output_0", TensorProto.INT32, [])

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

            # operator_layernorm output
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
                self.add_value_info(f"{prefix}/conv/split_sizes", TensorProto.INT64, [3])
                self.add_value_info(
                    f"{prefix}/conv/Shape_for_slice/output_0", TensorProto.INT64, [3]
                )
                self.add_value_info(
                    f"{prefix}/conv/Gather_for_slice/output_0", TensorProto.INT64, []
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
                # Attention layer outputs (q-projection width is nh*hd; may != H)
                self.add_value_info(
                    f"{prefix}/attn/q_proj/MatMul/output_0",
                    TensorProto.FLOAT,
                    ["batch_size", "sequence_length", q_hidden],
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
                    ["batch_size", "sequence_length", q_hidden],
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
                    ["batch_size", "sequence_length", q_hidden],
                )
                self.add_value_info(
                    f"{prefix}/attn/k_rotary/RotaryEmbedding/output_0",
                    TensorProto.FLOAT,
                    ["batch_size", "sequence_length", kv_hidden],
                )
                # GQA output (post-attention, pre-out_proj)
                self.add_value_info(
                    f"{prefix}/attn/GroupQueryAttention/output_0",
                    TensorProto.FLOAT,
                    ["batch_size", "sequence_length", q_hidden],
                )
                # Optional gated-output value infos
                if self.config.use_gated_attention:
                    gate_width = nh if self.config.gated_attention_style == "headwise" else q_hidden
                    self.add_value_info(
                        f"{prefix}/attn/gate_proj/MatMul/output_0",
                        TensorProto.FLOAT,
                        ["batch_size", "sequence_length", gate_width],
                    )
                    self.add_value_info(
                        f"{prefix}/attn/gate_proj/Sigmoid/output_0",
                        TensorProto.FLOAT,
                        ["batch_size", "sequence_length", gate_width],
                    )
                    if self.config.gated_attention_style == "headwise":
                        self.add_value_info(
                            f"{prefix}/attn/gate_proj/Reshape/output_0",
                            TensorProto.FLOAT,
                            ["batch_size", "sequence_length", nh, 1],
                        )
                        self.add_value_info(
                            f"{prefix}/attn/GroupQueryAttention/Reshape/output_0",
                            TensorProto.FLOAT,
                            ["batch_size", "sequence_length", nh, hd],
                        )
                        self.add_value_info(
                            f"{prefix}/attn/gate_proj/Mul/output_0",
                            TensorProto.FLOAT,
                            ["batch_size", "sequence_length", nh, hd],
                        )
                        self.add_value_info(
                            f"{prefix}/attn/gate_proj/Reshape_back/output_0",
                            TensorProto.FLOAT,
                            ["batch_size", "sequence_length", q_hidden],
                        )
                    else:
                        self.add_value_info(
                            f"{prefix}/attn/gate_proj/Mul/output_0",
                            TensorProto.FLOAT,
                            ["batch_size", "sequence_length", q_hidden],
                        )
                # Output projection (back to hidden width)
                self.add_value_info(
                    f"{prefix}/attn/o_proj/MatMul/output_0",
                    TensorProto.FLOAT,
                    ["batch_size", "sequence_length", H],
                )

            # Optional peri-LN on operator output (applies to all layer types)
            if self.config.use_peri_ln:
                self.add_value_info(
                    f"{prefix}/post_operator_layernorm/output_0",
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
            # Optional peri-LN on FFN output
            if self.config.use_peri_ln:
                self.add_value_info(
                    f"{prefix}/post_ffn_layernorm/output_0",
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
            ["batch_size", "num_logits", H],
        )
        self.add_value_info(
            "/lm_head/Transpose/output_0", TensorProto.FLOAT, [H, self.config.vocab_size]
        )

    def load_weights(self, model_path: str | pathlib.Path):
        """Load the float32 weights from the checkpoint directory's safetensors files.

        Read directly rather than through the remote modeling code, which only some
        transformers releases can construct.
        """
        import torch
        from safetensors import safe_open

        files = sorted(pathlib.Path(model_path).glob("model*.safetensors"))
        if not files:
            raise FileNotFoundError(f"no model*.safetensors in {model_path}")
        logger.info(f"Loading weights from {model_path}...")
        for path in files:
            with safe_open(str(path), framework="pt") as f:
                for name in f.keys():
                    self.weights[name] = f.get_tensor(name).to(torch.float32).numpy()
        logger.info(f"Loaded {len(self.weights)} weights")

    def build(self, model_path: str | pathlib.Path) -> onnx.ModelProto:
        """Build the complete ONNX model.

        Build phases:
            1. Load weights from the checkpoint directory
            2. Create graph inputs/outputs
            3. Build RoPE cache and attention mask preprocessing
            4. Build embedding layer
            5. For each layer: prepare weights, then build graph
            6. Build LM head with tied weights
            7. Create final ONNX model
        """
        logger.info("Building LFM3 ONNX model...")

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
            elif layer_type == "sliding_attention":
                hidden_state = self.build_attention_layer(layer_idx, hidden_state, is_swa=True)
            elif layer_type == "full_attention":
                hidden_state = self.build_attention_layer(layer_idx, hidden_state, is_swa=False)
            else:
                raise ValueError(f"Unsupported layer_type {layer_type!r} at layer {layer_idx}")

        # LM head
        self.build_lm_head(hidden_state)

        # Build ValueInfo for shape annotations
        self.build_value_info()

        model = self.build_graph("lfm3")
        logger.info(f"Model built: {len(self.nodes)} nodes, {len(self.value_info)} value_info")
        return model
