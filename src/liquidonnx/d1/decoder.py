"""
d1's language model: the LFM3-VL (or LFM3-Omni) decoder run once over a packed row, under d1's
"set" attention (the checkpoint's attention.py, Sees.rows), read at its answer slots. Nothing is
generated, so the graph has no KV cache or conv state I/O.

decoder.onnx

    inputs
      input_ids       [B, S]   int64     the state, then up to d1_set_size questions (pack_set)
      attention_mask  [B, S]   int64     all ones (the session runs one row at a time)
      position_ids    [B, S]   int64     0..S-1
      image_features  [Ni, H]  float32   vision_encoder.onnx rows, prompt order
      video_features  [Nv, H]  float32
      audio_features  [Na, H]  float32   (d1-Omni) audio_encoder.onnx rows, prompt order
      question_ids    [B, S]   int64     -1 on the state, k on question k's tokens
      slot_positions  [N]      int64     flat positions b * S + s of the answer slots
    outputs
      logits          [N, V]   float32   the tied LM head at the slots
      hidden_states   [B, S, H] float32  the final norm's output, which pointer reads score the
                                         state positions with (pointer.read_logits)

    input_ids → Gather → ScatterND image/video features (LFM3VLDecoderBuilder), audio features
      → conv layers: the zero conv state of a prompt with no past (causal)
      → sliding layers: GroupQueryAttention, local_window_size, zero-length past (causal)
      → full layers: MultiHeadAttention, kv heads repeated, attention_bias [B, 1, S, S]: key k is
        seen by query q when k is real and q or k is on the state, or both on one question
      → final norm → hidden_states
                   └→ Gather(slot_positions) → tied lm_head → logits

GroupQueryAttention is causal whatever its attention_bias, so the full layers take
MultiHeadAttention. The bias is built once and shared by the full layers; it is O(S²): 256 MB in
float32 at 8k tokens.

The LM head is LFM3Builder's MatMul over the transposed token table, named logits, so
lfm3.export.derive_quantized gives the q4 and q8 decoders their shared int8 head and token table.
"""

import pathlib

import numpy as np
from onnx import TensorProto, helper

from liquidonnx.d1.config import D1Config
from liquidonnx.lfm3_audio.weights import load_weights as load_audio_weights
from liquidonnx.lfm3_vl.decoder import LFM3VLDecoderBuilder

# Float graph I/O that stays float32 in the fp16 decoder
KEEP_IO_TYPES = ["image_features", "video_features", "audio_features", "logits", "hidden_states"]
SET_BIAS = "/model/set_mask/Unsqueeze_bias/output_0"


class D1DecoderBuilder(LFM3VLDecoderBuilder):
    def __init__(self, config: D1Config):
        super().__init__(config.language, config.tokens.image, config.tokens.video)
        self.audio = config.audio

    def load_weights(self, model_path: str | pathlib.Path):
        if self.audio is None:
            super().load_weights(model_path)
        else:
            self.weights = load_audio_weights(model_path, "language", self.audio)

    def build_inputs(self):
        super().build_inputs()
        self.inputs = [
            i for i in self.inputs if i.name != "logits_to_keep" and not i.name.startswith("past_")
        ]
        H = self.config.hidden_size
        if self.audio is not None:
            self.inputs.append(
                helper.make_tensor_value_info(
                    "audio_features", TensorProto.FLOAT, ["num_audio_tokens", H]
                )
            )
        self.inputs += [
            helper.make_tensor_value_info(
                "question_ids", TensorProto.INT64, ["batch_size", "sequence_length"]
            ),
            helper.make_tensor_value_info("slot_positions", TensorProto.INT64, ["num_slots"]),
        ]

    def build_outputs(self):
        self.outputs = [
            helper.make_tensor_value_info(
                "logits", TensorProto.FLOAT, ["num_slots", self.config.vocab_size]
            ),
            helper.make_tensor_value_info(
                "hidden_states",
                TensorProto.FLOAT,
                ["batch_size", "sequence_length", self.config.hidden_size],
            ),
        ]

    def build_attention_mask_subgraph(self):
        super().build_attention_mask_subgraph()
        self.build_empty_caches()
        self.build_set_bias()

    def build_empty_caches(self):
        """The past_conv.N and past_key_values.N.* LFM3Builder's layers read, for a prompt with
        no past: zero conv states [B, H, L] and zero-length sliding-layer KV [B, nkv, 0, hd]."""
        prefix = "/model/empty_cache"
        cfg = self.config
        batch = self.make_node("Shape", ["input_ids"], [f"{prefix}/Shape/output_0"], end=1)
        zero = self.get_constant(0.0, np.float32)
        shapes = {"conv": [cfg.hidden_size, cfg.conv_L_cache], "kv": [cfg.num_key_value_heads, 0, self.head_dim]}  # fmt: skip
        empty = {}
        for kind, dims in shapes.items():
            shape = self.make_node(
                "Concat",
                [batch, self.get_constant(dims)],
                [f"{prefix}/Concat_{kind}/output_0"],
                axis=0,
            )
            empty[kind] = self.make_node(
                "Expand", [zero, shape], [f"{prefix}/Expand_{kind}/output_0"]
            )
        names = [(empty["conv"], f"past_conv.{i}") for i in self.conv_indices]
        for i in self.swa_indices:
            names += [(empty["kv"], f"past_key_values.{i}.{part}") for part in ("key", "value")]
        for source, name in names:
            self.make_node("Identity", [source], [name], name=f"{prefix}/{name}")

    def build_set_bias(self):
        """SET_BIAS [B, 1, S, S]: 0 where a query sees a key, the float minimum elsewhere, as
        attention.Sees.rows masks the full layers under set (the state is question -1)."""
        # shortcut: a dense bias, O(S²) memory (256 MB in float32 at 8k tokens); long states need
        # Sees._plan's split (state rows unmasked, question rows over [Q, S]) instead
        prefix = "/model/set_mask"
        q = self.make_unsqueeze(
            "question_ids", self.get_constant([2]), f"{prefix}/Unsqueeze_q/output_0"
        )
        k = self.make_unsqueeze(
            "question_ids", self.get_constant([1]), f"{prefix}/Unsqueeze_k/output_0"
        )
        state = self.get_constant(-1)
        # [B, S, 1] and [B, 1, S] → [B, S, S]
        same = self.make_node("Equal", [q, k], [f"{prefix}/Equal/output_0"])
        q_state = self.make_node("Equal", [q, state], [f"{prefix}/Equal_q_state/output_0"])
        k_state = self.make_node("Equal", [k, state], [f"{prefix}/Equal_k_state/output_0"])
        seen = self.make_node("Or", [same, q_state], [f"{prefix}/Or/output_0"])
        seen = self.make_node("Or", [seen, k_state], [f"{prefix}/Or_1/output_0"])
        real = self.make_node(
            "Cast", ["attention_mask"], [f"{prefix}/Cast/output_0"], to=TensorProto.BOOL
        )
        real = self.make_unsqueeze(
            real, self.get_constant([1]), f"{prefix}/Unsqueeze_real/output_0"
        )
        seen = self.make_node("And", [seen, real], [f"{prefix}/And/output_0"])
        bias = self.make_node(
            "Where",
            [
                seen,
                self.get_constant(0.0, np.float32),
                self.get_constant(np.finfo(np.float32).min, np.float32),
            ],
            [f"{prefix}/Where/output_0"],
        )
        self.make_unsqueeze(bias, self.get_constant([1]), SET_BIAS)

    def build_embedding(self) -> str:
        embeds = super().build_embedding()
        if self.audio is None:
            return embeds
        return self.build_feature_scatter(embeds, self.audio.tokens.audio, "audio_features")

    def make_node(self, op_type, inputs, outputs, name=None, domain="", **attrs) -> str:
        # LFM3Builder.build_attention_layer runs causal GroupQueryAttention; the full layers
        # (local_window_size -1) attend under the set mask instead
        if op_type == "GroupQueryAttention" and attrs["local_window_size"] == -1:
            return self.make_set_attention(*inputs[:3], outputs[0])
        return super().make_node(op_type, inputs, outputs, name=name, domain=domain, **attrs)

    def make_set_attention(self, q: str, k: str, v: str, output: str) -> str:
        """MultiHeadAttention of q [B, S, nh*hd] over k and v [B, S, nkv*hd], query head h
        reading kv head h // (nh / nkv) as repeat_kv does, under SET_BIAS."""
        path = output.rsplit("/", 2)[0]
        k = self.repeat_kv(k, f"{path}/k_repeat")
        v = self.repeat_kv(v, f"{path}/v_repeat")
        return super().make_node(
            "MultiHeadAttention",
            [q, k, v, "", "", SET_BIAS],
            [output],
            name=f"{path}/MultiHeadAttention",
            domain="com.microsoft",
            num_heads=self.config.num_attention_heads,
            scale=float(self.head_dim**-0.5),
        )

    def repeat_kv(self, x: str, path: str) -> str:
        """[B, S, nkv*hd] → [B, S, nkv, groups, hd] → [B, S, nh*hd]"""
        nh, nkv, hd = (
            self.config.num_attention_heads,
            self.config.num_key_value_heads,
            self.head_dim,
        )
        split = self.make_reshape(
            x, self.get_constant([0, 0, nkv, 1, hd]), f"{path}/Reshape/output_0"
        )
        expanded = self.make_node(
            "Expand",
            [split, self.get_constant([1, 1, 1, nh // nkv, 1])],
            [f"{path}/Expand/output_0"],
        )
        return self.make_reshape(
            expanded, self.get_constant([0, 0, nh * hd]), f"{path}/Reshape_back/output_0"
        )

    def build_lm_head(self, hidden_state: str) -> str:
        """The final norm at every position (hidden_states), and the tied LM head at the slots."""
        n, H = self.config.num_hidden_layers, self.config.hidden_size
        weight = f"model.layers.{n}.final_norm_layernorm.weight"
        self.add_initializer(weight, self.weights["model.embedding_norm.weight"])
        normed = self.make_node(
            "SimplifiedLayerNormalization",
            [hidden_state, weight],
            [f"/model/layers.{n}/final_norm_layernorm/output_0"],
            name=f"/model/layers.{n}/final_norm_layernorm/LayerNorm",
            epsilon=self.config.norm_eps,
        )
        self.make_node(
            "Identity", [normed], ["hidden_states"], name="/model/hidden_states/Identity"
        )
        # [B, S, H] → [B*S, H] → [N, H]
        flat = self.make_reshape(
            normed, self.get_constant([-1, H]), "/model/slots/Reshape/output_0"
        )
        slots = self.make_gather(flat, "slot_positions", "/model/slots/Gather/output_0")
        head = self.make_node(
            "Transpose", ["model.embed_tokens.weight"], ["/lm_head/Transpose/output_0"], perm=[1, 0]
        )
        return self.make_matmul(slots, head, "logits")
