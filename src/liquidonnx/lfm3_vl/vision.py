"""
LFM3-VL vision encoder + projector: one patch grid (an image, or one temporal patch of a video)
→ its projected features, as Lfm3VlForConditionalGeneration.get_visual_features.

    pixel_values [h/m, w/m, m, m, patch_dim]   Qwen2-VL processor rows of one grid, block-major
        → Transpose [0,2,1,3,4] → [1, h*w, patch_dim]                         raster order
        → patchifier MatMul → embedding_norm (RMSNorm)                         [1, N, C]
        → layers: x += op(operator_norm(x)); x += mlp(ffn_norm(x))
              conv: in_proj → (a, b, c) → depthwise 3x3 Conv over the h x w grid of a*c → b * · → out_proj
              attention: q/k/v → per-head RMSNorm → 2D RoPE → MultiHeadAttention (kv heads repeated)
        → LayerNorm → m x m merge [N/m², m²C] → linear_1 → GELU → linear_2  image_features [N/m², H]

The grid size comes from the input shape, so the host passes rows[offset:offset + h*w] reshaped to
(h/m, w/m, m, m, patch_dim). 2D RoPE gathers each row's and column's cos and sin from tables
computed at export (rope_2d_tables), in the graph's float type: the fp16 version rounds those values
to fp16 rather than computing the angles in fp16.
"""

import numpy as np
import onnx
from onnx import TensorProto, helper

from liquidonnx.builder_base import ONNXBuilderBase
from liquidonnx.lfm3_vl.config import MAX_GRID_SIDE, LFM3VLVisionConfig


def rope_2d_tables(config: LFM3VLVisionConfig, max_side: int = MAX_GRID_SIDE):
    """cos/sin [max_side, head_dim/4] of position * frequency, the angles of _rope_2d."""
    import torch

    quarter = config.head_dim // 4
    exponents = 2.0 * torch.arange(quarter, dtype=torch.float32) / (config.head_dim // 2)
    if config.rope_negative_power:
        frequencies = config.rope_theta**-exponents
    else:
        frequencies = 1.0 / config.rope_theta**exponents
    angles = (torch.arange(max_side)[:, None] * frequencies).numpy().astype(np.float64)
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)


class LFM3VLVisionBuilder(ONNXBuilderBase):
    def __init__(self, config: LFM3VLVisionConfig):
        super().__init__()
        self.config = config
        self._used: set[str] = set()

    def build(self, weights: dict[str, np.ndarray]) -> onnx.ModelProto:
        self.weights = weights
        cfg = self.config
        m = cfg.merge_size
        self.inputs.append(
            helper.make_tensor_value_info(
                "pixel_values",
                TensorProto.FLOAT,
                ["grid_height_blocks", "grid_width_blocks", m, m, cfg.patch_dim],
            )
        )
        self.outputs.append(
            helper.make_tensor_value_info(
                "image_features", TensorProto.FLOAT, ["num_image_tokens", cfg.text_hidden_size]
            )
        )

        hidden = self.build_patches()
        cos, sin, position_ids = self.build_rope()
        for idx, layer_type in enumerate(cfg.layer_types):
            prefix = f"/vision_encoder/layers.{idx}"
            normed = self.rms_norm(
                hidden, f"vision_encoder.layers.{idx}.operator_norm", f"{prefix}/operator_norm"
            )
            if layer_type == "conv":
                out = self.build_conv(normed, idx, prefix)
            else:
                out = self.build_attention(normed, idx, prefix, cos, sin, position_ids)
            hidden = self.make_add(hidden, out, f"{prefix}/Add_1/output_0")
            normed = self.rms_norm(
                hidden, f"vision_encoder.layers.{idx}.ffn_norm", f"{prefix}/ffn_norm"
            )
            out = self.build_mlp(
                normed, f"vision_encoder.layers.{idx}.feed_forward", f"{prefix}/mlp"
            )
            hidden = self.make_add(hidden, out, f"{prefix}/Add_2/output_0")
        self.build_projector(hidden)
        unused = set(self.weights) - self._used
        if unused:
            raise ValueError(f"unused vision tensors: {sorted(unused)[:8]}")
        return self.build_graph("vision_encoder")

    # === Weights ===

    def weight(self, name: str, transpose: bool = False) -> str:
        """Initializer of a checkpoint tensor (transposed for MatMul)."""
        self._used.add(name)
        array = self.weights[name]
        init = f"{name}.T" if transpose else name
        self.add_initializer(init, array.T if transpose else array)
        return init

    def linear(self, x: str, name: str, path: str, bias: bool = False) -> str:
        out = self.make_matmul(
            x, self.weight(f"{name}.weight", transpose=True), f"{path}/MatMul/output_0"
        )
        if bias:
            out = self.make_add(out, self.weight(f"{name}.bias"), f"{path}/Add/output_0")
        return out

    def rms_norm(self, x: str, name: str, path: str) -> str:
        return self.make_layernorm(
            x, self.weight(f"{name}.weight"), None, path, epsilon=self.config.norm_eps
        )

    # === Graph pieces ===

    def build_patches(self) -> str:
        cfg = self.config
        m = cfg.merge_size
        shape = self.make_node("Shape", ["pixel_values"], ["/vision_encoder/grid/Shape/output_0"])
        blocks_h = self.make_node(
            "Slice",
            [shape, self.get_constant([0]), self.get_constant([1])],
            ["/vision_encoder/grid/blocks_h/output_0"],
        )
        blocks_w = self.make_node(
            "Slice",
            [shape, self.get_constant([1]), self.get_constant([2])],
            ["/vision_encoder/grid/blocks_w/output_0"],
        )
        self.grid_blocks_h = blocks_h
        self.grid_h = self.make_mul(
            blocks_h, self.get_constant([m]), "/vision_encoder/grid/h/output_0"
        )
        self.grid_w = self.make_mul(
            blocks_w, self.get_constant([m]), "/vision_encoder/grid/w/output_0"
        )

        raster = self.make_transpose(
            "pixel_values", "/vision_encoder/raster/Transpose/output_0", perm=[0, 2, 1, 3, 4]
        )
        patches = self.make_reshape(
            raster,
            self.get_constant([1, -1, cfg.patch_dim]),
            "/vision_encoder/raster/Reshape/output_0",
        )
        embeds = self.linear(patches, "vision_encoder.patchifier", "/vision_encoder/patchifier")
        return self.rms_norm(
            embeds, "vision_encoder.embedding_norm", "/vision_encoder/embedding_norm"
        )

    def build_rope(self) -> tuple[str, str, str]:
        """Per-token cos/sin [N, head_dim/2] (row angles, then column angles) and position_ids [1, N]."""
        cfg = self.config
        cos_table, sin_table = rope_2d_tables(cfg)
        self.add_initializer("vision_rope_cos_table", cos_table)
        self.add_initializer("vision_rope_sin_table", sin_table)
        zero, one = self.get_constant(0), self.get_constant(1)
        h = self.make_node("Squeeze", [self.grid_h], ["/vision_encoder/rope/h/output_0"])
        w = self.make_node("Squeeze", [self.grid_w], ["/vision_encoder/rope/w/output_0"])
        rows = self.make_node("Range", [zero, h, one], ["/vision_encoder/rope/rows/output_0"])
        cols = self.make_node("Range", [zero, w, one], ["/vision_encoder/rope/cols/output_0"])
        grid = self.make_node(
            "Concat",
            [self.grid_h, self.grid_w, self.get_constant([cfg.head_dim // 4])],
            ["/vision_encoder/rope/grid/output_0"],
            axis=0,
        )
        tables = []
        for kind, table in (("cos", "vision_rope_cos_table"), ("sin", "vision_rope_sin_table")):
            path = f"/vision_encoder/rope/{kind}"
            by_row = self.make_gather(table, rows, f"{path}/rows/output_0")
            by_row = self.make_unsqueeze(
                by_row, self.get_constant([1]), f"{path}/rows_unsqueeze/output_0"
            )
            by_row = self.make_node("Expand", [by_row, grid], [f"{path}/rows_expand/output_0"])
            by_col = self.make_gather(table, cols, f"{path}/cols/output_0")
            by_col = self.make_unsqueeze(
                by_col, self.get_constant([0]), f"{path}/cols_unsqueeze/output_0"
            )
            by_col = self.make_node("Expand", [by_col, grid], [f"{path}/cols_expand/output_0"])
            both = self.make_concat([by_row, by_col], f"{path}/Concat/output_0", axis=-1)
            tables.append(
                self.make_reshape(
                    both, self.get_constant([-1, cfg.head_dim // 2]), f"{path}/output_0"
                )
            )
        n = self.make_node("Mul", [h, w], ["/vision_encoder/rope/n/output_0"])
        positions = self.make_node(
            "Range", [zero, n, one], ["/vision_encoder/rope/positions/output_0"]
        )
        position_ids = self.make_unsqueeze(
            positions, self.get_constant([0]), "/vision_encoder/rope/position_ids/output_0"
        )
        return tables[0], tables[1], position_ids

    def build_conv(self, x: str, idx: int, prefix: str) -> str:
        C = self.config.hidden_size
        name = f"vision_encoder.layers.{idx}.conv"
        proj = self.linear(x, f"{name}.in_proj", f"{prefix}/conv/in_proj")
        parts = [f"{prefix}/conv/Split/output_{i}" for i in range(3)]
        self.make_node("Split", [proj, self.get_constant([C, C, C])], parts, axis=-1)
        gated = self.make_mul(parts[0], parts[2], f"{prefix}/conv/gate/output_0")
        channels_first = self.make_transpose(
            gated, f"{prefix}/conv/Transpose/output_0", perm=[0, 2, 1]
        )
        image_shape = self.make_node(
            "Concat",
            [self.get_constant([1, C]), self.grid_h, self.grid_w],
            [f"{prefix}/conv/image_shape/output_0"],
            axis=0,
        )
        image = self.make_reshape(channels_first, image_shape, f"{prefix}/conv/Reshape/output_0")
        conv = self.make_node(
            "Conv",
            [image, self.weight(f"{name}.spatial_conv")],
            [f"{prefix}/conv/Conv/output_0"],
            group=C,
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
        )
        flat = self.make_reshape(
            conv, self.get_constant([1, C, -1]), f"{prefix}/conv/Reshape_back/output_0"
        )
        tokens = self.make_transpose(flat, f"{prefix}/conv/Transpose_back/output_0", perm=[0, 2, 1])
        out = self.make_mul(parts[1], tokens, f"{prefix}/conv/out_gate/output_0")
        return self.linear(out, f"{name}.out_proj", f"{prefix}/conv/out_proj")

    def build_attention(
        self, x: str, idx: int, prefix: str, cos: str, sin: str, position_ids: str
    ) -> str:
        cfg = self.config
        hd, nh, nkv = cfg.head_dim, cfg.num_attention_heads, cfg.num_key_value_heads
        name = f"vision_encoder.layers.{idx}.self_attn"
        q = self.linear(x, f"{name}.q_proj", f"{prefix}/attn/q_proj")
        k = self.linear(x, f"{name}.k_proj", f"{prefix}/attn/k_proj")
        v = self.linear(x, f"{name}.v_proj", f"{prefix}/attn/v_proj")
        q = self.head_norm(q, f"{name}.q_layernorm", f"{prefix}/attn/q_norm", nh)
        k = self.head_norm(k, f"{name}.k_layernorm", f"{prefix}/attn/k_norm", nkv)
        rope = {
            "domain": "com.microsoft",
            "interleaved": 0,
            "num_heads": 0,
            "rotary_embedding_dim": 0,
        }
        q = self.make_node(
            "RotaryEmbedding",
            [q, position_ids, cos, sin],
            [f"{prefix}/attn/q_rotary/output_0"],
            **rope,
        )
        k = self.make_node(
            "RotaryEmbedding",
            [k, position_ids, cos, sin],
            [f"{prefix}/attn/k_rotary/output_0"],
            **rope,
        )
        # repeat_interleave of the kv heads: query head i reads kv head i // (nh // nkv)
        k = self.repeat_kv(k, f"{prefix}/attn/k_repeat")
        v = self.repeat_kv(v, f"{prefix}/attn/v_repeat")
        attn = self.make_node(
            "MultiHeadAttention",
            [q, k, v],
            [f"{prefix}/attn/MultiHeadAttention/output_0"],
            domain="com.microsoft",
            num_heads=nh,
            scale=float(hd**-0.5),
        )
        return self.linear(attn, f"{name}.out_proj", f"{prefix}/attn/out_proj")

    def head_norm(self, x: str, name: str, path: str, heads: int) -> str:
        hd = self.config.head_dim
        per_head = self.make_reshape(
            x, self.get_constant([1, -1, heads, hd]), f"{path}/Reshape/output_0"
        )
        normed = self.rms_norm(per_head, name, path)
        return self.make_reshape(
            normed, self.get_constant([1, -1, heads * hd]), f"{path}/Reshape_back/output_0"
        )

    def repeat_kv(self, x: str, path: str) -> str:
        cfg = self.config
        groups = cfg.num_attention_heads // cfg.num_key_value_heads
        if groups == 1:
            return x
        hd, nkv = cfg.head_dim, cfg.num_key_value_heads
        split = self.make_reshape(
            x, self.get_constant([1, -1, nkv, 1, hd]), f"{path}/Reshape/output_0"
        )
        expanded = self.make_node(
            "Expand", [split, self.get_constant([1, 1, 1, groups, 1])], [f"{path}/Expand/output_0"]
        )
        return self.make_reshape(
            expanded, self.get_constant([1, -1, cfg.num_attention_heads * hd]), f"{path}/output_0"
        )

    def build_mlp(self, x: str, name: str, path: str) -> str:
        inter = self.config.intermediate_size
        gate_up = self.linear(x, f"{name}.gate_up", f"{path}/gate_up")
        gate, up = f"{path}/Split/output_0", f"{path}/Split/output_1"
        self.make_node("Split", [gate_up, self.get_constant([inter, inter])], [gate, up], axis=-1)
        act = self.make_silu(gate, f"{path}/act")
        out = self.make_mul(act, up, f"{path}/Mul/output_0")
        return self.linear(out, f"{name}.w2", f"{path}/w2")

    def build_projector(self, hidden: str):
        cfg = self.config
        m, C = cfg.merge_size, cfg.hidden_size
        normed = self.make_layernorm(
            hidden,
            self.weight("vision_projector.layer_norm.weight"),
            self.weight("vision_projector.layer_norm.bias"),
            "/vision_projector/layer_norm",
            epsilon=cfg.projector_eps,
        )
        # [1, h*w, C] → [h/m, m, w/m, m, C] → [h/m, w/m, m, m, C] → [h*w/m², m²C]
        blocks = self.make_node(
            "Concat",
            [self.grid_blocks_h, self.get_constant([m, -1, m, C])],
            ["/vision_projector/merge/shape/output_0"],
            axis=0,
        )
        grid = self.make_reshape(normed, blocks, "/vision_projector/merge/Reshape/output_0")
        grid = self.make_transpose(
            grid, "/vision_projector/merge/Transpose/output_0", perm=[0, 2, 1, 3, 4]
        )
        merged = self.make_reshape(
            grid,
            self.get_constant([-1, m * m * C]),
            "/vision_projector/merge/Reshape_back/output_0",
        )
        out = self.linear(
            merged, "vision_projector.linear_1", "/vision_projector/linear_1", bias=True
        )
        out = self.make_gelu(out, "/vision_projector/act", approximate="none")
        out = self.linear(out, "vision_projector.linear_2", "/vision_projector/linear_2", bias=True)
        self.make_node("Identity", [out], ["image_features"])
