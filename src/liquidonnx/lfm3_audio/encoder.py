"""
LFM3-Audio speech encoder: log-mel of one clip -> one projected feature per 8 mel frames, as
Lfm3AudioModel.get_audio_features does for one unpadded clip (its padding masks are then no-ops).

audio_encoder.onnx

    input_features [1, 128, T] float32      (session.log_mel of one 16 kHz clip, T >= 1)
      → Unsqueeze                            [1, 1, 128, T]       (mel is the H axis, time W)
      → conv_in Conv(1→256, k3, s2, p1) + SiLU                     [1, 256, 64, ceil(T/2)]
      → conv_mid depthwise Conv(k3, s2, p1) → lin_mid 1x1 Conv → SiLU   [1, 256, 32, ceil(T/4)]
      → conv_end depthwise Conv(k3, s2, p1) → lin_end 1x1 Conv → SiLU   [1, 256, 16, N]
      → Reshape (channel-major flatten(1, 2))                       [1, 4096, N]
      → Transpose → lin_out (Conv1d of kernel 1, as a MatMul + bias) [1, N, 768]
      → LFM2 stack as LFM3Builder layers (config.encoder): 10 conv layers (L=3) and 7
        sliding_attention layers (window 32, 8/4 heads, hd 96, q/k RMSNorm, full RoPE theta 1e6
        over positions 0..N-1, no gate, no sandwich norms), SwiGLU 1792, no cache I/O
      → final RMSNorm (model.embedding_norm)                        [1, N, 768]
      → projector: LayerNorm(eps 1e-5, bias) → linear_1 (+b) → GELU (exact erf) → linear_2 (+b)
    audio_features [N, 2048] float32         N = ceil(T / 8)

One clip per run; the session concatenates the clips of a prompt in order. The encoder is causal
up to the downsampler's lookahead of 7 mel frames (streaming is out of scope for v1).

Weights: load_weights(checkpoint, "encoder", config): downsample.{conv_in, conv_mid, lin_mid,
conv_end, lin_end, lin_out}.{weight, bias}, model.layers.N.* (q_layernorm / k_layernorm / out_proj),
model.embedding_norm.weight, projector.{layer_norm, linear_1, linear_2}.{weight, bias}.

GroupQueryAttention's local_window_size=32 attends to the same 32 keys, the query's own included,
as HF's sliding window, on the CPU EP and CUDA's fp16 kernel (test_audio_encoder.test_gqa_window).
CUDA has no fp32 GQA kernel: the fp32 graph runs its attention on the CPU there.
"""

import pathlib

import numpy as np
import onnx
from onnx import TensorProto, helper

from liquidonnx.lfm3.builder import LFM3Builder
from liquidonnx.lfm3_audio import weights
from liquidonnx.lfm3_audio.config import ENCODER_STRIDE, LFM3AudioConfig

INPUT = "input_features"
OUTPUT = "audio_features"
# nn.LayerNorm's default
PROJECTOR_EPS = 1e-5


class LFM3AudioEncoderBuilder(LFM3Builder):
    def __init__(self, config: LFM3AudioConfig):
        """Builds from config.encoder (the LFM2 stack), config.encoder_conv_channels,
        config.mel.num_mel_bins and config.language.hidden_size (the projector's output)."""
        super().__init__(config.encoder)
        self.audio_config = config

    def load_weights(self, model_path: str | pathlib.Path):
        self.weights = weights.load_weights(model_path, "encoder", self.audio_config)

    def build_inputs(self):
        mels = self.audio_config.mel.num_mel_bins
        self.inputs.append(
            helper.make_tensor_value_info(INPUT, TensorProto.FLOAT, [1, mels, "num_frames"])
        )

    def build_outputs(self):
        H = self.audio_config.language.hidden_size
        self.outputs.append(
            helper.make_tensor_value_info(OUTPUT, TensorProto.FLOAT, ["num_audio_tokens", H])
        )

    def build_attention_mask_subgraph(self):
        """What the decoder's caller passes the LFM3Builder layers, for N = ceil(T / 8) tokens
        with no past: position_ids 0..N-1, GroupQueryAttention's sequence lengths N - 1 and N,
        and zero conv states."""
        prefix = "/model/audio_tokens"
        mask_prefix = "/model/attn_mask_reformat/attn_mask_subgraph"
        frames = self.make_node("Shape", [INPUT], [f"{prefix}/Shape/output_0"], start=2, end=3)
        stride = self.get_constant([ENCODER_STRIDE])
        padded = self.make_add(
            frames, self.get_constant([ENCODER_STRIDE - 1]), f"{prefix}/Add/output_0"
        )
        tokens = self.make_node("Div", [padded, stride], [f"{prefix}/Div/output_0"])  # [1]
        count = self.make_node(
            "Squeeze", [tokens, self.get_constant([0])], [f"{prefix}/Squeeze/output_0"]
        )
        positions = self.make_node(
            "Range",
            [self.get_constant(0), count, self.get_constant(1)],
            [f"{prefix}/Range/output_0"],
        )
        self.make_unsqueeze(positions, self.get_constant([0]), "position_ids")
        past = self.make_sub(tokens, self.get_constant([1]), f"{prefix}/Sub/output_0")
        self.make_node("Cast", [past], [f"{mask_prefix}/Sub/Cast/output_0"], to=TensorProto.INT32)
        self.make_node(
            "Cast", [count], [f"{mask_prefix}/Gather/Cast/output_0"], to=TensorProto.INT32
        )
        H, L = self.config.hidden_size, self.config.conv_L_cache
        for idx in self.conv_indices:
            self.add_initializer(f"past_conv.{idx}", np.zeros((1, H, L), dtype=np.float32))

    def build_embedding(self) -> str:
        """The downsampler: [1, mels, T] → [1, N, hidden]."""
        prefix = "/model/downsample"
        x = self.make_unsqueeze(INPUT, self.get_constant([1]), f"{prefix}/Unsqueeze/output_0")
        channels = self.audio_config.encoder_conv_channels
        for conv, depthwise, pointwise in (
            ("conv_in", False, None),
            ("conv_mid", True, "lin_mid"),
            ("conv_end", True, "lin_end"),
        ):
            x = self.make_conv2d(x, conv, group=channels if depthwise else 1, stride=2, pad=1)
            if pointwise:
                x = self.make_conv2d(x, pointwise)
            x = self.make_silu(x, f"{prefix}/{conv}/act")
        # [1, C, mels / 8, N] → [1, C * mels / 8, N], channel-major as flatten(1, 2)
        flat = channels * self.audio_config.mel.num_mel_bins // ENCODER_STRIDE
        x = self.make_reshape(x, self.get_constant([0, flat, -1]), f"{prefix}/Reshape/output_0")
        x = self.make_transpose(x, f"{prefix}/Transpose/output_0", perm=[0, 2, 1])
        # lin_out is a Conv1d of kernel 1: a linear layer over the flattened features
        return self.make_linear(
            x,
            self.weights["downsample.lin_out.weight"][:, :, 0],
            "downsample.lin_out.MatMul.weight",
            f"{prefix}/lin_out",
            self.weights["downsample.lin_out.bias"],
            "downsample.lin_out.bias",
        )

    def make_conv2d(self, x: str, name: str, group: int = 1, stride: int = 1, pad: int = 0) -> str:
        weight, bias = f"downsample.{name}.weight", f"downsample.{name}.bias"
        self.add_initializer(weight, self.weights[weight])
        self.add_initializer(bias, self.weights[bias])
        kernel = self.weights[weight].shape[-1]
        return self.make_node(
            "Conv",
            [x, weight, bias],
            [f"/model/downsample/{name}/Conv/output_0"],
            kernel_shape=[kernel, kernel],
            strides=[stride, stride],
            pads=[pad] * 4,
            group=group,
        )

    def build_lm_head(self, hidden_state: str) -> str:
        """The final norm, then the projector: LayerNorm → linear_1 → GELU (exact) → linear_2."""
        self.add_initializer(
            "model.embedding_norm.weight", self.weights["model.embedding_norm.weight"]
        )
        x = self.make_simple_layernorm(
            hidden_state, "model.embedding_norm.weight", "/model/final_norm"
        )
        for name in ("projector.layer_norm.weight", "projector.layer_norm.bias"):
            self.add_initializer(name, self.weights[name])
        x = self.make_layernorm(
            x,
            "projector.layer_norm.weight",
            "projector.layer_norm.bias",
            "/projector/layer_norm",
            epsilon=PROJECTOR_EPS,
        )
        for i, name in enumerate(("linear_1", "linear_2")):
            if i:
                x = self.make_gelu(x, "/projector/act", approximate="none")
            x = self.make_linear(
                x,
                self.weights[f"projector.{name}.weight"],
                f"projector.{name}.MatMul.weight",
                f"/projector/{name}",
                self.weights[f"projector.{name}.bias"],
                f"projector.{name}.bias",
            )
        H = self.audio_config.language.hidden_size
        return self.make_reshape(x, self.get_constant([-1, H]), OUTPUT)

    def build(self, model_path: str | pathlib.Path) -> onnx.ModelProto:
        """audio_encoder.onnx of the checkpoint directory (see the module docstring)."""
        model = super().build(model_path)
        for node in model.graph.node:
            if node.op_type == "GroupQueryAttention":
                # no past keys and values: the clip is the whole sequence
                node.input[3], node.input[4] = "", ""
        return model
