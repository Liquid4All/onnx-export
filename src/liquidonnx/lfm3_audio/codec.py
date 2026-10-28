"""
Liquid codec decoder: speech frame codes -> the STFT frames of their 24 kHz waveform, as
LiquidCodecModel.decode does before its iSTFT (which session.stft_to_wave runs in numpy).

codec_decoder.onnx

    audio_codes    [B, 12, F]      int64    codes in [0, 2016): end-of-audio frames dropped
    stft_features  [B, 6F, 1282]   float32  [..., :641] log-magnitude, [..., 641:] phase angle

      audio_codes → Transpose [B, F, 12] → Unsqueeze                        [B, F, 12, 1]
      → FSQ decode with int64 Div / Mod: ((codes // basis) % levels - half) / half,
        levels [8, 7, 6, 6], basis [1, 8, 56, 336], half [4, 3, 3, 3], Cast to float
                                                                            [B, F, 12, 4]
      → Reshape (codebook-major)                                            [B, F, 48]
      → quantizer.project_out Linear 48→256 → from_quantizer Linear 256→512 [B, F, 512]
      → repeat each frame 6 times (F.interpolate nearest-exact x6: Unsqueeze/Expand/Reshape)
                                                                            [B, 6F, 512]
      → LFM3Builder layers (config.codec): conv (L=7) at 0, 1, 3, 5, 7; sliding_attention at
        2, 4, 6 (window 30, causal, 8/2 heads, hd 64, q/k RMSNorm, headwise gate, full RoPE
        theta 1e6 over positions 0..6F-1), sandwich norms, SwiGLU 2304, no cache I/O
      → final RMSNorm (model.embedding_norm) → lin Linear 512→1282          [B, 6F, 1282]

Offline only (one utterance per run); no streaming caches in v1.

Weights: load_codec_weights(checkpoint, config): quantizer.project_out.{weight, bias},
from_quantizer.{weight, bias}, model.layers.N.*, model.embedding_norm.weight, lin.{weight, bias}
(istft.window is the numpy iSTFT's periodic Hann window).

Precision: fp32 and fp16 only (bf16 PyTorch is already 2-3% off fp32 on speech, 6-8% on random
codes). GroupQueryAttention's local_window_size = 30 attends to HF's 30 keys (query included) on
the CPU and CUDA EPs (tests/test_lfm3_audio/test_gqa_window.py), so no explicit mask is needed.
"""

import math
import pathlib

import numpy as np
from onnx import TensorProto, helper

from liquidonnx.lfm3.builder import LFM3Builder
from liquidonnx.lfm3_audio.config import LFM3AudioConfig
from liquidonnx.lfm3_audio.weights import load_codec_weights

INPUT = "audio_codes"
OUTPUT = "stft_features"


class LFM3AudioCodecBuilder(LFM3Builder):
    def __init__(self, config: LFM3AudioConfig):
        """Builds from config.codec, config.fsq_levels, config.codec_quantizer_dim,
        config.codec_frame_stack, config.codec_n_fft and config.num_codebooks."""
        super().__init__(config.codec)
        self.audio_config = config

    def load_weights(self, model_path: str | pathlib.Path):
        self.weights = load_codec_weights(model_path, self.audio_config)

    def build_inputs(self):
        K = self.audio_config.num_codebooks
        self.inputs.append(
            helper.make_tensor_value_info(INPUT, TensorProto.INT64, ["batch_size", K, "num_frames"])
        )

    def build_outputs(self):
        width = self.audio_config.codec_n_fft + 2
        self.outputs.append(
            helper.make_tensor_value_info(
                OUTPUT, TensorProto.FLOAT, ["batch_size", "sequence_length", width]
            )
        )

    def build_attention_mask_subgraph(self):
        """The decoder inputs LFM3Builder's layers read, made in-graph for one offline run over
        S = frame_stack * F positions: an all-ones attention_mask [B, S], position_ids 0..S-1 and
        empty caches (zero conv states, zero-length KV)."""
        cfg = self.config
        p = "/codec/inputs"
        shape = self.make_node("Shape", [INPUT], [f"{p}/Shape/output_0"])
        batch = self.make_slice(
            shape,
            self.get_constant([0]),
            self.get_constant([1]),
            self.get_constant([0]),
            f"{p}/batch/output_0",
        )
        frames = self.make_slice(
            shape,
            self.get_constant([2]),
            self.get_constant([3]),
            self.get_constant([0]),
            f"{p}/frames/output_0",
        )
        steps = self.make_mul(
            frames, self.get_constant([self.audio_config.codec_frame_stack]), f"{p}/steps/output_0"
        )
        batch_steps = self.make_concat([batch, steps], f"{p}/batch_steps/output_0", axis=0)
        self.make_node(
            "ConstantOfShape",
            [batch_steps],
            ["attention_mask"],
            value=helper.make_tensor("one", TensorProto.INT64, [1], [1]),
        )
        positions = self.make_node(
            "Range",
            [
                self.get_constant(0),
                self.make_node("Squeeze", [steps], [f"{p}/S/output_0"]),
                self.get_constant(1),
            ],
            [f"{p}/Range/output_0"],
        )
        self.make_node("Expand", [positions, batch_steps], ["position_ids"])

        conv_state = [cfg.hidden_size, cfg.conv_L_cache]
        kv_state = [cfg.num_key_value_heads, 0, self.head_dim]
        for name, dims in (("conv", conv_state), ("kv", kv_state)):
            self.add_initializer(f"codec.zero_{name}", np.zeros([1, *dims], np.float32))
            self.make_node(
                "Expand",
                [
                    f"codec.zero_{name}",
                    self.make_concat(
                        [batch, self.get_constant(dims)], f"{p}/{name}_shape/output_0", 0
                    ),
                ],
                [f"{p}/zero_{name}/output_0"],
            )
        for i in self.conv_indices:
            self.make_node("Identity", [f"{p}/zero_conv/output_0"], [f"past_conv.{i}"])
        for i in self.attn_indices:
            for kind in ("key", "value"):
                self.make_node(
                    "Identity", [f"{p}/zero_kv/output_0"], [f"past_key_values.{i}.{kind}"]
                )
        super().build_attention_mask_subgraph()

    def build_embedding(self) -> str:
        """FSQ decode, quantizer.project_out, from_quantizer and the x frame_stack repeat:
        audio_codes [B, K, F] → [B, stack * F, H]."""
        levels = np.array(self.audio_config.fsq_levels, np.int64)
        basis = np.array([math.prod(levels[:i]) for i in range(len(levels))], np.int64)
        half = levels // 2
        latent = len(levels) * self.audio_config.num_codebooks
        p = "/codec/quantizer"

        # [B, K, F] → [B, F, K, 1]
        codes = self.make_transpose(INPUT, f"{p}/Transpose/output_0", [0, 2, 1])
        codes = self.make_unsqueeze(codes, self.get_constant([3]), f"{p}/Unsqueeze/output_0")
        # ((codes // basis) % levels - half) / half, on non-negative int64 codes  [B, F, K, 4]
        digits = self.make_node("Div", [codes, self.get_constant(basis)], [f"{p}/Div/output_0"])
        digits = self.make_node(
            "Mod", [digits, self.get_constant(levels)], [f"{p}/Mod/output_0"], fmod=0
        )
        digits = self.make_sub(digits, self.get_constant(half), f"{p}/Sub/output_0")
        digits = self.make_node("Cast", [digits], [f"{p}/Cast/output_0"], to=TensorProto.FLOAT)
        latents = self.make_node(
            "Div", [digits, self.get_constant(half, np.float32)], [f"{p}/Scale/output_0"]
        )
        # [B, F, K, 4] → [B, F, K * 4] (codebook-major)
        hidden = self.make_reshape(
            latents, self.get_constant([0, 0, latent]), f"{p}/Reshape/output_0"
        )

        # project_out is absent when the latent is already quantizer_dim wide
        linears = [
            n for n in ("quantizer.project_out", "from_quantizer") if f"{n}.weight" in self.weights
        ]
        for name in linears:
            hidden = self.make_linear(
                hidden,
                self.weights[f"{name}.weight"],
                f"codec.{name}.MatMul.weight",
                f"/codec/{name}",
                bias=self.weights[f"{name}.bias"],
                bias_name=f"codec.{name}.bias",
            )

        # F.interpolate(nearest-exact) by an integer factor repeats each frame  [B, F, H] → [B, sF, H]
        H, stack = self.config.hidden_size, self.audio_config.codec_frame_stack
        p = "/codec/repeat"
        hidden = self.make_unsqueeze(hidden, self.get_constant([2]), f"{p}/Unsqueeze/output_0")
        hidden = self.make_node(
            "Expand", [hidden, self.get_constant([1, 1, stack, 1])], [f"{p}/Expand/output_0"]
        )
        return self.make_reshape(hidden, self.get_constant([0, -1, H]), f"{p}/Reshape/output_0")

    def build_lm_head(self, hidden_state: str) -> str:
        """Final RMSNorm, then lin: [B, S, H] → stft_features [B, S, n_fft + 2]."""
        num_layers = self.config.num_hidden_layers
        norm_weight = f"model.layers.{num_layers}.final_norm_layernorm.weight"
        self.add_initializer(norm_weight, self.weights["model.embedding_norm.weight"])
        normed = self.make_node(
            "SimplifiedLayerNormalization",
            [hidden_state, norm_weight],
            [f"/model/layers.{num_layers}/final_norm_layernorm/output_0"],
            name=f"/model/layers.{num_layers}/final_norm_layernorm/LayerNorm",
            epsilon=self.config.norm_eps,
        )
        self.add_initializer("codec.lin.MatMul.weight", self.weights["lin.weight"].T)
        self.add_initializer("codec.lin.bias", self.weights["lin.bias"])
        out = self.make_matmul(normed, "codec.lin.MatMul.weight", "/codec/lin/MatMul/output_0")
        return self.make_add(out, "codec.lin.bias", OUTPUT)

    def build_value_info(self):
        super().build_value_info()
        # LFM3Builder's token embedding and tied head do not exist here
        dropped = {"/model/embed_tokens/Gather/output_0", "/lm_head/Transpose/output_0"}
        self.value_info = [v for v in self.value_info if v.name not in dropped]
