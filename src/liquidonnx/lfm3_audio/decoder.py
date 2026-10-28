"""
LFM3-Audio language model: the LFM3 decoder with speech features and speech frames scattered into
its token embeddings, as Lfm3AudioModel.forward does, returning the final-norm hidden states the
depth decoder samples speech frames from as well as the logits.

decoder.onnx (an LFM3Builder subclass: LFM3Builder's cache names and node names, which
CachedDecoder, kv_feeds_attention, lfm3.export.derive_quantized and lfm3.quantize rely on)

    inputs, in this order
      input_ids        [B, S]        int64
      attention_mask   [B, P + S]    int64 (all ones)
      position_ids     [B, S]        int64
      image_features   [Ni, H]       float32   (only when built with image/video token ids:
      video_features   [Nv, H]       float32    lfm3_omni)
      audio_features   [Na, H]       float32   (audio_encoder.onnx output rows, prompt order)
      audio_out_codes  [Nf, 12]      int64     (one row of codes in [0, 2017) per <|audio_out|>)
      logits_to_keep   [1]           int64     (0 keeps every position)
      past_conv.N / past_key_values.N.{key,value}   as LFM3Builder
    outputs
      logits           [B, K, 128000] float32  K = logits_to_keep (or S)
      hidden_states    [B, K, H]      float32  the final RMSNorm's output at the same K positions
      present_conv.N / present.N.{key,value}

    input_ids → Gather(model.embed_tokens.weight)                    [B, S, H]
      → ScatterND image_features at image_token_id, video_features at video_token_id (omni)
      → ScatterND audio_features at tokens.audio (125031)
      → audio_out_codes + 2017 * [0..11] → Gather(codebook_embedding.weight [24204, H])
        [Nf, 12, H] → ReduceSum(axis 1) [Nf, H] → ScatterND at tokens.audio_out (125032)
      → LFM3 layers (config.language) → Slice(-logits_to_keep:) → final norm ─→ hidden_states
                                                                             └→ tied lm_head → logits

Row i of a feature (or code) input lands on the i-th placeholder of its token, as
LFM3VLDecoderBuilder.build_feature_scatter does; the session checks the counts and that codes are
in range (the graph fails only on too many rows, and a code >= 2017 silently reads the next
codebook's row). Decode steps pass zero-row features and codes ([0, H], [0, 12]); a speech frame
step passes input_ids [[125032]] with its codes [1, 12].

Weights: load_weights(checkpoint, "language", config): model.embed_tokens.weight,
model.embedding_norm.weight, model.layers.N.*, codebook_embedding.weight.

Derived precisions follow lfm3_vl.export.derive: fp16 via lfm3.export.convert_to_fp16 with
keep_io_types=KEEP_IO_TYPES, q4/q8/q4f16/q8f16 via lfm3.export.derive_quantized (int8 shared head
and token table). The codebook table stays a float Gather (fp16 in the fp16-activation graphs):
an int8 table (blocks of 32) would cost little next to the q8 decoder's own error, but would save
only a small part of q8's size.
"""

import pathlib

import numpy as np
from onnx import TensorProto, helper

from liquidonnx.lfm3.builder import LFM3Builder
from liquidonnx.lfm3_audio.config import LFM3AudioConfig
from liquidonnx.lfm3_audio.weights import load_weights
from liquidonnx.lfm3_vl.decoder import LFM3VLDecoderBuilder

# Float graph I/O that stays float32 in the fp16 decoder (caches become fp16)
KEEP_IO_TYPES = ["image_features", "video_features", "audio_features", "logits", "hidden_states"]
AUDIO_OUT_EMBEDS = "audio_out_embeds"


class LFM3AudioDecoderBuilder(LFM3VLDecoderBuilder):
    def __init__(
        self,
        config: LFM3AudioConfig,
        image_token_id: int | None = None,
        video_token_id: int | None = None,
    ):
        """Builds from config.language and config.tokens; with image_token_id and video_token_id
        (lfm3_omni) the graph also takes image_features and video_features."""
        if (image_token_id is None) != (video_token_id is None):
            raise ValueError("image_token_id and video_token_id go together")
        super().__init__(config.language, image_token_id, video_token_id)
        self.audio_config = config
        self.vision = image_token_id is not None

    def load_weights(self, model_path: str | pathlib.Path):
        self.weights = load_weights(model_path, "language", self.audio_config)

    def build_inputs(self):
        if self.vision:
            super().build_inputs()
        else:
            LFM3Builder.build_inputs(self)
        H, K = self.config.hidden_size, self.audio_config.num_codebooks
        extra = [
            helper.make_tensor_value_info(
                "audio_features", TensorProto.FLOAT, ["num_audio_tokens", H]
            ),
            helper.make_tensor_value_info(
                "audio_out_codes", TensorProto.INT64, ["num_audio_out_frames", K]
            ),
        ]
        before = [i.name for i in self.inputs].index("logits_to_keep")
        self.inputs[before:before] = extra

    def build_outputs(self):
        super().build_outputs()
        hidden = helper.make_tensor_value_info(
            "hidden_states",
            TensorProto.FLOAT,
            ["batch_size", "num_logits", self.config.hidden_size],
        )
        self.outputs.insert(1, hidden)

    def build_embedding(self) -> str:
        if self.vision:
            embeds = super().build_embedding()
        else:
            embeds = LFM3Builder.build_embedding(self)
        tokens = self.audio_config.tokens
        embeds = self.build_feature_scatter(embeds, tokens.audio, "audio_features")
        self.build_audio_out_embeds()
        return self.build_feature_scatter(embeds, tokens.audio_out, AUDIO_OUT_EMBEDS)

    def build_audio_out_embeds(self):
        """AUDIO_OUT_EMBEDS [Nf, H]: each frame's sum over k of codebook_embedding[code_k +
        codebook_size * k], as Lfm3AudioModel.get_audio_out_embeds."""
        prefix = "/model/audio_out_embed"
        K, size = self.audio_config.num_codebooks, self.audio_config.codebook_size
        self.add_initializer("codebook_embedding.weight", self.weights["codebook_embedding.weight"])
        rows = self.make_node(
            "Add",
            ["audio_out_codes", self.get_constant(np.arange(K) * size)],
            [f"{prefix}/Add/output_0"],
        )
        # [Nf, K] → [Nf, K, H] → [Nf, H]
        embeds = self.make_node(
            "Gather", ["codebook_embedding.weight", rows], [f"{prefix}/Gather/output_0"], axis=0
        )
        self.make_node(
            "ReduceSum",
            [embeds, self.get_constant([1])],
            [AUDIO_OUT_EMBEDS],
            name=f"{prefix}/ReduceSum",
            keepdims=0,
        )

    def build_lm_head(self, hidden_state: str) -> str:
        logits = super().build_lm_head(hidden_state)
        # The final norm's output at the kept positions: the depth decoder's input
        normed = f"/model/layers.{self.config.num_hidden_layers}/final_norm_layernorm/output_0"
        self.make_node(
            "Identity", [normed], ["hidden_states"], name="/model/hidden_states/Identity"
        )
        return logits
