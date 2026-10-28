"""
LFM3-VL language model: the LFM3 decoder with image and video features scattered into its token
embeddings, as Lfm3VlForConditionalGeneration.forward does.

    input_ids → Gather (embed_tokens) ─┐
    image_features [num_image_tokens, H] → ScatterND at input_ids == image_token_id
    video_features [num_video_tokens, H] → ScatterND at input_ids == video_token_id
        → LFM3 layers → Slice(-logits_to_keep:) → final norm → tied lm_head → logits  (LFM3Builder)

Row i of a feature input lands on the i-th placeholder of its token; the caller checks that the
counts match (the reference raises on a mismatch, the graph fails only on too many rows).
Decode steps pass zero-row features.
"""

import pathlib

from onnx import TensorProto, helper

from liquidonnx.lfm3.builder import LFM3Builder, LFM3Config
from liquidonnx.lfm3_vl.weights import language_weights, load_tensors


class LFM3VLDecoderBuilder(LFM3Builder):
    def __init__(self, config: LFM3Config, image_token_id: int, video_token_id: int):
        super().__init__(config)
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id

    def load_weights(self, model_path: str | pathlib.Path):
        self.weights = language_weights(load_tensors(model_path, "language_model."), self.config)

    def build_inputs(self):
        super().build_inputs()
        H = self.config.hidden_size
        extra = [
            helper.make_tensor_value_info(
                "image_features", TensorProto.FLOAT, ["num_image_tokens", H]
            ),
            helper.make_tensor_value_info(
                "video_features", TensorProto.FLOAT, ["num_video_tokens", H]
            ),
        ]
        after = [i.name for i in self.inputs].index("position_ids") + 1
        self.inputs[after:after] = extra

    def build_embedding(self) -> str:
        embeds = super().build_embedding()
        embeds = self.build_feature_scatter(embeds, self.image_token_id, "image_features")
        return self.build_feature_scatter(embeds, self.video_token_id, "video_features")

    def build_feature_scatter(self, embeds: str, token_id: int, feature: str) -> str:
        """embeds [B, S, H] with the rows of feature written over the token_id placeholders."""
        prefix = f"/model/{feature}_scatter"
        H = self.config.hidden_size
        shape = self.make_node("Shape", [embeds], [f"{prefix}/Shape/output_0"])
        flat = self.make_node(
            "Reshape", [embeds, self.get_constant([-1, H])], [f"{prefix}/Reshape/output_0"]
        )
        ids = self.make_node(
            "Reshape", ["input_ids", self.get_constant([-1])], [f"{prefix}/Reshape_ids/output_0"]
        )
        hit = self.make_node(
            "Equal", [ids, self.get_constant(token_id)], [f"{prefix}/Equal/output_0"]
        )
        positions = self.make_node("NonZero", [hit], [f"{prefix}/NonZero/output_0"])
        positions = self.make_node(
            "Transpose", [positions], [f"{prefix}/Transpose/output_0"], perm=[1, 0]
        )
        rows = self.make_node("Shape", [feature], [f"{prefix}/Shape_rows/output_0"], end=1)
        positions = self.make_node(
            "Slice",
            [positions, self.get_constant([0]), rows, self.get_constant([0])],
            [f"{prefix}/Slice/output_0"],
        )
        scattered = self.make_node(
            "ScatterND", [flat, positions, feature], [f"{prefix}/ScatterND/output_0"]
        )
        return self.make_node("Reshape", [scattered, shape], [f"{prefix}/Reshape_back/output_0"])
