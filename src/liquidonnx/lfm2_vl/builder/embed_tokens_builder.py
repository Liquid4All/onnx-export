"""Token embedding builder for LFM2-VL ONNX export.

This module contains the EmbedTokensBuilder class which creates a simple
ONNX graph that maps input_ids to embeddings via Gather.
"""

import logging

import numpy as np
import onnx
from onnx import TensorProto, helper

from liquidonnx.builder_base import ONNXBuilderBase
from liquidonnx.lfm2_vl.builder.config import LFM2VLConfig

logger = logging.getLogger(__name__)


class EmbedTokensBuilder(ONNXBuilderBase):
    """
    Simple token embedding builder for ONNX export.

    Creates an ONNX graph that maps input_ids to embeddings via Gather.
    This allows the decoder to take inputs_embeds, enabling clean
    text/image embedding fusion.

    Graph structure:
        input_ids [B, S]
            ↓
        Gather (weight, axis=0)
            ↓
        inputs_embeds [B, S, hidden_size]
    """

    def __init__(self, config: LFM2VLConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.text_config.hidden_size
        self.vocab_size = config.text_config.vocab_size
        self.embed_weight: np.ndarray | None = None

    def load_weights(self, weights: dict[str, np.ndarray]):
        prefixes = [
            "model.language_model.embed_tokens.weight",
            "language_model.embed_tokens.weight",
            "model.embed_tokens.weight",
        ]
        for prefix in prefixes:
            if prefix in weights:
                self.embed_weight = weights[prefix].astype(np.float32)
                logger.info(f"Loaded embed_tokens weight: {self.embed_weight.shape}")
                return

        raise ValueError("Could not find embed_tokens weight in model")

    def build(self) -> onnx.ModelProto:
        logger.info("Building embed_tokens...")

        self.inputs.append(
            helper.make_tensor_value_info(
                "input_ids", TensorProto.INT64, ["batch_size", "sequence_length"]
            )
        )

        self.outputs.append(
            helper.make_tensor_value_info(
                "inputs_embeds",
                TensorProto.FLOAT,
                ["batch_size", "sequence_length", self.hidden_size],
            )
        )

        # Community naming: model.embed_tokens.weight
        self.add_initializer("model.embed_tokens.weight", self.embed_weight)
        # Community node name: /model/embed_tokens/Gather
        self.make_node(
            "Gather",
            ["model.embed_tokens.weight", "input_ids"],
            ["inputs_embeds"],
            name="/model/embed_tokens/Gather",
            axis=0,
        )

        # Add ValueInfo for weight (community convention)
        self.add_value_info(
            "model.embed_tokens.weight", TensorProto.FLOAT, [self.vocab_size, self.hidden_size]
        )

        model = self.build_graph("embed_tokens", ms_domain=False)
        logger.info(
            f"embed_tokens built: {len(self.nodes)} nodes, "
            f"vocab_size={self.vocab_size}, hidden_size={self.hidden_size}"
        )
        return model
