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
        """Load embedding weights from model weights dict."""
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
        """Build the embed_tokens ONNX model."""
        logger.info("Building embed_tokens...")

        # Input: input_ids [batch_size, sequence_length]
        self.inputs.append(
            helper.make_tensor_value_info(
                "input_ids", TensorProto.INT64, ["batch_size", "sequence_length"]
            )
        )

        # Output: inputs_embeds [batch_size, sequence_length, hidden_size]
        self.outputs.append(
            helper.make_tensor_value_info(
                "inputs_embeds",
                TensorProto.FLOAT,
                ["batch_size", "sequence_length", self.hidden_size],
            )
        )

        # Add embedding weight and create Gather node
        self.add_initializer("weight", self.embed_weight)
        self.make_gather("weight", "input_ids", "inputs_embeds", axis=0)

        model = self.build_graph("embed_tokens", ms_domain=False, producer_name="lfm2-vl-builder")
        logger.info(
            f"embed_tokens built: {len(self.nodes)} nodes, "
            f"vocab_size={self.vocab_size}, hidden_size={self.hidden_size}"
        )
        return model
