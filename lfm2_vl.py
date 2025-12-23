"""
LFM2-VL Builder for ONNX export.

This builder exports LFM2-VL vision-language models as two ONNX models:
- embed_images.onnx: SigLIP2 vision encoder + MLP projector (fused)
- decoder.onnx: LFM2 language model backbone

Usage:
    # Export single model
    uv run lfm2_vl.py --model LiquidAI/LFM2-VL-1.6B --output LFM2-VL-1.6B-ONNX-builder

    # Available models:
    # - LiquidAI/LFM2-VL-450M  (350M backbone + 86M SigLIP2)
    # - LiquidAI/LFM2-VL-1.6B  (1.2B backbone + 400M SigLIP2)
    # - LiquidAI/LFM2-VL-3B    (2.6B backbone + 400M SigLIP2)
"""

import logging
from typing import List, Dict, Optional
from dataclasses import dataclass

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from lfm2 import LFM2Config, LFM2Builder

logger = logging.getLogger(__name__)


@dataclass
class SigLIP2Config:
    """Configuration for SigLIP2 vision encoder."""
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    patch_size: int
    num_channels: int = 3
    layer_norm_eps: float = 1e-6
    hidden_act: str = "gelu_pytorch_tanh"

    @classmethod
    def from_hf_config(cls, vision_config) -> "SigLIP2Config":
        return cls(
            hidden_size=vision_config.hidden_size,
            intermediate_size=vision_config.intermediate_size,
            num_hidden_layers=vision_config.num_hidden_layers,
            num_attention_heads=vision_config.num_attention_heads,
            patch_size=vision_config.patch_size,
            num_channels=getattr(vision_config, 'num_channels', 3),
            layer_norm_eps=getattr(vision_config, 'layer_norm_eps', 1e-6),
            hidden_act=getattr(vision_config, 'hidden_act', 'gelu_pytorch_tanh'),
        )


@dataclass
class LFM2VLConfig:
    """Configuration for LFM2-VL model."""
    text_config: LFM2Config
    vision_config: SigLIP2Config
    projector_hidden_size: int
    projector_hidden_act: str = "gelu"
    projector_bias: bool = True
    downsample_factor: int = 2
    image_token_id: int = 396
    tile_size: int = 512
    max_tiles: int = 10

    @classmethod
    def from_hf_config(cls, config) -> "LFM2VLConfig":
        return cls(
            text_config=LFM2Config.from_hf_config(config.text_config),
            vision_config=SigLIP2Config.from_hf_config(config.vision_config),
            projector_hidden_size=config.projector_hidden_size,
            projector_hidden_act=getattr(config, 'projector_hidden_act', 'gelu'),
            projector_bias=getattr(config, 'projector_bias', True),
            downsample_factor=getattr(config, 'downsample_factor', 2),
            image_token_id=getattr(config, 'image_token_id', 396),
            tile_size=getattr(config, 'tile_size', 512),
            max_tiles=getattr(config, 'max_tiles', 10),
        )


class VisionEmbedBuilder:
    """
    Fused vision encoder + projector builder for ONNX export.

    Creates an ONNX graph that combines:
    - SigLIP2 vision encoder (patch embedding + transformer layers)
    - MLP projector with pixel unshuffle

    Output: image embeddings in text embedding space
    """

    def __init__(self, config: LFM2VLConfig):
        self.config = config
        self.vision_config = config.vision_config
        self.head_dim = config.vision_config.hidden_size // config.vision_config.num_attention_heads

        # Projector dimensions
        self.vision_hidden = config.vision_config.hidden_size
        self.text_hidden = config.text_config.hidden_size
        self.proj_hidden = config.projector_hidden_size
        self.downsample = config.downsample_factor

        # Graph components
        self.nodes: List[onnx.NodeProto] = []
        self.inputs: List[onnx.ValueInfoProto] = []
        self.outputs: List[onnx.ValueInfoProto] = []
        self.initializers: List[onnx.TensorProto] = []

        # Weights storage
        self.weights: Dict[str, np.ndarray] = {}

        # Node counter
        self._node_count = 0

    def _unique_name(self, prefix: str) -> str:
        self._node_count += 1
        return f"{prefix}_{self._node_count}"

    def add_initializer(self, name: str, tensor: np.ndarray, dtype=None):
        """Add weight tensor as graph initializer."""
        if dtype is None:
            if tensor.dtype not in [np.int32, np.int64]:
                tensor = tensor.astype(np.float32)
        else:
            tensor = tensor.astype(dtype)
        self.initializers.append(numpy_helper.from_array(tensor, name))

    def make_node(self, op_type: str, inputs: List[str], outputs: List[str],
                  name: str = None, domain: str = "", **attrs) -> str:
        """Create an ONNX node and return the first output name."""
        if name is None:
            name = self._unique_name(op_type)

        node = helper.make_node(op_type, inputs, outputs, name=name, domain=domain, **attrs)
        self.nodes.append(node)
        return outputs[0] if outputs else None

    def make_layernorm(self, input_name: str, weight_name: str, bias_name: str,
                       output_name: str) -> str:
        """Create LayerNormalization node."""
        return self.make_node(
            "LayerNormalization",
            inputs=[input_name, weight_name, bias_name],
            outputs=[output_name],
            epsilon=self.vision_config.layer_norm_eps,
        )

    def make_gelu(self, input_name: str, output_name: str) -> str:
        """Create GELU activation (approximate tanh version)."""
        # GELU with tanh approximation
        return self.make_node("Gelu", [input_name], [output_name],
                              domain="com.microsoft", approximate="tanh")

    def build_inputs(self):
        """Create model inputs."""
        # pixel_values: [batch, num_patches, channels, patch_h, patch_w]
        # For flexibility, we use dynamic shapes
        self.inputs.append(helper.make_tensor_value_info(
            "pixel_values", TensorProto.FLOAT,
            ["batch_size", "num_patches", self.vision_config.num_channels,
             self.vision_config.patch_size, self.vision_config.patch_size]
        ))

        # patch_attention_mask: [batch, num_patches]
        self.inputs.append(helper.make_tensor_value_info(
            "patch_attention_mask", TensorProto.INT64,
            ["batch_size", "num_patches"]
        ))

    def build_outputs(self):
        """Create model outputs."""
        # Image embeddings in text space: [batch, num_image_tokens, text_hidden_size]
        self.outputs.append(helper.make_tensor_value_info(
            "image_embeddings", TensorProto.FLOAT,
            ["batch_size", "num_image_tokens", self.text_hidden]
        ))

    def build_patch_embedding(self) -> str:
        """Build patch embedding layer."""
        prefix = "vision_model.embeddings.patch_embedding"
        H = self.vision_config.hidden_size
        P = self.vision_config.patch_size
        C = self.vision_config.num_channels

        # Conv2d weights: [hidden_size, channels, patch_size, patch_size]
        self.add_initializer(f"{prefix}.weight", self.weights[f"{prefix}.weight"])
        self.add_initializer(f"{prefix}.bias", self.weights[f"{prefix}.bias"])

        # Reshape pixel_values: [B, N, C, P, P] -> [B*N, C, P, P]
        self.add_initializer("patch_embed/reshape_1", np.array([-1, C, P, P], dtype=np.int64))
        reshaped = self.make_node("Reshape", ["pixel_values", "patch_embed/reshape_1"],
                                  ["patch_embed/reshaped"])

        # Conv2d: [B*N, C, P, P] -> [B*N, H, 1, 1]
        conv_out = self.make_node("Conv", [reshaped, f"{prefix}.weight", f"{prefix}.bias"],
                                  ["patch_embed/conv"], kernel_shape=[P, P], strides=[P, P])

        # Flatten and reshape: [B*N, H, 1, 1] -> [B, N, H]
        squeezed = self.make_node("Squeeze", [conv_out], ["patch_embed/squeezed"],
                                  axes=[2, 3])

        # Get batch size from pixel_values shape
        self.make_node("Shape", ["pixel_values"], ["patch_embed/input_shape"])
        self.add_initializer("patch_embed/const_0", np.array(0, dtype=np.int64))
        self.add_initializer("patch_embed/const_1", np.array(1, dtype=np.int64))
        self.make_node("Gather", ["patch_embed/input_shape", "patch_embed/const_0"],
                       ["patch_embed/batch"], axis=0)
        self.make_node("Gather", ["patch_embed/input_shape", "patch_embed/const_1"],
                       ["patch_embed/num_patches"], axis=0)

        # Reshape to [B, N, H]
        self.add_initializer("patch_embed/h_dim", np.array(H, dtype=np.int64))
        self.make_node("Concat", ["patch_embed/batch", "patch_embed/num_patches", "patch_embed/h_dim"],
                       ["patch_embed/target_shape"], axis=0)
        return self.make_node("Reshape", [squeezed, "patch_embed/target_shape"],
                              ["patch_embeddings"])

    def build_encoder_layer(self, layer_idx: int, hidden_state: str) -> str:
        """Build a single transformer encoder layer."""
        prefix = f"vision_model.encoder.layers.{layer_idx}"
        H = self.vision_config.hidden_size
        nh = self.vision_config.num_attention_heads
        hd = self.head_dim
        I = self.vision_config.intermediate_size

        # Load weights
        # Layer norm 1
        self.add_initializer(f"{prefix}.layer_norm1.weight",
                             self.weights[f"{prefix}.layer_norm1.weight"])
        self.add_initializer(f"{prefix}.layer_norm1.bias",
                             self.weights[f"{prefix}.layer_norm1.bias"])

        # Self attention
        self.add_initializer(f"{prefix}.self_attn.q_proj.weight",
                             self.weights[f"{prefix}.self_attn.q_proj.weight"].T)
        self.add_initializer(f"{prefix}.self_attn.q_proj.bias",
                             self.weights[f"{prefix}.self_attn.q_proj.bias"])
        self.add_initializer(f"{prefix}.self_attn.k_proj.weight",
                             self.weights[f"{prefix}.self_attn.k_proj.weight"].T)
        self.add_initializer(f"{prefix}.self_attn.k_proj.bias",
                             self.weights[f"{prefix}.self_attn.k_proj.bias"])
        self.add_initializer(f"{prefix}.self_attn.v_proj.weight",
                             self.weights[f"{prefix}.self_attn.v_proj.weight"].T)
        self.add_initializer(f"{prefix}.self_attn.v_proj.bias",
                             self.weights[f"{prefix}.self_attn.v_proj.bias"])
        self.add_initializer(f"{prefix}.self_attn.out_proj.weight",
                             self.weights[f"{prefix}.self_attn.out_proj.weight"].T)
        self.add_initializer(f"{prefix}.self_attn.out_proj.bias",
                             self.weights[f"{prefix}.self_attn.out_proj.bias"])

        # Layer norm 2
        self.add_initializer(f"{prefix}.layer_norm2.weight",
                             self.weights[f"{prefix}.layer_norm2.weight"])
        self.add_initializer(f"{prefix}.layer_norm2.bias",
                             self.weights[f"{prefix}.layer_norm2.bias"])

        # MLP
        self.add_initializer(f"{prefix}.mlp.fc1.weight",
                             self.weights[f"{prefix}.mlp.fc1.weight"].T)
        self.add_initializer(f"{prefix}.mlp.fc1.bias",
                             self.weights[f"{prefix}.mlp.fc1.bias"])
        self.add_initializer(f"{prefix}.mlp.fc2.weight",
                             self.weights[f"{prefix}.mlp.fc2.weight"].T)
        self.add_initializer(f"{prefix}.mlp.fc2.bias",
                             self.weights[f"{prefix}.mlp.fc2.bias"])

        residual = hidden_state

        # Layer norm 1
        normed = self.make_layernorm(hidden_state, f"{prefix}.layer_norm1.weight",
                                     f"{prefix}.layer_norm1.bias", f"{prefix}/ln1")

        # Self attention
        # Q, K, V projections
        q = self.make_node("MatMul", [normed, f"{prefix}.self_attn.q_proj.weight"],
                           [f"{prefix}/q_matmul"])
        q = self.make_node("Add", [q, f"{prefix}.self_attn.q_proj.bias"], [f"{prefix}/q"])

        k = self.make_node("MatMul", [normed, f"{prefix}.self_attn.k_proj.weight"],
                           [f"{prefix}/k_matmul"])
        k = self.make_node("Add", [k, f"{prefix}.self_attn.k_proj.bias"], [f"{prefix}/k"])

        v = self.make_node("MatMul", [normed, f"{prefix}.self_attn.v_proj.weight"],
                           [f"{prefix}/v_matmul"])
        v = self.make_node("Add", [v, f"{prefix}.self_attn.v_proj.bias"], [f"{prefix}/v"])

        # Reshape to [B, N, nh, hd] then transpose to [B, nh, N, hd]
        self.add_initializer(f"{prefix}/reshape_qkv", np.array([0, -1, nh, hd], dtype=np.int64))
        q_4d = self.make_node("Reshape", [q, f"{prefix}/reshape_qkv"], [f"{prefix}/q_4d"])
        k_4d = self.make_node("Reshape", [k, f"{prefix}/reshape_qkv"], [f"{prefix}/k_4d"])
        v_4d = self.make_node("Reshape", [v, f"{prefix}/reshape_qkv"], [f"{prefix}/v_4d"])

        q_t = self.make_node("Transpose", [q_4d], [f"{prefix}/q_t"], perm=[0, 2, 1, 3])
        k_t = self.make_node("Transpose", [k_4d], [f"{prefix}/k_t"], perm=[0, 2, 1, 3])
        v_t = self.make_node("Transpose", [v_4d], [f"{prefix}/v_t"], perm=[0, 2, 1, 3])

        # Scaled dot-product attention
        scale = 1.0 / (hd ** 0.5)
        self.add_initializer(f"{prefix}/scale", np.array(scale, dtype=np.float32))

        # Q @ K^T
        k_t_transposed = self.make_node("Transpose", [k_t], [f"{prefix}/k_t_t"], perm=[0, 1, 3, 2])
        scores = self.make_node("MatMul", [q_t, k_t_transposed], [f"{prefix}/scores"])
        scores_scaled = self.make_node("Mul", [scores, f"{prefix}/scale"], [f"{prefix}/scores_scaled"])

        # Softmax
        attn_weights = self.make_node("Softmax", [scores_scaled], [f"{prefix}/attn_weights"], axis=-1)

        # Attention output
        attn_out = self.make_node("MatMul", [attn_weights, v_t], [f"{prefix}/attn_out"])

        # Transpose back and reshape
        attn_out_t = self.make_node("Transpose", [attn_out], [f"{prefix}/attn_out_t"], perm=[0, 2, 1, 3])
        self.add_initializer(f"{prefix}/reshape_out", np.array([0, -1, H], dtype=np.int64))
        attn_out_reshaped = self.make_node("Reshape", [attn_out_t, f"{prefix}/reshape_out"],
                                           [f"{prefix}/attn_out_reshaped"])

        # Output projection
        out_proj = self.make_node("MatMul", [attn_out_reshaped, f"{prefix}.self_attn.out_proj.weight"],
                                  [f"{prefix}/out_proj_matmul"])
        out_proj = self.make_node("Add", [out_proj, f"{prefix}.self_attn.out_proj.bias"],
                                  [f"{prefix}/out_proj"])

        # Residual 1
        hidden_state = self.make_node("Add", [residual, out_proj], [f"{prefix}/residual1"])

        # Layer norm 2
        residual2 = hidden_state
        normed2 = self.make_layernorm(hidden_state, f"{prefix}.layer_norm2.weight",
                                      f"{prefix}.layer_norm2.bias", f"{prefix}/ln2")

        # MLP
        fc1 = self.make_node("MatMul", [normed2, f"{prefix}.mlp.fc1.weight"], [f"{prefix}/fc1_matmul"])
        fc1 = self.make_node("Add", [fc1, f"{prefix}.mlp.fc1.bias"], [f"{prefix}/fc1"])
        fc1_act = self.make_gelu(fc1, f"{prefix}/fc1_act")

        fc2 = self.make_node("MatMul", [fc1_act, f"{prefix}.mlp.fc2.weight"], [f"{prefix}/fc2_matmul"])
        fc2 = self.make_node("Add", [fc2, f"{prefix}.mlp.fc2.bias"], [f"{prefix}/fc2"])

        # Residual 2
        return self.make_node("Add", [residual2, fc2], [f"{prefix}/residual2"])

    def build_post_layernorm(self, hidden_state: str) -> str:
        """Build post layer norm."""
        self.add_initializer("vision_model.post_layernorm.weight",
                             self.weights["vision_model.post_layernorm.weight"])
        self.add_initializer("vision_model.post_layernorm.bias",
                             self.weights["vision_model.post_layernorm.bias"])
        return self.make_layernorm(hidden_state,
                                   "vision_model.post_layernorm.weight",
                                   "vision_model.post_layernorm.bias",
                                   "vision_embeddings")

    def build_projector(self, vision_embeddings: str) -> str:
        """Build the MLP projector with pixel unshuffle."""
        ds = self.downsample
        input_dim = self.vision_hidden * ds * ds  # After pixel unshuffle

        # Load weights
        self.add_initializer("multi_modal_projector.linear_1.weight",
                             self.weights["multi_modal_projector.linear_1.weight"].T)
        if self.config.projector_bias:
            self.add_initializer("multi_modal_projector.linear_1.bias",
                                 self.weights["multi_modal_projector.linear_1.bias"])

        self.add_initializer("multi_modal_projector.linear_2.weight",
                             self.weights["multi_modal_projector.linear_2.weight"].T)
        if self.config.projector_bias:
            self.add_initializer("multi_modal_projector.linear_2.bias",
                                 self.weights["multi_modal_projector.linear_2.bias"])

        # Pixel unshuffle: [B, N, H] -> [B, N/(ds*ds), H*ds*ds]
        # Reshape to combine adjacent patches
        self.add_initializer("proj/reshape_target", np.array([0, -1, input_dim], dtype=np.int64))
        unshuffled = self.make_node("Reshape", [vision_embeddings, "proj/reshape_target"],
                                    ["proj/unshuffled"])

        # Linear 1
        fc1 = self.make_node("MatMul", [unshuffled, "multi_modal_projector.linear_1.weight"],
                             ["proj/fc1_matmul"])
        if self.config.projector_bias:
            fc1 = self.make_node("Add", [fc1, "multi_modal_projector.linear_1.bias"], ["proj/fc1"])

        # GELU
        fc1_act = self.make_gelu(fc1, "proj/fc1_act")

        # Linear 2
        fc2 = self.make_node("MatMul", [fc1_act, "multi_modal_projector.linear_2.weight"],
                             ["proj/fc2_matmul"])
        if self.config.projector_bias:
            fc2 = self.make_node("Add", [fc2, "multi_modal_projector.linear_2.bias"],
                                 ["image_embeddings"])
        else:
            self.make_node("Identity", [fc2], ["image_embeddings"])

        return "image_embeddings"

    def load_weights(self, weights: Dict[str, np.ndarray]):
        """Load weights from dict."""
        # Filter vision model and projector weights
        # Handle different prefixes: model.vision_tower.vision_model.* -> vision_model.*
        for name, weight in weights.items():
            if name.startswith("model.vision_tower.vision_model."):
                new_name = name.replace("model.vision_tower.", "")
                self.weights[new_name] = weight
            elif name.startswith("vision_model."):
                self.weights[name] = weight
            elif name.startswith("model.multi_modal_projector."):
                new_name = name.replace("model.", "")
                self.weights[new_name] = weight
            elif name.startswith("multi_modal_projector."):
                self.weights[name] = weight

        logger.info(f"Loaded {len(self.weights)} vision + projector weights")

    def build(self) -> onnx.ModelProto:
        """Build the fused vision encoder + projector ONNX model."""
        logger.info("Building fused vision encoder + projector...")

        # Build graph structure
        self.build_inputs()
        self.build_outputs()

        # Patch embedding
        hidden_state = self.build_patch_embedding()

        # Encoder layers
        for layer_idx in range(self.vision_config.num_hidden_layers):
            logger.info(f"Building vision layer {layer_idx}...")
            hidden_state = self.build_encoder_layer(layer_idx, hidden_state)

        # Post layer norm
        vision_embeddings = self.build_post_layernorm(hidden_state)

        # Projector (fused)
        logger.info("Building projector...")
        self.build_projector(vision_embeddings)

        # Create graph
        graph = helper.make_graph(
            self.nodes,
            "embed_images",
            self.inputs,
            self.outputs,
            self.initializers,
        )

        # Create model
        model = helper.make_model(
            graph,
            opset_imports=[
                helper.make_opsetid("", 21),
                helper.make_opsetid("com.microsoft", 1),
            ],
            ir_version=9,
        )
        model.producer_name = "lfm2-vl-builder"

        logger.info(f"Vision + projector model built: {len(self.nodes)} nodes")
        return model


def export_vl_model(model_path: str, output_dir: str):
    """Export LFM2-VL model to ONNX (embed_images + decoder)."""
    import os
    import json
    from transformers import AutoConfig, AutoTokenizer, AutoProcessor
    import torch

    # Load config
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    vl_config = LFM2VLConfig.from_hf_config(config)

    # Load model weights
    logger.info(f"Loading weights from {model_path}...")
    from transformers import AutoModelForImageTextToText
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, torch_dtype=torch.float32, trust_remote_code=True
    )

    weights = {}
    for name, param in model.named_parameters():
        weights[name] = param.detach().numpy()
        logger.debug(f"Loaded: {name} {param.shape}")

    logger.info(f"Loaded {len(weights)} total weights")

    del model
    import gc
    gc.collect()

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    onnx_dir = os.path.join(output_dir, "onnx")
    os.makedirs(onnx_dir, exist_ok=True)

    # Export fused vision encoder + projector
    logger.info("Exporting embed_images (vision encoder + projector)...")
    vision_builder = VisionEmbedBuilder(vl_config)
    vision_builder.load_weights(weights)
    vision_model = vision_builder.build()

    vision_path = os.path.join(onnx_dir, "embed_images.onnx")
    vision_data_path = os.path.join(onnx_dir, "embed_images.onnx_data")
    if os.path.exists(vision_data_path):
        os.remove(vision_data_path)
    onnx.save_model(vision_model, vision_path, save_as_external_data=True,
                    all_tensors_to_one_file=True, location="embed_images.onnx_data")
    logger.info(f"embed_images saved to {vision_path}")

    # Export decoder (reuse LFM2Builder)
    logger.info("Exporting decoder...")
    text_builder = LFM2Builder(vl_config.text_config)
    # Filter text model weights (they have "model.language_model." prefix in VL model)
    for name, weight in weights.items():
        if name.startswith("model.language_model."):
            # Remove prefix to match LFM2Builder expectations
            new_name = name.replace("model.language_model.", "model.")
            text_builder.weights[new_name] = weight
        elif name.startswith("language_model."):
            new_name = name.replace("language_model.", "model.")
            text_builder.weights[new_name] = weight

    # Build text model graph
    text_builder.build_inputs()
    text_builder.build_outputs()
    text_builder.build_rope_cache()
    text_builder.build_attention_mask_subgraph()

    # For VL, we need to add image embeddings input
    # The text model will receive merged embeddings (text + image)
    # For simplicity, we export the standard text model
    hidden_state = text_builder.build_embedding()

    for layer_idx in range(vl_config.text_config.num_hidden_layers):
        layer_type = vl_config.text_config.layer_types[layer_idx]
        logger.info(f"Building text layer {layer_idx} ({layer_type})...")

        if layer_type == "conv":
            hidden_state = text_builder.build_conv_layer(layer_idx, hidden_state)
        else:
            hidden_state = text_builder.build_attention_layer(layer_idx, hidden_state)

    text_builder.build_lm_head(hidden_state)

    text_graph = helper.make_graph(
        text_builder.nodes,
        "decoder",
        text_builder.inputs,
        text_builder.outputs,
        text_builder.initializers,
    )

    text_model = helper.make_model(
        text_graph,
        opset_imports=[
            helper.make_opsetid("", 21),
            helper.make_opsetid("com.microsoft", 1),
        ],
        ir_version=9,
    )
    text_model.producer_name = "lfm2-vl-builder"

    decoder_path = os.path.join(onnx_dir, "decoder.onnx")
    decoder_data_path = os.path.join(onnx_dir, "decoder.onnx_data")
    if os.path.exists(decoder_data_path):
        os.remove(decoder_data_path)
    onnx.save_model(text_model, decoder_path, save_as_external_data=True,
                    all_tensors_to_one_file=True, location="decoder.onnx_data")
    logger.info(f"decoder saved to {decoder_path}")

    # Copy tokenizer and config
    try:
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        processor.save_pretrained(output_dir)
    except Exception as e:
        logger.warning(f"Could not save processor: {e}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer.save_pretrained(output_dir)

    config.save_pretrained(output_dir)

    # Create generation_config.json
    gen_config = {
        "_from_model_config": True,
        "bos_token_id": config.text_config.bos_token_id if hasattr(config.text_config, 'bos_token_id') else 1,
        "eos_token_id": config.text_config.eos_token_id if hasattr(config.text_config, 'eos_token_id') else 7,
        "pad_token_id": 0,
        "transformers_version": "4.57.0"
    }
    gen_config_path = os.path.join(output_dir, "generation_config.json")
    with open(gen_config_path, "w") as f:
        json.dump(gen_config, f, indent=2)

    # Print summary
    total_size = 0
    for f in os.listdir(onnx_dir):
        fpath = os.path.join(onnx_dir, f)
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath)
            total_size += size
            logger.info(f"  {f}: {size / 1e6:.1f} MB")

    logger.info(f"Total ONNX size: {total_size / 1e9:.2f} GB")
    logger.info(f"Output directory: {output_dir}")

    return output_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export LFM2-VL to ONNX")
    parser.add_argument("--model", type=str, required=True,
                        help="Model path (e.g., LiquidAI/LFM2-VL-1.6B)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    export_vl_model(args.model, args.output)
