#!/usr/bin/env python3
"""
Full ONNX export for LFM2.5-Audio model supporting all 3 modes:
- ASR (Automatic Speech Recognition): Audio -> Text
- TTS (Text-to-Speech): Text -> Audio
- Interleaved: Mixed text and audio I/O

Exports the following ONNX models:
1. audio_encoder.onnx - Conformer encoder (mel-spectrogram -> audio embeddings)
2. embed_tokens.onnx - Text token embeddings
3. audio_embedding.onnx - Audio code embeddings
4. decoder.onnx - LFM2 backbone (embeddings -> logits/hidden states)
5. depthformer.onnx - Audio codebook prediction (8 codebooks)
6. audio_detokenizer.onnx - Audio synthesis (codes -> waveform)

Usage:
    uv run lfm2-audio-export-full LiquidAI/LFM2.5-Audio-1.5B
    uv run lfm2-audio-export-full LiquidAI/LFM2.5-Audio-1.5B --precision q4
"""

import argparse
import gc
import json
import logging
import pathlib
import shutil

import numpy as np
import onnx
import torch
import torch.nn as nn
from onnx import TensorProto, helper

from liquidonnx.external_data import split_external_data
from liquidonnx.lfm2.builder import LFM2Builder, LFM2Config
from liquidonnx.quantize import get_model_size, quantize_model

logger = logging.getLogger(__name__)


def get_model_name(model_path: str) -> str:
    if "/" in model_path:
        return model_path.split("/")[-1]
    return pathlib.Path(model_path).name


def load_audio_model_weights(model_path: str) -> dict[str, np.ndarray]:
    """Load all weights from HuggingFace audio model."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    logger.info(f"Loading weights from {model_path}...")
    safetensors_path = hf_hub_download(model_path, "model.safetensors")

    weights = {}
    with safe_open(safetensors_path, framework="np", device="cpu") as f:
        for key in f.keys():
            weights[key] = f.get_tensor(key)

    logger.info(f"Loaded {len(weights)} weights")
    return weights


def load_audio_config(model_path: str) -> dict:
    """Load config.json from HuggingFace model."""
    from huggingface_hub import hf_hub_download

    config_path = hf_hub_download(model_path, "config.json")
    with open(config_path) as f:
        return json.load(f)


# === 1. Audio Encoder Export (torch.onnx) ===


class AudioEncoderWrapper(nn.Module):
    """Wrapper for Conformer encoder + adapter for ONNX export."""

    def __init__(self, conformer, adapter):
        super().__init__()
        self.conformer = conformer
        self.adapter = adapter

    def forward(self, mel_features: torch.Tensor, mel_lengths: torch.Tensor):
        """
        Args:
            mel_features: [batch, time, features] mel-spectrogram
            mel_lengths: [batch] length of each sequence

        Returns:
            audio_embeddings: [batch, time', hidden] encoded audio
            output_lengths: [batch] output lengths
        """
        # Conformer expects [batch, features, time]
        mel_features = mel_features.transpose(1, 2)

        # Encode with conformer
        encoded, encoded_lens = self.conformer(audio_signal=mel_features, length=mel_lengths)

        # Transpose back to [batch, time, features]
        encoded = encoded.transpose(1, 2)

        # Apply adapter
        audio_embeddings = self.adapter(encoded)

        return audio_embeddings, encoded_lens


def export_audio_encoder(
    model, config: dict, onnx_dir: pathlib.Path, device: str = "cuda"
) -> pathlib.Path:
    """Export Conformer audio encoder to ONNX using torch.onnx."""
    logger.info("Exporting audio_encoder.onnx...")

    wrapper = AudioEncoderWrapper(model.conformer, model.audio_adapter).to(device)
    wrapper.eval()

    # Create dummy inputs
    batch_size = 1
    time_steps = 100
    features = config.get("preprocessor", {}).get("features", 128)

    mel_features = torch.randn(batch_size, time_steps, features, device=device)
    mel_lengths = torch.tensor([time_steps], dtype=torch.int64, device=device)

    output_path = onnx_dir / "audio_encoder.onnx"

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (mel_features, mel_lengths),
            str(output_path),
            input_names=["mel_features", "mel_lengths"],
            output_names=["audio_embeddings", "output_lengths"],
            dynamic_axes={
                "mel_features": {0: "batch", 1: "time"},
                "mel_lengths": {0: "batch"},
                "audio_embeddings": {0: "batch", 1: "time"},
                "output_lengths": {0: "batch"},
            },
            opset_version=18,
            do_constant_folding=True,
            dynamo=False,
        )

    logger.info(f"audio_encoder saved to {output_path}")
    return output_path


# === 2. Embed Tokens Export (builder) ===


class EmbedTokensBuilder:
    """Simple token embedding builder for audio model."""

    def __init__(self, vocab_size: int, hidden_size: int):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embed_weight: np.ndarray | None = None

    def load_weights(self, weights: dict[str, np.ndarray]):
        if "lfm.embed_tokens.weight" in weights:
            self.embed_weight = weights["lfm.embed_tokens.weight"].astype(np.float32)
        else:
            raise ValueError("Could not find embed_tokens weight")

    def build(self) -> onnx.ModelProto:
        nodes = []
        inputs = [
            helper.make_tensor_value_info(
                "input_ids", TensorProto.INT64, ["batch_size", "sequence_length"]
            )
        ]
        outputs = [
            helper.make_tensor_value_info(
                "inputs_embeds",
                TensorProto.FLOAT,
                ["batch_size", "sequence_length", self.hidden_size],
            )
        ]

        initializers = [
            onnx.numpy_helper.from_array(self.embed_weight, "model.embed_tokens.weight")
        ]

        nodes.append(
            helper.make_node(
                "Gather",
                ["model.embed_tokens.weight", "input_ids"],
                ["inputs_embeds"],
                name="/model/embed_tokens/Gather",
                axis=0,
            )
        )

        graph = helper.make_graph(nodes, "embed_tokens", inputs, outputs, initializers)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)], ir_version=10)
        model.producer_name = "liquidonnx"
        return model


def export_embed_tokens(
    weights: dict[str, np.ndarray], config: dict, onnx_dir: pathlib.Path
) -> pathlib.Path:
    """Export embed_tokens.onnx."""
    logger.info("Exporting embed_tokens.onnx...")

    lfm_config = config.get("lfm", {})
    vocab_size = lfm_config.get("vocab_size", 65536)
    hidden_size = lfm_config.get("hidden_size", 2048)

    builder = EmbedTokensBuilder(vocab_size, hidden_size)
    builder.load_weights(weights)
    model = builder.build()

    output_path = onnx_dir / "embed_tokens.onnx"
    onnx.save_model(model, str(output_path))
    logger.info(f"embed_tokens saved to {output_path}")
    return output_path


# === 3. Audio Embedding Export (builder) ===


def export_audio_embedding(
    weights: dict[str, np.ndarray], config: dict, onnx_dir: pathlib.Path
) -> pathlib.Path:
    """Export audio_embedding.onnx (audio token embedding lookup).

    Note: The embedding lookup does NOT apply normalization.
    The embedding_norm (RMSNorm) is only used in get_logits() for the inverse
    projection (embedding -> logits), not for the forward embedding lookup.

    Reference: liquid_audio/model/transformer.py SharedEmbedding.embed()
    """
    logger.info("Exporting audio_embedding.onnx...")

    nodes = []
    hidden_size = config.get("lfm", {}).get("hidden_size", 2048)

    embed_weight = weights["audio_embedding.embedding.weight"].astype(np.float32)

    inputs = [
        helper.make_tensor_value_info(
            "audio_codes", TensorProto.INT64, ["batch_size", "audio_length"]
        )
    ]
    outputs = [
        helper.make_tensor_value_info(
            "audio_embeds",
            TensorProto.FLOAT,
            ["batch_size", "audio_length", hidden_size],
        )
    ]

    initializers = [
        onnx.numpy_helper.from_array(embed_weight, "audio_embedding.weight"),
    ]

    # Just do embedding lookup - no normalization
    nodes.append(
        helper.make_node(
            "Gather",
            ["audio_embedding.weight", "audio_codes"],
            ["audio_embeds"],
            axis=0,
        )
    )

    graph = helper.make_graph(nodes, "audio_embedding", inputs, outputs, initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)], ir_version=10)
    model.producer_name = "liquidonnx"

    output_path = onnx_dir / "audio_embedding.onnx"
    onnx.save_model(model, str(output_path))
    logger.info(f"audio_embedding saved to {output_path}")
    return output_path


# === 4. Decoder Export (builder) ===


def export_decoder(
    weights: dict[str, np.ndarray], config: dict, onnx_dir: pathlib.Path
) -> pathlib.Path:
    """Export decoder.onnx (LFM2 backbone with inputs_embeds).

    Outputs both logits and hidden_states for audio generation.
    """
    logger.info("Exporting decoder.onnx...")

    lfm_config = config.get("lfm", {})
    lfm2_config = LFM2Config.from_hf_config(type("Config", (), lfm_config)())

    builder = LFM2Builder(lfm2_config, use_integrated_rope=True, vl_naming=True)

    # Load LFM weights (prefixed with "lfm.")
    for name, weight in weights.items():
        if name.startswith("lfm."):
            new_name = "model." + name[4:]
            builder.weights[new_name] = weight

    H = lfm2_config.hidden_size

    # Build inputs
    builder.inputs.append(
        helper.make_tensor_value_info(
            "inputs_embeds", TensorProto.FLOAT, ["batch_size", "sequence_length", H]
        )
    )
    builder.inputs.append(
        helper.make_tensor_value_info(
            "attention_mask", TensorProto.INT64, ["batch_size", "total_sequence_length"]
        )
    )

    # Cache inputs
    conv_set = set(builder.conv_indices)
    attn_set = set(builder.attn_indices)
    for idx in range(lfm2_config.num_hidden_layers):
        if idx in conv_set:
            builder.inputs.append(
                helper.make_tensor_value_info(
                    f"past_conv.{idx}",
                    TensorProto.FLOAT,
                    ["batch_size", H, lfm2_config.conv_L_cache],
                )
            )
        elif idx in attn_set:
            builder.inputs.append(
                helper.make_tensor_value_info(
                    f"past_key_values.{idx}.key",
                    TensorProto.FLOAT,
                    [
                        "batch_size",
                        lfm2_config.num_key_value_heads,
                        "past_sequence_length",
                        builder.head_dim,
                    ],
                )
            )
            builder.inputs.append(
                helper.make_tensor_value_info(
                    f"past_key_values.{idx}.value",
                    TensorProto.FLOAT,
                    [
                        "batch_size",
                        lfm2_config.num_key_value_heads,
                        "past_sequence_length",
                        builder.head_dim,
                    ],
                )
            )

    builder.build_outputs()

    # Add hidden_states output for audio generation
    builder.outputs.append(
        helper.make_tensor_value_info(
            "hidden_states", TensorProto.FLOAT, ["batch_size", "sequence_length", H]
        )
    )

    builder.build_rope_cache()
    builder.build_attention_mask_subgraph()

    builder.add_initializer(
        "model.embed_tokens.weight", builder.weights["model.embed_tokens.weight"]
    )
    hidden_state = "inputs_embeds"

    for layer_idx in range(lfm2_config.num_hidden_layers):
        layer_type = lfm2_config.layer_types[layer_idx]
        logger.info(f"Building decoder layer {layer_idx} ({layer_type})...")
        builder.prepare_layer_weights(layer_idx, layer_type)

        if layer_type == "conv":
            hidden_state = builder.build_conv_layer(layer_idx, hidden_state)
        else:
            hidden_state = builder.build_attention_layer(layer_idx, hidden_state)

    # Build lm_head and capture hidden states
    # The build_lm_head applies final norm then lm_head projection
    # We need the normed hidden states before projection
    builder.build_lm_head(hidden_state)

    # Add Identity node to output hidden states (final norm output)
    # The final norm output is at /model/layers.{num_layers}/final_norm_layernorm/output_0
    num_layers = lfm2_config.num_hidden_layers
    final_norm_output = f"/model/layers.{num_layers}/final_norm_layernorm/output_0"
    builder.nodes.append(
        helper.make_node(
            "Identity",
            [final_norm_output],
            ["hidden_states"],
            name="/hidden_states/Identity",
        )
    )

    builder.build_value_info()

    graph = helper.make_graph(
        builder.nodes,
        "decoder",
        builder.inputs,
        builder.outputs,
        builder.initializers,
        value_info=builder.value_info,
    )

    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 21),
            helper.make_opsetid("com.microsoft", 1),
        ],
        ir_version=10,
    )
    model.producer_name = "liquidonnx"

    output_path = onnx_dir / "decoder.onnx"
    output_data = onnx_dir / "decoder.onnx_data"
    if output_data.exists():
        output_data.unlink()

    onnx.save_model(
        model,
        str(output_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="decoder.onnx_data",
        size_threshold=1024,
    )
    logger.info(f"decoder saved to {output_path}")
    return output_path


# === 5. Depthformer Export (torch.onnx) ===


class DepthformerWrapper(nn.Module):
    """Wrapper for depthformer export that predicts 8 codebook tokens autoregressively.

    The depthformer takes the decoder hidden state and generates 8 audio codes.
    For each code position:
    1. Apply depth_linear to project from hidden_size to 8*depth_dim
    2. Pass through 6 transformer layers
    3. Use to_logits to predict the code for each codebook position
    """

    def __init__(self, model):
        super().__init__()
        self.depth_linear = model.depth_linear
        self.depthformer = model.depthformer
        self.depth_embeddings = model.depth_embeddings

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, hidden_size] - last hidden state from decoder

        Returns:
            logits: [batch, 8, 2049] - logits for each of 8 codebooks
        """
        batch_size = hidden_states.shape[0]

        # Project to depth dimension: [B, H] -> [B, 8*D]
        depth_hidden = self.depth_linear(hidden_states)  # [B, 8192]

        # Reshape to [B, 8, D]
        depth_dim = depth_hidden.shape[-1] // 8
        depth_hidden = depth_hidden.view(batch_size, 8, depth_dim)  # [B, 8, 1024]

        # Run through depthformer transformer layers
        # The depthformer expects [B, S, D] format
        depth_output = self.depthformer(depth_hidden)  # [B, 8, 1024]

        # Predict codebook logits for each position
        all_logits = []
        for i in range(8):
            # Get hidden state for this codebook position
            pos_hidden = depth_output[:, i, :]  # [B, 1024]

            # Apply get_logits for this codebook (includes RMSNorm before projection)
            logits_i = self.depth_embeddings[i].get_logits(pos_hidden)  # [B, 2049]
            all_logits.append(logits_i.unsqueeze(1))  # [B, 1, 2049]

        # Stack all codebook logits
        logits = torch.cat(all_logits, dim=1)  # [B, 8, 2049]

        return logits


class DepthformerAutoregressiveWrapper(nn.Module):
    """Autoregressive depthformer that predicts one codebook at a time.

    Takes hidden states + previously predicted codes to predict next code.
    """

    def __init__(self, model):
        super().__init__()
        self.depth_linear = model.depth_linear
        self.depthformer = model.depthformer
        self.depth_embeddings = model.depth_embeddings

    def forward(
        self,
        hidden_states: torch.Tensor,
        codebook_idx: torch.Tensor,
        prev_codes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, hidden_size] - last hidden state from decoder
            codebook_idx: scalar - which codebook to predict (0-7)
            prev_codes: [batch, codebook_idx] - previously predicted codes

        Returns:
            logits: [batch, 2049] - logits for the next codebook
        """
        batch_size = hidden_states.shape[0]
        idx = codebook_idx.item()

        # Project to depth dimension
        depth_hidden = self.depth_linear(hidden_states)  # [B, 8*D]
        depth_dim = depth_hidden.shape[-1] // 8
        depth_hidden = depth_hidden.view(batch_size, 8, depth_dim)  # [B, 8, 1024]

        # Add embeddings from previous codes
        for i in range(idx):
            prev_code = prev_codes[:, i]  # [B]
            code_embed = self.depth_embeddings[i].embedding(prev_code)  # [B, 1024]
            code_embed = self.depth_embeddings[i].embedding_norm(code_embed)
            depth_hidden[:, i + 1, :] = depth_hidden[:, i + 1, :] + code_embed

        # Run through depthformer
        depth_output = self.depthformer(depth_hidden)  # [B, 8, 1024]

        # Get logits for target codebook
        pos_hidden = depth_output[:, idx, :]  # [B, 1024]
        logits = self.depth_embeddings[idx].to_logits(pos_hidden)  # [B, 2049]

        return logits


def export_depthformer(
    model, config: dict, onnx_dir: pathlib.Path, device: str = "cuda"
) -> pathlib.Path:
    """Export depthformer.onnx using torch.onnx.

    Exports a simple non-autoregressive version that predicts all 8 codes at once.
    This is suitable for greedy/parallel decoding. For full autoregressive decoding,
    use the PyTorch model directly.
    """
    logger.info("Exporting depthformer.onnx...")

    wrapper = DepthformerWrapper(model).to(device)
    wrapper.eval()

    hidden_size = config.get("lfm", {}).get("hidden_size", 2048)
    batch_size = 1

    # Dummy input
    hidden_states = torch.randn(batch_size, hidden_size, device=device, dtype=torch.float32)

    output_path = onnx_dir / "depthformer.onnx"

    # Suppress verbose IR graph dump from PyTorch ONNX exporter
    import sys
    import io
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    try:
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                (hidden_states,),
                str(output_path),
                input_names=["hidden_states"],
                output_names=["codebook_logits"],
                dynamic_axes={
                    "hidden_states": {0: "batch"},
                    "codebook_logits": {0: "batch"},
                },
                opset_version=18,
                do_constant_folding=True,
                dynamo=False,
                verbose=False,
            )
    finally:
        sys.stdout = old_stdout

    logger.info(f"depthformer saved to {output_path}")
    return output_path


class DepthformerBuilder:
    """Builder for depthformer ONNX export with full transformer layers.

    The depthformer predicts 8 audio codebook tokens autoregressively:
    1. depth_linear: [B, 2048] -> [B, 8192] -> [B, 8, 1024]
    2. 6 transformer layers with bounded attention (causal within 8 positions)
    3. 8 output heads (to_logits for each codebook position)

    Architecture per layer:
    - operator_norm (LayerNorm)
    - bounded_attention: qkv_proj -> Q/K LayerNorm -> causal attention -> out_proj
    - residual connection
    - ffn_norm (LayerNorm)
    - MLP (SwiGLU): w1/w3 -> SiLU -> w2
    - residual connection
    """

    def __init__(self, weights: dict[str, np.ndarray], input_hidden_size: int = 2048):
        self.weights = weights
        self.input_hidden_size = input_hidden_size

        # Depthformer config (derived from weight shapes)
        self.hidden_size = 1024  # depth dimension
        self.num_codebooks = 8
        self.codebook_vocab = 2049
        self.num_layers = 6
        self.num_attention_heads = 32  # Q heads
        self.num_key_value_heads = 8  # KV heads
        self.head_dim = 32  # 1024 / 32 = 32
        self.intermediate_size = 2816
        self.norm_eps = 1e-5

        # Graph components
        self.nodes: list = []
        self.initializers: list = []
        self._initializer_names: set[str] = set()

    def add_initializer(self, name: str, tensor: np.ndarray, dtype=None):
        """Add weight tensor as initializer."""
        if name in self._initializer_names:
            return
        self._initializer_names.add(name)
        if dtype is None:
            if tensor.dtype not in [np.int32, np.int64]:
                tensor = tensor.astype(np.float32)
        else:
            tensor = tensor.astype(dtype)
        self.initializers.append(onnx.numpy_helper.from_array(tensor, name))

    def make_node(self, op_type: str, inputs: list, outputs: list, **attrs):
        """Create an ONNX node."""
        name = outputs[0].replace("/output_0", "")
        node = helper.make_node(op_type, inputs, outputs, name=name, **attrs)
        self.nodes.append(node)
        return outputs[0]

    def build_layernorm(self, input_name: str, weight_name: str, path: str) -> str:
        """Build SimplifiedLayerNormalization (no bias)."""
        output_name = f"{path}/output_0"
        node = helper.make_node(
            "SimplifiedLayerNormalization",
            [input_name, weight_name],
            [output_name],
            name=path,
            epsilon=self.norm_eps,
        )
        self.nodes.append(node)
        return output_name

    def build_input_projection(self) -> str:
        """Build depth_linear projection: [B, 2048] -> [B, 8, 1024]."""
        # depth_linear: [2048] -> [8192]
        depth_linear_w = self.weights["depth_linear.weight"].astype(np.float32).T
        depth_linear_b = self.weights.get(
            "depth_linear.bias", np.zeros(8 * self.hidden_size)
        ).astype(np.float32)
        self.add_initializer("depth_linear.weight", depth_linear_w)
        self.add_initializer("depth_linear.bias", depth_linear_b)

        self.make_node(
            "MatMul",
            ["hidden_states", "depth_linear.weight"],
            ["/depth_linear/matmul/output_0"],
        )
        self.make_node(
            "Add",
            ["/depth_linear/matmul/output_0", "depth_linear.bias"],
            ["/depth_linear/output_0"],
        )

        # Reshape to [B, 8, 1024]
        self.add_initializer(
            "reshape_to_seq",
            np.array([-1, self.num_codebooks, self.hidden_size], dtype=np.int64),
        )
        return self.make_node(
            "Reshape",
            ["/depth_linear/output_0", "reshape_to_seq"],
            ["/depth_linear/reshaped/output_0"],
        )

    def build_attention_layer(self, layer_idx: int, hidden_state: str) -> str:
        """Build a bounded attention layer."""
        prefix = f"/depthformer/layers.{layer_idx}"
        weight_prefix = f"depthformer.layers.{layer_idx}"
        H = self.hidden_size
        nh = self.num_attention_heads
        nkv = self.num_key_value_heads
        hd = self.head_dim

        residual = hidden_state

        # Operator LayerNorm
        self.add_initializer(
            f"{weight_prefix}.operator_norm.weight",
            self.weights[f"{weight_prefix}.operator_norm.weight"].astype(np.float32),
        )
        normed = self.build_layernorm(
            hidden_state, f"{weight_prefix}.operator_norm.weight", f"{prefix}/operator_norm"
        )

        # QKV projection (fused): [B, 8, 1024] -> [B, 8, 1536]
        qkv_w = self.weights[f"{weight_prefix}.operator.qkv_proj.weight"].astype(np.float32).T
        self.add_initializer(f"{weight_prefix}.qkv.weight", qkv_w)
        qkv = self.make_node(
            "MatMul", [normed, f"{weight_prefix}.qkv.weight"], [f"{prefix}/attn/qkv/output_0"]
        )

        # Split QKV: Q [B, 8, 1024], K [B, 8, 256], V [B, 8, 256]
        q_dim = nh * hd  # 32 * 32 = 1024
        kv_dim = nkv * hd  # 8 * 32 = 256
        self.add_initializer(
            f"qkv_split_sizes_{layer_idx}", np.array([q_dim, kv_dim, kv_dim], dtype=np.int64)
        )
        node = helper.make_node(
            "Split",
            [qkv, f"qkv_split_sizes_{layer_idx}"],
            [f"{prefix}/attn/q/output_0", f"{prefix}/attn/k/output_0", f"{prefix}/attn/v/output_0"],
            name=f"{prefix}/attn/split_qkv",
            axis=-1,
        )
        self.nodes.append(node)

        # Q/K LayerNorm (per-head)
        q_ln_w = self.weights[
            f"{weight_prefix}.operator.bounded_attention.q_layernorm.weight"
        ].astype(np.float32)
        k_ln_w = self.weights[
            f"{weight_prefix}.operator.bounded_attention.k_layernorm.weight"
        ].astype(np.float32)
        self.add_initializer(f"{weight_prefix}.q_ln.weight", q_ln_w)
        self.add_initializer(f"{weight_prefix}.k_ln.weight", k_ln_w)

        # Reshape Q for per-head norm: [B, 8, 1024] -> [B, 8*32, 32]
        # Use layer-specific names for reshape constants to help shape inference
        self.add_initializer(f"reshape_for_norm_{layer_idx}", np.array([0, -1, hd], dtype=np.int64))
        self.add_initializer(
            f"reshape_q_back_{layer_idx}", np.array([0, -1, q_dim], dtype=np.int64)
        )
        self.add_initializer(
            f"reshape_k_back_{layer_idx}", np.array([0, -1, kv_dim], dtype=np.int64)
        )

        q_reshaped = self.make_node(
            "Reshape",
            [f"{prefix}/attn/q/output_0", f"reshape_for_norm_{layer_idx}"],
            [f"{prefix}/attn/q_reshape1/output_0"],
        )
        q_normed = self.build_layernorm(
            q_reshaped, f"{weight_prefix}.q_ln.weight", f"{prefix}/attn/q_norm"
        )
        q_3d = self.make_node(
            "Reshape",
            [q_normed, f"reshape_q_back_{layer_idx}"],
            [f"{prefix}/attn/q_reshape2/output_0"],
        )

        k_reshaped = self.make_node(
            "Reshape",
            [f"{prefix}/attn/k/output_0", f"reshape_for_norm_{layer_idx}"],
            [f"{prefix}/attn/k_reshape1/output_0"],
        )
        k_normed = self.build_layernorm(
            k_reshaped, f"{weight_prefix}.k_ln.weight", f"{prefix}/attn/k_norm"
        )
        k_3d = self.make_node(
            "Reshape",
            [k_normed, f"reshape_k_back_{layer_idx}"],
            [f"{prefix}/attn/k_reshape2/output_0"],
        )

        # Reshape for attention: [B, 8, H] -> [B, nh, 8, hd]
        self.add_initializer(
            f"reshape_q_heads_{layer_idx}", np.array([0, -1, nh, hd], dtype=np.int64)
        )
        self.add_initializer(
            f"reshape_kv_heads_{layer_idx}", np.array([0, -1, nkv, hd], dtype=np.int64)
        )

        q_4d = self.make_node(
            "Reshape", [q_3d, f"reshape_q_heads_{layer_idx}"], [f"{prefix}/attn/q_4d/output_0"]
        )
        q_4d_t = self.make_node(
            "Transpose", [q_4d], [f"{prefix}/attn/q_4d_t/output_0"], perm=[0, 2, 1, 3]
        )

        k_4d = self.make_node(
            "Reshape", [k_3d, f"reshape_kv_heads_{layer_idx}"], [f"{prefix}/attn/k_4d/output_0"]
        )
        k_4d_t = self.make_node(
            "Transpose", [k_4d], [f"{prefix}/attn/k_4d_t/output_0"], perm=[0, 2, 1, 3]
        )

        v_4d = self.make_node(
            "Reshape",
            [f"{prefix}/attn/v/output_0", f"reshape_kv_heads_{layer_idx}"],
            [f"{prefix}/attn/v_4d/output_0"],
        )
        v_4d_t = self.make_node(
            "Transpose", [v_4d], [f"{prefix}/attn/v_4d_t/output_0"], perm=[0, 2, 1, 3]
        )

        # Scale
        scale = 1.0 / np.sqrt(hd)
        self.add_initializer(f"attn_scale_{layer_idx}", np.array([scale], dtype=np.float32))

        # K transpose for scores: [B, nkv, 8, hd] -> [B, nkv, hd, 8]
        k_t = self.make_node(
            "Transpose", [k_4d_t], [f"{prefix}/attn/k_t/output_0"], perm=[0, 1, 3, 2]
        )

        # Repeat KV heads to match Q heads (GQA)
        repeat_factor = nh // nkv  # 32 / 8 = 4
        self.add_initializer(f"unsq_axis_2_{layer_idx}", np.array([2], dtype=np.int64))
        k_t_exp = self.make_node(
            "Unsqueeze", [k_t, f"unsq_axis_2_{layer_idx}"], [f"{prefix}/attn/k_t_exp/output_0"]
        )
        repeat_shape = np.array([1, 1, repeat_factor, 1, 1], dtype=np.int64)
        self.add_initializer(f"repeat_shape_{layer_idx}", repeat_shape)
        k_t_rep = self.make_node(
            "Tile", [k_t_exp, f"repeat_shape_{layer_idx}"], [f"{prefix}/attn/k_t_rep/output_0"]
        )
        self.add_initializer(
            f"reshape_k_gqa_{layer_idx}", np.array([0, nh, hd, -1], dtype=np.int64)
        )
        k_t = self.make_node(
            "Reshape", [k_t_rep, f"reshape_k_gqa_{layer_idx}"], [f"{prefix}/attn/k_gqa/output_0"]
        )

        v_exp = self.make_node(
            "Unsqueeze", [v_4d_t, f"unsq_axis_2_{layer_idx}"], [f"{prefix}/attn/v_exp/output_0"]
        )
        v_rep = self.make_node(
            "Tile", [v_exp, f"repeat_shape_{layer_idx}"], [f"{prefix}/attn/v_rep/output_0"]
        )
        self.add_initializer(
            f"reshape_v_gqa_{layer_idx}", np.array([0, nh, -1, hd], dtype=np.int64)
        )
        v_4d_t = self.make_node(
            "Reshape", [v_rep, f"reshape_v_gqa_{layer_idx}"], [f"{prefix}/attn/v_gqa/output_0"]
        )

        # Attention scores: Q @ K^T [B, nh, 8, 8]
        scores = self.make_node("MatMul", [q_4d_t, k_t], [f"{prefix}/attn/scores/output_0"])
        scores_scaled = self.make_node(
            "Mul", [scores, f"attn_scale_{layer_idx}"], [f"{prefix}/attn/scores_scaled/output_0"]
        )

        # Causal mask for bounded attention (lower triangular)
        # Create causal mask: [1, 1, 8, 8]
        causal_mask = np.triu(np.ones((1, 1, 8, 8), dtype=np.float32) * -1e9, k=1)
        self.add_initializer(f"causal_mask_{layer_idx}", causal_mask)
        scores_masked = self.make_node(
            "Add",
            [scores_scaled, f"causal_mask_{layer_idx}"],
            [f"{prefix}/attn/scores_masked/output_0"],
        )

        attn_weights = self.make_node(
            "Softmax", [scores_masked], [f"{prefix}/attn/softmax/output_0"], axis=-1
        )

        # Attention output: [B, nh, 8, hd]
        attn_out = self.make_node(
            "MatMul", [attn_weights, v_4d_t], [f"{prefix}/attn/attn_out/output_0"]
        )

        # Reshape back: [B, nh, 8, hd] -> [B, 8, H]
        attn_out_t = self.make_node(
            "Transpose", [attn_out], [f"{prefix}/attn/attn_out_t/output_0"], perm=[0, 2, 1, 3]
        )
        self.add_initializer(f"reshape_out_{layer_idx}", np.array([0, -1, H], dtype=np.int64))
        attn_out_3d = self.make_node(
            "Reshape",
            [attn_out_t, f"reshape_out_{layer_idx}"],
            [f"{prefix}/attn/attn_out_3d/output_0"],
        )

        # Output projection
        o_w = self.weights[f"{weight_prefix}.operator.out_proj.weight"].astype(np.float32).T
        self.add_initializer(f"{weight_prefix}.o.weight", o_w)
        o_proj = self.make_node(
            "MatMul", [attn_out_3d, f"{weight_prefix}.o.weight"], [f"{prefix}/attn/o_proj/output_0"]
        )

        # Residual
        hidden_state = self.make_node(
            "Add", [residual, o_proj], [f"{prefix}/attn/residual/output_0"]
        )

        return self.build_mlp(layer_idx, hidden_state)

    def build_mlp(self, layer_idx: int, hidden_state: str) -> str:
        """Build MLP block (SwiGLU activation)."""
        prefix = f"/depthformer/layers.{layer_idx}"
        weight_prefix = f"depthformer.layers.{layer_idx}"

        residual = hidden_state

        # FFN LayerNorm
        self.add_initializer(
            f"{weight_prefix}.ffn_norm.weight",
            self.weights[f"{weight_prefix}.ffn_norm.weight"].astype(np.float32),
        )
        normed = self.build_layernorm(
            hidden_state, f"{weight_prefix}.ffn_norm.weight", f"{prefix}/ffn_norm"
        )

        # Gate projection: [B, 8, 1024] -> [B, 8, 2816]
        gate_w = self.weights[f"{weight_prefix}.feed_forward.w1.weight"].astype(np.float32).T
        self.add_initializer(f"{weight_prefix}.gate.weight", gate_w)
        gate = self.make_node(
            "MatMul", [normed, f"{weight_prefix}.gate.weight"], [f"{prefix}/mlp/gate/output_0"]
        )

        # Up projection
        up_w = self.weights[f"{weight_prefix}.feed_forward.w3.weight"].astype(np.float32).T
        self.add_initializer(f"{weight_prefix}.up.weight", up_w)
        up = self.make_node(
            "MatMul", [normed, f"{weight_prefix}.up.weight"], [f"{prefix}/mlp/up/output_0"]
        )

        # SiLU on gate
        gate_sig = self.make_node("Sigmoid", [gate], [f"{prefix}/mlp/sigmoid/output_0"])
        gate_silu = self.make_node("Mul", [gate, gate_sig], [f"{prefix}/mlp/silu/output_0"])

        # gate * up
        gated = self.make_node("Mul", [gate_silu, up], [f"{prefix}/mlp/gated/output_0"])

        # Down projection
        down_w = self.weights[f"{weight_prefix}.feed_forward.w2.weight"].astype(np.float32).T
        self.add_initializer(f"{weight_prefix}.down.weight", down_w)
        down = self.make_node(
            "MatMul", [gated, f"{weight_prefix}.down.weight"], [f"{prefix}/mlp/down/output_0"]
        )

        # Residual
        return self.make_node("Add", [residual, down], [f"{prefix}/mlp/residual/output_0"])

    def build_output_heads(self, hidden_state: str) -> str:
        """Build output heads for each codebook position."""
        # Split hidden_state [B, 8, 1024] into 8 parts of [B, 1, 1024] each
        # This has better shape inference than Slice with dynamic indices
        split_outputs = [f"/output/split_{i}/output_0" for i in range(self.num_codebooks)]
        self.add_initializer(
            "split_sizes_output", np.array([1] * self.num_codebooks, dtype=np.int64)
        )
        node = helper.make_node(
            "Split",
            [hidden_state, "split_sizes_output"],
            split_outputs,
            name="/output/split",
            axis=1,
        )
        self.nodes.append(node)

        all_logits = []
        for i in range(self.num_codebooks):
            # Squeeze: [B, 1, 1024] -> [B, 1024]
            self.add_initializer(f"squeeze_axis_{i}", np.array([1], dtype=np.int64))
            squeezed = self.make_node(
                "Squeeze",
                [f"/output/split_{i}/output_0", f"squeeze_axis_{i}"],
                [f"/output/sq_{i}/output_0"],
            )

            # to_logits projection: [B, 1024] -> [B, 2049]
            to_logits_w = (
                self.weights[f"depth_embeddings.{i}.to_logits.weight"].astype(np.float32).T
            )
            self.add_initializer(f"to_logits_{i}.weight", to_logits_w)

            logits = self.make_node(
                "MatMul", [squeezed, f"to_logits_{i}.weight"], [f"/output/logits_{i}/output_0"]
            )

            # Unsqueeze for concat: [B, 2049] -> [B, 1, 2049]
            self.add_initializer(f"unsq_axis_{i}", np.array([1], dtype=np.int64))
            logits_unsq = self.make_node(
                "Unsqueeze", [logits, f"unsq_axis_{i}"], [f"/output/logits_unsq_{i}/output_0"]
            )
            all_logits.append(logits_unsq)

        # Concat all logits: [B, 8, 2049]
        return self.make_node("Concat", all_logits, ["codebook_logits"], axis=1)

    def build(self) -> onnx.ModelProto:
        """Build the complete depthformer ONNX model."""
        # Input: last hidden state from decoder [B, 2048]
        inputs = [
            helper.make_tensor_value_info(
                "hidden_states", TensorProto.FLOAT, ["batch_size", self.input_hidden_size]
            )
        ]

        # Output: codebook logits [B, 8, 2049]
        # Use None for dimensions to let shape be inferred (avoids shape conflicts with ORT)
        outputs = [
            helper.make_tensor_value_info(
                "codebook_logits",
                TensorProto.FLOAT,
                [None, None, None],  # Let shape be inferred
            )
        ]

        # Build input projection
        hidden_state = self.build_input_projection()

        # Build 6 transformer layers
        for layer_idx in range(self.num_layers):
            logger.info(f"Building depthformer layer {layer_idx}...")
            hidden_state = self.build_attention_layer(layer_idx, hidden_state)

        # Build output heads
        self.build_output_heads(hidden_state)

        # Create graph
        graph = helper.make_graph(self.nodes, "depthformer", inputs, outputs, self.initializers)
        model = helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", 21)],
            ir_version=10,
        )
        model.producer_name = "liquidonnx"
        return model


def export_depthformer_from_weights(
    weights: dict[str, np.ndarray], config: dict, onnx_dir: pathlib.Path
) -> pathlib.Path:
    """Export depthformer using ONNX builder with full transformer layers."""
    logger.info("Exporting depthformer.onnx (full builder version)...")

    input_hidden_size = config.get("lfm", {}).get("hidden_size", 2048)

    builder = DepthformerBuilder(weights, input_hidden_size)
    model = builder.build()

    output_path = onnx_dir / "depthformer.onnx"
    onnx.save_model(model, str(output_path))
    logger.info(f"depthformer saved to {output_path}")
    return output_path


# === 6. Audio LM Head Export (builder) ===


def export_audio_lm_head(
    weights: dict[str, np.ndarray], config: dict, onnx_dir: pathlib.Path
) -> pathlib.Path:
    """Export audio_lm_head.onnx for predicting first audio token."""
    logger.info("Exporting audio_lm_head.onnx...")

    hidden_size = config.get("lfm", {}).get("hidden_size", 2048)
    audio_vocab_size = 16392  # 8 codebooks * 2049

    nodes = []
    initializers = []

    inputs = [
        helper.make_tensor_value_info(
            "hidden_states", TensorProto.FLOAT, ["batch_size", "sequence_length", hidden_size]
        )
    ]
    outputs = [
        helper.make_tensor_value_info(
            "audio_logits",
            TensorProto.FLOAT,
            ["batch_size", "sequence_length", audio_vocab_size],
        )
    ]

    # Use embedding weight transposed as lm_head (tied weights)
    if "audio_embedding.embedding.weight" in weights:
        embed_weight = weights["audio_embedding.embedding.weight"].astype(np.float32)
        # Transpose for MatMul: [hidden, vocab]
        lm_head_weight = embed_weight.T
        initializers.append(onnx.numpy_helper.from_array(lm_head_weight, "audio_lm_head.weight"))

        nodes.append(
            helper.make_node(
                "MatMul",
                ["hidden_states", "audio_lm_head.weight"],
                ["audio_logits"],
            )
        )

    graph = helper.make_graph(nodes, "audio_lm_head", inputs, outputs, initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)], ir_version=10)
    model.producer_name = "liquidonnx"

    output_path = onnx_dir / "audio_lm_head.onnx"
    onnx.save_model(model, str(output_path))
    logger.info(f"audio_lm_head saved to {output_path}")
    return output_path


# === Quantization ===


def do_quantize(onnx_dir: pathlib.Path, bits: int, block_size: int, symmetric: bool):
    """Quantize all exportable models to specified precision."""
    # Models to quantize with their settings
    # (model_name, exclude_lm_head)
    models_to_quantize = [
        ("decoder", True),
        ("audio_encoder", False),
        ("depthformer", False),
        ("audio_detokenizer", False),
        ("audio_detokenizer_lfm", False),  # PyTorch-exported version (preferred)
        ("embed_tokens", False),
        ("audio_embedding", False),
        ("audio_lm_head", False),
    ]

    for model_name, exclude_lm_head in models_to_quantize:
        fp32_path = onnx_dir / f"{model_name}.onnx"
        quant_path = onnx_dir / f"{model_name}_q{bits}.onnx"

        if not fp32_path.exists():
            continue
        if quant_path.exists():
            logger.info(f"  {model_name}_q{bits}.onnx already exists, skipping")
            continue

        try:
            _, orig_mb = get_model_size(fp32_path)
            quantize_model(
                fp32_path,
                quant_path,
                bits=bits,
                block_size=block_size,
                exclude_lm_head=exclude_lm_head,
                symmetric=symmetric,
            )
            _, quant_mb = get_model_size(quant_path)
            logger.info(f"  {model_name}: {orig_mb:.1f} -> {quant_mb:.1f} MB")
        except Exception as e:
            logger.warning(f"  Failed to quantize {model_name}: {e}")


# === 7. Audio Detokenizer Export (hybrid) ===


class AudioDetokenizerLFMWrapper(nn.Module):
    """Wrapper for the LFM (neural network) part of audio detokenizer.

    The full audio detokenizer has: FusedEmbedding -> LFM -> Linear -> ISTFT
    ISTFT uses unsupported ops, so we export just the neural network part
    and implement ISTFT in NumPy.
    """

    def __init__(self, detokenizer):
        super().__init__()
        self.emb = detokenizer.emb  # FusedEmbedding
        self.lfm = detokenizer.lfm  # Lfm2Model
        self.lin = detokenizer.lin  # Linear
        self.sliding_window_size = getattr(detokenizer, "sliding_window_size", 30)

    def forward(self, audio_codes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio_codes: [batch, 8, time] - audio codes from depthformer

        Returns:
            stft_features: [batch, time', 1282] - STFT features (log_magnitude + angle)

        Reference: liquid_audio/detokenizer.py LFM2AudioDetokenizer.forward()
        """
        # Embed audio codes
        x = self.emb(audio_codes)  # [B, T, 512]

        # 6x upsample (critical for correct output)
        upsample_size = 6 * x.shape[1]
        x = torch.nn.functional.interpolate(x.mT, upsample_size, mode="nearest-exact").mT

        # Create sliding window attention mask
        # Reference: liquid_audio/detokenizer.py lines 125-128
        idx = torch.arange(x.shape[1], device=x.device)
        d_idx = idx - idx[:, None]
        mask = torch.logical_and(d_idx <= 0, d_idx > -self.sliding_window_size)[None, None, ...]

        # Run through LFM with attention mask
        x = self.lfm(inputs_embeds=x, attention_mask=mask, use_cache=False).last_hidden_state

        # Project to STFT feature space (log_magnitude + angle)
        x = self.lin(x)  # [B, T, 1282]

        return x


def export_audio_detokenizer_lfm(
    model, config: dict, onnx_dir: pathlib.Path, device: str = "cuda"
) -> pathlib.Path | None:
    """Export the neural network part of audio detokenizer.

    Returns None if export fails (e.g., due to unsupported ops).
    """
    logger.info("Exporting audio_detokenizer_lfm.onnx...")

    try:
        wrapper = AudioDetokenizerLFMWrapper(model.detokenizer).to(device)
        wrapper.eval()

        # Dummy input: [batch, 8, time]
        batch_size = 1
        num_codebooks = 8
        seq_len = 10
        audio_codes = torch.randint(0, 2048, (batch_size, num_codebooks, seq_len), device=device)

        output_path = onnx_dir / "audio_detokenizer_lfm.onnx"

        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                (audio_codes,),
                str(output_path),
                input_names=["audio_codes"],
                output_names=["stft_features"],
                dynamic_axes={
                    "audio_codes": {0: "batch", 2: "time"},
                    "stft_features": {0: "batch", 1: "time"},
                },
                opset_version=18,
                do_constant_folding=True,
                dynamo=False,
            )

        logger.info(f"audio_detokenizer_lfm saved to {output_path}")
        return output_path
    except Exception as e:
        logger.warning(f"Failed to export audio_detokenizer_lfm: {e}")
        return None


def save_istft_config(config: dict, onnx_dir: pathlib.Path):
    """Save ISTFT configuration for NumPy-based decoding."""
    import json

    istft_config = {
        "n_fft": 1280,
        "hop_length": 320,
        "win_length": 1280,
        "sample_rate": 24000,
        "center": True,
    }

    config_path = onnx_dir / "istft_config.json"
    with open(config_path, "w") as f:
        json.dump(istft_config, f, indent=2)

    logger.info(f"ISTFT config saved to {config_path}")


def export_audio_detokenizer_pytorch(model_path: str, onnx_dir: pathlib.Path) -> pathlib.Path | None:
    """Export audio detokenizer using PyTorch/transformers (more accurate than builder).

    This creates audio_detokenizer_lfm.onnx which uses the transformers Lfm2Model.
    The inference code will prefer this over the builder-based model.
    """
    import json
    import os

    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from transformers import Lfm2Config, Lfm2Model

    logger.info("Exporting audio_detokenizer_lfm.onnx (PyTorch/transformers)...")

    try:
        # Download audio_detokenizer weights
        cache_path = pathlib.Path(
            snapshot_download(model_path, allow_patterns=["audio_detokenizer/*"])
        )
        detok_path = cache_path / "audio_detokenizer"

        if not detok_path.exists():
            logger.warning("Audio detokenizer not found in model, skipping PyTorch export")
            return None

        # Load config
        with open(detok_path / "config.json") as f:
            config_dict = json.load(f)

        # Convert sliding_attention to full_attention for transformers compatibility
        # The sliding window attention mask is manually applied in forward()
        sliding_window = config_dict.get("sliding_window", 30)
        layer_types = config_dict.get("layer_types", [])
        config_dict["layer_types"] = [
            "full_attention" if lt == "sliding_attention" else lt
            for lt in layer_types
        ]
        lfm_config = Lfm2Config(**config_dict)

        # Create FusedEmbedding
        class FusedEmbedding(torch.nn.Module):
            def __init__(self, dim: int, codebooks: int = 8, vocab_size: int = 2048):
                super().__init__()
                self.emb = torch.nn.Embedding(codebooks * vocab_size, dim)
                self.codebooks = codebooks
                self.vocab_size = vocab_size

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                offsets = torch.arange(self.codebooks, device=x.device) * self.vocab_size
                offset_x = offsets[:, None] + x
                return self.emb(offset_x).mean(1)

        # Create detokenizer wrapper
        class AudioDetokPyTorch(torch.nn.Module):
            def __init__(self, config, sliding_window: int):
                super().__init__()
                self.emb = FusedEmbedding(config.hidden_size)
                self.lfm = Lfm2Model(config)
                self.lin = torch.nn.Linear(config.hidden_size, 1282)
                self.sliding_window = sliding_window

            def forward(self, audio_codes: torch.Tensor) -> torch.Tensor:
                x = self.emb(audio_codes)
                # 6x upsample (critical) - use transpose instead of .mT for ONNX compatibility
                # Use "nearest" instead of "nearest-exact" for ONNX opset 14 compatibility
                upsample_size = 6 * x.shape[1]
                x = x.transpose(-1, -2)  # [B, T, H] -> [B, H, T]
                x = torch.nn.functional.interpolate(x, upsample_size, mode="nearest")
                x = x.transpose(-1, -2)  # [B, H, T*6] -> [B, T*6, H]

                # Create sliding window attention mask (critical for audio quality)
                # Each position attends to at most sliding_window previous positions
                seq_len = x.shape[1]
                idx = torch.arange(seq_len, device=x.device)
                d_idx = idx - idx[:, None]
                mask = torch.logical_and(d_idx <= 0, d_idx > -self.sliding_window)
                mask = mask[None, None, ...]  # [1, 1, S, S]

                x = self.lfm(inputs_embeds=x, attention_mask=mask, use_cache=False).last_hidden_state
                x = self.lin(x)
                return x

        logger.info("Creating PyTorch model...")
        model = AudioDetokPyTorch(lfm_config, sliding_window)

        # Load weights
        weights = load_file(str(detok_path / "model.safetensors"))
        model.load_state_dict(weights, strict=False)
        model.eval()

        # Export to ONNX
        logger.info("Exporting to ONNX...")
        codes = torch.randint(0, 2048, (1, 8, 10), dtype=torch.long)
        output_path = onnx_dir / "audio_detokenizer_lfm.onnx"

        # Use legacy exporter (dynamo=False) because dynamo can't handle
        # dynamic attention mask creation in the forward pass
        with torch.no_grad():
            torch.onnx.export(
                model,
                (codes,),
                str(output_path),
                input_names=["audio_codes"],
                output_names=["stft_features"],
                dynamic_axes={
                    "audio_codes": {0: "batch", 2: "time"},
                    "stft_features": {0: "batch", 1: "time"},
                },
                opset_version=17,
                do_constant_folding=True,
                dynamo=False,
                verbose=False,
            )
        # Clean up model
        del model
        gc.collect()

        logger.info(f"audio_detokenizer_lfm saved to {output_path}")
        return output_path

    except Exception as e:
        logger.warning(f"Failed to export audio_detokenizer_lfm: {e}")
        import traceback
        traceback.print_exc()
        return None


# === 8. Audio Detokenizer Export (builder) ===


class AudioDetokenizerBuilder:
    """Builder for audio detokenizer ONNX export.

    The audio detokenizer has the following architecture:
    1. FusedEmbedding: 8 codebooks (2048 vocab each) → [B, T, 512]
    2. LFM (8 layers): Mix of conv and sliding_attention layers
    3. Linear: [B, T, 512] → [B, T, 1282] (STFT space)

    Layer types: ["conv", "conv", "sliding_attention", "conv",
                  "sliding_attention", "conv", "sliding_attention", "conv"]
    """

    def __init__(self, config: dict, weights: dict[str, np.ndarray]):
        self.config = config
        self.weights = weights

        # Model configuration
        self.hidden_size = config.get("hidden_size", 512)
        self.num_attention_heads = config.get("num_attention_heads", 16)
        self.num_key_value_heads = config.get("num_key_value_heads", 8)
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.intermediate_size = config.get("intermediate_size") or (self.hidden_size * 9 // 2)
        self.output_size = config.get("output_size", 1282)
        self.norm_eps = config.get("norm_eps", 1e-5)
        self.num_codebooks = 8
        self.codebook_vocab = 2048
        self.conv_L_cache = 3

        # Layer types: 4 conv, 4 sliding_attention
        self.layer_types = config.get(
            "layer_types",
            [
                "conv",
                "conv",
                "sliding_attention",
                "conv",
                "sliding_attention",
                "conv",
                "sliding_attention",
                "conv",
            ],
        )
        self.num_layers = len(self.layer_types)

        # Graph components
        self.nodes: list = []
        self.initializers: list = []
        self._initializer_names: set[str] = set()

    def add_initializer(self, name: str, tensor: np.ndarray, dtype=None):
        """Add weight tensor as initializer."""
        if name in self._initializer_names:
            return
        self._initializer_names.add(name)
        if dtype is None:
            if tensor.dtype not in [np.int32, np.int64]:
                tensor = tensor.astype(np.float32)
        else:
            tensor = tensor.astype(dtype)
        self.initializers.append(onnx.numpy_helper.from_array(tensor, name))

    def get_constant(self, value, dtype=np.int64) -> str:
        """Add constant and return its name."""
        arr = np.asarray(value, dtype=dtype)
        name = f"/constants/{str(value).replace(' ', '')}"
        self.add_initializer(name, arr)
        return name

    def make_node(self, op_type: str, inputs: list, outputs: list, **attrs):
        """Create an ONNX node."""
        name = outputs[0].replace("/output_0", "")
        node = helper.make_node(op_type, inputs, outputs, name=name, **attrs)
        self.nodes.append(node)
        return outputs[0]

    def build_embedding(self) -> str:
        """Build fused codebook embedding.

        Input: audio_codes [B, 8, T]
        Output: embedded [B, T, 512]
        """
        # Embedding weight: [16384, 512] (8 codebooks * 2048)
        emb_weight = self.weights["emb.emb.weight"].astype(np.float32)
        self.add_initializer("emb.weight", emb_weight)

        # Codebook offsets: [0, 2048, 4096, ...]
        offsets = np.array(
            [i * self.codebook_vocab for i in range(self.num_codebooks)], dtype=np.int64
        ).reshape(1, self.num_codebooks, 1)
        self.add_initializer("codebook_offsets", offsets)

        # Add offsets to codes: [B, 8, T]
        self.make_node("Add", ["audio_codes", "codebook_offsets"], ["/emb/offset_codes/output_0"])

        # Transpose: [B, 8, T] -> [B, T, 8]
        self.make_node(
            "Transpose",
            ["/emb/offset_codes/output_0"],
            ["/emb/transposed/output_0"],
            perm=[0, 2, 1],
        )

        # Get shape for reshape back
        self.make_node("Shape", ["/emb/transposed/output_0"], ["/emb/shape/output_0"])

        # Flatten for gather: [B, T, 8] -> [B*T*8]
        self.add_initializer("flatten_shape", np.array([-1], dtype=np.int64))
        self.make_node(
            "Reshape", ["/emb/transposed/output_0", "flatten_shape"], ["/emb/flat/output_0"]
        )

        # Gather embeddings: [B*T*8, 512]
        self.make_node("Gather", ["emb.weight", "/emb/flat/output_0"], ["/emb/gathered/output_0"])

        # Get batch and time dimensions
        self.add_initializer("zero_idx", np.array([0], dtype=np.int64))
        self.add_initializer("one_idx", np.array([1], dtype=np.int64))
        self.add_initializer("two_idx", np.array([2], dtype=np.int64))
        self.add_initializer("eight_const", np.array([8], dtype=np.int64))
        self.add_initializer("hidden_const", np.array([self.hidden_size], dtype=np.int64))

        self.make_node(
            "Slice", ["/emb/shape/output_0", "zero_idx", "one_idx"], ["/emb/batch_dim/output_0"]
        )
        self.make_node(
            "Slice", ["/emb/shape/output_0", "one_idx", "two_idx"], ["/emb/time_dim/output_0"]
        )

        # Build reshape shape [B, T, 8, 512]
        self.make_node(
            "Concat",
            ["/emb/batch_dim/output_0", "/emb/time_dim/output_0", "eight_const", "hidden_const"],
            ["/emb/reshape_shape/output_0"],
            axis=0,
        )

        # Reshape: [B*T*8, 512] -> [B, T, 8, 512]
        self.make_node(
            "Reshape",
            ["/emb/gathered/output_0", "/emb/reshape_shape/output_0"],
            ["/emb/reshaped/output_0"],
        )

        # Mean across codebooks: [B, T, 8, 512] -> [B, T, 512]
        # Reference: liquid_audio/detokenizer.py FusedEmbedding.forward() uses .mean(1)
        self.add_initializer("mean_axis", np.array([2], dtype=np.int64))
        self.make_node(
            "ReduceMean",
            ["/emb/reshaped/output_0", "mean_axis"],
            ["/emb/summed/output_0"],
            keepdims=0,
        )

        # Apply embedding norm (critical for correct output scaling)
        emb_output = "/emb/summed/output_0"
        if "lfm.embedding_norm.weight" in self.weights:
            self.add_initializer(
                "lfm.embedding_norm.weight",
                self.weights["lfm.embedding_norm.weight"].astype(np.float32),
            )
            emb_output = self.build_layernorm(
                "/emb/summed/output_0", "lfm.embedding_norm.weight", "/emb/norm"
            )

        # === 6x Upsampling ===
        # Reference: liquid_audio/detokenizer.py LFM2AudioDetokenizer.forward()
        # upsample_size = 6 * x.shape[1]
        # x = nn.functional.interpolate(x.mT, upsample_size, mode="nearest-exact").mT
        #
        # Flow: [B, T, H] → transpose → [B, H, T] → resize 6x → [B, H, 6T] → transpose → [B, 6T, H]

        # Transpose [B, T, H] → [B, H, T]
        self.make_node(
            "Transpose", [emb_output], ["/emb/pre_upsample_t/output_0"], perm=[0, 2, 1]
        )

        # Resize: [B, H, T] → [B, H, 6*T]
        # Using Resize with scales [1, 1, 6] for nearest-neighbor interpolation
        self.add_initializer("upsample_scales", np.array([1.0, 1.0, 6.0], dtype=np.float32))
        # Empty roi and sizes as per ONNX spec (use scales instead)
        self.add_initializer("empty_roi", np.array([], dtype=np.float32))
        self.add_initializer("empty_sizes", np.array([], dtype=np.int64))

        node = helper.make_node(
            "Resize",
            ["/emb/pre_upsample_t/output_0", "empty_roi", "upsample_scales"],
            ["/emb/upsampled/output_0"],
            name="/emb/upsample",
            mode="nearest",
            coordinate_transformation_mode="asymmetric",
            nearest_mode="floor",
        )
        self.nodes.append(node)

        # Transpose back: [B, H, 6T] → [B, 6T, H]
        return self.make_node(
            "Transpose", ["/emb/upsampled/output_0"], ["/emb/post_upsample_t/output_0"], perm=[0, 2, 1]
        )

    def build_layernorm(self, input_name: str, weight_name: str, path: str) -> str:
        """Build SimplifiedLayerNormalization (no bias)."""
        output_name = f"{path}/output_0"
        node = helper.make_node(
            "SimplifiedLayerNormalization",
            [input_name, weight_name],
            [output_name],
            name=path,
            epsilon=self.norm_eps,
        )
        self.nodes.append(node)
        return output_name

    def build_mlp(self, layer_idx: int, hidden_state: str) -> str:
        """Build MLP block (SwiGLU activation)."""
        prefix = f"/lfm/layers.{layer_idx}"
        weight_prefix = f"lfm.layers.{layer_idx}"

        residual = hidden_state

        # FFN LayerNorm
        self.add_initializer(
            f"{weight_prefix}.ffn_norm.weight",
            self.weights[f"{weight_prefix}.ffn_norm.weight"].astype(np.float32),
        )
        normed = self.build_layernorm(
            hidden_state, f"{weight_prefix}.ffn_norm.weight", f"{prefix}/ffn_norm"
        )

        # Gate projection: [B, T, H] -> [B, T, intermediate]
        gate_w = self.weights[f"{weight_prefix}.feed_forward.w1.weight"].astype(np.float32).T
        self.add_initializer(f"{weight_prefix}.gate.weight", gate_w)
        gate = self.make_node(
            "MatMul",
            [normed, f"{weight_prefix}.gate.weight"],
            [f"{prefix}/mlp/gate/output_0"],
        )

        # Up projection
        up_w = self.weights[f"{weight_prefix}.feed_forward.w3.weight"].astype(np.float32).T
        self.add_initializer(f"{weight_prefix}.up.weight", up_w)
        up = self.make_node(
            "MatMul",
            [normed, f"{weight_prefix}.up.weight"],
            [f"{prefix}/mlp/up/output_0"],
        )

        # SiLU on gate: gate * sigmoid(gate)
        gate_sig = self.make_node("Sigmoid", [gate], [f"{prefix}/mlp/sigmoid/output_0"])
        gate_silu = self.make_node("Mul", [gate, gate_sig], [f"{prefix}/mlp/silu/output_0"])

        # gate * up
        gated = self.make_node("Mul", [gate_silu, up], [f"{prefix}/mlp/gated/output_0"])

        # Down projection
        down_w = self.weights[f"{weight_prefix}.feed_forward.w2.weight"].astype(np.float32).T
        self.add_initializer(f"{weight_prefix}.down.weight", down_w)
        down = self.make_node(
            "MatMul",
            [gated, f"{weight_prefix}.down.weight"],
            [f"{prefix}/mlp/down/output_0"],
        )

        # Residual
        return self.make_node("Add", [residual, down], [f"{prefix}/mlp/residual/output_0"])

    def build_conv_layer(self, layer_idx: int, hidden_state: str) -> str:
        """Build a conv layer (short convolution with gating).

        Note: For the detokenizer, we don't use caching - we just apply the convolution
        to the full sequence with padding.
        """
        prefix = f"/lfm/layers.{layer_idx}"
        weight_prefix = f"lfm.layers.{layer_idx}"
        H = self.hidden_size
        L = self.conv_L_cache  # kernel size

        residual = hidden_state

        # Operator LayerNorm
        self.add_initializer(
            f"{weight_prefix}.operator_norm.weight",
            self.weights[f"{weight_prefix}.operator_norm.weight"].astype(np.float32),
        )
        normed = self.build_layernorm(
            hidden_state, f"{weight_prefix}.operator_norm.weight", f"{prefix}/operator_norm"
        )

        # In projection: [B, T, H] -> [B, T, 3H]
        in_proj_w = self.weights[f"{weight_prefix}.conv.in_proj.weight"].astype(np.float32).T
        self.add_initializer(f"{weight_prefix}.in_proj.weight", in_proj_w)
        in_proj = self.make_node(
            "MatMul",
            [normed, f"{weight_prefix}.in_proj.weight"],
            [f"{prefix}/conv/in_proj/output_0"],
        )

        # Transpose: [B, T, 3H] -> [B, 3H, T]
        in_proj_t = self.make_node(
            "Transpose", [in_proj], [f"{prefix}/conv/transpose1/output_0"], perm=[0, 2, 1]
        )

        # Split into B, C, x (each [B, H, T])
        self.add_initializer("split_sizes", np.array([H, H, H], dtype=np.int64))
        node = helper.make_node(
            "Split",
            [in_proj_t, "split_sizes"],
            [
                f"{prefix}/conv/B/output_0",
                f"{prefix}/conv/C/output_0",
                f"{prefix}/conv/x/output_0",
            ],
            name=f"{prefix}/conv/split",
            axis=1,
        )
        self.nodes.append(node)

        # Bx = B * x (input gating, no sigmoid)
        Bx = self.make_node(
            "Mul",
            [f"{prefix}/conv/B/output_0", f"{prefix}/conv/x/output_0"],
            [f"{prefix}/conv/Bx/output_0"],
        )

        # Pad Bx for causal convolution: [B, H, T] -> [B, H, L-1 + T]
        # Pad format: [x1_begin, x2_begin, x3_begin, x1_end, x2_end, x3_end]
        self.add_initializer("conv_pads", np.array([0, 0, L - 1, 0, 0, 0], dtype=np.int64))
        Bx_padded = self.make_node(
            "Pad", [Bx, "conv_pads"], [f"{prefix}/conv/padded/output_0"], mode="constant"
        )

        # Depthwise Conv1D (kernel=L, groups=H)
        conv_w = self.weights[f"{weight_prefix}.conv.conv.weight"].astype(np.float32)
        self.add_initializer(f"{weight_prefix}.conv.weight", conv_w)
        conv_out = self.make_node(
            "Conv",
            [Bx_padded, f"{weight_prefix}.conv.weight"],
            [f"{prefix}/conv/conv1d/output_0"],
            kernel_shape=[L],
            group=H,
        )

        # Output gating: y = C * conv_out
        y = self.make_node(
            "Mul",
            [f"{prefix}/conv/C/output_0", conv_out],
            [f"{prefix}/conv/y/output_0"],
        )

        # Transpose: [B, H, T] -> [B, T, H]
        y_t = self.make_node(
            "Transpose", [y], [f"{prefix}/conv/transpose2/output_0"], perm=[0, 2, 1]
        )

        # Out projection
        out_proj_w = self.weights[f"{weight_prefix}.conv.out_proj.weight"].astype(np.float32).T
        self.add_initializer(f"{weight_prefix}.out_proj.weight", out_proj_w)
        out_proj = self.make_node(
            "MatMul",
            [y_t, f"{weight_prefix}.out_proj.weight"],
            [f"{prefix}/conv/out_proj/output_0"],
        )

        # Residual
        hidden_state = self.make_node(
            "Add", [residual, out_proj], [f"{prefix}/conv/residual/output_0"]
        )

        # MLP
        return self.build_mlp(layer_idx, hidden_state)

    def build_attention_layer(self, layer_idx: int, hidden_state: str) -> str:
        """Build a sliding attention layer.

        For the detokenizer, we use standard attention (no KV cache) with a causal mask.
        sliding_attention typically uses a local window but here we just use full attention
        since the sequences are short.
        """
        prefix = f"/lfm/layers.{layer_idx}"
        weight_prefix = f"lfm.layers.{layer_idx}"
        H = self.hidden_size
        nh = self.num_attention_heads
        nkv = self.num_key_value_heads
        hd = self.head_dim

        residual = hidden_state

        # Operator LayerNorm
        self.add_initializer(
            f"{weight_prefix}.operator_norm.weight",
            self.weights[f"{weight_prefix}.operator_norm.weight"].astype(np.float32),
        )
        normed = self.build_layernorm(
            hidden_state, f"{weight_prefix}.operator_norm.weight", f"{prefix}/operator_norm"
        )

        # Q/K/V projections
        q_w = self.weights[f"{weight_prefix}.self_attn.q_proj.weight"].astype(np.float32).T
        k_w = self.weights[f"{weight_prefix}.self_attn.k_proj.weight"].astype(np.float32).T
        v_w = self.weights[f"{weight_prefix}.self_attn.v_proj.weight"].astype(np.float32).T

        self.add_initializer(f"{weight_prefix}.q.weight", q_w)
        self.add_initializer(f"{weight_prefix}.k.weight", k_w)
        self.add_initializer(f"{weight_prefix}.v.weight", v_w)

        q = self.make_node(
            "MatMul", [normed, f"{weight_prefix}.q.weight"], [f"{prefix}/attn/q/output_0"]
        )
        k = self.make_node(
            "MatMul", [normed, f"{weight_prefix}.k.weight"], [f"{prefix}/attn/k/output_0"]
        )
        v = self.make_node(
            "MatMul", [normed, f"{weight_prefix}.v.weight"], [f"{prefix}/attn/v/output_0"]
        )

        # Q/K LayerNorm (per-head)
        q_ln_w = self.weights[f"{weight_prefix}.self_attn.q_layernorm.weight"].astype(np.float32)
        k_ln_w = self.weights[f"{weight_prefix}.self_attn.k_layernorm.weight"].astype(np.float32)
        self.add_initializer(f"{weight_prefix}.q_ln.weight", q_ln_w)
        self.add_initializer(f"{weight_prefix}.k_ln.weight", k_ln_w)

        # Reshape Q for per-head norm: [B, T, H] -> [B, T*nh, hd]
        self.add_initializer("reshape_for_norm", np.array([0, -1, hd], dtype=np.int64))
        self.add_initializer("reshape_q_back", np.array([0, -1, H], dtype=np.int64))
        self.add_initializer("reshape_k_back", np.array([0, -1, nkv * hd], dtype=np.int64))

        q_reshaped = self.make_node(
            "Reshape", [q, "reshape_for_norm"], [f"{prefix}/attn/q_reshape1/output_0"]
        )
        q_normed = self.build_layernorm(
            q_reshaped, f"{weight_prefix}.q_ln.weight", f"{prefix}/attn/q_norm"
        )
        q_3d = self.make_node(
            "Reshape", [q_normed, "reshape_q_back"], [f"{prefix}/attn/q_reshape2/output_0"]
        )

        k_reshaped = self.make_node(
            "Reshape", [k, "reshape_for_norm"], [f"{prefix}/attn/k_reshape1/output_0"]
        )
        k_normed = self.build_layernorm(
            k_reshaped, f"{weight_prefix}.k_ln.weight", f"{prefix}/attn/k_norm"
        )
        k_3d = self.make_node(
            "Reshape", [k_normed, "reshape_k_back"], [f"{prefix}/attn/k_reshape2/output_0"]
        )

        # Reshape for attention: [B, T, H] -> [B, nh, T, hd]
        self.add_initializer("reshape_q_heads", np.array([0, -1, nh, hd], dtype=np.int64))
        self.add_initializer("reshape_k_heads", np.array([0, -1, nkv, hd], dtype=np.int64))
        self.add_initializer("reshape_v_heads", np.array([0, -1, nkv, hd], dtype=np.int64))

        q_4d = self.make_node(
            "Reshape", [q_3d, "reshape_q_heads"], [f"{prefix}/attn/q_4d/output_0"]
        )
        q_4d_t = self.make_node(
            "Transpose", [q_4d], [f"{prefix}/attn/q_4d_t/output_0"], perm=[0, 2, 1, 3]
        )

        k_4d = self.make_node(
            "Reshape", [k_3d, "reshape_k_heads"], [f"{prefix}/attn/k_4d/output_0"]
        )
        k_4d_t = self.make_node(
            "Transpose", [k_4d], [f"{prefix}/attn/k_4d_t/output_0"], perm=[0, 2, 1, 3]
        )

        v_4d = self.make_node("Reshape", [v, "reshape_v_heads"], [f"{prefix}/attn/v_4d/output_0"])
        v_4d_t = self.make_node(
            "Transpose", [v_4d], [f"{prefix}/attn/v_4d_t/output_0"], perm=[0, 2, 1, 3]
        )

        # Scaled dot product attention (SDPA)
        # For simplicity, use the SDPA op if available, otherwise manual implementation
        # Note: ONNX opset 21 doesn't have SDPA, but we can use com.microsoft.Attention
        # or implement manually
        scale = 1.0 / np.sqrt(hd)
        self.add_initializer("attn_scale", np.array([scale], dtype=np.float32))

        # K transpose: [B, nkv, T, hd] -> [B, nkv, hd, T]
        k_t = self.make_node(
            "Transpose", [k_4d_t], [f"{prefix}/attn/k_t/output_0"], perm=[0, 1, 3, 2]
        )

        # Repeat KV heads to match Q heads if needed (GQA)
        if nkv != nh:
            repeat_factor = nh // nkv
            # Expand K: [B, nkv, hd, T] -> [B, nkv, 1, hd, T] -> [B, nkv, repeat, hd, T]
            self.add_initializer("unsq_axis", np.array([2], dtype=np.int64))
            k_t_exp = self.make_node(
                "Unsqueeze", [k_t, "unsq_axis"], [f"{prefix}/attn/k_t_exp/output_0"]
            )
            repeat_shape = np.array([1, 1, repeat_factor, 1, 1], dtype=np.int64)
            self.add_initializer("repeat_shape", repeat_shape)
            k_t_rep = self.make_node(
                "Tile", [k_t_exp, "repeat_shape"], [f"{prefix}/attn/k_t_rep/output_0"]
            )
            self.add_initializer("reshape_k_gqa", np.array([0, nh, hd, -1], dtype=np.int64))
            k_t = self.make_node(
                "Reshape", [k_t_rep, "reshape_k_gqa"], [f"{prefix}/attn/k_gqa/output_0"]
            )

            # Expand V similarly
            v_exp = self.make_node(
                "Unsqueeze", [v_4d_t, "unsq_axis"], [f"{prefix}/attn/v_exp/output_0"]
            )
            v_rep = self.make_node(
                "Tile", [v_exp, "repeat_shape"], [f"{prefix}/attn/v_rep/output_0"]
            )
            self.add_initializer("reshape_v_gqa", np.array([0, nh, -1, hd], dtype=np.int64))
            v_4d_t = self.make_node(
                "Reshape", [v_rep, "reshape_v_gqa"], [f"{prefix}/attn/v_gqa/output_0"]
            )

        # Attention scores: Q @ K^T [B, nh, T, T]
        scores = self.make_node("MatMul", [q_4d_t, k_t], [f"{prefix}/attn/scores/output_0"])
        scores_scaled = self.make_node(
            "Mul", [scores, "attn_scale"], [f"{prefix}/attn/scores_scaled/output_0"]
        )

        # Causal mask: lower triangular (for audio this is typically bidirectional,
        # but we'll use non-causal for now since audio tokens are all given)
        # For now, just apply softmax without mask
        attn_weights = self.make_node(
            "Softmax", [scores_scaled], [f"{prefix}/attn/softmax/output_0"], axis=-1
        )

        # Attention output: [B, nh, T, hd]
        attn_out = self.make_node(
            "MatMul", [attn_weights, v_4d_t], [f"{prefix}/attn/attn_out/output_0"]
        )

        # Reshape back: [B, nh, T, hd] -> [B, T, H]
        attn_out_t = self.make_node(
            "Transpose", [attn_out], [f"{prefix}/attn/attn_out_t/output_0"], perm=[0, 2, 1, 3]
        )
        self.add_initializer("reshape_out", np.array([0, -1, H], dtype=np.int64))
        attn_out_3d = self.make_node(
            "Reshape", [attn_out_t, "reshape_out"], [f"{prefix}/attn/attn_out_3d/output_0"]
        )

        # Output projection
        o_w = self.weights[f"{weight_prefix}.self_attn.out_proj.weight"].astype(np.float32).T
        self.add_initializer(f"{weight_prefix}.o.weight", o_w)
        o_proj = self.make_node(
            "MatMul", [attn_out_3d, f"{weight_prefix}.o.weight"], [f"{prefix}/attn/o_proj/output_0"]
        )

        # Residual
        hidden_state = self.make_node(
            "Add", [residual, o_proj], [f"{prefix}/attn/residual/output_0"]
        )

        # MLP
        return self.build_mlp(layer_idx, hidden_state)

    def build_output_linear(self, hidden_state: str) -> str:
        """Build final linear projection to STFT space."""
        # Final layer norm (optional, some models have it)
        if "lfm.norm.weight" in self.weights:
            self.add_initializer(
                "lfm.norm.weight",
                self.weights["lfm.norm.weight"].astype(np.float32),
            )
            hidden_state = self.build_layernorm(hidden_state, "lfm.norm.weight", "/lfm/final_norm")

        # Linear projection: [B, T, H] -> [B, T, output_size]
        lin_w = self.weights["lin.weight"].astype(np.float32).T
        lin_b = self.weights.get("lin.bias", np.zeros(self.output_size)).astype(np.float32)
        self.add_initializer("lin.weight", lin_w)
        self.add_initializer("lin.bias", lin_b)

        lin_out = self.make_node("MatMul", [hidden_state, "lin.weight"], ["/lin/matmul/output_0"])
        return self.make_node("Add", [lin_out, "lin.bias"], ["stft_features"])

    def build(self) -> onnx.ModelProto:
        """Build the complete audio detokenizer ONNX model."""
        # Input
        inputs = [
            helper.make_tensor_value_info(
                "audio_codes", TensorProto.INT64, ["batch_size", self.num_codebooks, "time"]
            )
        ]

        # Output
        outputs = [
            helper.make_tensor_value_info(
                "stft_features",
                TensorProto.FLOAT,
                ["batch_size", "time", self.output_size],
            )
        ]

        # Build embedding
        hidden_state = self.build_embedding()

        # Build LFM layers
        for layer_idx in range(self.num_layers):
            layer_type = self.layer_types[layer_idx]
            logger.info(f"Building detokenizer layer {layer_idx} ({layer_type})...")

            if layer_type == "conv":
                hidden_state = self.build_conv_layer(layer_idx, hidden_state)
            else:  # sliding_attention
                hidden_state = self.build_attention_layer(layer_idx, hidden_state)

        # Build output linear
        self.build_output_linear(hidden_state)

        # Create graph
        graph = helper.make_graph(
            self.nodes, "audio_detokenizer", inputs, outputs, self.initializers
        )
        model = helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", 21)],
            ir_version=10,
        )
        model.producer_name = "liquidonnx"
        return model


def export_audio_detokenizer_builder(model_path: str, onnx_dir: pathlib.Path) -> pathlib.Path:
    """Export audio detokenizer using ONNX builder with full LFM layers.

    The audio detokenizer converts audio codes to waveform:
    1. Embedding: [B, 8, T] -> [B, T, 512] (fused codebook embedding)
    2. LFM: [B, T, 512] -> [B, T, 512] (8-layer transformer with conv and attention)
    3. Linear: [B, T, 512] -> [B, T, 1282] (STFT space projection)
    4. ISTFT: [B, T, 1282] -> waveform (done in numpy/scipy)

    This exports steps 1-3 to ONNX. Step 4 is done in numpy.
    """
    logger.info("Exporting audio_detokenizer.onnx (full LFM builder version)...")

    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    # Download audio_detokenizer from HuggingFace
    try:
        cache_path = pathlib.Path(
            snapshot_download(
                model_path,
                allow_patterns=["audio_detokenizer/*"],
            )
        )
        detok_path = cache_path / "audio_detokenizer"
    except Exception as e:
        logger.warning(f"Could not download audio_detokenizer: {e}")
        return None

    if not detok_path.exists():
        logger.warning("Audio detokenizer not found, skipping export")
        return None

    # Load config
    import json as json_module

    with open(detok_path / "config.json") as f:
        detok_config = json_module.load(f)

    logger.info(f"Audio detokenizer config: {detok_config}")

    # Load weights
    detok_weights = {}
    with safe_open(str(detok_path / "model.safetensors"), framework="np", device="cpu") as f:
        for key in f.keys():
            detok_weights[key] = f.get_tensor(key)

    logger.info(f"Loaded {len(detok_weights)} audio detokenizer weights")

    # Build the model using AudioDetokenizerBuilder
    builder = AudioDetokenizerBuilder(detok_config, detok_weights)
    model = builder.build()

    # Save the model
    output_path = onnx_dir / "audio_detokenizer.onnx"
    onnx.save_model(model, str(output_path))
    logger.info(f"audio_detokenizer saved to {output_path}")

    # Save ISTFT window for scipy
    if "istft.window" in detok_weights:
        window = detok_weights["istft.window"].astype(np.float32)
        np.save(str(onnx_dir / "istft_window.npy"), window)
        logger.info(f"ISTFT window saved to {onnx_dir / 'istft_window.npy'}")

    return output_path


# === Main Export ===


def export_full_model(
    model_path: str, output_dir: pathlib.Path, export_audio_encoder_flag: bool = True
):
    """Export all components of LFM2.5-Audio to ONNX."""
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_dir = output_dir / "onnx"
    onnx_dir.mkdir(exist_ok=True)

    # Load config and weights
    config = load_audio_config(model_path)
    weights = load_audio_model_weights(model_path)

    # Export builder-based components (no torch model needed)
    export_embed_tokens(weights, config, onnx_dir)
    export_audio_embedding(weights, config, onnx_dir)
    export_decoder(weights, config, onnx_dir)
    export_audio_lm_head(weights, config, onnx_dir)

    # Export torch-based components (require liquid_audio)
    pytorch_model = None
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        from liquid_audio import LFM2AudioModel

        logger.info(f"Loading PyTorch model for torch exports (device: {device})...")
        pytorch_model = LFM2AudioModel.from_pretrained(
            model_path, dtype=torch.float32, device=device
        )
        pytorch_model.eval()

        # Export audio encoder
        if export_audio_encoder_flag:
            with torch.no_grad():
                export_audio_encoder(pytorch_model, config, onnx_dir, device)

        # Export depthformer (with full transformer layers)
        with torch.no_grad():
            export_depthformer(pytorch_model, config, onnx_dir, device)

        # Export audio detokenizer neural network part
        with torch.no_grad():
            export_audio_detokenizer_lfm(pytorch_model, config, onnx_dir, device)
            save_istft_config(config, onnx_dir)

    except ImportError:
        logger.warning("=" * 60)
        logger.warning("liquid_audio package not available")
        logger.warning("  - audio_encoder.onnx will NOT be exported (ASR mode unavailable)")
        logger.warning("  - Using builder fallback for depthformer and audio_detokenizer")
        logger.warning("  - TTS and text modes will still work")
        logger.warning("To enable ASR: pip install liquid-audio")
        logger.warning("=" * 60)
        export_depthformer_from_weights(weights, config, onnx_dir)
    except Exception as e:
        logger.warning(f"Failed to load PyTorch model: {e}")
        logger.warning("Using builder fallback for depthformer")
        export_depthformer_from_weights(weights, config, onnx_dir)

    # Cleanup PyTorch model
    if pytorch_model is not None:
        del pytorch_model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    # Export audio detokenizer using builder (no liquid_audio runtime needed)
    try:
        export_audio_detokenizer_builder(model_path, onnx_dir)
        save_istft_config(config, onnx_dir)
    except Exception as e:
        logger.warning(f"Failed to export audio_detokenizer: {e}")

    # Export audio detokenizer using PyTorch/transformers (preferred, more accurate)
    # This creates audio_detokenizer_lfm.onnx which inference prefers over the builder version
    try:
        export_audio_detokenizer_pytorch(model_path, onnx_dir)
    except Exception as e:
        logger.warning(f"Failed to export audio_detokenizer_lfm: {e}")

    # Clean up
    weights.clear()
    gc.collect()

    # Copy config and tokenizer
    from huggingface_hub import hf_hub_download

    for filename in ["config.json", "tokenizer.json", "tokenizer_config.json"]:
        try:
            src = hf_hub_download(model_path, filename)
            shutil.copy(src, output_dir / filename)
        except Exception as e:
            logger.warning(f"Could not copy {filename}: {e}")

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("Export Summary")
    logger.info("=" * 60)
    total_size = 0
    for fpath in sorted(onnx_dir.iterdir()):
        if fpath.is_file():
            size = fpath.stat().st_size
            total_size += size
            logger.info(f"  {fpath.name}: {size / 1e6:.1f} MB")
    logger.info(f"Total: {total_size / 1e9:.2f} GB")

    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Full ONNX export for LFM2.5-Audio (all modes)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "model",
        help="HuggingFace model ID (e.g., LiquidAI/LFM2.5-Audio-1.5B)",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("."),
        help="Output base directory",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        help="Output folder name (default: {model-name}-ONNX-Full)",
    )
    parser.add_argument(
        "--precision",
        nargs="*",
        metavar="PRECISION",
        help="Output precisions: q4, q8 (default if no args)",
    )
    parser.add_argument(
        "--skip-audio-encoder",
        action="store_true",
        help="Skip audio encoder export (requires liquid_audio)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=32,
        help="Block size for quantization (default: 32)",
    )
    parser.add_argument(
        "--split-data",
        type=float,
        default=2.0,
        metavar="GB",
        help="Split external data into chunks (default: 2GB per chunk)",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    model_name = get_model_name(args.model)
    output_name = args.output_name or f"{model_name}-ONNX-Full"
    output_dir = args.output_dir / "exports" / output_name
    onnx_dir = output_dir / "onnx"

    logger.info("=" * 60)
    logger.info("Full ONNX Export for LFM2.5-Audio")
    logger.info("=" * 60)

    export_full_model(args.model, output_dir, not args.skip_audio_encoder)

    # Quantize
    quant_bits = []
    if args.precision is not None:
        if len(args.precision) == 0:
            quant_bits = [4, 8]
        else:
            for p in args.precision:
                p = p.lower()
                if p in ("q4", "q8"):
                    quant_bits.append(int(p[1]))

    for bits in quant_bits:
        logger.info("=" * 60)
        logger.info(f"Quantizing to Q{bits}")
        logger.info("=" * 60)
        do_quantize(onnx_dir, bits, args.block_size, symmetric=(bits == 4))

    # Split data
    chunk_size_bytes = int(args.split_data * 1024 * 1024 * 1024)
    for onnx_file in onnx_dir.glob("*.onnx"):
        data_file = onnx_file.with_suffix(".onnx_data")
        if data_file.exists() and data_file.stat().st_size > chunk_size_bytes:
            logger.info(f"Splitting {onnx_file.name}...")
            split_external_data(onnx_file, chunk_size=chunk_size_bytes)

    logger.info("=" * 60)
    logger.info("Export complete!")
    logger.info("=" * 60)
    logger.info(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
