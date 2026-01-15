"""
FastConformer encoder builder for ONNX export.

The FastConformer architecture processes mel-spectrograms through:
1. Subsampling: Reduces temporal resolution (factor 8)
2. Conformer blocks: Self-attention + convolution + feed-forward

Each Conformer block has:
    x → FFN1 (half) → MHA → Conv → FFN2 (half) → LayerNorm → out
    with residual connections

Note: This is a simplified export that removes dropout and uses
standard attention instead of relative position attention for
ONNX compatibility. The adapter MLP is included at the end.
"""

import logging

import numpy as np
import onnx
from onnx import TensorProto, helper

from liquidonnx.builder_base import ONNXBuilderBase

from .config import ConformerConfig

logger = logging.getLogger(__name__)


class ConformerEncoderBuilder(ONNXBuilderBase):
    """Builds ONNX graph for FastConformer encoder + adapter."""

    def __init__(self, config: ConformerConfig, adapter_output_dim: int = 2048):
        super().__init__()
        self.config = config
        self.adapter_output_dim = adapter_output_dim

    def build_inputs(self):
        """Build graph inputs for mel-spectrogram."""
        # Input: mel-spectrogram [batch, time, features]
        self.inputs.append(
            helper.make_tensor_value_info(
                "mel_spectrogram",
                TensorProto.FLOAT,
                ["batch_size", "time_steps", self.config.feat_in],
            )
        )
        # Length of each sequence in the batch
        self.inputs.append(
            helper.make_tensor_value_info("mel_lengths", TensorProto.INT64, ["batch_size"])
        )

    def build_outputs(self):
        """Build graph outputs for audio embeddings."""
        # Output: audio embeddings [batch, reduced_time, hidden]
        self.outputs.append(
            helper.make_tensor_value_info(
                "audio_embeddings",
                TensorProto.FLOAT,
                ["batch_size", "reduced_time", self.adapter_output_dim],
            )
        )
        # Output lengths after subsampling
        self.outputs.append(
            helper.make_tensor_value_info("audio_lengths", TensorProto.INT64, ["batch_size"])
        )

    def build_subsampling(self, input_name: str) -> str:
        """Build depthwise-striding subsampling layer.

        Subsampling reduces temporal resolution by factor of 8:
            [B, T, 128] → [B, T//8, 512]

        Architecture (pre_encode with depthwise separable convs):
            [B, T, 128] → reshape [B, 1, T, 128]
            → DepthwiseConv(256, k=3, s=2) → ReLU
            → DepthwiseConv(256, k=3, s=2) → PointwiseConv(256) → ReLU
            → DepthwiseConv(256, k=3, s=2) → PointwiseConv(256) → ReLU
            → reshape [B, T//8, 256*F'] → Linear(d_model)

        Weight mapping:
            conformer.pre_encode.conv.0 → depthwise conv (stride 2)
            conformer.pre_encode.conv.2 → depthwise conv (stride 2)
            conformer.pre_encode.conv.3 → pointwise conv
            conformer.pre_encode.conv.5 → depthwise conv (stride 2)
            conformer.pre_encode.conv.6 → pointwise conv
            conformer.pre_encode.out → linear projection
        """
        prefix = "/encoder/pre_encode"
        C = self.config.subsampling_conv_channels  # 256

        # Reshape for Conv2d: [B, T, F] → [B, 1, T, F]
        reshaped = self.make_unsqueeze(
            input_name, self.get_constant([1]), f"{prefix}/Unsqueeze/output_0"
        )

        # Expand to C channels: [B, 1, T, F] → [B, C, T, F]
        # We need to tile the input to match the depthwise conv groups
        # The depthwise conv weight is [C, 1, 3, 3] meaning C groups, 1 channel each
        expanded = self.make_node(
            "Expand",
            [reshaped, self.get_constant([1, C, 1, 1])],
            [f"{prefix}/Expand/output_0"],
        )

        # === Block 1: Depthwise conv (stride 2) + ReLU ===
        conv0 = self.make_node(
            "Conv",
            [expanded, "encoder.pre_encode.conv.0.weight", "encoder.pre_encode.conv.0.bias"],
            [f"{prefix}/conv0/Conv/output_0"],
            kernel_shape=[3, 3],
            strides=[2, 2],
            pads=[1, 1, 1, 1],
            group=C,
        )
        relu0 = self.make_node("Relu", [conv0], [f"{prefix}/conv0/Relu/output_0"])

        # === Block 2: Depthwise conv (stride 2) + Pointwise conv + ReLU ===
        conv2 = self.make_node(
            "Conv",
            [relu0, "encoder.pre_encode.conv.2.weight", "encoder.pre_encode.conv.2.bias"],
            [f"{prefix}/conv2/Conv/output_0"],
            kernel_shape=[3, 3],
            strides=[2, 2],
            pads=[1, 1, 1, 1],
            group=C,
        )
        conv3 = self.make_node(
            "Conv",
            [conv2, "encoder.pre_encode.conv.3.weight", "encoder.pre_encode.conv.3.bias"],
            [f"{prefix}/conv3/Conv/output_0"],
            kernel_shape=[1, 1],
            strides=[1, 1],
        )
        relu3 = self.make_node("Relu", [conv3], [f"{prefix}/conv3/Relu/output_0"])

        # === Block 3: Depthwise conv (stride 2) + Pointwise conv + ReLU ===
        conv5 = self.make_node(
            "Conv",
            [relu3, "encoder.pre_encode.conv.5.weight", "encoder.pre_encode.conv.5.bias"],
            [f"{prefix}/conv5/Conv/output_0"],
            kernel_shape=[3, 3],
            strides=[2, 2],
            pads=[1, 1, 1, 1],
            group=C,
        )
        conv6 = self.make_node(
            "Conv",
            [conv5, "encoder.pre_encode.conv.6.weight", "encoder.pre_encode.conv.6.bias"],
            [f"{prefix}/conv6/Conv/output_0"],
            kernel_shape=[1, 1],
            strides=[1, 1],
        )
        relu6 = self.make_node("Relu", [conv6], [f"{prefix}/conv6/Relu/output_0"])

        # Reshape: [B, C, T', F'] → [B, T', C*F']
        self.make_node("Shape", [relu6], [f"{prefix}/Shape/output_0"])
        batch = self.make_node(
            "Gather",
            [f"{prefix}/Shape/output_0", self.get_constant(0)],
            [f"{prefix}/batch/output_0"],
            axis=0,
        )
        time = self.make_node(
            "Gather",
            [f"{prefix}/Shape/output_0", self.get_constant(2)],
            [f"{prefix}/time/output_0"],
            axis=0,
        )

        # Transpose: [B, C, T, F] → [B, T, C, F]
        transposed = self.make_transpose(relu6, f"{prefix}/Transpose/output_0", perm=[0, 2, 1, 3])

        # Flatten last two dims: [B, T, C*F]
        new_shape = self.make_concat(
            [
                self.make_unsqueeze(batch, self.get_constant([0]), f"{prefix}/batch_u/output_0"),
                self.make_unsqueeze(time, self.get_constant([0]), f"{prefix}/time_u/output_0"),
                self.get_constant([-1]),
            ],
            f"{prefix}/new_shape/output_0",
            axis=0,
        )
        flattened = self.make_reshape(transposed, new_shape, f"{prefix}/Reshape/output_0")

        # Linear projection to d_model
        return self.make_linear(
            flattened,
            self.weights["conformer.pre_encode.out.weight"],
            "encoder.pre_encode.out.weight",
            f"{prefix}/out",
            bias=self.weights["conformer.pre_encode.out.bias"],
            bias_name="encoder.pre_encode.out.bias",
        )

    def build_conformer_block(self, layer_idx: int, hidden_state: str) -> str:
        """Build a single Conformer block.

        Structure:
            x → FFN1(half residual) → Self-Attn → Conv → FFN2(half residual) → LayerNorm
        """
        prefix = f"/encoder/layers.{layer_idx}"

        # === Feed-forward 1 (half residual) ===
        ffn1_out = self.build_feed_forward(hidden_state, layer_idx, "feed_forward1")
        # Half residual: x + 0.5 * ffn1_out
        half_const = self.get_constant(0.5, dtype=np.float32)
        ffn1_scaled = self.make_mul(ffn1_out, half_const, f"{prefix}/ffn1/Mul/output_0")
        hidden_state = self.make_add(hidden_state, ffn1_scaled, f"{prefix}/ffn1/Add/output_0")

        # === Self-Attention (simplified, no relative position) ===
        attn_out = self.build_self_attention(hidden_state, layer_idx)
        hidden_state = self.make_add(hidden_state, attn_out, f"{prefix}/attn/Add/output_0")

        # === Convolution module ===
        conv_out = self.build_conv_module(hidden_state, layer_idx)
        hidden_state = self.make_add(hidden_state, conv_out, f"{prefix}/conv/Add/output_0")

        # === Feed-forward 2 (half residual) ===
        ffn2_out = self.build_feed_forward(hidden_state, layer_idx, "feed_forward2")
        ffn2_scaled = self.make_mul(ffn2_out, half_const, f"{prefix}/ffn2/Mul/output_0")
        hidden_state = self.make_add(hidden_state, ffn2_scaled, f"{prefix}/ffn2/Add/output_0")

        # === Final LayerNorm ===
        return self.make_layernorm(
            hidden_state,
            f"encoder.layers.{layer_idx}.norm_out.weight",
            f"encoder.layers.{layer_idx}.norm_out.bias",
            f"{prefix}/norm_out",
        )

    def build_feed_forward(self, hidden_state: str, layer_idx: int, name: str) -> str:
        """Build feed-forward module: LayerNorm → Linear → SiLU → Linear."""
        prefix = f"/encoder/layers.{layer_idx}/{name}"
        weight_prefix = f"conformer.layers.{layer_idx}.{name}"

        # LayerNorm
        normed = self.make_layernorm(
            hidden_state,
            f"encoder.layers.{layer_idx}.norm_{name}.weight",
            f"encoder.layers.{layer_idx}.norm_{name}.bias",
            f"{prefix}/norm",
        )

        # Linear 1 (expand)
        linear1 = self.make_linear(
            normed,
            self.weights[f"{weight_prefix}.linear1.weight"],
            f"encoder.layers.{layer_idx}.{name}.linear1.weight",
            f"{prefix}/linear1",
            bias=self.weights[f"{weight_prefix}.linear1.bias"],
            bias_name=f"encoder.layers.{layer_idx}.{name}.linear1.bias",
        )

        # SiLU activation
        silu = self.make_silu(linear1, f"{prefix}/act")

        # Linear 2 (project back)
        return self.make_linear(
            silu,
            self.weights[f"{weight_prefix}.linear2.weight"],
            f"encoder.layers.{layer_idx}.{name}.linear2.weight",
            f"{prefix}/linear2",
            bias=self.weights[f"{weight_prefix}.linear2.bias"],
            bias_name=f"encoder.layers.{layer_idx}.{name}.linear2.bias",
        )

    def build_self_attention(self, hidden_state: str, layer_idx: int) -> str:
        """Build self-attention module (simplified without relative position)."""
        prefix = f"/encoder/layers.{layer_idx}/self_attn"
        weight_prefix = f"conformer.layers.{layer_idx}.self_attn"
        d_model = self.config.d_model
        n_heads = self.config.n_heads
        head_dim = d_model // n_heads

        # LayerNorm
        normed = self.make_layernorm(
            hidden_state,
            f"encoder.layers.{layer_idx}.norm_self_att.weight",
            f"encoder.layers.{layer_idx}.norm_self_att.bias",
            f"{prefix}/norm",
        )

        # Q, K, V projections
        q = self.make_linear(
            normed,
            self.weights[f"{weight_prefix}.linear_q.weight"],
            f"encoder.layers.{layer_idx}.self_attn.q.weight",
            f"{prefix}/q_proj",
            bias=self.weights[f"{weight_prefix}.linear_q.bias"],
            bias_name=f"encoder.layers.{layer_idx}.self_attn.q.bias",
        )
        k = self.make_linear(
            normed,
            self.weights[f"{weight_prefix}.linear_k.weight"],
            f"encoder.layers.{layer_idx}.self_attn.k.weight",
            f"{prefix}/k_proj",
            bias=self.weights[f"{weight_prefix}.linear_k.bias"],
            bias_name=f"encoder.layers.{layer_idx}.self_attn.k.bias",
        )
        v = self.make_linear(
            normed,
            self.weights[f"{weight_prefix}.linear_v.weight"],
            f"encoder.layers.{layer_idx}.self_attn.v.weight",
            f"{prefix}/v_proj",
            bias=self.weights[f"{weight_prefix}.linear_v.bias"],
            bias_name=f"encoder.layers.{layer_idx}.self_attn.v.bias",
        )

        # Reshape for multi-head attention: [B, T, D] → [B, T, H, D/H] → [B, H, T, D/H]
        reshape_const = self.get_constant([0, -1, n_heads, head_dim])
        q_4d = self.make_reshape(q, reshape_const, f"{prefix}/q_reshape/output_0")
        k_4d = self.make_reshape(k, reshape_const, f"{prefix}/k_reshape/output_0")
        v_4d = self.make_reshape(v, reshape_const, f"{prefix}/v_reshape/output_0")

        q_t = self.make_transpose(q_4d, f"{prefix}/q_transpose/output_0", perm=[0, 2, 1, 3])
        k_t = self.make_transpose(k_4d, f"{prefix}/k_transpose/output_0", perm=[0, 2, 1, 3])
        v_t = self.make_transpose(v_4d, f"{prefix}/v_transpose/output_0", perm=[0, 2, 1, 3])

        # Scaled dot-product attention
        scale = 1.0 / (head_dim**0.5)
        k_t_t = self.make_transpose(k_t, f"{prefix}/k_t_transpose/output_0", perm=[0, 1, 3, 2])
        scores = self.make_matmul(q_t, k_t_t, f"{prefix}/scores/output_0")
        scaled_scores = self.make_mul(
            scores, self.get_constant(scale, dtype=np.float32), f"{prefix}/scaled_scores/output_0"
        )
        attn_weights = self.make_node(
            "Softmax", [scaled_scores], [f"{prefix}/softmax/output_0"], axis=-1
        )
        attn_out = self.make_matmul(attn_weights, v_t, f"{prefix}/attn_out/output_0")

        # Reshape back: [B, H, T, D/H] → [B, T, H, D/H] → [B, T, D]
        attn_t = self.make_transpose(
            attn_out, f"{prefix}/attn_transpose/output_0", perm=[0, 2, 1, 3]
        )
        reshape_back = self.get_constant([0, -1, d_model])
        attn_flat = self.make_reshape(attn_t, reshape_back, f"{prefix}/attn_reshape/output_0")

        # Output projection
        return self.make_linear(
            attn_flat,
            self.weights[f"{weight_prefix}.linear_out.weight"],
            f"encoder.layers.{layer_idx}.self_attn.out.weight",
            f"{prefix}/out_proj",
            bias=self.weights[f"{weight_prefix}.linear_out.bias"],
            bias_name=f"encoder.layers.{layer_idx}.self_attn.out.bias",
        )

    def build_conv_module(self, hidden_state: str, layer_idx: int) -> str:
        """Build convolution module: LayerNorm → Conv1d (pointwise) → GLU → DepthConv → BN → SiLU → Conv1d."""
        prefix = f"/encoder/layers.{layer_idx}/conv"

        # LayerNorm
        normed = self.make_layernorm(
            hidden_state,
            f"encoder.layers.{layer_idx}.norm_conv.weight",
            f"encoder.layers.{layer_idx}.norm_conv.bias",
            f"{prefix}/norm",
        )

        # Transpose for Conv1d: [B, T, C] → [B, C, T]
        normed_t = self.make_transpose(normed, f"{prefix}/transpose1/output_0", perm=[0, 2, 1])

        # Pointwise conv 1 (expand to 2*d_model for GLU)
        pw1 = self.make_node(
            "Conv",
            [
                normed_t,
                f"encoder.layers.{layer_idx}.conv.pointwise_conv1.weight",
                f"encoder.layers.{layer_idx}.conv.pointwise_conv1.bias",
            ],
            [f"{prefix}/pw1/Conv/output_0"],
            kernel_shape=[1],
        )

        # GLU: split in half, sigmoid one half, multiply
        d_model = self.config.d_model
        split_const = self.get_constant([d_model, d_model])
        self.make_node(
            "Split",
            [pw1, split_const],
            [f"{prefix}/glu/Split/output_0", f"{prefix}/glu/Split/output_1"],
            axis=1,
        )
        glu_sigmoid = self.make_sigmoid(
            f"{prefix}/glu/Split/output_1", f"{prefix}/glu/Sigmoid/output_0"
        )
        glu_out = self.make_mul(
            f"{prefix}/glu/Split/output_0", glu_sigmoid, f"{prefix}/glu/Mul/output_0"
        )

        # Depthwise conv
        dw = self.make_node(
            "Conv",
            [
                glu_out,
                f"encoder.layers.{layer_idx}.conv.depthwise_conv.weight",
                f"encoder.layers.{layer_idx}.conv.depthwise_conv.bias",
            ],
            [f"{prefix}/dw/Conv/output_0"],
            kernel_shape=[self.config.conv_kernel_size],
            pads=[self.config.conv_kernel_size // 2, self.config.conv_kernel_size // 2],
            group=d_model,
        )

        # Batch normalization (inference mode)
        bn = self.make_node(
            "BatchNormalization",
            [
                dw,
                f"encoder.layers.{layer_idx}.conv.batch_norm.weight",
                f"encoder.layers.{layer_idx}.conv.batch_norm.bias",
                f"encoder.layers.{layer_idx}.conv.batch_norm.running_mean",
                f"encoder.layers.{layer_idx}.conv.batch_norm.running_var",
            ],
            [f"{prefix}/bn/BatchNormalization/output_0"],
            epsilon=1e-5,
        )

        # SiLU
        silu = self.make_silu(bn, f"{prefix}/act")

        # Pointwise conv 2 (project back)
        pw2 = self.make_node(
            "Conv",
            [
                silu,
                f"encoder.layers.{layer_idx}.conv.pointwise_conv2.weight",
                f"encoder.layers.{layer_idx}.conv.pointwise_conv2.bias",
            ],
            [f"{prefix}/pw2/Conv/output_0"],
            kernel_shape=[1],
        )

        # Transpose back: [B, C, T] → [B, T, C]
        return self.make_transpose(pw2, f"{prefix}/transpose2/output_0", perm=[0, 2, 1])

    def build_adapter(self, hidden_state: str) -> str:
        """Build adapter MLP: LayerNorm → Linear → Linear."""
        prefix = "/encoder/adapter"

        # LayerNorm (from audio_adapter.model.0)
        normed = self.make_layernorm(
            hidden_state,
            "encoder.adapter.norm.weight",
            "encoder.adapter.norm.bias",
            f"{prefix}/norm",
        )

        # Linear 1
        linear1 = self.make_linear(
            normed,
            self.weights["audio_adapter.model.1.weight"],
            "encoder.adapter.linear1.weight",
            f"{prefix}/linear1",
            bias=self.weights["audio_adapter.model.1.bias"],
            bias_name="encoder.adapter.linear1.bias",
        )

        # ReLU (implied by typical adapter design)
        relu = self.make_node("Relu", [linear1], [f"{prefix}/Relu/output_0"])

        # Linear 2
        return self.make_linear(
            relu,
            self.weights["audio_adapter.model.3.weight"],
            "encoder.adapter.linear2.weight",
            f"{prefix}/linear2",
            bias=self.weights["audio_adapter.model.3.bias"],
            bias_name="encoder.adapter.linear2.bias",
        )

    def build_length_output(self) -> str:
        """Compute output lengths after subsampling."""
        # Output length = input_length // subsampling_factor
        factor = self.get_constant(self.config.subsampling_factor)
        return self.make_node(
            "Div", ["mel_lengths", factor], ["audio_lengths"], name="/encoder/length_div"
        )

    def prepare_weights(self):
        """Register all weights as initializers."""
        # Pre-encode (subsampling) weights - depthwise separable convolutions
        # conv.0, conv.2, conv.5 are depthwise convs
        # conv.3, conv.6 are pointwise convs
        for idx in [0, 2, 3, 5, 6]:
            w_name = f"conformer.pre_encode.conv.{idx}.weight"
            b_name = f"conformer.pre_encode.conv.{idx}.bias"
            if w_name in self.weights:
                self.add_initializer(f"encoder.pre_encode.conv.{idx}.weight", self.weights[w_name])
                self.add_initializer(f"encoder.pre_encode.conv.{idx}.bias", self.weights[b_name])

        # Linear projection
        if "conformer.pre_encode.out.weight" in self.weights:
            self.add_initializer(
                "encoder.pre_encode.out.weight",
                self.weights["conformer.pre_encode.out.weight"].T,
            )
            self.add_initializer(
                "encoder.pre_encode.out.bias",
                self.weights["conformer.pre_encode.out.bias"],
            )

        # Conformer layer weights
        for layer_idx in range(self.config.n_layers):
            prefix = f"conformer.layers.{layer_idx}"
            out_prefix = f"encoder.layers.{layer_idx}"

            # Layer norms
            for norm_name in [
                "norm_feed_forward1",
                "norm_feed_forward2",
                "norm_self_att",
                "norm_conv",
                "norm_out",
            ]:
                w_name = f"{prefix}.{norm_name}.weight"
                b_name = f"{prefix}.{norm_name}.bias"
                if w_name in self.weights:
                    self.add_initializer(f"{out_prefix}.{norm_name}.weight", self.weights[w_name])
                    self.add_initializer(f"{out_prefix}.{norm_name}.bias", self.weights[b_name])

            # Feed-forward weights
            for ff_name in ["feed_forward1", "feed_forward2"]:
                for lin_name in ["linear1", "linear2"]:
                    w_name = f"{prefix}.{ff_name}.{lin_name}.weight"
                    b_name = f"{prefix}.{ff_name}.{lin_name}.bias"
                    if w_name in self.weights:
                        self.add_initializer(
                            f"{out_prefix}.{ff_name}.{lin_name}.weight",
                            self.weights[w_name].T,
                        )
                        self.add_initializer(
                            f"{out_prefix}.{ff_name}.{lin_name}.bias",
                            self.weights[b_name],
                        )

            # Attention weights (renamed for clarity)
            for proj in ["q", "k", "v"]:
                w_name = f"{prefix}.self_attn.linear_{proj}.weight"
                b_name = f"{prefix}.self_attn.linear_{proj}.bias"
                if w_name in self.weights:
                    self.add_initializer(
                        f"{out_prefix}.self_attn.{proj}.weight", self.weights[w_name].T
                    )
                    self.add_initializer(
                        f"{out_prefix}.self_attn.{proj}.bias", self.weights[b_name]
                    )
            w_name = f"{prefix}.self_attn.linear_out.weight"
            b_name = f"{prefix}.self_attn.linear_out.bias"
            if w_name in self.weights:
                self.add_initializer(f"{out_prefix}.self_attn.out.weight", self.weights[w_name].T)
                self.add_initializer(f"{out_prefix}.self_attn.out.bias", self.weights[b_name])

            # Conv module weights
            for conv_name in ["pointwise_conv1", "pointwise_conv2", "depthwise_conv"]:
                w_name = f"{prefix}.conv.{conv_name}.weight"
                b_name = f"{prefix}.conv.{conv_name}.bias"
                if w_name in self.weights:
                    self.add_initializer(
                        f"{out_prefix}.conv.{conv_name}.weight", self.weights[w_name]
                    )
                    self.add_initializer(
                        f"{out_prefix}.conv.{conv_name}.bias", self.weights[b_name]
                    )

            # Batch norm
            for bn_param in ["weight", "bias", "running_mean", "running_var"]:
                name = f"{prefix}.conv.batch_norm.{bn_param}"
                if name in self.weights:
                    self.add_initializer(
                        f"{out_prefix}.conv.batch_norm.{bn_param}", self.weights[name]
                    )

        # Adapter weights
        if "audio_adapter.model.0.weight" in self.weights:
            self.add_initializer(
                "encoder.adapter.norm.weight", self.weights["audio_adapter.model.0.weight"]
            )
            self.add_initializer(
                "encoder.adapter.norm.bias", self.weights["audio_adapter.model.0.bias"]
            )

    def load_weights(self, model_path: str):
        """Load weights from HuggingFace model."""
        from huggingface_hub import hf_hub_download
        from safetensors import safe_open

        logger.info(f"Loading conformer weights from {model_path}...")

        # Download safetensors file
        safetensors_path = hf_hub_download(model_path, "model.safetensors")

        with safe_open(safetensors_path, framework="np", device="cpu") as f:
            for key in f.keys():
                if key.startswith("conformer.") or key.startswith("audio_adapter."):
                    self.weights[key] = f.get_tensor(key)

        logger.info(f"Loaded {len(self.weights)} weights")

    def build(self, model_path: str) -> onnx.ModelProto:
        """Build the complete ONNX model for audio encoder."""
        logger.info("Building Conformer encoder ONNX model...")

        # Load weights
        self.load_weights(model_path)

        # Build graph structure
        self.build_inputs()
        self.build_outputs()

        # Prepare all weights as initializers
        self.prepare_weights()

        # Build subsampling
        hidden_state = self.build_subsampling("mel_spectrogram")

        # Build conformer layers
        for layer_idx in range(self.config.n_layers):
            logger.info(f"Building conformer layer {layer_idx}...")
            hidden_state = self.build_conformer_block(layer_idx, hidden_state)

        # Build adapter (projects to LFM2 hidden size)
        hidden_state = self.build_adapter(hidden_state)

        # Final output assignment
        self.make_node("Identity", [hidden_state], ["audio_embeddings"], name="/encoder/output")

        # Build length output
        self.build_length_output()

        model = self.build_graph("conformer_encoder")
        logger.info(f"Model built: {len(self.nodes)} nodes")
        return model
