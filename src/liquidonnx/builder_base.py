"""
Base ONNX builder with shared utilities.

Provides common infrastructure for building ONNX graphs:
- Graph component management (nodes, inputs, outputs, initializers)
- Helper methods for common operations (MatMul, Add, LayerNorm, etc.)
- Weight management utilities

Used by:
- LFM2Builder (text model)
- VisionEmbedBuilder (vision encoder + projector)
- EmbedTokensBuilder (token embeddings)
"""

import logging

import numpy as np
import onnx
from onnx import helper, numpy_helper

logger = logging.getLogger(__name__)

SLICE_END = np.array([np.iinfo(np.int64).max], dtype=np.int64)


class ONNXBuilderBase:
    """
    Base class for ONNX model builders.

    Provides common utilities for building ONNX graphs:
    - Node creation with automatic naming
    - Initializer management
    - Helper methods for common operations
    """

    def __init__(self):
        self.nodes: list[onnx.NodeProto] = []
        self.inputs: list[onnx.ValueInfoProto] = []
        self.outputs: list[onnx.ValueInfoProto] = []
        self.initializers: list[onnx.TensorProto] = []
        self.weights: dict[str, np.ndarray] = {}
        self._node_count = 0

    def _unique_name(self, prefix: str) -> str:
        """Generate unique node name."""
        self._node_count += 1
        return f"{prefix}_{self._node_count}"

    def add_initializer(self, name: str, tensor: np.ndarray, dtype=None):
        """Add weight tensor as graph initializer.

        Args:
            name: Initializer name
            tensor: Weight tensor
            dtype: Override dtype (default: float32 for floats, preserve int types)
        """
        if dtype is None:
            if tensor.dtype not in [np.int32, np.int64]:
                tensor = tensor.astype(np.float32)
        else:
            tensor = tensor.astype(dtype)
        self.initializers.append(numpy_helper.from_array(tensor, name))

    def make_node(
        self,
        op_type: str,
        inputs: list[str],
        outputs: list[str],
        name: str = None,
        domain: str = "",
        **attrs,
    ) -> str:
        """Create an ONNX node and return the first output name.

        Args:
            op_type: ONNX operator type
            inputs: Input tensor names
            outputs: Output tensor names
            name: Node name (auto-generated if None)
            domain: Operator domain (empty for standard ops)
            **attrs: Operator attributes

        Returns:
            First output tensor name
        """
        if name is None:
            name = self._unique_name(op_type)

        node = helper.make_node(op_type, inputs, outputs, name=name, domain=domain, **attrs)
        self.nodes.append(node)
        return outputs[0] if outputs else None

    def make_matmul(self, a: str, b: str, output_name: str) -> str:
        """Create MatMul node: output = a @ b."""
        return self.make_node("MatMul", [a, b], [output_name])

    def make_add(self, a: str, b: str, output_name: str) -> str:
        """Create Add node: output = a + b."""
        return self.make_node("Add", [a, b], [output_name])

    def make_mul(self, a: str, b: str, output_name: str) -> str:
        """Create Mul node: output = a * b."""
        return self.make_node("Mul", [a, b], [output_name])

    def make_sigmoid(self, input_name: str, output_name: str) -> str:
        """Create Sigmoid node."""
        return self.make_node("Sigmoid", [input_name], [output_name])

    def make_silu(self, input_name: str, output_name: str) -> str:
        """Create SiLU activation: x * sigmoid(x)."""
        sigmoid_out = self.make_sigmoid(input_name, f"{output_name}_sigmoid")
        return self.make_mul(input_name, sigmoid_out, output_name)

    def make_gelu(self, input_name: str, output_name: str, approximate: str = "tanh") -> str:
        """Create GELU activation.

        Args:
            input_name: Input tensor name
            output_name: Output tensor name
            approximate: "tanh" for fast approximation, "none" for exact GELU

        Returns:
            Output tensor name
        """
        return self.make_node("Gelu", [input_name], [output_name], approximate=approximate)

    def make_layernorm(
        self,
        input_name: str,
        weight_name: str,
        bias_name: str | None,
        output_name: str,
        epsilon: float = 1e-5,
    ) -> str:
        """Create LayerNormalization node.

        Args:
            input_name: Input tensor
            weight_name: Scale weight
            bias_name: Bias (None for SimplifiedLayerNorm)
            output_name: Output tensor
            epsilon: Epsilon for numerical stability

        Returns:
            Output tensor name
        """
        if bias_name is None:
            return self.make_node(
                "SimplifiedLayerNormalization",
                inputs=[input_name, weight_name],
                outputs=[output_name],
                epsilon=epsilon,
            )
        return self.make_node(
            "LayerNormalization",
            inputs=[input_name, weight_name, bias_name],
            outputs=[output_name],
            epsilon=epsilon,
        )

    def make_reshape(self, input_name: str, shape_name: str, output_name: str) -> str:
        """Create Reshape node."""
        return self.make_node("Reshape", [input_name, shape_name], [output_name])

    def make_transpose(self, input_name: str, output_name: str, perm: list[int]) -> str:
        """Create Transpose node."""
        return self.make_node("Transpose", [input_name], [output_name], perm=perm)

    def make_gather(self, data: str, indices: str, output_name: str, axis: int = 0) -> str:
        """Create Gather node."""
        return self.make_node("Gather", [data, indices], [output_name], axis=axis)

    def make_concat(self, inputs: list[str], output_name: str, axis: int) -> str:
        """Create Concat node."""
        return self.make_node("Concat", inputs, [output_name], axis=axis)

    def make_unsqueeze(self, input_name: str, axes_name: str, output_name: str) -> str:
        """Create Unsqueeze node."""
        return self.make_node("Unsqueeze", [input_name, axes_name], [output_name])

    def make_slice(
        self,
        input_name: str,
        starts: str,
        ends: str,
        axes: str,
        output_name: str,
        steps: str = None,
    ) -> str:
        """Create Slice node."""
        inputs = [input_name, starts, ends, axes]
        if steps:
            inputs.append(steps)
        return self.make_node("Slice", inputs, [output_name])

    def make_linear(
        self,
        input_name: str,
        weight: np.ndarray,
        weight_name: str,
        output_prefix: str,
        bias: np.ndarray = None,
        bias_name: str = None,
        transpose_weight: bool = True,
    ) -> str:
        """Create linear projection: output = input @ weight + bias.

        Args:
            input_name: Input tensor name
            weight: Weight matrix
            weight_name: Initializer name for weight
            output_prefix: Prefix for output names
            bias: Optional bias vector
            bias_name: Initializer name for bias
            transpose_weight: Transpose weight from [out, in] to [in, out]

        Returns:
            Output tensor name
        """
        if transpose_weight:
            weight = weight.T
        self.add_initializer(weight_name, weight)

        matmul_out = self.make_matmul(input_name, weight_name, f"{output_prefix}/matmul")

        if bias is not None and bias_name is not None:
            self.add_initializer(bias_name, bias)
            return self.make_add(matmul_out, bias_name, f"{output_prefix}/out")

        return matmul_out

    def make_slice_last_n(
        self, input_name: str, n_elements: str, output_name: str, axis: int = 2
    ) -> str:
        """Slice last N elements along axis (dynamic N).

        Args:
            input_name: Input tensor name
            n_elements: Name of scalar tensor containing N
            output_name: Output tensor name
            axis: Axis to slice along

        Returns:
            Output tensor name
        """
        prefix = f"{output_name}_slice"
        self.add_initializer(f"{prefix}/neg1", np.array(-1, dtype=np.int64))
        neg_n = self.make_mul(n_elements, f"{prefix}/neg1", f"{prefix}/neg_n")

        self.add_initializer(f"{prefix}/axes0", np.array([0], dtype=np.int64))
        start = self.make_unsqueeze(neg_n, f"{prefix}/axes0", f"{prefix}/start")

        self.add_initializer(f"{prefix}/end", SLICE_END.copy())
        self.add_initializer(f"{prefix}/axis", np.array([axis], dtype=np.int64))

        return self.make_slice(input_name, start, f"{prefix}/end", f"{prefix}/axis", output_name)

    def build_graph(
        self,
        name: str,
        opset_version: int = 21,
        ms_domain: bool = True,
        ir_version: int = 9,
        producer_name: str = "liquidonnx",
    ) -> onnx.ModelProto:
        """Build the ONNX model from accumulated graph components.

        Args:
            name: Graph name
            opset_version: ONNX opset version
            ms_domain: Include com.microsoft domain
            ir_version: IR version
            producer_name: Producer name

        Returns:
            ONNX ModelProto
        """
        graph = helper.make_graph(
            self.nodes,
            name,
            self.inputs,
            self.outputs,
            self.initializers,
        )

        opset_imports = [helper.make_opsetid("", opset_version)]
        if ms_domain:
            opset_imports.append(helper.make_opsetid("com.microsoft", 1))

        model = helper.make_model(graph, opset_imports=opset_imports, ir_version=ir_version)
        model.producer_name = producer_name

        return model
