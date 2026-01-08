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
        self.value_info: list[onnx.ValueInfoProto] = []
        self._initializer_names: set[str] = set()
        self._value_info_names: set[str] = set()
        self.weights: dict[str, np.ndarray] = {}
        self._node_count = 0
        self._constants: dict[tuple, str] = {}  # (value_bytes, dtype) -> name

    def _unique_name(self, prefix: str) -> str:
        """Generate unique node name."""
        self._node_count += 1
        return f"{prefix}_{self._node_count}"

    def _strip_output_suffix(self, name: str) -> str:
        """Strip /output_N suffix from tensor name to get node name."""
        if "/output_" in name:
            return name.rsplit("/output_", 1)[0]
        return name

    def _parent_path(self, path: str) -> str:
        """Get parent path by removing last component."""
        return path.rsplit("/", 1)[0] if "/" in path else path

    def _output_name(self, path: str, op_type: str, idx: int = 0) -> str:
        """Generate output name from logical path and op type.

        Args:
            path: Logical path (e.g., "/model/layers.0/input_layernorm")
            op_type: ONNX operator type (e.g., "LayerNormalization")
            idx: Output index (default 0)

        Returns:
            Output name like "/model/layers.0/input_layernorm/LayerNormalization/output_0"
        """
        return f"{path}/{op_type}/output_{idx}"

    def add_initializer(self, name: str, tensor: np.ndarray, dtype=None):
        """Add weight tensor as graph initializer.

        Args:
            name: Initializer name
            tensor: Weight tensor
            dtype: Override dtype (default: float32 for floats, preserve int types)

        Note: Skips if initializer with same name already exists.
        """
        if name in self._initializer_names:
            return
        self._initializer_names.add(name)

        if dtype is None:
            if tensor.dtype not in [np.int32, np.int64]:
                tensor = tensor.astype(np.float32)
        else:
            tensor = tensor.astype(dtype)
        self.initializers.append(numpy_helper.from_array(tensor, name))

    def add_value_info(
        self, name: str, elem_type: int, shape: list[int | str]
    ) -> None:
        """Add shape annotation for an intermediate tensor.

        Args:
            name: Tensor name
            elem_type: ONNX TensorProto element type (e.g., TensorProto.FLOAT)
            shape: Shape as list of ints (concrete) or strings (symbolic dims)
        """
        if name in self._value_info_names:
            return
        self._value_info_names.add(name)
        self.value_info.append(helper.make_tensor_value_info(name, elem_type, shape))

    def get_constant(self, value: int | float | list | np.ndarray, dtype=np.int64) -> str:
        """Get or create a shared constant via Constant node.

        Returns the output name of a Constant node. If the same value was already
        added, returns the existing output name (deduplication).

        Args:
            value: Scalar or array value
            dtype: NumPy dtype (default: int64)

        Returns:
            Constant node output name like "/model/constants/INT64/[2]"
        """
        arr = np.asarray(value, dtype=dtype)

        # Create cache key from bytes + dtype + shape
        key = (arr.tobytes(), str(arr.dtype), arr.shape)

        if key in self._constants:
            return self._constants[key]

        # Generate name matching community convention
        # Community: node="/model/constant_nodes/...", output="/model/constants/..."
        dtype_name = str(arr.dtype).upper().replace("FLOAT32", "FLOAT").replace("FLOAT64", "FLOAT")
        if arr.ndim == 0:
            value_str = str(arr.item())
        else:
            value_str = str(arr.tolist())
        output_name = f"/model/constants/{dtype_name}/{value_str}"

        # Add as initializer (matches community convention)
        self.add_initializer(output_name, arr)

        self._constants[key] = output_name
        return output_name

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
            name: Node name (derived from first output if None, stripping /output_N suffix)
            domain: Operator domain (empty for standard ops)
            **attrs: Operator attributes

        Returns:
            First output tensor name
        """
        if name is None:
            # Derive node name from first output, stripping /output_N suffix
            if outputs and outputs[0]:
                name = self._strip_output_suffix(outputs[0])
            else:
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

    def make_silu(self, input_name: str, path: str) -> str:
        """Create SiLU activation: x * sigmoid(x).

        Args:
            input_name: Input tensor name
            path: Logical path (e.g., "/model/layers.0/mlp/act_fn")

        Returns:
            Output name "{path}/Mul/output_0"
        """
        sigmoid_out = self.make_sigmoid(input_name, self._output_name(path, "Sigmoid"))
        return self.make_mul(input_name, sigmoid_out, self._output_name(path, "Mul"))

    def make_gelu(self, input_name: str, path: str, approximate: str = "tanh") -> str:
        """Create GELU activation.

        Args:
            input_name: Input tensor name
            path: Logical path (e.g., "/model/layers.0/mlp/act_fn")
            approximate: "tanh" for fast approximation, "none" for exact GELU

        Returns:
            Output name "{path}/Gelu/output_0"
        """
        return self.make_node("Gelu", [input_name], [self._output_name(path, "Gelu")], approximate=approximate)

    def make_layernorm(
        self,
        input_name: str,
        weight_name: str,
        bias_name: str | None,
        path: str,
        epsilon: float = 1e-5,
        name: str = None,
    ) -> str:
        """Create LayerNormalization node.

        Automatically selects SimplifiedLayerNormalization (no bias) or
        LayerNormalization (with bias) based on bias_name.

        Args:
            input_name: Input tensor
            weight_name: Scale weight
            bias_name: Bias (None for SimplifiedLayerNorm)
            path: Logical path (e.g., "/model/layers.0/input_layernorm")
            epsilon: Epsilon for numerical stability
            name: Override name in path (default: op type). Use "LayerNorm" for
                  community convention where SimplifiedLayerNormalization uses
                  shorter name in path.

        Returns:
            Output name "{path}/{name}/output_0"
        """
        if bias_name is None:
            op_type = "SimplifiedLayerNormalization"
            return self.make_node(
                op_type,
                inputs=[input_name, weight_name],
                outputs=[self._output_name(path, name or op_type)],
                epsilon=epsilon,
            )
        op_type = "LayerNormalization"
        return self.make_node(
            op_type,
            inputs=[input_name, weight_name, bias_name],
            outputs=[self._output_name(path, name or op_type)],
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
        self, input_name: str, n_elements: str, path: str, axis: int = 2
    ) -> str:
        """Slice last N elements along axis (dynamic N).

        Args:
            input_name: Input tensor name
            n_elements: Name of scalar tensor containing N
            path: Logical path (e.g., "/model/layers.0/conv/slice_last")
            axis: Axis to slice along

        Returns:
            Output name "{path}/Slice/output_0"
        """
        neg_n = self.make_mul(n_elements, self.get_constant(-1), self._output_name(path, "Mul"))
        start = self.make_unsqueeze(neg_n, self.get_constant([0]), self._output_name(path, "Unsqueeze"))

        return self.make_slice(
            input_name,
            start,
            self.get_constant([np.iinfo(np.int64).max]),
            self.get_constant([axis]),
            self._output_name(path, "Slice"),
        )

    def build_graph(
        self,
        name: str,
        opset_version: int = 21,
        ms_domain: bool = True,
        ir_version: int = 10,
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
            value_info=self.value_info,
        )

        opset_imports = [helper.make_opsetid("", opset_version)]
        if ms_domain:
            opset_imports.append(helper.make_opsetid("com.microsoft", 1))

        model = helper.make_model(graph, opset_imports=opset_imports, ir_version=ir_version)
        model.producer_name = producer_name

        return model
