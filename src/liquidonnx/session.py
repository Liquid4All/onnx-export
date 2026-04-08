"""Shared ONNX inference utilities."""

import logging
import pathlib

import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)


# Mapping from legacy component names to Transformers.js v4 names
_COMPONENT_ALIASES = {
    "embed_images": "vision_encoder",
    "decoder": "decoder_model_merged",
}


def get_onnx_file(
    onnx_dir: pathlib.Path, precision: str | None, name: str = "model"
) -> pathlib.Path:
    """Get ONNX file path for given precision.

    Args:
        onnx_dir: Directory containing ONNX files
        precision: None for fp32, "fp16", "q4", "q8"
        name: Base name of the model file (default: "model").
              Accepts legacy names (embed_images, decoder) and resolves
              to new names (vision_encoder, decoder_model_merged) with
              fallback to legacy if the new file doesn't exist.

    Returns:
        Path to the ONNX file (e.g., model.onnx, model_q4.onnx, decoder_fp16.onnx)
    """
    new_name = _COMPONENT_ALIASES.get(name)
    if new_name:
        suffix = f"_{precision}.onnx" if precision else ".onnx"
        new_path = onnx_dir / f"{new_name}{suffix}"
        if new_path.exists():
            return new_path
        # Fall back to legacy name
    if precision:
        return onnx_dir / f"{name}_{precision}.onnx"
    return onnx_dir / f"{name}.onnx"


_cuda_works = None  # Cache CUDA availability check


def get_providers() -> list[str]:
    """Get available execution providers, preferring CUDA if it works."""
    global _cuda_works

    available = ort.get_available_providers()
    if "CUDAExecutionProvider" not in available:
        return ["CPUExecutionProvider"]

    # Check if CUDA actually works (cuDNN available, etc.)
    if _cuda_works is None:
        try:
            # Create a minimal session to test CUDA
            import tempfile

            from onnx import TensorProto, helper

            # Minimal valid ONNX model (use IR version 8 for compatibility)
            X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1])
            Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1])
            node = helper.make_node("Identity", ["X"], ["Y"])
            graph = helper.make_graph([node], "test", [X], [Y])
            model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
            model.ir_version = 8

            with tempfile.NamedTemporaryFile(suffix=".onnx", delete=True) as f:
                import onnx

                onnx.save(model, f.name)
                ort.InferenceSession(f.name, providers=["CUDAExecutionProvider"])
            _cuda_works = True
            logger.info("CUDA execution provider available")
        except Exception as e:
            _cuda_works = False
            logger.info(f"CUDA not available, using CPU: {e}")

    if _cuda_works:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def load_onnx_session(
    path: pathlib.Path, providers: list[str] | None = None
) -> ort.InferenceSession:
    """Load ONNX model as inference session.

    Args:
        path: Path to ONNX model
        providers: Execution providers to use. If None, auto-detects (CUDA if available, else CPU).

    Returns:
        ONNX Runtime InferenceSession
    """
    if not path.exists():
        raise FileNotFoundError(f"ONNX file not found: {path}")
    if providers is None:
        providers = get_providers()

    # Try with preferred providers, fallback to CPU if CUDA fails
    try:
        logger.info(f"Loading {path.name} with {providers[0]}...")
        return ort.InferenceSession(str(path), providers=providers)
    except Exception as e:
        if "CUDAExecutionProvider" in providers and "CPUExecutionProvider" in providers:
            logger.warning(f"CUDA failed ({e}), falling back to CPU...")
            return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        raise


ONNX_TYPE_TO_NUMPY = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
}


def initialize_cache(session: ort.InferenceSession) -> dict:
    """Initialize KV cache tensors for an ONNX inference session.

    Automatically detects cache inputs (past_*) and initializes them with zeros.
    Infers dtype from the ONNX model input specification.
    """
    skip_inputs = {"input_ids", "inputs_embeds", "attention_mask", "position_ids"}
    cache = {}

    for inp in session.get_inputs():
        if inp.name in skip_inputs:
            continue
        shape = [d if isinstance(d, int) else 1 for d in inp.shape]
        for i, d in enumerate(inp.shape):
            if isinstance(d, str) and "sequence" in d.lower():
                shape[i] = 0
        dtype = ONNX_TYPE_TO_NUMPY.get(inp.type, np.float32)
        cache[inp.name] = np.zeros(shape, dtype=dtype)

    return cache


def update_cache(cache: dict, outputs: list, output_infos: list) -> None:
    """Update cache from model outputs.

    Handles present_conv -> past_conv and present. -> past_key_values. mappings.
    """
    for i, out_info in enumerate(output_infos[1:], 1):  # Skip logits
        name = out_info.name
        if "present_conv" in name:
            cache_name = name.replace("present_conv", "past_conv")
        elif "present." in name:
            cache_name = name.replace("present.", "past_key_values.")
        else:
            continue
        if cache_name in cache:
            cache[cache_name] = outputs[i]


class ONNXTextModel:
    """Shared ONNX inference for LFM2 text models (dense and MoE)."""

    def __init__(self, model_path: str, force_cpu: bool = False):
        self.model_path = pathlib.Path(model_path)
        self.tokenizer = None
        self.session = None
        self.input_names = set()
        self.force_cpu = force_cpu

    def load(self):
        """Load tokenizer and ONNX model."""
        from transformers import AutoTokenizer

        logger.info(f"Loading model from {self.model_path}...")

        # Handle both directory and direct ONNX file paths
        if self.model_path.suffix == ".onnx":
            onnx_path = self.model_path
            tokenizer_path = self.model_path.parent.parent
        else:
            tokenizer_path = self.model_path
            # Try decoder_model_merged.onnx first, then decoder.onnx (legacy), then model.onnx
            onnx_path = self.model_path / "onnx" / "decoder_model_merged.onnx"
            if not onnx_path.exists():
                onnx_path = self.model_path / "onnx" / "decoder.onnx"
            if not onnx_path.exists():
                onnx_path = self.model_path / "onnx" / "model.onnx"

        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=True)

        providers = ["CPUExecutionProvider"] if self.force_cpu else None
        logger.info(f"Loading ONNX from {onnx_path}...")
        self.session = load_onnx_session(onnx_path, providers=providers)

        self.input_names = {inp.name for inp in self.session.get_inputs()}
        logger.info(f"Model loaded. Inputs: {len(self.input_names)} tensors")

    def generate(
        self,
        messages: list,
        max_new_tokens: int = 100,
        stream: bool = True,
    ) -> str:
        """Generate response for chat messages."""
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = np.array(
            [self.tokenizer.encode(prompt, add_special_tokens=False)], dtype=np.int64
        )

        cache = initialize_cache(self.session)
        output_infos = self.session.get_outputs()

        seq_len = input_ids.shape[1]
        position_ids = np.arange(seq_len, dtype=np.int64).reshape(1, -1)

        generated_tokens = []
        cur_len = seq_len

        for step in range(max_new_tokens):
            if step == 0:
                ids = input_ids
                pos = position_ids
            else:
                ids = np.array([[generated_tokens[-1]]], dtype=np.int64)
                pos = np.array([[cur_len - 1]], dtype=np.int64)

            attn_mask = np.ones((1, cur_len), dtype=np.int64)

            feed = {"input_ids": ids, "attention_mask": attn_mask}
            if "position_ids" in self.input_names:
                feed["position_ids"] = pos
            feed.update(cache)

            outputs = self.session.run(None, feed)
            logits = outputs[0][0, -1]

            next_token = int(np.argmax(logits))
            generated_tokens.append(next_token)

            update_cache(cache, outputs, output_infos)
            cur_len += 1

            if stream:
                token_str = self.tokenizer.decode([next_token])
                print(token_str, end="", flush=True)

            if next_token == self.tokenizer.eos_token_id:
                break

        if stream:
            print()

        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)


def run_chat_loop(model: ONNXTextModel, args) -> None:
    """Run interactive chat loop."""
    print("\n" + "=" * 50)
    print("LFM2 Model - ONNX Inference")
    print("Type 'quit' or 'exit' to stop")
    print("=" * 50 + "\n")

    messages = []

    if args.prompt:
        messages.append({"role": "user", "content": args.prompt})
        print(f"User: {args.prompt}")
        print("Assistant: ", end="")
        response = model.generate(
            messages, max_new_tokens=args.max_tokens, stream=not args.no_stream
        )
        messages.append({"role": "assistant", "content": response})
        if args.no_stream:
            print(response)

    while True:
        try:
            user_input = input("\nUser: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break
        if user_input.lower() == "clear":
            messages = []
            print("Chat history cleared.")
            continue

        messages.append({"role": "user", "content": user_input})
        print("Assistant: ", end="")
        response = model.generate(
            messages, max_new_tokens=args.max_tokens, stream=not args.no_stream
        )
        messages.append({"role": "assistant", "content": response})
        if args.no_stream:
            print(response)
