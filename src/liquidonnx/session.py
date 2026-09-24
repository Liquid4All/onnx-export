"""Shared ONNX inference utilities."""

import json
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


def cache_input_name(output_name: str) -> str | None:
    """Past-cache input fed by a present-cache output, for both decoder layouts.

    onnx-community:      present_conv.N -> past_conv.N, present.N.key -> past_key_values.N.key
    onnxruntime-genai:   present.N.conv -> past.N.conv, present.N.key -> past_key_values.N.key
    """
    if output_name.startswith("present_conv."):
        return output_name.replace("present_conv.", "past_conv.", 1)
    if output_name.startswith("present.") and output_name.endswith(".conv"):
        return output_name.replace("present.", "past.", 1)
    if output_name.startswith("present."):
        return output_name.replace("present.", "past_key_values.", 1)
    return None


def update_cache(cache: dict, outputs: list, output_infos: list) -> None:
    for out_info, value in zip(output_infos, outputs, strict=True):
        cache_name = cache_input_name(out_info.name)
        if cache_name in cache:
            cache[cache_name] = value


def decoder_inputs(
    session: ort.InferenceSession, inputs: np.ndarray, cache: dict, past_len: int
) -> dict:
    """Feed for one decoder step: token ids [B, S] or embeddings [B, S, H], a full attention
    mask, positions if the graph takes them (genai graphs derive them from the mask), and the
    cache."""
    names = {inp.name for inp in session.get_inputs()}
    batch, seq_len = inputs.shape[:2]
    feed = {"attention_mask": np.ones((batch, past_len + seq_len), dtype=np.int64)}
    if inputs.ndim == 3:
        feed["inputs_embeds"] = inputs.astype(np.float32)
    else:
        feed["input_ids"] = inputs.astype(np.int64)
    if "position_ids" in names:
        positions = np.arange(past_len, past_len + seq_len, dtype=np.int64)
        feed["position_ids"] = np.broadcast_to(positions, (batch, seq_len)).copy()
    feed.update(cache)
    return feed


def default_decoder(model_dir: pathlib.Path) -> pathlib.Path:
    """Decoder of an export folder: genai_config.json's choice, else the legacy file names."""
    genai_config = model_dir / "genai_config.json"
    if genai_config.exists():
        return model_dir / json.loads(genai_config.read_text())["model"]["decoder"]["filename"]
    for name in ("decoder_model_merged.onnx", "decoder.onnx", "model.onnx"):
        if (model_dir / "onnx" / name).exists():
            return model_dir / "onnx" / name
    return model_dir / "onnx" / "model.onnx"


class ONNXTextModel:
    """Shared ONNX inference for LFM2 text models (dense and MoE)."""

    def __init__(self, model_path: str, force_cpu: bool = False):
        self.model_path = pathlib.Path(model_path)
        self.tokenizer = None
        self.session = None
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
            onnx_path = default_decoder(self.model_path)

        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=True)

        providers = ["CPUExecutionProvider"] if self.force_cpu else None
        logger.info(f"Loading ONNX from {onnx_path}...")
        self.session = load_onnx_session(onnx_path, providers=providers)

        logger.info(f"Model loaded. Inputs: {len(self.session.get_inputs())} tensors")

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

        generated_tokens = []
        past_len = 0

        for step in range(max_new_tokens):
            ids = input_ids if step == 0 else np.array([[generated_tokens[-1]]], dtype=np.int64)
            feed = decoder_inputs(self.session, ids, cache, past_len)
            past_len += ids.shape[1]

            outputs = self.session.run(None, feed)
            logits = outputs[0][0, -1]

            next_token = int(np.argmax(logits))
            generated_tokens.append(next_token)

            update_cache(cache, outputs, output_infos)

            if stream:
                token_str = self.tokenizer.decode([next_token])
                print(token_str, end="", flush=True)

            if next_token == self.tokenizer.eos_token_id:
                break

        if stream:
            print()

        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)


class GenaiTextModel:
    """ONNXTextModel's interface on onnxruntime-genai, driven by the export's genai_config.json."""

    def __init__(self, model_path: str):
        self.model_path = pathlib.Path(model_path)
        self.tokenizer = None
        self.model = None

    def load(self):
        """Load tokenizer and the onnxruntime-genai model (a .onnx path picks the precision)."""
        import onnxruntime_genai as og
        from transformers import AutoTokenizer

        if self.model_path.suffix == ".onnx":
            model_dir = self.model_path.parent.parent
            decoder = self.model_path.resolve().relative_to(model_dir.resolve()).as_posix()
            config = og.Config(str(model_dir))
            config.overlay(json.dumps({"model": {"decoder": {"filename": decoder}}}))
        else:
            model_dir = self.model_path
            config = og.Config(str(model_dir))

        logger.info(f"Loading {model_dir} with onnxruntime-genai {og.__version__}...")
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
        self.model = og.Model(config)

    def generate(self, messages: list, max_new_tokens: int = 100, stream: bool = True) -> str:
        import onnxruntime_genai as og

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)

        params = og.GeneratorParams(self.model)
        params.set_search_options(do_sample=False, max_length=len(input_ids) + max_new_tokens)
        generator = og.Generator(self.model, params)
        generator.append_tokens(np.array(input_ids, dtype=np.int32))

        generated_tokens = []
        while not generator.is_done():
            generator.generate_next_token()
            next_token = int(generator.get_next_tokens()[0])
            generated_tokens.append(next_token)
            if stream:
                print(self.tokenizer.decode([next_token]), end="", flush=True)

        if stream:
            print()

        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)


def run_chat_loop(model: ONNXTextModel | GenaiTextModel, args) -> None:
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
