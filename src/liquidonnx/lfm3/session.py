"""
onnxruntime generation for LFM3 exports.

onnxruntime-genai has no LFM3 model type, so LFM3 graphs run on plain onnxruntime with their own
cache names: past_conv.N / past_key_values.N.key / .value in, present_conv.N / present.N.* out.
"""

import logging
import pathlib

import numpy as np
import onnxruntime as ort

from liquidonnx.session import initialize_cache, load_onnx_session

logger = logging.getLogger(__name__)

CHECKPOINT_PATTERNS = ["*.json", "*.jinja", "*.py", "model*.safetensors"]

# Revisions the exporter and tests were validated on (README 3.1b); other models follow main
REVISIONS = {
    "LiquidAI/LFM3-1.7B": "40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd",
    "LiquidAI/LFM3-1.7B-Base": "545fc8f4c7b5fa9f405105274cd29c4f8f44b526",
}


def load_tokenizer(path: str | pathlib.Path):
    """The checkpoint's tokenizer, without loading its model config.

    AutoTokenizer otherwise builds the remote-code config, which transformers 5.3+ rejects for
    older LFM3 checkpoints ("sliding_window_attention"); tokenizer_config.json names the class.
    """
    from transformers import AutoTokenizer, PretrainedConfig

    return AutoTokenizer.from_pretrained(str(path), config=PretrainedConfig())


def get_onnx_file(onnx_dir: pathlib.Path, precision: str | None) -> pathlib.Path:
    """model.onnx for fp32 (precision None or "fp32"), model_{precision}.onnx otherwise."""
    if precision in (None, "fp32"):
        return onnx_dir / "model.onnx"
    return onnx_dir / f"model_{precision}.onnx"


# Preferred precisions per device: CUDA has no float32 GroupQueryAttention kernel, so it wants
# fp16 activations; the CPU runs int4/int8 MatMulNBits fastest with float32 activations.
PREFERRED = {
    "cuda": ("q4f16", "fp16", "q8f16", "q4", "q8", "fp32"),
    "cpu": ("q4", "q8", "q4f32", "fp32", "q4f16", "fp16", "q8f16"),
}


# CUDA provider options for LFM3 graphs. On SM 9.0+ GroupQueryAttention otherwise picks cuDNN
# SDPA, which builds a new cuDNN graph for every KV length, i.e. every decoded token; flash and
# memory-efficient attention don't. cuDNN's default exhaustive algorithm search re-benchmarks the
# depthwise convolutions for every new input shape.
CUDA_OPTIONS = {"sdpa_kernel": "3", "cudnn_conv_algo_search": "HEURISTIC"}


def load_session(path: pathlib.Path, ep: str = "cpu", tf32: bool = True) -> ort.InferenceSession:
    return load_onnx_session(path, ep=ep, tf32=tf32, cuda_options=CUDA_OPTIONS)


def cache_input_name(output: str) -> str | None:
    """present_conv.N -> past_conv.N, present.N.key -> past_key_values.N.key."""
    if output.startswith("present_conv."):
        return output.replace("present_conv.", "past_conv.", 1)
    if output.startswith("present."):
        return output.replace("present.", "past_key_values.", 1)
    return None


def update_cache(cache: dict, outputs: list, output_infos: list) -> None:
    for out_info, value in zip(output_infos, outputs, strict=True):
        cache_name = cache_input_name(out_info.name)
        if cache_name in cache:
            cache[cache_name] = value


def kv_feeds_attention(path: pathlib.Path) -> bool:
    """Whether every KV cache input goes straight into GroupQueryAttention (no Cast), so one
    buffer can be bound as both past and present."""
    import onnx

    graph = onnx.load(str(path), load_external_data=False).graph
    kv = {i.name for i in graph.input if i.name.startswith("past_key_values.")}
    consumers = {name: [] for name in kv}
    for node in graph.node:
        for name in node.input:
            if name in consumers:
                consumers[name].append(node.op_type)
    return bool(kv) and all(ops == ["GroupQueryAttention"] for ops in consumers.values())


class CachedDecoder:
    """The decoder steps of one generation, with the caches bound in place between steps.

    Caches stay where their consumers run: on the GPU in CUDA sessions, except float32 KV, whose
    GroupQueryAttention has no CUDA kernel and runs on the CPU. With a capacity (and KV that
    feeds GroupQueryAttention directly), each KV cache is one [B, kv_heads, capacity, head_dim]
    buffer bound as both past and present, so attention appends in place; on CUDA that also
    enables its XQA decode kernel.
    """

    def __init__(self, session: ort.InferenceSession, capacity: int | None = None):
        self.session = session
        self.capacity = capacity
        cuda = "CUDAExecutionProvider" in session.get_providers()
        self.outputs = [o.name for o in session.get_outputs()]
        self.inputs = {i.name for i in session.get_inputs()}
        self.cache, self.shared = {}, set()
        for name, array in initialize_cache(session).items():
            kv = name.startswith("past_key_values.")
            device = "cuda" if cuda and not (kv and array.dtype == np.float32) else "cpu"
            if kv and capacity:
                array = np.zeros((*array.shape[:2], capacity, array.shape[3]), dtype=array.dtype)
                self.shared.add(name)
            self.cache[name] = (ort.OrtValue.ortvalue_from_numpy(array, device, 0), device)
        self.past = 0

    def step(self, ids: list[int], **inputs: np.ndarray) -> np.ndarray:
        """logits of ids after the tokens so far; inputs are the graph's other inputs."""
        n = len(ids)
        if self.capacity and self.past + n > self.capacity:
            raise ValueError(f"{self.past + n} tokens exceed the cache capacity {self.capacity}")
        binding = self.session.io_binding()
        binding.bind_cpu_input("input_ids", np.array([ids], dtype=np.int64))
        binding.bind_cpu_input("attention_mask", np.ones((1, self.past + n), dtype=np.int64))
        positions = np.arange(self.past, self.past + n, dtype=np.int64)[None, :]
        binding.bind_cpu_input("position_ids", positions)
        for name, value in inputs.items():
            if name in self.inputs:
                binding.bind_cpu_input(name, value)
        for name, (value, _) in self.cache.items():
            binding.bind_ortvalue_input(name, value)
        for name in self.outputs:
            past = cache_input_name(name)
            if past in self.shared:
                binding.bind_ortvalue_output(name, self.cache[past][0])
            elif past in self.cache:
                binding.bind_output(name, self.cache[past][1])
            else:
                binding.bind_output(name, "cpu")
        self.session.run_with_iobinding(binding)
        values = binding.get_outputs()
        for name, value in zip(self.outputs, values, strict=True):
            past = cache_input_name(name)
            if past in self.cache and past not in self.shared:
                self.cache[past] = (value, self.cache[past][1])
        self.past += n
        # Every output of the step by name (OrtValues), e.g. an audio decoder's hidden_states
        self.last_outputs = dict(zip(self.outputs, values, strict=True))
        return self.last_outputs["logits"].numpy()


def default_ep(force_cpu: bool = False) -> str:
    if not force_cpu and "CUDAExecutionProvider" in ort.get_available_providers():
        return "cuda"
    return "cpu"


def log_speed(start: float, encoded: float, slots: list[tuple[str, float]]):
    """Encoder time (image or speech), prefill (to the first slot) and mean ms per text token and
    per speech frame of the CLIs (end-of-audio frames count in the slot after them)."""
    if not slots:
        return
    prefill = slots[0][1] - encoded
    message = f"Encoder {1000 * (encoded - start):.0f} ms, prefill {1000 * prefill:.0f} ms"
    for kind in ("token", "frame"):
        times = [t - slots[i - 1][1] for i, (k, t) in enumerate(slots) if i and k == kind]
        if times:
            message += f", {len(times)} {kind}s at {1000 * np.mean(times):.1f} ms"
    logger.info(message)


class ONNXTextModel:
    def __init__(self, model_path: str, force_cpu: bool = False, precision: str | None = None):
        """precision picks model_{precision}.onnx in an export folder; by default the first of
        PREFERRED for the device that the folder has."""
        self.model_path = pathlib.Path(model_path)
        self.tokenizer = None
        self.session = None
        self.input_names = set()
        self.force_cpu = force_cpu
        self.precision = precision

    def load(self):
        logger.info(f"Loading model from {self.model_path}...")

        if self.model_path.suffix == ".onnx":
            onnx_path = self.model_path
            tokenizer_path = self.model_path.parent.parent
        else:
            onnx_dir = self.model_path / "onnx"
            precision = self.precision or next(
                (
                    p
                    for p in PREFERRED[default_ep(self.force_cpu)]
                    if get_onnx_file(onnx_dir, p).exists()
                ),
                None,
            )
            onnx_path = get_onnx_file(onnx_dir, precision)
            tokenizer_path = self.model_path

        self.tokenizer = load_tokenizer(tokenizer_path)
        self.session = load_session(onnx_path, ep=default_ep(self.force_cpu))
        self.share_kv = kv_feeds_attention(onnx_path)
        self.input_names = {inp.name for inp in self.session.get_inputs()}
        logger.info(f"Model loaded. Inputs: {len(self.input_names)} tensors")

    def generate(
        self,
        messages: list,
        max_new_tokens: int = 100,
        stream: bool = True,
        enable_thinking: bool = True,
    ) -> str:
        """Greedy answer to chat messages."""
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
        )
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        capacity = len(ids) + max_new_tokens if self.share_kv else None
        decoder = CachedDecoder(self.session, capacity)

        generated_tokens = []
        for _ in range(max_new_tokens):
            logits = decoder.step(ids, logits_to_keep=np.array([1], dtype=np.int64))
            next_token = int(np.argmax(logits[0, -1]))
            generated_tokens.append(next_token)
            ids = [next_token]

            if stream:
                print(self.tokenizer.decode([next_token]), end="", flush=True)

            if next_token == self.tokenizer.eos_token_id:
                break

        if stream:
            print()

        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)


def run_chat_loop(model: ONNXTextModel, args) -> None:
    print("\n" + "=" * 50)
    print("LFM3 Model - ONNX Inference")
    print("Type 'quit' or 'exit' to stop")
    print("=" * 50 + "\n")

    messages = []

    if args.prompt:
        messages.append({"role": "user", "content": args.prompt})
        print(f"User: {args.prompt}")
        print("Assistant: ", end="")
        response = model.generate(
            messages,
            max_new_tokens=args.max_tokens,
            stream=not args.no_stream,
            enable_thinking=not args.no_think,
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
            messages,
            max_new_tokens=args.max_tokens,
            stream=not args.no_stream,
            enable_thinking=not args.no_think,
        )
        messages.append({"role": "assistant", "content": response})
        if args.no_stream:
            print(response)
