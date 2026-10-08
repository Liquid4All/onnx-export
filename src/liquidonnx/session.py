"""
onnxruntime sessions and decoder steps for the graphs of an export.

Generation runs on onnxruntime-genai (liquidonnx.genai_runtime); these helpers run single graphs,
for tests, the comparison against PyTorch and the audio detokenizer. Decoder caches follow the
onnxruntime-genai layout: past_key_values.N.key / .value and past.N.conv in, present.* out.
"""

import logging
import pathlib

import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)


SESSION_PROVIDERS = {
    "cpu": ["CPUExecutionProvider"],
    "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
}


def preload_cuda_libraries():
    """Load the CUDA and cuDNN libraries of the nvidia pip packages into the process.

    Otherwise the CUDA EP finds them only if torch has loaded them: onnxruntime-genai fails with
    libcublasLt.so.13 and onnxruntime sessions silently fall back to CPU.
    """
    # onnxruntime-gpu is a separate distribution, so the onnxruntime floor does not apply to it
    preload = getattr(ort, "preload_dlls", None)
    if preload:
        preload()


def load_onnx_session(
    path: pathlib.Path, ep: str = "cpu", tf32: bool = True, threads: int = 0
) -> ort.InferenceSession:
    """The graph at path on an execution provider of SESSION_PROVIDERS.

    onnxruntime runs on CPU when the CUDA EP fails to load, so a CUDA session without it fails.
    The CUDA EP runs fp32 matmuls and convolutions in TF32 unless tf32 is False. threads sets
    the intra-op thread pool (0: onnxruntime's default, one thread per physical core).
    """
    if not path.exists():
        raise FileNotFoundError(f"ONNX file not found: {path}")
    providers = SESSION_PROVIDERS[ep]
    provider = providers[0]
    if ep == "cuda":
        preload_cuda_libraries()
        if not tf32:
            providers = [(provider, {"use_tf32": "0"}), *providers[1:]]
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    logger.info(f"Loading {path.name} ({ep})...")
    session = ort.InferenceSession(str(path), sess_options=options, providers=providers)
    if provider not in session.get_providers():
        raise RuntimeError(
            f"onnxruntime {ort.__version__} ran {path.name} without {provider} (available: "
            f"{', '.join(ort.get_available_providers())}); CUDA needs onnxruntime-gpu and the "
            "CUDA and cuDNN libraries, see onnxruntime's log above"
        )
    return session


ONNX_TYPE_TO_NUMPY = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
}


def initialize_cache(session: ort.InferenceSession, batch_size: int = 1) -> dict:
    """Empty conv and KV caches (past_* inputs), typed as the graph declares them."""
    cache = {}
    for inp in session.get_inputs():
        if not inp.name.startswith("past"):
            continue
        shape = [d if isinstance(d, int) else 1 for d in inp.shape]
        for i, d in enumerate(inp.shape):
            if isinstance(d, str) and "sequence" in d.lower():
                shape[i] = 0
        shape[0] = batch_size
        dtype = ONNX_TYPE_TO_NUMPY.get(inp.type, np.float32)
        cache[inp.name] = np.zeros(shape, dtype=dtype)
    return cache


def cache_input_name(output_name: str) -> str | None:
    """Past-cache input fed by a present-cache output.

    present.N.conv -> past.N.conv, present.N.key -> past_key_values.N.key
    """
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


def decoder_inputs(inputs: np.ndarray, cache: dict, past_len: int) -> dict:
    """Feed for one decoder step: token ids [B, S] or embeddings [B, S, H], a full attention
    mask (the graph derives positions from it) and the cache."""
    batch, seq_len = inputs.shape[:2]
    feed = {"attention_mask": np.ones((batch, past_len + seq_len), dtype=np.int64)}
    if inputs.ndim == 3:
        feed["inputs_embeds"] = inputs.astype(np.float32)
    else:
        feed["input_ids"] = inputs.astype(np.int64)
    feed.update(cache)
    return feed


def cached_outputs(
    session: ort.InferenceSession,
    inputs: np.ndarray,
    prefill: int,
    names: tuple[str, ...] = ("logits",),
) -> dict[str, np.ndarray]:
    """Last-position outputs of a prefill of inputs[:, :prefill], then of one call per position.

    inputs is [1, S] token ids or [1, S, H] embeddings; each result has S - prefill + 1 fp32 rows,
    row i coming from position prefill + i - 1 (logits: predicting position prefill + i).
    """
    cache, outputs = initialize_cache(session), session.get_outputs()
    rows = {name: [] for name in names}
    for start, end in [(0, prefill), *((p, p + 1) for p in range(prefill, inputs.shape[1]))]:
        result = session.run(None, decoder_inputs(inputs[:, start:end], cache, start))
        update_cache(cache, result, outputs)
        named = dict(zip([o.name for o in outputs], result, strict=True))
        for name in names:
            rows[name].append(named[name][0, -1])
    return {name: np.stack(r).astype(np.float32) for name, r in rows.items()}
