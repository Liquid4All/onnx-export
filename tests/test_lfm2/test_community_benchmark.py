"""
Benchmark comparison: local ONNX exports vs onnx-community versions.

Compares performance metrics: load time, prefill time, tokens/second.

Run with:
    uv run pytest tests/test_lfm2/test_community_benchmark.py -v -s
    uv run pytest tests/test_lfm2/test_community_benchmark.py -v -s -k "350M"

Set ONNX_COMMUNITY_DIR environment variable to the directory containing community models:
    export ONNX_COMMUNITY_DIR=/path/to/onnx-community
"""

import logging
import pathlib
import time
from dataclasses import dataclass

import numpy as np
import pytest
from helpers import get_community_onnx_dir, get_community_onnx_file, skip_if_missing

from liquidonnx.lfm2 import MODELS
from liquidonnx.lfm2.generate import get_onnx_dir
from liquidonnx.quantize import get_total_model_size_mb
from liquidonnx.session import get_onnx_file, initialize_cache, update_cache

logger = logging.getLogger(__name__)

QUANT_CONFIGS = [
    pytest.param(None, id="fp32"),
    pytest.param(4, id="q4"),
    pytest.param(8, id="q8"),
]

PREFILL_TOKENS = 256
DECODE_TOKENS = 100
WARMUP_RUNS = 2


@dataclass
class BenchmarkResult:
    name: str
    size_mb: float
    load_time_s: float
    prefill_ms: float
    decode_ms_per_token: float
    tokens_per_sec: float
    total_tokens: int


def run_benchmark(
    sess,
    tokenizer,
    input_ids: list[int],
    max_tokens: int,
    warmup: int = 2,
) -> tuple[int, float, float]:
    """Run generation benchmark. Returns (tokens_generated, prefill_ms, total_time_s)."""
    generated = input_ids.copy()

    input_names = {inp.name for inp in sess.get_inputs()}
    has_position_ids = "position_ids" in input_names
    cache = initialize_cache(sess)

    # Warmup
    for _ in range(warmup):
        ids = np.array([input_ids], dtype=np.int64)
        attn = np.ones((1, len(input_ids)), dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": attn}
        if has_position_ids:
            feed["position_ids"] = np.arange(len(input_ids), dtype=np.int64).reshape(1, -1)
        feed.update(cache)
        sess.run(None, feed)

    cache = initialize_cache(sess)

    prefill_ms = 0.0
    gen_start = None

    for step in range(max_tokens):
        cur_len = len(generated)

        if step == 0:
            ids = np.array([generated], dtype=np.int64)
            pos = np.arange(cur_len, dtype=np.int64).reshape(1, -1)
        else:
            ids = np.array([[generated[-1]]], dtype=np.int64)
            pos = np.array([[cur_len - 1]], dtype=np.int64)

        attn = np.ones((1, cur_len), dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": attn}
        if has_position_ids:
            feed["position_ids"] = pos
        feed.update(cache)

        start = time.perf_counter()
        result = sess.run(None, feed)
        elapsed = time.perf_counter() - start

        if step == 0:
            prefill_ms = elapsed * 1000
            gen_start = time.perf_counter()

        update_cache(cache, result, sess.get_outputs())

        next_token = int(np.argmax(result[0][0, -1]))
        generated.append(next_token)

        if next_token == tokenizer.eos_token_id:
            break

    total_time = time.perf_counter() - gen_start if gen_start else 0
    new_tokens = len(generated) - len(input_ids)
    return new_tokens, prefill_ms, total_time


def benchmark_model(
    onnx_path: pathlib.Path,
    tokenizer,
    name: str,
) -> BenchmarkResult:
    """Benchmark a single model."""
    import onnxruntime as ort

    size_mb = get_total_model_size_mb(onnx_path)

    # Load model
    load_start = time.perf_counter()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    load_time = time.perf_counter() - load_start

    # Generate input sequence of PREFILL_TOKENS length
    # Use a repeated pattern to reach target length
    base_text = "The quick brown fox jumps over the lazy dog. "
    input_ids = tokenizer.encode(base_text * 50)[:PREFILL_TOKENS]

    # Run benchmark
    tokens, prefill_ms, total_time = run_benchmark(
        sess, tokenizer, input_ids, DECODE_TOKENS, WARMUP_RUNS
    )

    tokens_per_sec = tokens / total_time if total_time > 0 else 0
    decode_ms_per_token = (total_time * 1000) / tokens if tokens > 0 else 0

    return BenchmarkResult(
        name=name,
        size_mb=size_mb,
        load_time_s=load_time,
        prefill_ms=prefill_ms,
        decode_ms_per_token=decode_ms_per_token,
        tokens_per_sec=tokens_per_sec,
        total_tokens=tokens,
    )


@pytest.mark.parametrize("model_tokenizer", MODELS.keys(), indirect=True)
@pytest.mark.parametrize("bits", QUANT_CONFIGS)
def test_benchmark_comparison(
    exports_dir: pathlib.Path,
    community_dir: pathlib.Path,
    model_tokenizer,
    bits: int,
):
    """Benchmark local vs community ONNX models."""
    size, tokenizer = model_tokenizer
    logger.info(f"Benchmarking {size}/q{bits}")

    # Check local export exists
    local_onnx_dir = get_onnx_dir(exports_dir, size)
    skip_if_missing(local_onnx_dir, "Local export not found")

    local_onnx_file = get_onnx_file(local_onnx_dir, bits)
    skip_if_missing(local_onnx_file, f"Local ONNX not found: {local_onnx_file.name}")

    # Check community export exists
    community_onnx_dir = get_community_onnx_dir(community_dir, size)
    skip_if_missing(community_onnx_dir, "Community export not found")

    community_onnx_file = get_community_onnx_file(community_onnx_dir, bits)
    skip_if_missing(community_onnx_file, f"Community ONNX not found: {community_onnx_file.name}")

    # Benchmark both
    local_result = benchmark_model(local_onnx_file, tokenizer, f"local-{size}-q{bits}")
    community_result = benchmark_model(community_onnx_file, tokenizer, f"community-{size}-q{bits}")

    # Log results
    logger.info(f"  Prefill: {PREFILL_TOKENS} tokens, Decode: {DECODE_TOKENS} tokens")
    logger.info("")
    logger.info(f"  {'Metric':<20} {'Local':>12} {'Community':>12} {'Winner':>12}")
    logger.info(f"  {'-' * 56}")

    # Size comparison
    size_winner = "LOCAL" if local_result.size_mb < community_result.size_mb else "COMMUNITY"
    if local_result.size_mb == community_result.size_mb:
        size_winner = "TIE"
    logger.info(
        f"  {'Size (MB)':<20} {local_result.size_mb:>12.1f} {community_result.size_mb:>12.1f} {size_winner:>12}"
    )

    # Load time comparison
    load_winner = (
        "LOCAL" if local_result.load_time_s < community_result.load_time_s else "COMMUNITY"
    )
    logger.info(
        f"  {'Load time (s)':<20} {local_result.load_time_s:>12.2f} {community_result.load_time_s:>12.2f} {load_winner:>12}"
    )

    # Prefill comparison
    prefill_winner = (
        "LOCAL" if local_result.prefill_ms < community_result.prefill_ms else "COMMUNITY"
    )
    logger.info(
        f"  {'Prefill (ms)':<20} {local_result.prefill_ms:>12.1f} {community_result.prefill_ms:>12.1f} {prefill_winner:>12}"
    )

    # Decode comparison (ms per token - lower is better)
    decode_winner = (
        "LOCAL"
        if local_result.decode_ms_per_token < community_result.decode_ms_per_token
        else "COMMUNITY"
    )
    logger.info(
        f"  {'Decode (ms/tok)':<20} {local_result.decode_ms_per_token:>12.1f} {community_result.decode_ms_per_token:>12.1f} {decode_winner:>12}"
    )

    # Tokens/sec comparison
    speed_winner = (
        "LOCAL" if local_result.tokens_per_sec > community_result.tokens_per_sec else "COMMUNITY"
    )
    logger.info(
        f"  {'Speed (tok/s)':<20} {local_result.tokens_per_sec:>12.1f} {community_result.tokens_per_sec:>12.1f} {speed_winner:>12}"
    )

    logger.info(f"  {'-' * 56}")

    # Overall winner (based on speed)
    logger.info(f"  Overall: {speed_winner} is faster")
