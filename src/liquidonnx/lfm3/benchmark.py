#!/usr/bin/env python3
"""
Performance benchmark for a single LFM3 ONNX model.

Usage:
    uv run lfm3-bench --model LiquidAI/LFM3-1.7B --onnx exports/LFM3-1.7B-ONNX
    uv run lfm3-bench --model LiquidAI/LFM3-1.7B --onnx .../onnx/model.onnx
    uv run lfm3-bench --model LiquidAI/LFM3-1.7B --onnx model.onnx --max-tokens 50
"""

import argparse
import logging
import pathlib
import time
from dataclasses import dataclass

import numpy as np

from liquidonnx.lfm3.session import (
    CachedDecoder,
    default_ep,
    kv_feeds_attention,
    load_session,
    load_tokenizer,
)
from liquidonnx.quantize import get_total_model_size_mb

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result of benchmarking a model."""

    name: str
    file_size_mb: float
    load_time_s: float
    prefill_time_ms: float
    tokens_per_second: float
    total_tokens: int
    total_time_s: float
    generated_text: str


class ONNXBenchmark:
    """Benchmarks ONNX model inference speed."""

    def __init__(self, model_path: str, force_cpu: bool = False):
        self.model_path = model_path
        self.force_cpu = force_cpu
        self.tokenizer = None

    def load_tokenizer(self):
        """Load tokenizer."""
        logger.info(f"Loading tokenizer: {self.model_path}")
        self.tokenizer = load_tokenizer(self.model_path)

    def load_onnx_model(self, onnx_path: str):
        """Load ONNX model and return session + load time."""
        onnx_path = pathlib.Path(onnx_path)
        if onnx_path.suffix == ".onnx":
            model_file = onnx_path
        else:
            model_file = onnx_path / "onnx" / "model.onnx"
            if not model_file.exists():
                model_file = onnx_path / "model.onnx"

        start = time.perf_counter()
        sess = load_session(model_file, ep=default_ep(self.force_cpu))
        load_time = time.perf_counter() - start
        self.model_file = model_file

        return sess, load_time, model_file

    def run_generation(
        self,
        sess,
        input_ids: list[int],
        max_tokens: int,
        warmup: int = 2,
    ) -> tuple[list[int], float, float]:
        """Run generation and return tokens, prefill time, and total generation time."""
        share_kv = kv_feeds_attention(self.model_file)
        keep = {"logits_to_keep": np.array([1], dtype=np.int64)}

        # Warmup: a prefill and two decode steps, so one-time kernel setup isn't timed
        for _ in range(warmup):
            decoder = CachedDecoder(sess, len(input_ids) + 2 if share_kv else None)
            token = int(np.argmax(decoder.step(input_ids, **keep)[0, -1]))
            for _ in range(2):
                token = int(np.argmax(decoder.step([token], **keep)[0, -1]))

        generated = input_ids.copy()
        decoder = CachedDecoder(sess, len(input_ids) + max_tokens if share_kv else None)
        ids = input_ids
        prefill_time = 0.0
        generation_start = None
        for step in range(max_tokens):
            start = time.perf_counter()
            logits = decoder.step(ids, **keep)
            if step == 0:
                prefill_time = (time.perf_counter() - start) * 1000
                generation_start = time.perf_counter()
            next_token = int(np.argmax(logits[0, -1]))
            generated.append(next_token)
            ids = [next_token]
            if next_token == self.tokenizer.eos_token_id:
                break

        total_gen_time = time.perf_counter() - generation_start if generation_start else 0
        return generated, prefill_time, total_gen_time

    def benchmark(
        self,
        onnx_path: str,
        prompt: str = "Hello, how are",
        max_tokens: int = 20,
    ) -> BenchmarkResult:
        """Benchmark the model."""
        self.load_tokenizer()

        # Load model
        sess, load_time, model_file = self.load_onnx_model(onnx_path)

        # Get file size
        file_size_mb = get_total_model_size_mb(model_file)

        # Prepare inputs
        input_ids = self.tokenizer.encode(prompt)

        # Run generation
        generated, prefill_time, gen_time = self.run_generation(sess, input_ids, max_tokens)

        # Calculate metrics
        new_tokens = len(generated) - len(input_ids)
        tokens_per_second = new_tokens / gen_time if gen_time > 0 else 0
        generated_text = self.tokenizer.decode(generated)

        return BenchmarkResult(
            name=onnx_path,
            file_size_mb=file_size_mb,
            load_time_s=load_time,
            prefill_time_ms=prefill_time,
            tokens_per_second=tokens_per_second,
            total_tokens=new_tokens,
            total_time_s=gen_time,
            generated_text=generated_text,
        )


def print_result(result: BenchmarkResult):
    """Print benchmark result."""
    print("\n" + "=" * 70)
    print("BENCHMARK RESULT")
    print("=" * 70)
    print(f"Model: {result.name}")
    print(f"Size: {result.file_size_mb:.1f} MB")
    print(f"Load time: {result.load_time_s:.2f}s")
    print(f"Prefill: {result.prefill_time_ms:.1f}ms")
    print(
        f"Generation: {result.tokens_per_second:.1f} tok/s ({result.total_tokens} tokens in {result.total_time_s:.2f}s)"
    )
    print(f"\nGenerated: {result.generated_text}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Benchmark ONNX model performance")
    parser.add_argument(
        "--model", type=str, required=True, help="HuggingFace model path (for tokenizer)"
    )
    parser.add_argument("--onnx", type=str, required=True, help="ONNX model path or directory")
    parser.add_argument("--prompt", type=str, default="Hello, how are", help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=20, help="Max tokens to generate")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution (skip CUDA)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    benchmark = ONNXBenchmark(args.model, args.cpu)
    result = benchmark.benchmark(args.onnx, args.prompt, args.max_tokens)
    print_result(result)


if __name__ == "__main__":
    main()
