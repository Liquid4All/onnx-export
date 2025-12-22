#!/usr/bin/env python3
"""
Performance benchmark for a single LFM2 ONNX model.

Usage:
    python benchmark.py --model LiquidAI/LFM2-1.2B --onnx LFM2-1.2B-ONNX-builder
    python benchmark.py --model LiquidAI/LFM2-1.2B --onnx LFM2-1.2B-ONNX-builder/onnx/model.onnx
    python benchmark.py --model LiquidAI/LFM2-1.2B --onnx model.onnx --max-tokens 50
"""

import argparse
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)


def get_model_size_mb(onnx_path: str) -> float:
    """Get total model size in MB (including external data)."""
    path = Path(onnx_path)
    total = path.stat().st_size if path.exists() else 0

    # Check for external data files (different naming conventions)
    data_path = Path(str(path) + ".data")
    if data_path.exists():
        total += data_path.stat().st_size
    else:
        data_path = path.parent / (path.stem + ".onnx_data")
        if data_path.exists():
            total += data_path.stat().st_size

    return total / (1024 * 1024)


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

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.tokenizer = None

    def load_tokenizer(self):
        """Load tokenizer."""
        from transformers import AutoTokenizer

        logger.info(f"Loading tokenizer: {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )

    def load_onnx_model(self, onnx_path: str):
        """Load ONNX model and return session + load time."""
        import onnxruntime as ort

        # Resolve path
        if onnx_path.endswith(".onnx"):
            model_file = onnx_path
        else:
            model_file = os.path.join(onnx_path, "onnx", "model.onnx")
            if not os.path.exists(model_file):
                model_file = os.path.join(onnx_path, "model.onnx")

        if not os.path.exists(model_file):
            raise FileNotFoundError(f"ONNX model not found: {model_file}")

        logger.info(f"Loading ONNX model: {model_file}")
        start = time.perf_counter()
        sess = ort.InferenceSession(model_file, providers=["CPUExecutionProvider"])
        load_time = time.perf_counter() - start

        return sess, load_time, model_file

    def run_generation(
        self,
        sess,
        input_ids: List[int],
        max_tokens: int,
        warmup: int = 2,
    ) -> tuple[List[int], float, float]:
        """Run generation and return tokens, prefill time, and total generation time."""
        generated = input_ids.copy()

        # Check which inputs the model expects
        input_names = {inp.name for inp in sess.get_inputs()}
        has_position_ids = "position_ids" in input_names

        # Initialize caches
        cache = {}
        for inp in sess.get_inputs():
            if inp.name not in ["input_ids", "attention_mask", "position_ids"]:
                shape = [d if isinstance(d, int) else 1 for d in inp.shape]
                cache[inp.name] = np.zeros(shape, dtype=np.float32)

        outputs_info = sess.get_outputs()

        # Warmup runs
        for _ in range(warmup):
            ids = np.array([input_ids], dtype=np.int64)
            attn = np.ones((1, len(input_ids)), dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": attn}
            if has_position_ids:
                feed["position_ids"] = np.arange(len(input_ids), dtype=np.int64).reshape(1, -1)
            feed.update(cache)
            sess.run(None, feed)

        # Reset cache after warmup
        for inp in sess.get_inputs():
            if inp.name not in ["input_ids", "attention_mask", "position_ids"]:
                shape = [d if isinstance(d, int) else 1 for d in inp.shape]
                cache[inp.name] = np.zeros(shape, dtype=np.float32)

        prefill_time = 0.0
        generation_start = None

        for step in range(max_tokens):
            cur_len = len(generated)

            if step == 0:
                ids = np.array([generated], dtype=np.int64)
                pos = np.arange(cur_len, dtype=np.int64).reshape(1, -1)
            else:
                ids = np.array([[generated[-1]]], dtype=np.int64)
                pos = np.array([[cur_len - 1]], dtype=np.int64)

            attn_mask = np.ones((1, cur_len), dtype=np.int64)

            feed = {"input_ids": ids, "attention_mask": attn_mask}
            if has_position_ids:
                feed["position_ids"] = pos
            feed.update(cache)

            start = time.perf_counter()
            result = sess.run(None, feed)
            elapsed = time.perf_counter() - start

            if step == 0:
                prefill_time = elapsed * 1000  # Convert to ms
                generation_start = time.perf_counter()

            # Update caches
            for i, out_info in enumerate(outputs_info[1:], 1):
                out_name = out_info.name
                if "present_conv" in out_name:
                    cache_name = out_name.replace("present_conv", "past_conv")
                elif "present." in out_name:
                    cache_name = out_name.replace("present.", "past_key_values.")
                else:
                    continue
                if cache_name in cache:
                    cache[cache_name] = result[i]

            next_token = int(np.argmax(result[0][0, -1]))
            generated.append(next_token)

            if next_token == self.tokenizer.eos_token_id:
                break

        total_gen_time = time.perf_counter() - generation_start if generation_start else 0
        return generated, prefill_time, total_gen_time

    def benchmark(
        self,
        onnx_path: str,
        prompt: str = "The capital of France is",
        max_tokens: int = 20,
    ) -> BenchmarkResult:
        """Benchmark the model."""
        self.load_tokenizer()

        # Load model
        sess, load_time, model_file = self.load_onnx_model(onnx_path)

        # Get file size
        file_size_mb = get_model_size_mb(model_file)

        # Prepare inputs
        input_ids = self.tokenizer.encode(prompt)

        # Run generation
        generated, prefill_time, gen_time = self.run_generation(
            sess, input_ids, max_tokens
        )

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
    print(f"Generation: {result.tokens_per_second:.1f} tok/s ({result.total_tokens} tokens in {result.total_time_s:.2f}s)")
    print(f"\nGenerated: {result.generated_text}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Benchmark ONNX model performance")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model path (for tokenizer)")
    parser.add_argument("--onnx", type=str, required=True, help="ONNX model path or directory")
    parser.add_argument("--prompt", type=str, default="The capital of France is", help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=20, help="Max tokens to generate")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    benchmark = ONNXBenchmark(args.model)
    result = benchmark.benchmark(args.onnx, args.prompt, args.max_tokens)
    print_result(result)


if __name__ == "__main__":
    main()
