#!/usr/bin/env python3
"""
Benchmark all LFM2 ONNX Q4 models.

Measures inference speed for Builder Q4 and Community Q4 quantized versions.

Usage:
    python benchmark_all.py
    python benchmark_all.py --models 350M 1.2B
    python benchmark_all.py --prompt "Hello world" --max-tokens 50
"""

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)

# ============================================================================
# MODEL PATHS - Edit these if your paths differ
# ============================================================================

BUILDER_Q4_MODELS = {
    "350M": "LFM2-350M-ONNX-builder-Q4-fp32head/onnx/model.onnx",
    "700M": "LFM2-700M-ONNX-builder-Q4-fp32head/onnx/model.onnx",
    "1.2B": "LFM2-1.2B-ONNX-builder-Q4-fp32head/onnx/model.onnx",
    "2.6B": "LFM2-2.6B-ONNX-builder-Q4-fp32head/onnx/model.onnx",
}

COMMUNITY_Q4_MODELS = {
    "350M": "/Users/ykhrustalev/workplace/models/onnx-community/LFM2-350M-ONNX/onnx/model_q4.onnx",
    "700M": "/Users/ykhrustalev/workplace/models/onnx-community/LFM2-700M-ONNX/onnx/model_q4.onnx",
    "1.2B": "/Users/ykhrustalev/workplace/models/onnx-community/LFM2-1.2B-ONNX/onnx/model_q4.onnx",
    "2.6B": "/Users/ykhrustalev/workplace/models/onnx-community/LFM2-2.6B-ONNX/onnx/model_q4.onnx",
}

TOKENIZER_PATHS = {
    "350M": "LiquidAI/LFM2-350M",
    "700M": "LiquidAI/LFM2-700M",
    "1.2B": "LiquidAI/LFM2-1.2B",
    "2.6B": "LiquidAI/LFM2-2.6B",
}

# ============================================================================


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
    model_size: str
    source: str  # "builder" or "community"
    file_size_mb: float
    load_time_s: float
    prefill_time_ms: float  # Time to process prompt
    tokens_per_second: float  # Generation speed
    total_tokens: int
    total_time_s: float


class ONNXBenchmark:
    """Benchmarks ONNX model inference speed."""

    def __init__(self):
        self.tokenizer = None
        self.current_tokenizer_path = None

    def load_tokenizer(self, model_path: str):
        """Load tokenizer."""
        if self.current_tokenizer_path == model_path:
            return

        from transformers import AutoTokenizer

        logger.info(f"Loading tokenizer: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.current_tokenizer_path = model_path

    def load_onnx_model(self, onnx_path: str):
        """Load ONNX model and return load time."""
        import onnxruntime as ort

        logger.info(f"Loading ONNX model: {onnx_path}")
        start = time.perf_counter()
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        load_time = time.perf_counter() - start
        return sess, load_time

    def prepare_inputs(self, prompt: str) -> Dict[str, np.ndarray]:
        """Prepare input tensors."""
        input_ids = self.tokenizer.encode(prompt, return_tensors="np")
        seq_len = input_ids.shape[1]

        return {
            "input_ids": input_ids.astype(np.int64),
            "attention_mask": np.ones((1, seq_len), dtype=np.int64),
            "position_ids": np.arange(seq_len, dtype=np.int64).reshape(1, -1),
        }

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
        model_size: str,
        onnx_path: str,
        tokenizer_path: str,
        source: str,
        prompt: str = "Hello, how are",
        max_tokens: int = 20,
    ) -> BenchmarkResult:
        """Benchmark a single model."""
        self.load_tokenizer(tokenizer_path)

        # Get file size
        file_size_mb = get_model_size_mb(onnx_path)

        # Load model
        sess, load_time = self.load_onnx_model(onnx_path)

        # Prepare inputs
        input_ids = self.tokenizer.encode(prompt)

        # Run generation
        generated, prefill_time, gen_time = self.run_generation(
            sess, input_ids, max_tokens
        )

        # Calculate metrics
        new_tokens = len(generated) - len(input_ids)
        tokens_per_second = new_tokens / gen_time if gen_time > 0 else 0

        return BenchmarkResult(
            model_size=model_size,
            source=source,
            file_size_mb=file_size_mb,
            load_time_s=load_time,
            prefill_time_ms=prefill_time,
            tokens_per_second=tokens_per_second,
            total_tokens=new_tokens,
            total_time_s=gen_time,
        )


def print_results(results: List[BenchmarkResult]):
    """Print benchmark results as a table."""
    print("\n" + "=" * 120)
    print("ONNX Q4 MODEL PERFORMANCE BENCHMARK")
    print("=" * 120)

    # Group by model size
    by_size = {}
    for r in results:
        if r.model_size not in by_size:
            by_size[r.model_size] = {}
        by_size[r.model_size][r.source] = r

    # Print header
    print(f"\n{'Model':<12} | {'Builder Q4':<50} | {'Community Q4':<50}")
    print(f"{'':<12} | {'Size MB':<10} {'Load(s)':<10} {'Prefill(ms)':<12} {'Tok/s':<10} | {'Size MB':<10} {'Load(s)':<10} {'Prefill(ms)':<12} {'Tok/s':<10}")
    print("-" * 120)

    # Print each model
    for size in ["350M", "700M", "1.2B", "2.6B"]:
        if size not in by_size:
            continue

        builder = by_size[size].get("builder")
        community = by_size[size].get("community")

        if builder and community:
            print(
                f"LFM2-{size:<6} | "
                f"{builder.file_size_mb:<10.1f} {builder.load_time_s:<10.2f} {builder.prefill_time_ms:<12.1f} {builder.tokens_per_second:<10.1f} | "
                f"{community.file_size_mb:<10.1f} {community.load_time_s:<10.2f} {community.prefill_time_ms:<12.1f} {community.tokens_per_second:<10.1f}"
            )
        elif builder:
            print(
                f"LFM2-{size:<6} | "
                f"{builder.file_size_mb:<10.1f} {builder.load_time_s:<10.2f} {builder.prefill_time_ms:<12.1f} {builder.tokens_per_second:<10.1f} | "
                f"{'N/A':<10} {'N/A':<10} {'N/A':<12} {'N/A':<10}"
            )
        elif community:
            print(
                f"LFM2-{size:<6} | "
                f"{'N/A':<10} {'N/A':<10} {'N/A':<12} {'N/A':<10} | "
                f"{community.file_size_mb:<10.1f} {community.load_time_s:<10.2f} {community.prefill_time_ms:<12.1f} {community.tokens_per_second:<10.1f}"
            )

    print("=" * 120)


def main():
    parser = argparse.ArgumentParser(description="Benchmark ONNX Q4 models")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["350M", "700M", "1.2B", "2.6B"],
        default=["350M", "700M", "1.2B", "2.6B"],
        help="Model sizes to benchmark",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Hello, how are",
        help="Prompt for generation",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=20,
        help="Maximum tokens to generate",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    benchmark = ONNXBenchmark()
    results = []

    for size in args.models:
        print(f"\n{'='*60}")
        print(f"BENCHMARKING LFM2-{size}")
        print(f"{'='*60}")

        tokenizer_path = TOKENIZER_PATHS[size]
        builder_path = BUILDER_Q4_MODELS[size]
        community_path = COMMUNITY_Q4_MODELS[size]

        # Benchmark Builder Q4
        print(f"\n--- Builder Q4 ---")
        try:
            result = benchmark.benchmark(
                size, builder_path, tokenizer_path, "builder",
                args.prompt, args.max_tokens
            )
            results.append(result)
            print(f"  Size: {result.file_size_mb:.1f} MB")
            print(f"  Load time: {result.load_time_s:.2f}s")
            print(f"  Prefill: {result.prefill_time_ms:.1f}ms")
            print(f"  Generation: {result.tokens_per_second:.1f} tok/s ({result.total_tokens} tokens)")
        except Exception as e:
            print(f"  ERROR: {e}")

        # Benchmark Community Q4
        print(f"\n--- Community Q4 ---")
        try:
            result = benchmark.benchmark(
                size, community_path, tokenizer_path, "community",
                args.prompt, args.max_tokens
            )
            results.append(result)
            print(f"  Size: {result.file_size_mb:.1f} MB")
            print(f"  Load time: {result.load_time_s:.2f}s")
            print(f"  Prefill: {result.prefill_time_ms:.1f}ms")
            print(f"  Generation: {result.tokens_per_second:.1f} tok/s ({result.total_tokens} tokens)")
        except Exception as e:
            print(f"  ERROR: {e}")

    # Print final comparison table
    if results:
        print_results(results)


if __name__ == "__main__":
    main()
