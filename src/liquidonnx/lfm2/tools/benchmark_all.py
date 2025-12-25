#!/usr/bin/env python3
"""
Benchmark all LFM2 ONNX quantized models.

Measures inference speed for Builder and Community quantized versions.

Usage:
    lfm2-bench-all
    lfm2-bench-all --models 350M 1.2B
    lfm2-bench-all --quant q4       # Only Q4
    lfm2-bench-all --quant q8       # Only Q8
    lfm2-bench-all --prompt "Hello world" --max-tokens 50
"""

import argparse
import logging
from dataclasses import dataclass

from liquidonnx.lfm2.benchmark import ONNXBenchmark

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

BUILDER_Q8_MODELS = {
    "350M": "LFM2-350M-ONNX-builder-Q8-fp32head/onnx/model.onnx",
    "700M": "LFM2-700M-ONNX-builder-Q8-fp32head/onnx/model.onnx",
    "1.2B": "LFM2-1.2B-ONNX-builder-Q8-fp32head/onnx/model.onnx",
    "2.6B": "LFM2-2.6B-ONNX-builder-Q8-fp32head/onnx/model.onnx",
}

COMMUNITY_Q4_MODELS = {
    "350M": "/Users/ykhrustalev/workplace/models/onnx-community/LFM2-350M-ONNX/onnx/model_q4.onnx",
    "700M": "/Users/ykhrustalev/workplace/models/onnx-community/LFM2-700M-ONNX/onnx/model_q4.onnx",
    "1.2B": "/Users/ykhrustalev/workplace/models/onnx-community/LFM2-1.2B-ONNX/onnx/model_q4.onnx",
    "2.6B": "/Users/ykhrustalev/workplace/models/onnx-community/LFM2-2.6B-ONNX/onnx/model_q4.onnx",
}

COMMUNITY_Q8_MODELS = {
    "350M": "/Users/ykhrustalev/workplace/models/onnx-community/LFM2-350M-ONNX/onnx/model_q8.onnx",
    "700M": "/Users/ykhrustalev/workplace/models/onnx-community/LFM2-700M-ONNX/onnx/model_q8.onnx",
    "1.2B": "/Users/ykhrustalev/workplace/models/onnx-community/LFM2-1.2B-ONNX/onnx/model_q8.onnx",
    "2.6B": "/Users/ykhrustalev/workplace/models/onnx-community/LFM2-2.6B-ONNX/onnx/model_q8.onnx",
}

TOKENIZER_PATHS = {
    "350M": "LiquidAI/LFM2-350M",
    "700M": "LiquidAI/LFM2-700M",
    "1.2B": "LiquidAI/LFM2-1.2B",
    "2.6B": "LiquidAI/LFM2-2.6B",
}


# ============================================================================


@dataclass
class BenchmarkResult:
    """Result of benchmarking a model."""

    model_size: str
    source: str  # "builder" or "community"
    quant_type: str  # "q4" or "q8"
    file_size_mb: float
    load_time_s: float
    prefill_time_ms: float  # Time to process prompt
    tokens_per_second: float  # Generation speed
    total_tokens: int
    total_time_s: float


def print_results(results: list[BenchmarkResult], quant_type: str):
    """Print benchmark results as a table."""
    quant_label = quant_type.upper()

    print("\n" + "=" * 120)
    print(f"ONNX {quant_label} MODEL PERFORMANCE BENCHMARK")
    print("=" * 120)

    # Group by model size
    by_size = {}
    for r in results:
        if r.quant_type != quant_type:
            continue
        if r.model_size not in by_size:
            by_size[r.model_size] = {}
        by_size[r.model_size][r.source] = r

    # Print header
    print(f"\n{'Model':<12} | {'Builder ' + quant_label:<50} | {'Community ' + quant_label:<50}")
    print(
        f"{'':<12} | {'Size MB':<10} {'Load(s)':<10} {'Prefill(ms)':<12} {'Tok/s':<10} | {'Size MB':<10} {'Load(s)':<10} {'Prefill(ms)':<12} {'Tok/s':<10}"
    )
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
    parser = argparse.ArgumentParser(description="Benchmark ONNX quantized models")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["350M", "700M", "1.2B", "2.6B"],
        default=["350M", "700M", "1.2B", "2.6B"],
        help="Model sizes to benchmark",
    )
    parser.add_argument(
        "--quant",
        nargs="+",
        choices=["q4", "q8"],
        default=["q4", "q8"],
        help="Quantization types to benchmark",
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

    benchmark = ONNXBenchmark(TOKENIZER_PATHS["1.2B"])  # Will reload per model
    results = []

    for size in args.models:
        print(f"\n{'=' * 60}")
        print(f"BENCHMARKING LFM2-{size}")
        print(f"{'=' * 60}")

        tokenizer_path = TOKENIZER_PATHS[size]
        benchmark.model_path = tokenizer_path

        # Q4 benchmarks
        if "q4" in args.quant:
            builder_path = BUILDER_Q4_MODELS[size]
            community_path = COMMUNITY_Q4_MODELS[size]

            print("\n--- Builder Q4 ---")
            try:
                res = benchmark.benchmark(builder_path, args.prompt, args.max_tokens)
                result = BenchmarkResult(
                    model_size=size,
                    source="builder",
                    quant_type="q4",
                    file_size_mb=res.file_size_mb,
                    load_time_s=res.load_time_s,
                    prefill_time_ms=res.prefill_time_ms,
                    tokens_per_second=res.tokens_per_second,
                    total_tokens=res.total_tokens,
                    total_time_s=res.total_time_s,
                )
                results.append(result)
                print(f"  Size: {result.file_size_mb:.1f} MB")
                print(f"  Load time: {result.load_time_s:.2f}s")
                print(f"  Prefill: {result.prefill_time_ms:.1f}ms")
                print(
                    f"  Generation: {result.tokens_per_second:.1f} tok/s ({result.total_tokens} tokens)"
                )
            except Exception as e:
                print(f"  ERROR: {e}")

            print("\n--- Community Q4 ---")
            try:
                res = benchmark.benchmark(community_path, args.prompt, args.max_tokens)
                result = BenchmarkResult(
                    model_size=size,
                    source="community",
                    quant_type="q4",
                    file_size_mb=res.file_size_mb,
                    load_time_s=res.load_time_s,
                    prefill_time_ms=res.prefill_time_ms,
                    tokens_per_second=res.tokens_per_second,
                    total_tokens=res.total_tokens,
                    total_time_s=res.total_time_s,
                )
                results.append(result)
                print(f"  Size: {result.file_size_mb:.1f} MB")
                print(f"  Load time: {result.load_time_s:.2f}s")
                print(f"  Prefill: {result.prefill_time_ms:.1f}ms")
                print(
                    f"  Generation: {result.tokens_per_second:.1f} tok/s ({result.total_tokens} tokens)"
                )
            except Exception as e:
                print(f"  ERROR: {e}")

        # Q8 benchmarks
        if "q8" in args.quant:
            builder_path = BUILDER_Q8_MODELS[size]
            community_path = COMMUNITY_Q8_MODELS[size]

            print("\n--- Builder Q8 ---")
            try:
                res = benchmark.benchmark(builder_path, args.prompt, args.max_tokens)
                result = BenchmarkResult(
                    model_size=size,
                    source="builder",
                    quant_type="q8",
                    file_size_mb=res.file_size_mb,
                    load_time_s=res.load_time_s,
                    prefill_time_ms=res.prefill_time_ms,
                    tokens_per_second=res.tokens_per_second,
                    total_tokens=res.total_tokens,
                    total_time_s=res.total_time_s,
                )
                results.append(result)
                print(f"  Size: {result.file_size_mb:.1f} MB")
                print(f"  Load time: {result.load_time_s:.2f}s")
                print(f"  Prefill: {result.prefill_time_ms:.1f}ms")
                print(
                    f"  Generation: {result.tokens_per_second:.1f} tok/s ({result.total_tokens} tokens)"
                )
            except Exception as e:
                print(f"  ERROR: {e}")

            print("\n--- Community Q8 ---")
            try:
                res = benchmark.benchmark(community_path, args.prompt, args.max_tokens)
                result = BenchmarkResult(
                    model_size=size,
                    source="community",
                    quant_type="q8",
                    file_size_mb=res.file_size_mb,
                    load_time_s=res.load_time_s,
                    prefill_time_ms=res.prefill_time_ms,
                    tokens_per_second=res.tokens_per_second,
                    total_tokens=res.total_tokens,
                    total_time_s=res.total_time_s,
                )
                results.append(result)
                print(f"  Size: {result.file_size_mb:.1f} MB")
                print(f"  Load time: {result.load_time_s:.2f}s")
                print(f"  Prefill: {result.prefill_time_ms:.1f}ms")
                print(
                    f"  Generation: {result.tokens_per_second:.1f} tok/s ({result.total_tokens} tokens)"
                )
            except Exception as e:
                print(f"  ERROR: {e}")

    # Print final comparison tables
    if results:
        if "q4" in args.quant:
            print_results(results, "q4")
        if "q8" in args.quant:
            print_results(results, "q8")


if __name__ == "__main__":
    main()
