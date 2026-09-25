#!/usr/bin/env python3
"""
Speed of an LFM2 or LFM2-MoE export on onnxruntime-genai: load, prefill and decode.

Usage:
    uv run lfm2-bench --model exports/LFM2.5-1.2B-Instruct-ONNX
    uv run lfm2-bench --model exports/LFM2.5-1.2B-Instruct-ONNX --precision q8 --max-tokens 50
"""

import argparse
import json
import logging
import pathlib
import time
from dataclasses import dataclass

import numpy as np
import onnxruntime_genai as og

from liquidonnx.genai_runtime import add_runtime_arguments, load_model
from liquidonnx.lfm2.export import ALL_PRECISIONS, genai_files, model_file
from liquidonnx.quantize import get_total_model_size_mb

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    decoder: str
    file_size_mb: float
    load_time_s: float
    prefill_time_ms: float
    tokens_per_second: float
    total_tokens: int
    total_time_s: float
    generated_text: str


def benchmark(
    model_dir: pathlib.Path,
    precision: str | None = None,
    prompt: str = "Hello, how are",
    max_tokens: int = 20,
    ep: str = "cpu",
    warmup: int = 2,
) -> BenchmarkResult:
    """Greedy continuation of prompt (no chat template), exactly max_tokens long, after warmup runs."""
    if precision:
        decoder = f"onnx/{model_file(precision)}"
    else:
        config = json.loads((model_dir / "genai_config.json").read_text())
        decoder = config["model"]["decoder"]["filename"]

    start = time.perf_counter()
    model = load_model(model_dir, precision and genai_files(precision), ep)
    load_time = time.perf_counter() - start

    from transformers import AutoTokenizer

    tokenizer = og.Tokenizer(model)
    # transformers adds the BOS token, which onnxruntime-genai's encode leaves out.
    ids = np.array(AutoTokenizer.from_pretrained(model_dir).encode(prompt), dtype=np.int32)

    def run() -> tuple[list[int], float, float]:
        params = og.GeneratorParams(model)
        length = len(ids) + max_tokens
        params.set_search_options(do_sample=False, min_length=length, max_length=length)
        generator = og.Generator(model, params)
        start = time.perf_counter()
        generator.append_tokens(ids)
        prefill = time.perf_counter() - start
        start = time.perf_counter()
        while not generator.is_done():
            generator.generate_next_token()
        return list(generator.get_sequence(0)[len(ids) :]), prefill, time.perf_counter() - start

    for _ in range(warmup):
        run()
    tokens, prefill, decode = run()
    # append_tokens computed the first token's logits; decode ran one step per later token.
    steps = len(tokens) - 1

    return BenchmarkResult(
        decoder=decoder,
        file_size_mb=get_total_model_size_mb(model_dir / decoder),
        load_time_s=load_time,
        prefill_time_ms=prefill * 1000,
        tokens_per_second=steps / decode if decode > 0 else 0,
        total_tokens=len(tokens),
        total_time_s=decode,
        generated_text=prompt + tokenizer.decode(np.array(tokens, dtype=np.int32)),
    )


def log_result(result: BenchmarkResult):
    logger.info("=" * 70)
    logger.info(f"Decoder: {result.decoder} ({result.file_size_mb:.1f} MB)")
    logger.info(f"Load time: {result.load_time_s:.2f}s")
    logger.info(f"Prefill: {result.prefill_time_ms:.1f}ms")
    logger.info(
        f"Generation: {result.tokens_per_second:.1f} tok/s "
        f"({result.total_tokens} tokens in {result.total_time_s:.2f}s)"
    )
    logger.info(f"Generated: {result.generated_text}")
    logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Benchmark an LFM2 export on onnxruntime-genai")
    parser.add_argument("--model", required=True, type=pathlib.Path, help="Export folder")
    add_runtime_arguments(parser, ALL_PRECISIONS)
    parser.add_argument("--prompt", type=str, default="Hello, how are", help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=20, help="Max tokens to generate")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    log_result(benchmark(args.model, args.precision, args.prompt, args.max_tokens, args.ep))


if __name__ == "__main__":
    main()
