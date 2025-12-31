#!/usr/bin/env python3
"""
Performance benchmark for LFM2-VL ONNX models.

Usage:
    uv run python -m liquidonnx.lfm2_vl.benchmark --model exports/LFM2-VL-450M-ONNX-tiled
    uv run python -m liquidonnx.lfm2_vl.benchmark --model exports/LFM2-VL-450M-ONNX-tiled --image photo.jpg
    uv run python -m liquidonnx.lfm2_vl.benchmark --model exports/LFM2-VL-450M-ONNX-tiled --compare /path/to/community
    uv run python -m liquidonnx.lfm2_vl.benchmark --model exports/LFM2-VL-450M-ONNX-tiled --components
"""

import argparse
import logging
import pathlib
import time
from dataclasses import dataclass

import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image
from transformers import AutoProcessor

from liquidonnx.lfm2_vl import VISION_MODE_CONV2D, VISION_MODE_TILED
from liquidonnx.lfm2_vl.preprocessing import (
    build_inputs_embeds,
    detect_vision_format,
    get_image_token_id,
    preprocess_conv2d,
    preprocess_tiled,
)
from liquidonnx.quantize import get_total_model_size_mb
from liquidonnx.session import initialize_cache, update_cache

logger = logging.getLogger(__name__)


@dataclass
class ComponentTiming:
    """Timing for individual model components."""

    embed_tokens_ms: float
    embed_images_ms: float
    decoder_prefill_ms: float
    decoder_per_token_ms: float


@dataclass
class BenchmarkResult:
    """Result of benchmarking a VL model."""

    name: str
    vision_format: str
    total_size_mb: float
    load_time_s: float
    vision_encoder_ms: float
    prefill_ms: float
    tokens_per_second: float
    total_tokens: int
    total_time_s: float
    generated_text: str
    node_counts: dict[str, int] | None = None
    component_timing: ComponentTiming | None = None


def load_onnx_session(path: pathlib.Path) -> ort.InferenceSession:
    """Load ONNX session with optimizations."""
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.log_severity_level = 3
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def count_nodes(model_path: pathlib.Path) -> int:
    """Count nodes in ONNX model."""
    model = onnx.load(str(model_path))
    return len(model.graph.node)


def benchmark_component(
    sess: ort.InferenceSession,
    inputs: dict,
    num_runs: int = 10,
    warmup: int = 3,
) -> float:
    """Benchmark a single component and return mean time in ms."""
    for _ in range(warmup):
        sess.run(None, inputs)

    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        sess.run(None, inputs)
        times.append(time.perf_counter() - start)

    return np.mean(times) * 1000


class VLBenchmark:
    """Benchmarks LFM2-VL ONNX model performance."""

    def __init__(self, model_path: str):
        self.model_path = pathlib.Path(model_path)
        self.onnx_dir = self.model_path / "onnx"
        self.processor = None
        self.tokenizer = None
        self.embed_tokens_sess = None
        self.embed_images_sess = None
        self.decoder_sess = None
        self.vision_format = VISION_MODE_TILED
        self.image_token_id = None

    def _get_hf_model_name(self) -> str:
        """Get HuggingFace model name from directory."""
        dir_name = self.model_path.name
        if "450M" in dir_name:
            return "LiquidAI/LFM2-VL-450M"
        elif "1.6B" in dir_name:
            return "LiquidAI/LFM2-VL-1.6B"
        elif "3B" in dir_name:
            return "LiquidAI/LFM2-VL-3B"
        return str(self.model_path)

    def _find_onnx_file(self, *names: str) -> pathlib.Path:
        """Find first existing ONNX file from list of names."""
        for name in names:
            path = self.onnx_dir / name
            if path.exists():
                return path
        raise FileNotFoundError(f"None of {names} found in {self.onnx_dir}")

    def load(self) -> float:
        """Load all models and return total load time."""
        start = time.perf_counter()

        hf_model = self._get_hf_model_name()
        logger.info(f"Loading processor from {hf_model}...")
        self.processor = AutoProcessor.from_pretrained(hf_model, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer
        self.image_token_id = get_image_token_id(self.tokenizer)

        logger.info("Loading ONNX models...")
        # Handle both local and community naming conventions
        embed_tokens_path = self._find_onnx_file("embed_tokens.onnx")
        embed_images_path = self._find_onnx_file("embed_images.onnx", "vision_encoder.onnx")
        decoder_path = self._find_onnx_file("decoder.onnx", "decoder_model_merged.onnx")

        self.embed_tokens_sess = load_onnx_session(embed_tokens_path)
        self.embed_images_sess = load_onnx_session(embed_images_path)
        self.decoder_sess = load_onnx_session(decoder_path)

        self.vision_format = detect_vision_format(self.embed_images_sess)
        logger.info(f"Vision format: {self.vision_format}")

        return time.perf_counter() - start

    def get_total_size_mb(self) -> float:
        """Get total size of all ONNX files."""
        total = 0.0
        # Try both local and community naming
        file_options = [
            ["embed_tokens.onnx"],
            ["embed_images.onnx", "vision_encoder.onnx"],
            ["decoder.onnx", "decoder_model_merged.onnx"],
        ]
        for names in file_options:
            for name in names:
                path = self.onnx_dir / name
                if path.exists():
                    total += get_total_model_size_mb(path)
                    break
        return total

    def get_node_counts(self) -> dict[str, int]:
        """Get node counts for each component."""
        counts = {}
        # Map component names to possible file names
        components = {
            "embed_tokens": ["embed_tokens.onnx"],
            "embed_images": ["embed_images.onnx", "vision_encoder.onnx"],
            "decoder": ["decoder.onnx", "decoder_model_merged.onnx"],
        }
        for comp_name, file_names in components.items():
            for name in file_names:
                path = self.onnx_dir / name
                if path.exists():
                    counts[comp_name] = count_nodes(path)
                    break
        return counts

    def _prepare_vision_inputs(self, image: Image.Image) -> dict:
        """Prepare vision encoder inputs based on model's expected input names."""
        input_names = {inp.name for inp in self.embed_images_sess.get_inputs()}

        if self.vision_format == VISION_MODE_CONV2D:
            pixel_values, spatial_h, spatial_w = preprocess_conv2d(image)
            return {
                "pixel_values": pixel_values,
                "spatial_h": np.array(spatial_h, dtype=np.int64),
                "spatial_w": np.array(spatial_w, dtype=np.int64),
            }

        pixel_values, pixel_attention_mask, spatial_shapes = preprocess_tiled(
            image, self.processor, do_image_splitting=False, do_pad_to_square=True
        )

        inputs = {"pixel_values": pixel_values}

        if "pixel_attention_mask" in input_names:
            inputs["pixel_attention_mask"] = pixel_attention_mask

        # Handle spatial_shapes (optional for some models)
        if "spatial_shapes" in input_names:
            inputs["spatial_shapes"] = spatial_shapes

        return inputs

    def _get_image_embeddings(self, image: Image.Image) -> tuple[np.ndarray, float]:
        """Get image embeddings and return (embeddings, time_ms)."""
        inputs = self._prepare_vision_inputs(image)

        start = time.perf_counter()
        outputs = self.embed_images_sess.run(None, inputs)
        elapsed = (time.perf_counter() - start) * 1000

        return outputs[0][0], elapsed

    def _get_text_embeddings(self, input_ids: np.ndarray) -> np.ndarray:
        """Get text embeddings."""
        outputs = self.embed_tokens_sess.run(None, {"input_ids": input_ids.astype(np.int64)})
        return outputs[0]

    def benchmark_components(
        self, image: Image.Image | None = None, num_runs: int = 10
    ) -> ComponentTiming:
        """Benchmark individual components."""
        # Create dummy inputs
        dummy_ids = np.array([[1, 2, 3, 4, 5]], dtype=np.int64)

        # Embed tokens
        embed_tokens_ms = benchmark_component(
            self.embed_tokens_sess, {"input_ids": dummy_ids}, num_runs
        )

        # Embed images
        if image is None:
            image = Image.new("RGB", (384, 384), color="red")

        vision_inputs = self._prepare_vision_inputs(image)
        embed_images_ms = benchmark_component(self.embed_images_sess, vision_inputs, num_runs)

        # Decoder prefill (100 tokens)
        seq_len = 100
        dummy_embeds = np.random.randn(1, seq_len, 1024).astype(np.float32)
        cache = initialize_cache(self.decoder_sess)
        has_position_ids = "position_ids" in {inp.name for inp in self.decoder_sess.get_inputs()}

        decoder_inputs = {
            "inputs_embeds": dummy_embeds,
            "attention_mask": np.ones((1, seq_len), dtype=np.int64),
        }
        if has_position_ids:
            decoder_inputs["position_ids"] = np.arange(seq_len, dtype=np.int64).reshape(1, -1)
        decoder_inputs.update(cache)

        decoder_prefill_ms = benchmark_component(self.decoder_sess, decoder_inputs, num_runs)

        # Decoder per-token (single token)
        single_embeds = np.random.randn(1, 1, 1024).astype(np.float32)
        cache = initialize_cache(self.decoder_sess)
        # Run prefill first to populate cache
        self.decoder_sess.run(None, decoder_inputs)

        single_inputs = {
            "inputs_embeds": single_embeds,
            "attention_mask": np.ones((1, seq_len + 1), dtype=np.int64),
        }
        if has_position_ids:
            single_inputs["position_ids"] = np.array([[seq_len]], dtype=np.int64)
        single_inputs.update(cache)

        decoder_per_token_ms = benchmark_component(self.decoder_sess, single_inputs, num_runs)

        return ComponentTiming(
            embed_tokens_ms=embed_tokens_ms,
            embed_images_ms=embed_images_ms,
            decoder_prefill_ms=decoder_prefill_ms,
            decoder_per_token_ms=decoder_per_token_ms,
        )

    def benchmark(
        self,
        image: Image.Image | None = None,
        prompt: str = "What do you see?",
        max_tokens: int = 20,
        warmup: int = 2,
        benchmark_components: bool = False,
    ) -> BenchmarkResult:
        """Run full benchmark."""
        load_time = self.load()
        total_size = self.get_total_size_mb()
        node_counts = self.get_node_counts()

        # Component timing
        component_timing = None
        if benchmark_components:
            logger.info("Benchmarking components...")
            component_timing = self.benchmark_components(image)

        # Use dummy image if none provided
        if image is None:
            image = Image.new("RGB", (384, 384), color="blue")

        # Prepare inputs
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=text, images=[image], return_tensors="pt", do_image_splitting=False)
        input_ids = inputs["input_ids"].numpy()

        # Warmup
        logger.info("Warmup runs...")
        for _ in range(warmup):
            self._get_image_embeddings(image)

        # Benchmark vision encoder
        logger.info("Benchmarking vision encoder...")
        image_embeds, vision_time = self._get_image_embeddings(image)

        # Build inputs_embeds
        text_embeds = self._get_text_embeddings(input_ids)[0]
        inputs_embeds = build_inputs_embeds(
            text_embeds, [image_embeds], self.image_token_id, input_ids
        )

        # Generation
        cache = initialize_cache(self.decoder_sess)
        has_position_ids = "position_ids" in {inp.name for inp in self.decoder_sess.get_inputs()}

        seq_len = inputs_embeds.shape[1]
        generated_tokens = []
        cur_len = seq_len

        prefill_time = 0.0
        gen_start = None

        logger.info("Running generation...")
        for step in range(max_tokens):
            if step == 0:
                embeds = inputs_embeds
                pos = np.arange(seq_len, dtype=np.int64).reshape(1, -1)
            else:
                last_token = np.array([[generated_tokens[-1]]], dtype=np.int64)
                embeds = self._get_text_embeddings(last_token)
                pos = np.array([[cur_len - 1]], dtype=np.int64)

            feed = {
                "inputs_embeds": embeds.astype(np.float32),
                "attention_mask": np.ones((1, cur_len), dtype=np.int64),
            }
            if has_position_ids:
                feed["position_ids"] = pos
            feed.update(cache)

            start = time.perf_counter()
            outputs = self.decoder_sess.run(None, feed)
            elapsed = time.perf_counter() - start

            if step == 0:
                prefill_time = elapsed * 1000
                gen_start = time.perf_counter()

            update_cache(cache, outputs, self.decoder_sess.get_outputs())

            next_token = int(np.argmax(outputs[0][0, -1]))
            generated_tokens.append(next_token)
            cur_len += 1

            if next_token == self.tokenizer.eos_token_id:
                break

        total_gen_time = time.perf_counter() - gen_start if gen_start else 0
        tokens_per_second = len(generated_tokens) / total_gen_time if total_gen_time > 0 else 0
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return BenchmarkResult(
            name=str(self.model_path),
            vision_format=self.vision_format,
            total_size_mb=total_size,
            load_time_s=load_time,
            vision_encoder_ms=vision_time,
            prefill_ms=prefill_time,
            tokens_per_second=tokens_per_second,
            total_tokens=len(generated_tokens),
            total_time_s=total_gen_time,
            generated_text=generated_text,
            node_counts=node_counts,
            component_timing=component_timing,
        )


def print_result(result: BenchmarkResult):
    """Print benchmark result."""
    print("\n" + "=" * 70)
    print("BENCHMARK RESULT")
    print("=" * 70)
    print(f"Model: {result.name}")
    print(f"Vision format: {result.vision_format}")
    print(f"Total size: {result.total_size_mb:.1f} MB")
    print(f"Load time: {result.load_time_s:.2f}s")

    if result.node_counts:
        print("\nNode counts:")
        for name, count in result.node_counts.items():
            print(f"  {name}: {count}")
        print(f"  Total: {sum(result.node_counts.values())}")

    print("\nPerformance:")
    print(f"  Vision encoder: {result.vision_encoder_ms:.1f} ms")
    print(f"  Decoder prefill: {result.prefill_ms:.1f} ms")
    print(f"  Generation: {result.tokens_per_second:.1f} tok/s")
    print(f"  ({result.total_tokens} tokens in {result.total_time_s:.2f}s)")

    if result.component_timing:
        ct = result.component_timing
        print("\nComponent breakdown:")
        print(f"  embed_tokens: {ct.embed_tokens_ms:.2f} ms")
        print(f"  embed_images: {ct.embed_images_ms:.1f} ms")
        print(f"  decoder prefill (100 tok): {ct.decoder_prefill_ms:.1f} ms")
        print(f"  decoder per token: {ct.decoder_per_token_ms:.2f} ms")

    print(f"\nGenerated: {result.generated_text[:100]}...")
    print("=" * 70)


def compare_models(local_path: str, community_path: str, image: Image.Image | None = None):
    """Compare local and community model performance."""
    print("=" * 70)
    print("MODEL COMPARISON: Local vs Community")
    print("=" * 70)

    local_benchmark = VLBenchmark(local_path)
    local_result = local_benchmark.benchmark(image, benchmark_components=True)

    community_benchmark = VLBenchmark(community_path)
    community_result = community_benchmark.benchmark(image, benchmark_components=True)

    print(f"\n{'Metric':<30} {'Local':>15} {'Community':>15} {'Diff':>10}")
    print("-" * 70)

    metrics = [
        ("Total size (MB)", local_result.total_size_mb, community_result.total_size_mb),
        ("Vision encoder (ms)", local_result.vision_encoder_ms, community_result.vision_encoder_ms),
        ("Decoder prefill (ms)", local_result.prefill_ms, community_result.prefill_ms),
        ("Tokens/second", local_result.tokens_per_second, community_result.tokens_per_second),
    ]

    for name, local_val, comm_val in metrics:
        diff = local_val - comm_val
        diff_pct = (diff / comm_val * 100) if comm_val != 0 else 0
        print(f"{name:<30} {local_val:>15.1f} {comm_val:>15.1f} {diff_pct:>+9.1f}%")

    if local_result.node_counts and community_result.node_counts:
        print(f"\n{'Component nodes':<30} {'Local':>15} {'Community':>15} {'Diff':>10}")
        print("-" * 70)
        for comp in ["embed_tokens", "embed_images", "decoder"]:
            local_n = local_result.node_counts.get(comp, 0)
            comm_n = community_result.node_counts.get(comp, 0)
            diff = local_n - comm_n
            print(f"{comp:<30} {local_n:>15} {comm_n:>15} {diff:>+10}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark LFM2-VL ONNX model performance")
    parser.add_argument("--model", required=True, help="Path to ONNX model directory")
    parser.add_argument("--image", help="Image path for benchmark")
    parser.add_argument("--prompt", default="What do you see?", help="Prompt for generation")
    parser.add_argument("--max-tokens", type=int, default=20, help="Max tokens to generate")
    parser.add_argument("--components", action="store_true", help="Benchmark individual components")
    parser.add_argument("--compare", help="Path to community model for comparison")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    image = None
    if args.image:
        image = Image.open(args.image).convert("RGB")
        logger.info(f"Using image: {args.image}")

    if args.compare:
        compare_models(args.model, args.compare, image)
    else:
        benchmark = VLBenchmark(args.model)
        result = benchmark.benchmark(
            image=image,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            benchmark_components=args.components,
        )
        print_result(result)


if __name__ == "__main__":
    main()
