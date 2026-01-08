"""
Benchmark VL pipeline: embed_images + embed_tokens + decoder.

Measures per-component and end-to-end performance for single image inference.

Run with:
    uv run pytest tests/test_lfm2_vl/test_benchmark.py -v -s
    uv run pytest tests/test_lfm2_vl/test_benchmark.py -v -s -k "450M and tiled and q4"
"""

import logging
import pathlib
import time
from dataclasses import dataclass

import numpy as np
import onnxruntime as ort
import pytest
from helpers import get_onnx_dir
from PIL import Image

from liquidonnx.lfm2_vl import VISION_MODE_CONV2D
from liquidonnx.lfm2_vl.preprocessing import (
    detect_vision_format,
    get_image_token_id,
    preprocess_conv2d,
)
from liquidonnx.quantize import get_total_model_size_mb
from liquidonnx.session import get_onnx_file, initialize_cache, update_cache

logger = logging.getLogger(__name__)

# HuggingFace model IDs to test
MODELS = [
    "LiquidAI/LFM2-VL-450M",
    "LiquidAI/LFM2-VL-1.6B",
]

QUANT_CONFIGS = [
    pytest.param("q4", "q4", id="q4"),
    pytest.param("q4", "q8", id="q4d-q8v"),
    pytest.param("q8", "q8", id="q8"),
]

DECODE_TOKENS = 50
WARMUP_RUNS = 1


@dataclass
class VLBenchmarkResult:
    name: str
    embed_images_mb: float
    embed_tokens_mb: float
    decoder_mb: float
    total_mb: float
    load_time_s: float
    image_encode_ms: float
    text_encode_ms: float
    prefill_ms: float
    decode_ms_per_token: float
    tokens_per_sec: float
    total_tokens: int
    e2e_time_s: float


def get_image_embeddings(embed_images_sess, image, processor):
    """Get image embeddings from ONNX model."""
    vision_format = detect_vision_format(embed_images_sess)

    if vision_format == VISION_MODE_CONV2D:
        pixel_values, spatial_h, spatial_w = preprocess_conv2d(image)
        outputs = embed_images_sess.run(
            None,
            {
                "pixel_values": pixel_values,
                "spatial_h": np.array(spatial_h, dtype=np.int64),
                "spatial_w": np.array(spatial_w, dtype=np.int64),
            },
        )
        return outputs[0][0]
    else:
        inputs = processor.image_processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].numpy().astype(np.float32)
        pixel_attention_mask = inputs["pixel_attention_mask"].numpy().astype(np.int64)
        spatial_shapes = inputs["spatial_shapes"].numpy().astype(np.int64)

        outputs = embed_images_sess.run(
            None,
            {
                "pixel_values": pixel_values,
                "pixel_attention_mask": pixel_attention_mask,
                "spatial_shapes": spatial_shapes,
            },
        )
        onnx_embeds = outputs[0]
        num_tiles, tokens_per_tile, hidden = onnx_embeds.shape
        return onnx_embeds.reshape(-1, hidden)


def run_benchmark(
    embed_images_sess,
    embed_tokens_sess,
    decoder_sess,
    processor,
    image,
    prompt: str,
    max_tokens: int,
    warmup: int = 1,
) -> tuple[VLBenchmarkResult, str]:
    """Run VL generation benchmark."""
    tokenizer = processor.tokenizer

    # Build chat message with image
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, images=[image], return_tensors="pt")
    input_ids = inputs["input_ids"].numpy().astype(np.int64)

    # Warmup runs
    for _ in range(warmup):
        _ = get_image_embeddings(embed_images_sess, image, processor)
        _ = embed_tokens_sess.run(None, {"input_ids": input_ids})[0]

    # Timed image encoding
    start = time.perf_counter()
    image_embeds = get_image_embeddings(embed_images_sess, image, processor)
    image_encode_ms = (time.perf_counter() - start) * 1000

    # Timed text encoding
    start = time.perf_counter()
    text_embeds = embed_tokens_sess.run(None, {"input_ids": input_ids})[0][0]
    text_encode_ms = (time.perf_counter() - start) * 1000

    # Merge embeddings
    image_token_id = get_image_token_id(tokenizer)
    image_mask = input_ids[0] == image_token_id

    if image_mask.sum() > 0:
        result_embeds = []
        img_idx = 0
        for i, is_image in enumerate(image_mask):
            if is_image and img_idx < len(image_embeds):
                result_embeds.append(image_embeds[img_idx])
                img_idx += 1
            else:
                result_embeds.append(text_embeds[i])
        inputs_embeds = np.stack(result_embeds, axis=0)[np.newaxis, ...].astype(np.float32)
    else:
        inputs_embeds = text_embeds[np.newaxis, ...].astype(np.float32)

    # Decoder generation
    seq_len = inputs_embeds.shape[1]
    input_names = {inp.name for inp in decoder_sess.get_inputs()}
    has_position_ids = "position_ids" in input_names
    cache = initialize_cache(decoder_sess)
    generated_tokens = []
    cur_len = seq_len

    prefill_ms = 0.0
    decode_start = None

    for step in range(max_tokens):
        if step == 0:
            embeds = inputs_embeds
            pos = np.arange(seq_len, dtype=np.int64).reshape(1, -1)
        else:
            last_token = np.array([[generated_tokens[-1]]], dtype=np.int64)
            embeds = embed_tokens_sess.run(None, {"input_ids": last_token})[0]
            pos = np.array([[cur_len - 1]], dtype=np.int64)

        attn_mask = np.ones((1, cur_len), dtype=np.int64)
        feed = {"inputs_embeds": embeds.astype(np.float32), "attention_mask": attn_mask}
        if has_position_ids:
            feed["position_ids"] = pos
        feed.update(cache)

        start = time.perf_counter()
        result = decoder_sess.run(None, feed)
        elapsed = (time.perf_counter() - start) * 1000

        if step == 0:
            prefill_ms = elapsed
            decode_start = time.perf_counter()

        update_cache(cache, result, decoder_sess.get_outputs())

        next_token = int(np.argmax(result[0][0, -1]))
        generated_tokens.append(next_token)
        cur_len += 1

        if next_token == tokenizer.eos_token_id:
            break

    decode_time = time.perf_counter() - decode_start if decode_start else 0
    num_tokens = len(generated_tokens)
    tokens_per_sec = num_tokens / decode_time if decode_time > 0 else 0
    decode_ms_per_token = (decode_time * 1000) / num_tokens if num_tokens > 0 else 0

    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return (
        image_encode_ms,
        text_encode_ms,
        prefill_ms,
        decode_ms_per_token,
        tokens_per_sec,
        num_tokens,
        generated_text,
    )


@pytest.mark.parametrize("model_processor", MODELS, indirect=True)
@pytest.mark.parametrize("decoder_type,vision_type", QUANT_CONFIGS)
def test_benchmark(
    exports_dir: pathlib.Path,
    cardinal_image: pathlib.Path,
    model_processor,
    decoder_type: str,
    vision_type: str,
):
    """Benchmark single-image VL inference."""
    model_id, processor = model_processor
    logger.info(f"Benchmarking {model_id}/{decoder_type}d-{vision_type}v")

    onnx_dir = get_onnx_dir(exports_dir, model_id)
    if not onnx_dir.exists():
        pytest.skip(f"Export not found: {onnx_dir}")

    embed_tokens_file = onnx_dir / "embed_tokens.onnx"
    embed_images_file = get_onnx_file(onnx_dir, vision_type, "embed_images")
    decoder_file = get_onnx_file(onnx_dir, decoder_type, "decoder")

    if not embed_tokens_file.exists():
        pytest.skip(f"embed_tokens not found: {embed_tokens_file}")
    if not embed_images_file.exists():
        pytest.skip(f"embed_images not found: {embed_images_file}")
    if not decoder_file.exists():
        pytest.skip(f"decoder not found: {decoder_file}")

    # Measure model sizes
    embed_tokens_mb = get_total_model_size_mb(embed_tokens_file)
    embed_images_mb = get_total_model_size_mb(embed_images_file)
    decoder_mb = get_total_model_size_mb(decoder_file)
    total_mb = embed_tokens_mb + embed_images_mb + decoder_mb

    # Load models (timed)
    load_start = time.perf_counter()
    embed_tokens_sess = ort.InferenceSession(
        str(embed_tokens_file), providers=["CPUExecutionProvider"]
    )
    embed_images_sess = ort.InferenceSession(
        str(embed_images_file), providers=["CPUExecutionProvider"]
    )
    decoder_sess = ort.InferenceSession(str(decoder_file), providers=["CPUExecutionProvider"])
    load_time = time.perf_counter() - load_start

    # Load test image
    image = Image.open(cardinal_image).convert("RGB")

    # Run benchmark
    prompt = "Describe this image in detail."
    e2e_start = time.perf_counter()
    (
        image_encode_ms,
        text_encode_ms,
        prefill_ms,
        decode_ms_per_token,
        tokens_per_sec,
        total_tokens,
        generated_text,
    ) = run_benchmark(
        embed_images_sess,
        embed_tokens_sess,
        decoder_sess,
        processor,
        image,
        prompt,
        DECODE_TOKENS,
        WARMUP_RUNS,
    )
    e2e_time = time.perf_counter() - e2e_start

    # Log results
    logger.info("")
    logger.info(f"  {'Model Sizes':<25}")
    logger.info(f"  {'-' * 40}")
    logger.info(f"  {'embed_tokens':<25} {embed_tokens_mb:>10.1f} MB")
    logger.info(f"  {'embed_images':<25} {embed_images_mb:>10.1f} MB")
    logger.info(f"  {'decoder':<25} {decoder_mb:>10.1f} MB")
    logger.info(f"  {'total':<25} {total_mb:>10.1f} MB")
    logger.info("")
    logger.info(f"  {'Performance':<25}")
    logger.info(f"  {'-' * 40}")
    logger.info(f"  {'Load time':<25} {load_time:>10.2f} s")
    logger.info(f"  {'Image encode':<25} {image_encode_ms:>10.1f} ms")
    logger.info(f"  {'Text encode':<25} {text_encode_ms:>10.1f} ms")
    logger.info(f"  {'Prefill':<25} {prefill_ms:>10.1f} ms")
    logger.info(f"  {'Decode':<25} {decode_ms_per_token:>10.1f} ms/tok")
    logger.info(f"  {'Speed':<25} {tokens_per_sec:>10.1f} tok/s")
    logger.info(f"  {'Tokens generated':<25} {total_tokens:>10d}")
    logger.info(f"  {'E2E time':<25} {e2e_time:>10.2f} s")
    logger.info("")
    logger.info(f"  Generated: {generated_text[:100]}...")
