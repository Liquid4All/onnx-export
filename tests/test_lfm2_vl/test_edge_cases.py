"""
Edge case tests for LFM2-VL ONNX exports.

Tests different image sizes, aspect ratios, and batching scenarios.

Run with:
    uv run pytest tests/test_lfm2_vl/test_edge_cases.py -v
"""

import logging
import pathlib

import numpy as np
import pytest
from helpers import get_onnx_dir
from PIL import Image

from liquidonnx.embeddings import embed
from liquidonnx.lfm2_vl.export import bundle
from liquidonnx.session import decoder_inputs, initialize_cache, load_onnx_session

logger = logging.getLogger(__name__)

# HuggingFace model IDs to test
MODELS = [
    "LiquidAI/LFM2-VL-450M",
]


def run_embed_images(embed_images_sess, processor, image):
    """Get image embeddings for a single image."""
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
    return outputs[0]  # [num_tokens, hidden]


def run_full_inference(embeddings_sess, embed_images_sess, decoder_sess, processor, image, prompt):
    """Run full VL inference and return logits."""
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, images=[image], return_tensors="pt")

    input_ids = inputs["input_ids"].numpy().astype(np.int64)
    image_embeds = run_embed_images(embed_images_sess, processor, image)
    inputs_embeds = embed(embeddings_sess, input_ids, image_embeds)

    feed = decoder_inputs(inputs_embeds, initialize_cache(decoder_sess), past_len=0)
    return decoder_sess.run(None, feed)[0]


# === Image Size Edge Cases ===

# Model hidden sizes: 450M=1024, 1.6B=1536, 3B=2048
MODEL_HIDDEN_SIZES = {
    "LiquidAI/LFM2-VL-450M": 1024,
    "LiquidAI/LFM2-VL-1.6B": 1536,
    "LiquidAI/LFM2-VL-3B": 2048,
}

IMAGE_SIZE_CASES = [
    pytest.param((64, 64), id="tiny_64x64"),
    pytest.param((128, 128), id="small_128x128"),
    pytest.param((256, 256), id="medium_256x256"),
    pytest.param((512, 512), id="large_512x512"),
    pytest.param((768, 768), id="xlarge_768x768"),
]

# Note: Very wide/tall ratios that don't tile evenly may cause issues
ASPECT_RATIO_CASES = [
    pytest.param((512, 256), id="wide_2_1"),
    pytest.param((256, 512), id="tall_1_2"),
    pytest.param((640, 480), id="4_3_ratio"),
    pytest.param((384, 512), id="3_4_ratio"),
]


@pytest.mark.parametrize("pytorch_model", MODELS, indirect=True)
@pytest.mark.parametrize("size", IMAGE_SIZE_CASES)
def test_image_sizes(
    exports_dir: pathlib.Path,
    pytorch_model,
    size: tuple[int, int],
):
    """Test that different image sizes process correctly."""
    model_id, model, processor = pytorch_model

    onnx_dir = get_onnx_dir(exports_dir, model_id)
    if not onnx_dir.exists():
        pytest.skip(f"Export not found: {onnx_dir}")

    embed_images_file = onnx_dir / "vision_encoder.onnx"
    if not embed_images_file.exists():
        pytest.skip(f"vision_encoder not found: {embed_images_file}")

    embed_images_sess = load_onnx_session(embed_images_file)

    # Create test image with gradient pattern
    w, h = size
    img = Image.new("RGB", (w, h))
    pixels = img.load()
    for y in range(h):
        for x in range(w):
            pixels[x, y] = (
                int(255 * x / w),
                int(255 * y / h),
                128,
            )

    # Process image
    embeddings = run_embed_images(embed_images_sess, processor, img)

    # Verify output shape (hidden size varies by model)
    expected_hidden = MODEL_HIDDEN_SIZES.get(model_id, 1024)
    assert embeddings.ndim == 2, f"Expected 2D output, got {embeddings.ndim}D"
    assert embeddings.shape[1] == expected_hidden, (
        f"Expected hidden={expected_hidden}, got {embeddings.shape[1]}"
    )
    assert embeddings.shape[0] > 0, "Expected at least one token"

    logger.info(f"  {w}x{h} -> {embeddings.shape[0]} tokens, hidden={embeddings.shape[1]}")


@pytest.mark.parametrize("pytorch_model", MODELS, indirect=True)
@pytest.mark.parametrize("size", ASPECT_RATIO_CASES)
def test_aspect_ratios(
    exports_dir: pathlib.Path,
    pytorch_model,
    size: tuple[int, int],
):
    """Test that different aspect ratios process correctly."""
    model_id, model, processor = pytorch_model

    onnx_dir = get_onnx_dir(exports_dir, model_id)
    if not onnx_dir.exists():
        pytest.skip(f"Export not found: {onnx_dir}")

    embed_images_file = onnx_dir / "vision_encoder.onnx"
    if not embed_images_file.exists():
        pytest.skip(f"vision_encoder not found: {embed_images_file}")

    embed_images_sess = load_onnx_session(embed_images_file)

    # Create test image
    w, h = size
    img = Image.new("RGB", (w, h), color=(100, 150, 200))

    # Process image
    embeddings = run_embed_images(embed_images_sess, processor, img)

    # Verify output (hidden size varies by model)
    expected_hidden = MODEL_HIDDEN_SIZES.get(model_id, 1024)
    assert embeddings.ndim == 2
    assert embeddings.shape[1] == expected_hidden
    assert embeddings.shape[0] > 0

    logger.info(f"  {w}x{h} (ratio={w / h:.2f}) -> {embeddings.shape[0]} tokens")


@pytest.mark.parametrize("pytorch_model", MODELS, indirect=True)
def test_batch_different_sizes(
    exports_dir: pathlib.Path,
    pytorch_model,
):
    """Test processing multiple images with different sizes in sequence.

    Note: The current ONNX model processes images one at a time (batch=1).
    This test verifies that different-sized images can be processed in
    the same session without issues.
    """
    model_id, model, processor = pytorch_model

    onnx_dir = get_onnx_dir(exports_dir, model_id)
    if not onnx_dir.exists():
        pytest.skip(f"Export not found: {onnx_dir}")

    embed_images_file = onnx_dir / "vision_encoder.onnx"
    if not embed_images_file.exists():
        pytest.skip(f"vision_encoder not found: {embed_images_file}")

    embed_images_sess = load_onnx_session(embed_images_file)

    # Create images with different sizes
    images = [
        Image.new("RGB", (512, 512), color=(255, 0, 0)),  # Square red
        Image.new("RGB", (768, 512), color=(0, 255, 0)),  # Wide green
        Image.new("RGB", (512, 768), color=(0, 0, 255)),  # Tall blue
        Image.new("RGB", (256, 256), color=(255, 255, 0)),  # Small yellow
    ]

    results = []
    for i, img in enumerate(images):
        embeddings = run_embed_images(embed_images_sess, processor, img)
        results.append(embeddings)
        logger.info(f"  Image {i + 1} ({img.size[0]}x{img.size[1]}): {embeddings.shape[0]} tokens")

    # Verify all processed successfully
    expected_hidden = MODEL_HIDDEN_SIZES.get(model_id, 1024)
    assert len(results) == 4
    for emb in results:
        assert emb.ndim == 2
        assert emb.shape[1] == expected_hidden
        assert emb.shape[0] > 0


@pytest.mark.parametrize("pytorch_model", MODELS, indirect=True)
def test_full_inference_different_sizes(
    exports_dir: pathlib.Path,
    pytorch_model,
):
    """Test full inference pipeline with different image sizes."""
    model_id, model, processor = pytorch_model

    onnx_dir = get_onnx_dir(exports_dir, model_id)
    if not onnx_dir.exists():
        pytest.skip(f"Export not found: {onnx_dir}")

    files = bundle("fp32")
    for name in files.values():
        if not (onnx_dir / name).exists():
            pytest.skip(f"{name} not found in {onnx_dir}")

    embeddings_sess = load_onnx_session(onnx_dir / files["embedding"])
    embed_images_sess = load_onnx_session(onnx_dir / files["vision"])
    decoder_sess = load_onnx_session(onnx_dir / files["decoder"])

    prompt = "What do you see?"

    # Test with different image sizes
    sizes = [(256, 256), (512, 384), (640, 480)]

    for w, h in sizes:
        img = Image.new("RGB", (w, h), color=(128, 128, 128))

        logits = run_full_inference(
            embeddings_sess, embed_images_sess, decoder_sess, processor, img, prompt
        )

        # Verify logits shape
        assert logits.ndim == 3, f"Expected 3D logits, got {logits.ndim}D"
        assert logits.shape[0] == 1, "Expected batch=1"
        assert logits.shape[2] == 65536, f"Expected vocab=65536, got {logits.shape[2]}"

        # Get top prediction
        top_token = int(np.argmax(logits[0, -1]))
        logger.info(f"  {w}x{h}: seq_len={logits.shape[1]}, top_token={top_token}")
