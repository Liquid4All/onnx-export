"""
Compare local ONNX VL exports against onnx-community versions.

Both are compared against PyTorch reference to show which is closer.

Run with:
    uv run pytest tests/test_lfm2_vl/test_community.py -v
    uv run pytest tests/test_lfm2_vl/test_community.py -v -k "fp32"

Set ONNX_COMMUNITY_DIR environment variable to the directory containing community models:
    export ONNX_COMMUNITY_DIR=/path/to/onnx-community
"""

import logging
import pathlib

import numpy as np
import pytest
import torch
from helpers import get_community_vl_files, get_community_vl_onnx_dir, skip_if_missing
from PIL import Image

from liquidonnx.lfm2_vl import MODELS
from liquidonnx.lfm2_vl.infer import get_onnx_dir
from liquidonnx.lfm2_vl.preprocessing import get_image_token_id, pad_to_square
from liquidonnx.session import get_onnx_file, initialize_cache, load_onnx_session

logger = logging.getLogger(__name__)

QUANT_CONFIGS = [
    pytest.param(None, None, False, id="fp32"),
    pytest.param("q4", None, False, id="q4d"),
    pytest.param("q8", "q8", False, id="q8"),
    pytest.param(None, None, True, id="fp16-community"),
]

PROMPTS = ["What do you see?", "Describe the colors.", "What is the main subject?"]

MULTI_IMAGE_PROMPTS = [
    "Describe the main difference between these two images.",
    "What do you see in each image?",
]


def compute_metrics(expected: np.ndarray, actual: np.ndarray) -> dict:
    """Compute comparison metrics between two logit arrays."""
    diff = np.abs(expected - actual)

    exp_last = expected[0, -1]
    act_last = actual[0, -1]

    exp_top5 = np.argsort(exp_last)[-5:][::-1]
    act_top5 = np.argsort(act_last)[-5:][::-1]

    top1_match = exp_top5[0] == act_top5[0]
    top5_overlap = len(set(exp_top5) & set(act_top5))

    return {
        "max_diff": float(diff.max()),
        "mean_diff": float(diff.mean()),
        "top1_match": top1_match,
        "top5_overlap": top5_overlap,
        "expected_top5": exp_top5.tolist(),
        "actual_top5": act_top5.tolist(),
    }


def run_pytorch_vl(model, processor, image, prompt):
    """Run PyTorch VL model and return logits."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        tokenize=True,
    )

    with torch.no_grad():
        outputs = model(**inputs)
        return outputs.logits.numpy()


def run_local_onnx_vl(embed_tokens_sess, embed_images_sess, decoder_sess, processor, image, prompt):
    """Run local ONNX VL model and return logits."""
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, images=[image], return_tensors="pt")

    input_ids = inputs["input_ids"].numpy().astype(np.int64)
    pixel_values = inputs["pixel_values"].numpy().astype(np.float32)
    pixel_attention_mask = inputs["pixel_attention_mask"].numpy().astype(np.int64)
    spatial_shapes = inputs["spatial_shapes"].numpy().astype(np.int64)

    # Get image embeddings
    image_outputs = embed_images_sess.run(
        None,
        {
            "pixel_values": pixel_values,
            "pixel_attention_mask": pixel_attention_mask,
            "spatial_shapes": spatial_shapes,
        },
    )
    # Output is 2D [total_tokens, hidden] after Compress
    image_embeds_flat = image_outputs[0]

    # Get text embeddings
    text_embeds = embed_tokens_sess.run(None, {"input_ids": input_ids})[0][0]

    # Merge embeddings
    image_token_id = get_image_token_id(processor.tokenizer)
    image_mask = input_ids[0] == image_token_id

    result_embeds = []
    img_idx = 0
    for i, is_image in enumerate(image_mask):
        if is_image and img_idx < len(image_embeds_flat):
            result_embeds.append(image_embeds_flat[img_idx])
            img_idx += 1
        else:
            result_embeds.append(text_embeds[i])

    inputs_embeds = np.stack(result_embeds, axis=0)[np.newaxis, ...].astype(np.float32)

    # Run decoder
    seq_len = inputs_embeds.shape[1]
    attention_mask = np.ones((1, seq_len), dtype=np.int64)

    decoder_inputs = {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
    }

    # Only add position_ids if decoder expects it (not needed with integrated RoPE)
    input_names = {inp.name for inp in decoder_sess.get_inputs()}
    if "position_ids" in input_names:
        position_ids = np.arange(seq_len, dtype=np.int64).reshape(1, -1)
        decoder_inputs["position_ids"] = position_ids

    # Initialize KV cache
    cache = initialize_cache(decoder_sess)
    decoder_inputs.update(cache)

    logits = decoder_sess.run(None, decoder_inputs)[0]
    return logits


def run_community_onnx_vl(
    embed_tokens_sess, vision_encoder_sess, decoder_sess, processor, image, prompt
):
    """Run community ONNX VL model and return logits."""
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, images=[image], return_tensors="pt")

    input_ids = inputs["input_ids"].numpy().astype(np.int64)
    pixel_values = inputs["pixel_values"].numpy().astype(np.float32)
    pixel_attention_mask = inputs["pixel_attention_mask"].numpy().astype(np.int64)
    spatial_shapes = inputs["spatial_shapes"].numpy().astype(np.int64)

    # Get image embeddings (community uses different input names)
    image_outputs = vision_encoder_sess.run(
        None,
        {
            "pixel_values": pixel_values,
            "pixel_attention_mask": pixel_attention_mask,
            "spatial_shapes": spatial_shapes,
        },
    )
    image_embeds_flat = image_outputs[0]

    # Get text embeddings
    text_embeds = embed_tokens_sess.run(None, {"input_ids": input_ids})[0][0]

    # Merge embeddings
    image_token_id = get_image_token_id(processor.tokenizer)
    image_mask = input_ids[0] == image_token_id

    result_embeds = []
    img_idx = 0
    for i, is_image in enumerate(image_mask):
        if is_image and img_idx < len(image_embeds_flat):
            result_embeds.append(image_embeds_flat[img_idx])
            img_idx += 1
        else:
            result_embeds.append(text_embeds[i])

    inputs_embeds = np.stack(result_embeds, axis=0)[np.newaxis, ...].astype(np.float32)

    # Run decoder (community decoder doesn't have position_ids)
    seq_len = inputs_embeds.shape[1]
    attention_mask = np.ones((1, seq_len), dtype=np.int64)

    decoder_inputs = {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
    }

    # Initialize KV cache (auto-detects dtype from model)
    cache = initialize_cache(decoder_sess)
    decoder_inputs.update(cache)

    logits = decoder_sess.run(None, decoder_inputs)[0]
    return logits


def run_pytorch_vl_multi(model, processor, images, prompt):
    """Run PyTorch VL model with multiple images and return logits."""
    content = [{"type": "image", "image": img} for img in images]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        tokenize=True,
    )

    with torch.no_grad():
        outputs = model(**inputs)
        return outputs.logits.numpy()


def run_local_onnx_vl_multi(
    embed_tokens_sess, embed_images_sess, decoder_sess, processor, images, prompt
):
    """Run local ONNX VL model with multiple images and return logits."""
    content = [{"type": "image"} for _ in images]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, images=images, return_tensors="pt")

    input_ids = inputs["input_ids"].numpy().astype(np.int64)
    pixel_values = inputs["pixel_values"].numpy().astype(np.float32)
    pixel_attention_mask = inputs["pixel_attention_mask"].numpy().astype(np.int64)
    spatial_shapes = inputs["spatial_shapes"].numpy().astype(np.int64)

    # Get image embeddings for all images at once
    image_outputs = embed_images_sess.run(
        None,
        {
            "pixel_values": pixel_values,
            "pixel_attention_mask": pixel_attention_mask,
            "spatial_shapes": spatial_shapes,
        },
    )
    image_embeds_flat = image_outputs[0]

    # Get text embeddings
    text_embeds = embed_tokens_sess.run(None, {"input_ids": input_ids})[0][0]

    # Merge embeddings
    image_token_id = get_image_token_id(processor.tokenizer)
    image_mask = input_ids[0] == image_token_id

    result_embeds = []
    img_idx = 0
    for i, is_image in enumerate(image_mask):
        if is_image and img_idx < len(image_embeds_flat):
            result_embeds.append(image_embeds_flat[img_idx])
            img_idx += 1
        else:
            result_embeds.append(text_embeds[i])

    inputs_embeds = np.stack(result_embeds, axis=0)[np.newaxis, ...].astype(np.float32)

    # Run decoder
    seq_len = inputs_embeds.shape[1]
    attention_mask = np.ones((1, seq_len), dtype=np.int64)

    decoder_inputs = {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
    }

    # Only add position_ids if decoder expects it (not needed with integrated RoPE)
    input_names = {inp.name for inp in decoder_sess.get_inputs()}
    if "position_ids" in input_names:
        position_ids = np.arange(seq_len, dtype=np.int64).reshape(1, -1)
        decoder_inputs["position_ids"] = position_ids

    cache = initialize_cache(decoder_sess)
    decoder_inputs.update(cache)

    logits = decoder_sess.run(None, decoder_inputs)[0]
    return logits


def run_community_onnx_vl_multi(
    embed_tokens_sess, vision_encoder_sess, decoder_sess, processor, images, prompt
):
    """Run community ONNX VL model with multiple images and return logits."""
    content = [{"type": "image"} for _ in images]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, images=images, return_tensors="pt")

    input_ids = inputs["input_ids"].numpy().astype(np.int64)
    pixel_values = inputs["pixel_values"].numpy().astype(np.float32)
    pixel_attention_mask = inputs["pixel_attention_mask"].numpy().astype(np.int64)
    spatial_shapes = inputs["spatial_shapes"].numpy().astype(np.int64)

    # Get image embeddings for all images
    image_outputs = vision_encoder_sess.run(
        None,
        {
            "pixel_values": pixel_values,
            "pixel_attention_mask": pixel_attention_mask,
            "spatial_shapes": spatial_shapes,
        },
    )
    image_embeds_flat = image_outputs[0]

    # Get text embeddings
    text_embeds = embed_tokens_sess.run(None, {"input_ids": input_ids})[0][0]

    # Merge embeddings
    image_token_id = get_image_token_id(processor.tokenizer)
    image_mask = input_ids[0] == image_token_id

    result_embeds = []
    img_idx = 0
    for i, is_image in enumerate(image_mask):
        if is_image and img_idx < len(image_embeds_flat):
            result_embeds.append(image_embeds_flat[img_idx])
            img_idx += 1
        else:
            result_embeds.append(text_embeds[i])

    inputs_embeds = np.stack(result_embeds, axis=0)[np.newaxis, ...].astype(np.float32)

    # Run decoder (community decoder doesn't have position_ids)
    seq_len = inputs_embeds.shape[1]
    attention_mask = np.ones((1, seq_len), dtype=np.int64)

    decoder_inputs = {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
    }

    cache = initialize_cache(decoder_sess)
    decoder_inputs.update(cache)

    logits = decoder_sess.run(None, decoder_inputs)[0]
    return logits


@pytest.mark.parametrize("pytorch_model", MODELS.keys(), indirect=True)
@pytest.mark.parametrize("decoder_type,vision_type,use_fp16", QUANT_CONFIGS)
@pytest.mark.parametrize("prompt", PROMPTS)
def test_community_comparison(
    exports_dir: pathlib.Path,
    community_dir: pathlib.Path,
    cardinal_image: pathlib.Path,
    pytorch_model,
    decoder_type: str | None,
    vision_type: str | None,
    use_fp16: bool,
    prompt: str,
):
    """Compare local and community ONNX VL exports against PyTorch reference."""
    size, model, processor = pytorch_model
    quant_str = f"d{decoder_type or 'fp32'}/v{vision_type or 'fp32'}"
    if use_fp16:
        quant_str = "fp16"
    logger.info(f"Comparing VL {size}/{quant_str}: '{prompt}'")

    # Check local export exists
    local_onnx_dir = get_onnx_dir(exports_dir, size)
    skip_if_missing(local_onnx_dir, "Local VL export not found")

    local_decoder_file = get_onnx_file(local_onnx_dir, decoder_type, "decoder")
    local_vision_file = get_onnx_file(local_onnx_dir, vision_type, "embed_images")
    skip_if_missing(local_decoder_file, f"Local decoder not found: {local_decoder_file.name}")
    skip_if_missing(local_vision_file, f"Local vision encoder not found: {local_vision_file.name}")

    # Check community export exists
    community_onnx_dir = get_community_vl_onnx_dir(community_dir, size)
    skip_if_missing(community_onnx_dir, f"Community VL export not found: {community_onnx_dir}")

    community_files = get_community_vl_files(community_onnx_dir, use_fp16)
    for name, path in community_files.items():
        skip_if_missing(path, f"Community {name} not found: {path}")

    # Load local models
    local_embed_tokens = load_onnx_session(local_onnx_dir / "embed_tokens.onnx")
    local_embed_images = load_onnx_session(local_vision_file)
    local_decoder = load_onnx_session(local_decoder_file)

    # Load community models
    community_embed_tokens = load_onnx_session(community_files["embed_tokens"])
    community_vision = load_onnx_session(community_files["vision_encoder"])
    community_decoder = load_onnx_session(community_files["decoder"])

    # Load and preprocess image
    image = Image.open(cardinal_image).convert("RGB")
    image = pad_to_square(image)

    # Run PyTorch
    pytorch_logits = run_pytorch_vl(model, processor, image, prompt)
    logger.info(f"  PyTorch logits: shape={pytorch_logits.shape}")

    # Run local ONNX
    local_logits = run_local_onnx_vl(
        local_embed_tokens, local_embed_images, local_decoder, processor, image, prompt
    )
    logger.info(f"  Local ONNX logits: shape={local_logits.shape}")

    # Run community ONNX
    community_logits = run_community_onnx_vl(
        community_embed_tokens, community_vision, community_decoder, processor, image, prompt
    )
    logger.info(f"  Community ONNX logits: shape={community_logits.shape}")

    # Compare both against PyTorch
    local_metrics = compute_metrics(pytorch_logits, local_logits)
    community_metrics = compute_metrics(pytorch_logits, community_logits)

    # Log comparison results
    logger.info(f"  PyTorch top-5: {local_metrics['expected_top5']}")
    logger.info(
        f"  Local vs PyTorch:     max_diff={local_metrics['max_diff']:.4f}, "
        f"mean_diff={local_metrics['mean_diff']:.4f}, "
        f"top1={'✓' if local_metrics['top1_match'] else '✗'}, "
        f"top5={local_metrics['top5_overlap']}/5"
    )
    logger.info(
        f"  Community vs PyTorch: max_diff={community_metrics['max_diff']:.4f}, "
        f"mean_diff={community_metrics['mean_diff']:.4f}, "
        f"top1={'✓' if community_metrics['top1_match'] else '✗'}, "
        f"top5={community_metrics['top5_overlap']}/5"
    )

    # Determine winner
    if local_metrics["max_diff"] < community_metrics["max_diff"]:
        winner = "LOCAL"
    elif community_metrics["max_diff"] < local_metrics["max_diff"]:
        winner = "COMMUNITY"
    else:
        winner = "TIE"
    logger.info(f"  Winner: {winner} (lower max_diff)")

    # Assert both produce reasonable results
    min_overlap = 3

    assert local_metrics["top5_overlap"] >= min_overlap, (
        f"Local top-5 overlap too low: {local_metrics['top5_overlap']}/5"
    )
    assert community_metrics["top5_overlap"] >= min_overlap, (
        f"Community top-5 overlap too low: {community_metrics['top5_overlap']}/5"
    )


@pytest.mark.parametrize("pytorch_model", MODELS.keys(), indirect=True)
@pytest.mark.parametrize("decoder_type,vision_type,use_fp16", QUANT_CONFIGS)
@pytest.mark.parametrize("prompt", MULTI_IMAGE_PROMPTS)
def test_community_comparison_multi_image(
    exports_dir: pathlib.Path,
    community_dir: pathlib.Path,
    cardinal_image: pathlib.Path,
    bluejay_image: pathlib.Path,
    pytorch_model,
    decoder_type: str | None,
    vision_type: str | None,
    use_fp16: bool,
    prompt: str,
):
    """Compare local and community ONNX VL exports with multiple images."""
    size, model, processor = pytorch_model
    quant_str = f"d{decoder_type or 'fp32'}/v{vision_type or 'fp32'}"
    if use_fp16:
        quant_str = "fp16"
    logger.info(f"Comparing VL multi-image {size}/{quant_str}: '{prompt}'")

    # Check local export exists
    local_onnx_dir = get_onnx_dir(exports_dir, size)
    skip_if_missing(local_onnx_dir, "Local VL export not found")

    local_decoder_file = get_onnx_file(local_onnx_dir, decoder_type, "decoder")
    local_vision_file = get_onnx_file(local_onnx_dir, vision_type, "embed_images")
    skip_if_missing(local_decoder_file, f"Local decoder not found: {local_decoder_file.name}")
    skip_if_missing(local_vision_file, f"Local vision encoder not found: {local_vision_file.name}")

    # Check community export exists
    community_onnx_dir = get_community_vl_onnx_dir(community_dir, size)
    skip_if_missing(community_onnx_dir, f"Community VL export not found: {community_onnx_dir}")

    community_files = get_community_vl_files(community_onnx_dir, use_fp16)
    for name, path in community_files.items():
        skip_if_missing(path, f"Community {name} not found: {path}")

    # Load local models
    local_embed_tokens = load_onnx_session(local_onnx_dir / "embed_tokens.onnx")
    local_embed_images = load_onnx_session(local_vision_file)
    local_decoder = load_onnx_session(local_decoder_file)

    # Load community models
    community_embed_tokens = load_onnx_session(community_files["embed_tokens"])
    community_vision = load_onnx_session(community_files["vision_encoder"])
    community_decoder = load_onnx_session(community_files["decoder"])

    # Load and preprocess images
    images = [
        pad_to_square(Image.open(cardinal_image).convert("RGB")),
        pad_to_square(Image.open(bluejay_image).convert("RGB")),
    ]

    # Run PyTorch
    pytorch_logits = run_pytorch_vl_multi(model, processor, images, prompt)
    logger.info(f"  PyTorch logits: shape={pytorch_logits.shape}")

    # Run local ONNX
    local_logits = run_local_onnx_vl_multi(
        local_embed_tokens, local_embed_images, local_decoder, processor, images, prompt
    )
    logger.info(f"  Local ONNX logits: shape={local_logits.shape}")

    # Run community ONNX
    community_logits = run_community_onnx_vl_multi(
        community_embed_tokens, community_vision, community_decoder, processor, images, prompt
    )
    logger.info(f"  Community ONNX logits: shape={community_logits.shape}")

    # Compare both against PyTorch
    local_metrics = compute_metrics(pytorch_logits, local_logits)
    community_metrics = compute_metrics(pytorch_logits, community_logits)

    # Log comparison results
    logger.info(f"  PyTorch top-5: {local_metrics['expected_top5']}")
    logger.info(
        f"  Local vs PyTorch:     max_diff={local_metrics['max_diff']:.4f}, "
        f"mean_diff={local_metrics['mean_diff']:.4f}, "
        f"top1={'✓' if local_metrics['top1_match'] else '✗'}, "
        f"top5={local_metrics['top5_overlap']}/5"
    )
    logger.info(
        f"  Community vs PyTorch: max_diff={community_metrics['max_diff']:.4f}, "
        f"mean_diff={community_metrics['mean_diff']:.4f}, "
        f"top1={'✓' if community_metrics['top1_match'] else '✗'}, "
        f"top5={community_metrics['top5_overlap']}/5"
    )

    # Determine winner
    if local_metrics["max_diff"] < community_metrics["max_diff"]:
        winner = "LOCAL"
    elif community_metrics["max_diff"] < local_metrics["max_diff"]:
        winner = "COMMUNITY"
    else:
        winner = "TIE"
    logger.info(f"  Winner: {winner} (lower max_diff)")

    # Assert both produce reasonable results
    min_overlap = 3

    assert local_metrics["top5_overlap"] >= min_overlap, (
        f"Local top-5 overlap too low: {local_metrics['top5_overlap']}/5"
    )
    assert community_metrics["top5_overlap"] >= min_overlap, (
        f"Community top-5 overlap too low: {community_metrics['top5_overlap']}/5"
    )
