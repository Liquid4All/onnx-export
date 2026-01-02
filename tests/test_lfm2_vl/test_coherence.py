"""
Multi-turn coherence tests for LFM2-VL ONNX exports.

Tests whether ONNX models maintain coherent multi-turn conversations
with image context compared to PyTorch reference.

Run with:
    uv run pytest tests/test_lfm2_vl/test_coherence.py -v
    uv run pytest tests/test_lfm2_vl/test_coherence.py -v -k "450M and tiled"
"""

import logging
import pathlib

import numpy as np
import pytest
import torch
from helpers import skip_if_missing
from PIL import Image

from liquidonnx.lfm2_vl import MODELS, VISION_MODE_CONV2D, VISION_MODE_TILED
from liquidonnx.lfm2_vl.infer import get_onnx_dir
from liquidonnx.lfm2_vl.preprocessing import (
    detect_vision_format,
    get_image_token_id,
    pad_to_square,
    preprocess_conv2d,
)
from liquidonnx.session import get_onnx_file, initialize_cache, load_onnx_session, update_cache
from liquidonnx.verify import compare_logits_similarity

logger = logging.getLogger(__name__)

QUANT_CONFIGS = [
    pytest.param(None, None, id="fp32"),
    pytest.param("fp16", "fp16", id="fp16"),
    pytest.param("q4", "q4", id="q4"),
    pytest.param("q8", "q8", id="q8"),
]

MAX_NEW_TOKENS = 20
SIMILARITY_THRESHOLD_FP32 = 0.75  # Higher threshold for fp32 (no quantization error)
SIMILARITY_THRESHOLD_QUANT = 0.7  # Lower threshold for quantized models

SINGLE_IMAGE_PROMPTS = [
    "What do you see in this image? Describe the main elements.",
    "What colors are present in the image?",
    "Can you identify any shapes or patterns?",
]

MULTI_IMAGE_PROMPTS = [
    "Which one most important thing do you see on each image? Be concise and exact.",
    "What are the similarities between these images?",
    "What are the differences between these images?",
]

COHERENCE_SCENARIOS = [
    pytest.param("single", SINGLE_IMAGE_PROMPTS, id="single"),
    pytest.param("multi", MULTI_IMAGE_PROMPTS, id="multi"),
]


def get_onnx_image_embeddings(embed_images_sess, images, processor):
    """Get image embeddings from ONNX model.

    Note: Caller should pad images to square for tiled format to ensure all
    tiles have regular 32x32 patches (avoids ONNX/PyTorch pixel_unshuffle mismatch).
    """
    vision_format = detect_vision_format(embed_images_sess)
    embeddings = []

    for image in images:
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
            embeddings.append(outputs[0][0])
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
            # Output is 2D [total_tokens, hidden] after Compress
            embeddings.append(outputs[0])

    return embeddings


def generate_pytorch(model, processor, messages, images, max_new_tokens):
    has_images = any(
        isinstance(item, dict) and item.get("type") == "image"
        for msg in messages
        for item in (msg.get("content", []) if isinstance(msg.get("content"), list) else [])
    )

    if has_images and images:
        # Inject actual images into message content
        img_idx = 0
        messages_with_images = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                new_content = [{"type": "text", "text": content}]
            else:
                new_content = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image":
                        if img_idx < len(images):
                            new_content.append({"type": "image", "image": images[img_idx]})
                            img_idx += 1
                        else:
                            new_content.append(item)
                    else:
                        new_content.append(item)
            messages_with_images.append({**msg, "content": new_content})

        inputs = processor.apply_chat_template(
            messages_with_images,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
        )
    else:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=text, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_logits=True,
        )

    input_len = inputs["input_ids"].shape[1]
    tokens = output.sequences[0, input_len:].tolist()
    logits = np.stack([x[0].numpy() for x in output.logits]) if output.logits else np.array([])
    text = processor.tokenizer.decode(tokens, skip_special_tokens=True)

    return tokens, logits, text


def generate_onnx(
    embed_tokens_sess, embed_images_sess, decoder_sess, processor, messages, images, max_new_tokens
):
    tokenizer = processor.tokenizer
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    has_images = any(
        isinstance(item, dict) and item.get("type") == "image"
        for msg in messages
        for item in (msg.get("content", []) if isinstance(msg.get("content"), list) else [])
    )

    if has_images and images:
        inputs = processor(text=text, images=images, return_tensors="pt")
        input_ids = inputs["input_ids"].numpy().astype(np.int64)

        image_embeds_list = get_onnx_image_embeddings(embed_images_sess, images, processor)
        all_image_embeds = np.concatenate(image_embeds_list, axis=0)

        text_embeds = embed_tokens_sess.run(None, {"input_ids": input_ids})[0][0]

        image_token_id = get_image_token_id(tokenizer)
        image_mask = input_ids[0] == image_token_id

        if image_mask.sum() > 0 and len(all_image_embeds) > 0:
            result_embeds = []
            img_idx = 0
            for i, is_image in enumerate(image_mask):
                if is_image and img_idx < len(all_image_embeds):
                    result_embeds.append(all_image_embeds[img_idx])
                    img_idx += 1
                else:
                    result_embeds.append(text_embeds[i])
            inputs_embeds = np.stack(result_embeds, axis=0)[np.newaxis, ...].astype(np.float32)
        else:
            inputs_embeds = text_embeds[np.newaxis, ...].astype(np.float32)
    else:
        input_ids = np.array([tokenizer.encode(text, add_special_tokens=False)], dtype=np.int64)
        inputs_embeds = embed_tokens_sess.run(None, {"input_ids": input_ids})[0]

    seq_len = inputs_embeds.shape[1]
    input_names = {inp.name for inp in decoder_sess.get_inputs()}
    has_position_ids = "position_ids" in input_names
    cache = initialize_cache(decoder_sess)
    all_logits = []
    generated_tokens = []
    cur_len = seq_len

    for step in range(max_new_tokens):
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

        result = decoder_sess.run(None, feed)
        logits = result[0][0, -1]
        all_logits.append(logits)
        update_cache(cache, result, decoder_sess.get_outputs())

        next_token = int(np.argmax(logits))
        generated_tokens.append(next_token)
        cur_len += 1

        if next_token == tokenizer.eos_token_id:
            break

    text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return generated_tokens, np.stack(all_logits) if all_logits else np.array([]), text


def run_multi_turn_coherence(
    model,
    processor,
    embed_tokens_sess,
    embed_images_sess,
    decoder_sess,
    images: list,
    prompts: list[str],
) -> float:
    # For tiled format, pad images to square to ensure all tiles are regular.
    # This avoids ONNX/PyTorch mismatch on irregular tiles due to pixel_unshuffle ordering.
    vision_format = detect_vision_format(embed_images_sess)
    if vision_format == VISION_MODE_TILED:
        images = [pad_to_square(img) for img in images]

    messages_pytorch = []
    messages_onnx = []
    similarities = []

    for turn, prompt in enumerate(prompts, 1):
        is_first = turn == 1

        # Build user message
        if is_first and images:
            content = [{"type": "image"} for _ in images]
            content.append({"type": "text", "text": prompt})
            user_msg = {"role": "user", "content": content}
        else:
            user_msg = {"role": "user", "content": prompt}

        current_pytorch = messages_pytorch + [user_msg]
        current_onnx = messages_onnx + [user_msg]

        # Generate responses
        pt_tokens, pt_logits, pt_text = generate_pytorch(
            model, processor, current_pytorch, images, MAX_NEW_TOKENS
        )
        ox_tokens, ox_logits, ox_text = generate_onnx(
            embed_tokens_sess,
            embed_images_sess,
            decoder_sess,
            processor,
            current_onnx,
            images,
            MAX_NEW_TOKENS,
        )

        similarity = compare_logits_similarity(pt_logits, ox_logits)
        similarities.append(similarity)

        logger.info(f"  Turn {turn}: similarity={similarity:.4f}")
        logger.info(f"    Prompt: {prompt[:60]}...")
        logger.info(f"    PyTorch: {pt_text}")
        logger.info(f"    ONNX:    {ox_text}")

        # Update conversation history
        messages_pytorch = current_pytorch + [{"role": "assistant", "content": pt_text}]
        messages_onnx = current_onnx + [{"role": "assistant", "content": ox_text}]

    return float(np.mean(similarities)) if similarities else 0.0


# pytorch_model outermost so same model runs consecutively (memory optimization)
@pytest.mark.parametrize("pytorch_model", MODELS.keys(), indirect=True)
@pytest.mark.parametrize("decoder_type,vision_type", QUANT_CONFIGS)
@pytest.mark.parametrize("scenario,prompts", COHERENCE_SCENARIOS)
def test_coherence(
    exports_dir: pathlib.Path,
    cardinal_image: pathlib.Path,
    bluejay_image: pathlib.Path,
    pytorch_model,
    decoder_type: str | None,
    vision_type: str | None,
    scenario: str,
    prompts: list[str],
):
    size, model, processor = pytorch_model

    onnx_dir = get_onnx_dir(exports_dir, size)
    skip_if_missing(onnx_dir, "Export not found")

    decoder_file = get_onnx_file(onnx_dir, decoder_type, "decoder")
    embed_images_file = get_onnx_file(onnx_dir, vision_type, "embed_images")
    skip_if_missing(decoder_file, "Decoder not found")
    skip_if_missing(embed_images_file, "Vision encoder not found")
    embed_tokens_sess = load_onnx_session(onnx_dir / "embed_tokens.onnx")
    embed_images_sess = load_onnx_session(embed_images_file)
    decoder_sess = load_onnx_session(decoder_file)

    if scenario == "single":
        images = [Image.open(cardinal_image).convert("RGB")]
    else:
        images = [
            Image.open(cardinal_image).convert("RGB"),
            Image.open(bluejay_image).convert("RGB"),
        ]

    avg_similarity = run_multi_turn_coherence(
        model,
        processor,
        embed_tokens_sess,
        embed_images_sess,
        decoder_sess,
        images,
        prompts,
    )

    # Use stricter threshold for fp32/fp16 (no quantization error)
    is_float = decoder_type is None or decoder_type == "fp16"
    threshold = SIMILARITY_THRESHOLD_FP32 if is_float else SIMILARITY_THRESHOLD_QUANT

    assert avg_similarity > threshold, (
        f"Semantic similarity too low: {avg_similarity:.4f} (threshold={threshold})"
    )
