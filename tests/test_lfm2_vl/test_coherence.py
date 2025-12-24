"""
Multi-turn coherence tests for LFM2-VL ONNX exports.

Tests whether ONNX models maintain coherent multi-turn conversations
with image context compared to PyTorch reference.

Run with:
    pytest tests/test_lfm2_vl/test_coherence.py -v
    pytest tests/test_lfm2_vl/test_coherence.py -v -k "450M and tiled"
"""

import pathlib

import numpy as np
import pytest
import torch
from PIL import Image

from liquidonnx.lfm2_vl import MODELS, VISION_MODES, VISION_MODE_CONV2D
from liquidonnx.lfm2_vl.preprocessing import detect_vision_format, preprocess_conv2d, preprocess_tiled
from test_lfm2_vl.helpers import (
    QUANT_CONFIGS,
    skip_if_missing,
    get_onnx_file,
    get_vl_onnx_dir,
    load_onnx_session,
)

MAX_NEW_TOKENS = 20
SIMILARITY_THRESHOLD = 0.7

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


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_flat, b_flat = a.flatten(), b.flatten()
    dot = np.dot(a_flat, b_flat)
    norm_a, norm_b = np.linalg.norm(a_flat), np.linalg.norm(b_flat)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def get_image_token_id(tokenizer) -> int:
    for token_name in ["<image>", "<|image|>", "[IMG]"]:
        token_id = tokenizer.convert_tokens_to_ids(token_name)
        if token_id != tokenizer.unk_token_id:
            return token_id
    if hasattr(tokenizer, "image_token_id"):
        return tokenizer.image_token_id
    raise ValueError("Could not find image token ID")


def get_onnx_image_embeddings(embed_images_sess, images, processor):
    vision_format = detect_vision_format(embed_images_sess)
    embeddings = []

    for image in images:
        if vision_format == VISION_MODE_CONV2D:
            pixel_values, spatial_h, spatial_w = preprocess_conv2d(image)
            outputs = embed_images_sess.run(None, {
                "pixel_values": pixel_values,
                "spatial_h": np.array(spatial_h, dtype=np.int64),
                "spatial_w": np.array(spatial_w, dtype=np.int64),
            })
            embeddings.append(outputs[0][0])
        else:
            pixel_values, patch_attention_mask, _ = preprocess_tiled(
                image, processor, do_image_splitting=False, pad_to_square=True
            )
            outputs = embed_images_sess.run(None, {
                "pixel_values": pixel_values,
                "patch_attention_mask": patch_attention_mask,
            })
            onnx_embeds = outputs[0]
            num_tiles, tokens_per_tile, hidden = onnx_embeds.shape
            embeddings.append(onnx_embeds.reshape(-1, hidden))

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

    input_len = inputs['input_ids'].shape[1]
    tokens = output.sequences[0, input_len:].tolist()
    logits = np.stack([l[0].numpy() for l in output.logits]) if output.logits else np.array([])
    text = processor.tokenizer.decode(tokens, skip_special_tokens=True)

    return tokens, logits, text


def generate_onnx(embed_tokens_sess, embed_images_sess, decoder_sess, processor, messages, images, max_new_tokens):
    tokenizer = processor.tokenizer
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    has_images = any(
        isinstance(item, dict) and item.get("type") == "image"
        for msg in messages
        for item in (msg.get("content", []) if isinstance(msg.get("content"), list) else [])
    )

    if has_images and images:
        inputs = processor(text=text, images=images, return_tensors="pt")
        input_ids = inputs['input_ids'].numpy().astype(np.int64)

        image_embeds_list = get_onnx_image_embeddings(embed_images_sess, images, processor)
        all_image_embeds = np.concatenate(image_embeds_list, axis=0)

        text_embeds = embed_tokens_sess.run(None, {"input_ids": input_ids})[0][0]

        image_token_id = get_image_token_id(tokenizer)
        image_mask = (input_ids[0] == image_token_id)

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

    # Initialize caches
    cache = {}
    for inp in decoder_sess.get_inputs():
        if inp.name not in ["inputs_embeds", "attention_mask", "position_ids"]:
            shape = [d if isinstance(d, int) else 1 for d in inp.shape]
            cache[inp.name] = np.zeros(shape, dtype=np.float32)

    outputs_info = decoder_sess.get_outputs()
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

        next_token = int(np.argmax(logits))
        generated_tokens.append(next_token)
        cur_len += 1

        if next_token == tokenizer.eos_token_id:
            break

    text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return generated_tokens, np.stack(all_logits) if all_logits else np.array([]), text


def compare_logits(pytorch_logits, onnx_logits) -> float:
    if len(pytorch_logits) == 0 or len(onnx_logits) == 0:
        return 1.0
    min_steps = min(len(pytorch_logits), len(onnx_logits))
    similarities = [cosine_similarity(pytorch_logits[i], onnx_logits[i]) for i in range(min_steps)]
    return float(np.mean(similarities))


def run_multi_turn_coherence(
    model,
    processor,
    embed_tokens_sess,
    embed_images_sess,
    decoder_sess,
    images: list,
    prompts: list[str],
) -> float:
    messages_pytorch = []
    messages_onnx = []
    similarities = []

    for turn, prompt in enumerate(prompts, 1):
        is_first = (turn == 1)

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
            embed_tokens_sess, embed_images_sess, decoder_sess,
            processor, current_onnx, images, MAX_NEW_TOKENS
        )

        similarity = compare_logits(pt_logits, ox_logits)
        similarities.append(similarity)

        # Update conversation history
        messages_pytorch = current_pytorch + [{"role": "assistant", "content": pt_text}]
        messages_onnx = current_onnx + [{"role": "assistant", "content": ox_text}]

    return float(np.mean(similarities)) if similarities else 0.0


# pytorch_model outermost so same model runs consecutively (memory optimization)
@pytest.mark.parametrize("pytorch_model", MODELS.keys(), indirect=True)
@pytest.mark.parametrize("vision_mode", VISION_MODES)
@pytest.mark.parametrize("decoder_bits,vision_bits", QUANT_CONFIGS)
def test_coherence_single_image(
    exports_dir: pathlib.Path,
    cardinal_image: pathlib.Path,
    pytorch_model,
    vision_mode: str,
    decoder_bits: int | None,
    vision_bits: int | None,
):
    size, model, processor = pytorch_model

    onnx_dir = get_vl_onnx_dir(exports_dir, size, vision_mode)
    skip_if_missing(onnx_dir, "Export not found")

    decoder_file = get_onnx_file(onnx_dir, "decoder", decoder_bits)
    embed_images_file = get_onnx_file(onnx_dir, "embed_images", vision_bits)
    skip_if_missing(decoder_file, "Decoder not found")
    skip_if_missing(embed_images_file, "Vision encoder not found")
    embed_tokens_sess = load_onnx_session(onnx_dir, "embed_tokens.onnx")
    embed_images_sess = load_onnx_session(onnx_dir, embed_images_file.name)
    decoder_sess = load_onnx_session(onnx_dir, decoder_file.name)

    images = [Image.open(cardinal_image).convert("RGB")]

    avg_similarity = run_multi_turn_coherence(
        model, processor,
        embed_tokens_sess, embed_images_sess, decoder_sess,
        images, SINGLE_IMAGE_PROMPTS,
    )

    assert avg_similarity > SIMILARITY_THRESHOLD, (
        f"Semantic similarity too low: {avg_similarity:.4f}"
    )


# pytorch_model outermost so same model runs consecutively (memory optimization)
@pytest.mark.parametrize("pytorch_model", MODELS.keys(), indirect=True)
@pytest.mark.parametrize("vision_mode", VISION_MODES)
@pytest.mark.parametrize("decoder_bits,vision_bits", QUANT_CONFIGS)
def test_coherence_multi_image(
    exports_dir: pathlib.Path,
    cardinal_image: pathlib.Path,
    bluejay_image: pathlib.Path,
    pytorch_model,
    vision_mode: str,
    decoder_bits: int | None,
    vision_bits: int | None,
):
    size, model, processor = pytorch_model

    onnx_dir = get_vl_onnx_dir(exports_dir, size, vision_mode)
    skip_if_missing(onnx_dir, "Export not found")

    decoder_file = get_onnx_file(onnx_dir, "decoder", decoder_bits)
    embed_images_file = get_onnx_file(onnx_dir, "embed_images", vision_bits)
    skip_if_missing(decoder_file, "Decoder not found")
    skip_if_missing(embed_images_file, "Vision encoder not found")
    embed_tokens_sess = load_onnx_session(onnx_dir, "embed_tokens.onnx")
    embed_images_sess = load_onnx_session(onnx_dir, embed_images_file.name)
    decoder_sess = load_onnx_session(onnx_dir, decoder_file.name)

    images = [
        Image.open(cardinal_image).convert("RGB"),
        Image.open(bluejay_image).convert("RGB"),
    ]

    avg_similarity = run_multi_turn_coherence(
        model, processor,
        embed_tokens_sess, embed_images_sess, decoder_sess,
        images, MULTI_IMAGE_PROMPTS,
    )

    assert avg_similarity > SIMILARITY_THRESHOLD, (
        f"Semantic similarity too low: {avg_similarity:.4f}"
    )
