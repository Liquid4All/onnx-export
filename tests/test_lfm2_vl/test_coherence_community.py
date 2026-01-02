"""
Multi-turn coherence tests comparing local and community ONNX VL models.

Compares both local and onnx-community models against PyTorch reference
in multi-turn conversation coherence tests.

Run with:
    uv run pytest tests/test_lfm2_vl/test_coherence_community.py -v
    uv run pytest tests/test_lfm2_vl/test_coherence_community.py -v -k "450M"
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
from liquidonnx.session import get_onnx_file, initialize_cache, load_onnx_session, update_cache
from liquidonnx.verify import compare_logits_similarity

logger = logging.getLogger(__name__)

MAX_NEW_TOKENS = 20
SIMILARITY_THRESHOLD = 0.75

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


def generate_pytorch(model, processor, messages, images, max_new_tokens):
    """Generate response using PyTorch model."""
    has_images = any(
        isinstance(item, dict) and item.get("type") == "image"
        for msg in messages
        for item in (msg.get("content", []) if isinstance(msg.get("content"), list) else [])
    )

    if has_images and images:
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


def get_image_embeddings_local(embed_images_sess, processor, images):
    """Get image embeddings from local ONNX model."""
    embeddings = []
    for image in images:
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
        embeddings.append(outputs[0])
    return embeddings


def get_image_embeddings_community(vision_sess, processor, images):
    """Get image embeddings from community ONNX model."""
    embeddings = []
    for image in images:
        inputs = processor.image_processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].numpy().astype(np.float32)
        pixel_attention_mask = inputs["pixel_attention_mask"].numpy().astype(np.int64)
        spatial_shapes = inputs["spatial_shapes"].numpy().astype(np.int64)

        outputs = vision_sess.run(
            None,
            {
                "pixel_values": pixel_values,
                "pixel_attention_mask": pixel_attention_mask,
                "spatial_shapes": spatial_shapes,
            },
        )
        # Output is 2D [num_tokens, hidden] - same as local
        embeddings.append(outputs[0])
    return embeddings


def generate_onnx(
    embed_tokens_sess,
    embed_images_sess,
    decoder_sess,
    processor,
    messages,
    images,
    max_new_tokens,
    get_image_embeds_fn,
):
    """Generate response using ONNX model."""
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

        image_embeds_list = get_image_embeds_fn(embed_images_sess, processor, images)
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


def run_coherence_comparison(
    model,
    processor,
    local_sessions,
    community_sessions,
    images: list,
    prompts: list[str],
) -> tuple[float, float]:
    """Run multi-turn coherence test comparing local and community against PyTorch.

    Returns:
        Tuple of (local_avg_similarity, community_avg_similarity)
    """
    # Pad images to square for consistent tile processing
    images = [pad_to_square(img) for img in images]

    local_embed_tokens, local_embed_images, local_decoder = local_sessions
    community_embed_tokens, community_vision, community_decoder = community_sessions

    messages_pytorch = []
    messages_local = []
    messages_community = []

    local_similarities = []
    community_similarities = []

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
        current_local = messages_local + [user_msg]
        current_community = messages_community + [user_msg]

        # Generate PyTorch response
        pt_tokens, pt_logits, pt_text = generate_pytorch(
            model, processor, current_pytorch, images, MAX_NEW_TOKENS
        )

        # Generate local ONNX response
        local_tokens, local_logits, local_text = generate_onnx(
            local_embed_tokens,
            local_embed_images,
            local_decoder,
            processor,
            current_local,
            images,
            MAX_NEW_TOKENS,
            get_image_embeddings_local,
        )

        # Generate community ONNX response
        community_tokens, community_logits, community_text = generate_onnx(
            community_embed_tokens,
            community_vision,
            community_decoder,
            processor,
            current_community,
            images,
            MAX_NEW_TOKENS,
            get_image_embeddings_community,
        )

        # Compute similarities
        local_sim = compare_logits_similarity(pt_logits, local_logits)
        community_sim = compare_logits_similarity(pt_logits, community_logits)

        local_similarities.append(local_sim)
        community_similarities.append(community_sim)

        # Determine winner
        if local_sim > community_sim:
            winner = "LOCAL"
        elif community_sim > local_sim:
            winner = "COMMUNITY"
        else:
            winner = "TIE"

        logger.info(
            f"  Turn {turn}: local={local_sim:.4f}, community={community_sim:.4f} -> {winner}"
        )
        logger.info(f"    Prompt: {prompt[:60]}...")
        logger.info(f"    PyTorch:   {pt_text[:80]}...")
        logger.info(f"    Local:     {local_text[:80]}...")
        logger.info(f"    Community: {community_text[:80]}...")

        # Update conversation histories
        messages_pytorch = current_pytorch + [{"role": "assistant", "content": pt_text}]
        messages_local = current_local + [{"role": "assistant", "content": local_text}]
        messages_community = current_community + [{"role": "assistant", "content": community_text}]

    local_avg = float(np.mean(local_similarities)) if local_similarities else 0.0
    community_avg = float(np.mean(community_similarities)) if community_similarities else 0.0

    return local_avg, community_avg


@pytest.mark.parametrize("pytorch_model", MODELS.keys(), indirect=True)
@pytest.mark.parametrize("scenario,prompts", COHERENCE_SCENARIOS)
def test_coherence_community(
    exports_dir: pathlib.Path,
    community_dir: pathlib.Path,
    cardinal_image: pathlib.Path,
    bluejay_image: pathlib.Path,
    pytorch_model,
    scenario: str,
    prompts: list[str],
):
    """Compare multi-turn coherence: local vs community ONNX against PyTorch.

    Uses fp32 precision for both local and community models.
    """
    size, model, processor = pytorch_model

    # Check local exports
    local_onnx_dir = get_onnx_dir(exports_dir, size)
    skip_if_missing(local_onnx_dir, "Local export not found")

    local_decoder = get_onnx_file(local_onnx_dir, None, "decoder")
    local_embed_images = get_onnx_file(local_onnx_dir, None, "embed_images")
    skip_if_missing(local_decoder, "Local decoder not found")
    skip_if_missing(local_embed_images, "Local embed_images not found")

    # Check community exports
    community_onnx_dir = get_community_vl_onnx_dir(community_dir, size)
    skip_if_missing(community_onnx_dir, f"Community export not found: {community_onnx_dir}")

    community_files = get_community_vl_files(community_onnx_dir, use_fp16=False)
    skip_if_missing(community_files["decoder"], "Community decoder not found")
    skip_if_missing(community_files["vision_encoder"], "Community vision encoder not found")

    logger.info(f"Testing coherence {size}/{scenario}: local vs community")

    # Load local sessions
    local_embed_tokens_sess = load_onnx_session(local_onnx_dir / "embed_tokens.onnx")
    local_embed_images_sess = load_onnx_session(local_embed_images)
    local_decoder_sess = load_onnx_session(local_decoder)

    # Load community sessions
    community_embed_tokens_sess = load_onnx_session(community_files["embed_tokens"])
    community_vision_sess = load_onnx_session(community_files["vision_encoder"])
    community_decoder_sess = load_onnx_session(community_files["decoder"])

    # Prepare images
    if scenario == "single":
        images = [Image.open(cardinal_image).convert("RGB")]
    else:
        images = [
            Image.open(cardinal_image).convert("RGB"),
            Image.open(bluejay_image).convert("RGB"),
        ]

    # Run comparison
    local_avg, community_avg = run_coherence_comparison(
        model,
        processor,
        (local_embed_tokens_sess, local_embed_images_sess, local_decoder_sess),
        (community_embed_tokens_sess, community_vision_sess, community_decoder_sess),
        images,
        prompts,
    )

    # Log final results
    if local_avg > community_avg:
        winner = "LOCAL"
    elif community_avg > local_avg:
        winner = "COMMUNITY"
    else:
        winner = "TIE"

    logger.info(f"  Final: local={local_avg:.4f}, community={community_avg:.4f} -> {winner}")

    # Assert both meet minimum threshold
    assert local_avg > SIMILARITY_THRESHOLD, (
        f"Local similarity too low: {local_avg:.4f} (threshold={SIMILARITY_THRESHOLD})"
    )
    assert community_avg > SIMILARITY_THRESHOLD, (
        f"Community similarity too low: {community_avg:.4f} (threshold={SIMILARITY_THRESHOLD})"
    )
