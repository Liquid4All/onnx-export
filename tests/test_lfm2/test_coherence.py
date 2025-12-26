"""
Multi-turn coherence tests for LFM2 ONNX exports.

Tests whether ONNX models maintain coherent multi-turn conversations
compared to PyTorch reference.

Run with:
    pytest tests/test_lfm2/test_coherence.py -v
    pytest tests/test_lfm2/test_coherence.py -v -k "1.2B and q4"
"""

import logging
import pathlib

import numpy as np
import pytest
import torch
from helpers import skip_if_missing

from liquidonnx.lfm2 import MODELS
from liquidonnx.lfm2.generate import get_onnx_dir
from liquidonnx.session import get_onnx_file, initialize_cache, load_onnx_session, update_cache
from liquidonnx.verify import compare_logits_similarity

logger = logging.getLogger(__name__)

QUANT_CONFIGS = [
    pytest.param(None, id="fp32"),
    pytest.param(4, id="q4"),
    pytest.param(8, id="q8"),
]

MAX_NEW_TOKENS = 20
SIMILARITY_THRESHOLD_FP32 = 0.95
SIMILARITY_THRESHOLD_QUANT = 0.70

DEFAULT_PROMPTS = [
    "My name is Sarah and I work as a software engineer. Can you remember this?",
    "What is my name?",
    "What is my profession?",
]


def generate_pytorch(
    model, tokenizer, input_ids: list[int], max_new_tokens: int
) -> tuple[list[int], np.ndarray]:
    """Generate tokens with PyTorch model using KV cache."""
    generated = input_ids.copy()
    all_logits = []
    past_key_values = None

    with torch.no_grad():
        for step in range(max_new_tokens):
            if step == 0:
                ids = torch.tensor([generated], dtype=torch.long)
                pos = torch.arange(len(generated), dtype=torch.long).unsqueeze(0)
                attn = torch.ones(1, len(generated), dtype=torch.long)
            else:
                ids = torch.tensor([[generated[-1]]], dtype=torch.long)
                pos = torch.tensor([[len(generated) - 1]], dtype=torch.long)
                attn = torch.ones(1, len(generated), dtype=torch.long)

            outputs = model(
                input_ids=ids,
                attention_mask=attn,
                position_ids=pos,
                past_key_values=past_key_values,
                use_cache=True,
            )

            past_key_values = outputs.past_key_values
            logits = outputs.logits[0, -1].numpy()
            all_logits.append(logits)
            next_token = int(np.argmax(logits))
            generated.append(next_token)

            if next_token == tokenizer.eos_token_id:
                break

    return generated, np.stack(all_logits) if all_logits else np.array([])


def generate_onnx(
    session, tokenizer, input_ids: list[int], max_new_tokens: int
) -> tuple[list[int], np.ndarray]:
    """Generate tokens with ONNX model."""
    generated = input_ids.copy()
    all_logits = []

    input_names = {inp.name for inp in session.get_inputs()}
    has_position_ids = "position_ids" in input_names
    cache = initialize_cache(session)

    for step in range(max_new_tokens):
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

        result = session.run(None, feed)
        logits = result[0][0, -1]
        all_logits.append(logits)
        update_cache(cache, result, session.get_outputs())

        next_token = int(np.argmax(logits))
        generated.append(next_token)

        if next_token == tokenizer.eos_token_id:
            break

    return generated, np.stack(all_logits) if all_logits else np.array([])


def run_multi_turn_coherence(
    model,
    tokenizer,
    onnx_session,
    prompts: list[str],
) -> float:
    """Run multi-turn coherence test, return average similarity."""
    messages_pytorch = []
    messages_onnx = []
    similarities = []

    for turn, prompt in enumerate(prompts, 1):
        # Add user message
        messages_pytorch = messages_pytorch + [{"role": "user", "content": prompt}]
        messages_onnx = messages_onnx + [{"role": "user", "content": prompt}]

        # Apply chat template
        pytorch_text = tokenizer.apply_chat_template(
            messages_pytorch, tokenize=False, add_generation_prompt=True
        )
        onnx_text = tokenizer.apply_chat_template(
            messages_onnx, tokenize=False, add_generation_prompt=True
        )

        pytorch_input = tokenizer.encode(pytorch_text, add_special_tokens=False)
        onnx_input = tokenizer.encode(onnx_text, add_special_tokens=False)

        # Generate responses
        pytorch_output, pytorch_logits = generate_pytorch(
            model, tokenizer, pytorch_input, MAX_NEW_TOKENS
        )
        onnx_output, onnx_logits = generate_onnx(
            onnx_session, tokenizer, onnx_input, MAX_NEW_TOKENS
        )

        similarity = compare_logits_similarity(pytorch_logits, onnx_logits)
        similarities.append(similarity)

        # Decode responses
        pytorch_new = pytorch_output[len(pytorch_input) :]
        onnx_new = onnx_output[len(onnx_input) :]
        pytorch_response = tokenizer.decode(pytorch_new, skip_special_tokens=True)
        onnx_response = tokenizer.decode(onnx_new, skip_special_tokens=True)

        logger.info(f"  Turn {turn}: similarity={similarity:.4f}")
        logger.info(f"    Prompt: {prompt[:60]}...")
        logger.info(f"    PyTorch: {pytorch_response[:80]}")
        logger.info(f"    ONNX:    {onnx_response[:80]}")

        # Update conversation history
        messages_pytorch = messages_pytorch + [{"role": "assistant", "content": pytorch_response}]
        messages_onnx = messages_onnx + [{"role": "assistant", "content": onnx_response}]

    return float(np.mean(similarities)) if similarities else 0.0


@pytest.mark.parametrize("pytorch_model", MODELS.keys(), indirect=True)
@pytest.mark.parametrize("bits", QUANT_CONFIGS)
def test_coherence(
    exports_dir: pathlib.Path,
    pytorch_model,
    bits: int | None,
):
    """Test multi-turn coherence between PyTorch and ONNX."""
    size, model, tokenizer = pytorch_model

    onnx_dir = get_onnx_dir(exports_dir, size)
    skip_if_missing(onnx_dir, "Export not found")

    onnx_file = get_onnx_file(onnx_dir, bits)
    skip_if_missing(onnx_file, f"ONNX file not found: {onnx_file.name}")

    onnx_session = load_onnx_session(onnx_file)

    avg_similarity = run_multi_turn_coherence(
        model,
        tokenizer,
        onnx_session,
        DEFAULT_PROMPTS,
    )

    # Use stricter threshold for fp32 (no quantization error)
    threshold = SIMILARITY_THRESHOLD_FP32 if bits is None else SIMILARITY_THRESHOLD_QUANT

    assert avg_similarity > threshold, (
        f"Semantic similarity too low: {avg_similarity:.4f} (threshold={threshold})"
    )
