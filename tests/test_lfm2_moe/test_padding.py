"""
Padded-batch regression tests for LFM2-MoE ONNX exports.

These cover the published MoE graph behavior for padded batches by checking
that the fp32 ONNX export stays close to the PyTorch reference at the last
valid token for each row.
"""

import logging
import pathlib

import numpy as np
import pytest
import torch
from helpers import get_onnx_dir

from liquidonnx.session import get_onnx_file, load_onnx_session
from liquidonnx.verify import cosine_similarity

logger = logging.getLogger(__name__)

MODEL_ID = "LiquidAI/LFM2.5-8B-A1B"
PROMPTS = [
    "Name the capital of France.",
    "Explain in one sentence why compilers use intermediate representations.",
]
PADDING_SIDES = ["left", "right"]
MIN_COSINE = 0.95
MIN_TOP5_OVERLAP = 4


def _build_padded_batch(tokenizer, prompts: list[str], padding_side: str):
    tokenized = [tokenizer.encode(prompt, add_special_tokens=False) for prompt in prompts]
    max_len = max(len(tokens) for tokens in tokenized)

    if tokenizer.pad_token_id is None:
        pytest.skip("Tokenizer does not define a pad_token_id")

    input_rows = []
    mask_rows = []
    for tokens in tokenized:
        pad = max_len - len(tokens)
        if padding_side == "left":
            input_rows.append([tokenizer.pad_token_id] * pad + tokens)
            mask_rows.append([0] * pad + [1] * len(tokens))
        else:
            input_rows.append(tokens + [tokenizer.pad_token_id] * pad)
            mask_rows.append([1] * len(tokens) + [0] * pad)

    input_ids = torch.tensor(input_rows, dtype=torch.long)
    attention_mask = torch.tensor(mask_rows, dtype=torch.long)

    # Match the integrated-RoPE ONNX graph, which uses absolute sequence indices.
    position_ids = torch.arange(max_len, dtype=torch.long).unsqueeze(0).expand(len(prompts), -1)

    return tokenized, input_ids, attention_mask, position_ids


def _init_onnx_cache(session, batch_size: int) -> dict[str, np.ndarray]:
    cache = {}
    skip_inputs = {"input_ids", "attention_mask", "position_ids"}

    for inp in session.get_inputs():
        if inp.name in skip_inputs:
            continue

        shape = []
        for dim in inp.shape:
            if isinstance(dim, int):
                shape.append(dim)
            elif isinstance(dim, str) and dim == "batch_size":
                shape.append(batch_size)
            elif isinstance(dim, str) and "sequence" in dim.lower():
                shape.append(0)
            else:
                shape.append(1)

        dtype = np.float16 if "float16" in inp.type else np.float32
        cache[inp.name] = np.zeros(shape, dtype=dtype)

    return cache


def _top5_overlap(expected: np.ndarray, actual: np.ndarray) -> int:
    exp_top5 = np.argsort(expected)[-5:]
    act_top5 = np.argsort(actual)[-5:]
    return len(set(exp_top5.tolist()) & set(act_top5.tolist()))


@pytest.mark.parametrize("pytorch_model", [MODEL_ID], indirect=True)
@pytest.mark.parametrize("padding_side", PADDING_SIDES)
def test_padded_batch_matches_pytorch(
    exports_dir: pathlib.Path,
    pytorch_model,
    padding_side: str,
):
    """Compare padded fp32 batch logits against PyTorch at each row's last valid token."""
    model_id, model, tokenizer = pytorch_model
    onnx_dir = get_onnx_dir(exports_dir, model_id)
    onnx_file = get_onnx_file(onnx_dir, None)

    if not onnx_file.exists():
        pytest.skip(f"ONNX file not found: {onnx_file}")

    try:
        onnx_sess = load_onnx_session(onnx_file)
    except Exception as e:
        pytest.skip(f"ONNX model failed to load: {e}")

    _, input_ids, attention_mask, position_ids = _build_padded_batch(
        tokenizer, PROMPTS, padding_side
    )

    with torch.no_grad():
        pytorch_logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        ).logits.numpy()

    onnx_inputs = {
        "input_ids": input_ids.numpy().astype(np.int64),
        "attention_mask": attention_mask.numpy().astype(np.int64),
    }
    if any(inp.name == "position_ids" for inp in onnx_sess.get_inputs()):
        onnx_inputs["position_ids"] = position_ids.numpy().astype(np.int64)
    onnx_inputs.update(_init_onnx_cache(onnx_sess, batch_size=input_ids.shape[0]))

    onnx_logits = onnx_sess.run(None, onnx_inputs)[0]

    for row_idx in range(input_ids.shape[0]):
        last_valid_idx = int(np.flatnonzero(attention_mask[row_idx].numpy())[-1])
        expected = pytorch_logits[row_idx, last_valid_idx]
        actual = onnx_logits[row_idx, last_valid_idx]
        cosine = cosine_similarity(expected, actual)
        overlap = _top5_overlap(expected, actual)
        top1_match = int(np.argmax(expected)) == int(np.argmax(actual))

        logger.info(
            "%s row %d: cosine=%.4f top5=%d/5 top1=%s",
            padding_side,
            row_idx,
            cosine,
            overlap,
            top1_match,
        )

        assert cosine >= MIN_COSINE, (
            f"{padding_side} row {row_idx} cosine {cosine:.4f} < {MIN_COSINE:.2f}"
        )
        assert overlap >= MIN_TOP5_OVERLAP, (
            f"{padding_side} row {row_idx} top-5 overlap {overlap}/5 < {MIN_TOP5_OVERLAP}/5"
        )
