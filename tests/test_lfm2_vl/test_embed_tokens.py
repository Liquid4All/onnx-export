"""
Verify embed_tokens ONNX export against PyTorch reference.

Run with:
    uv run pytest tests/test_lfm2_vl/test_embed_tokens.py -v
    uv run pytest tests/test_lfm2_vl/test_embed_tokens.py -v -k "450M"
"""

import logging
import pathlib

import numpy as np
import pytest
import torch
from helpers import get_model_name, get_onnx_dir

from liquidonnx.session import load_onnx_session
from liquidonnx.verify import check_results, compare_arrays, get_tolerances

logger = logging.getLogger(__name__)

# HuggingFace model IDs to test
MODELS = [
    "LiquidAI/LFM2-VL-450M",
    "LiquidAI/LFM2-VL-1.6B",
    "LiquidAI/LFM2-VL-3B",
    "LiquidAI/LFM2.5-VL-1.6B",
]

PROMPTS = ["Hello, how are you?", "The quick brown fox", "Describe this image:"]


# pytorch_model outermost so same model runs consecutively (memory optimization)
@pytest.mark.parametrize("pytorch_model", MODELS, indirect=True)
@pytest.mark.parametrize("prompt", PROMPTS)
def test_embed_tokens(exports_dir: pathlib.Path, pytorch_model, prompt: str):
    model_id, model, processor = pytorch_model
    model_name = get_model_name(model_id)
    logger.info(f"Testing {model_name}: '{prompt}'")

    onnx_dir = get_onnx_dir(exports_dir, model_id)
    if not onnx_dir.exists():
        pytest.skip(f"Export not found: {onnx_dir}")

    embed_tokens_file = onnx_dir / "embed_tokens.onnx"
    if not embed_tokens_file.exists():
        pytest.skip(f"embed_tokens not found: {embed_tokens_file}")

    embed_tokens_sess = load_onnx_session(embed_tokens_file)

    input_ids = processor.tokenizer.encode(prompt, return_tensors="pt")

    with torch.no_grad():
        pytorch_embeds = model.model.language_model.embed_tokens(input_ids).numpy()

    onnx_embeds = embed_tokens_sess.run(
        None,
        {
            "input_ids": input_ids.numpy().astype(np.int64),
        },
    )[0]

    atol, rtol = get_tolerances(None)
    result = compare_arrays(
        f"embed_tokens: '{prompt[:20]}...'", pytorch_embeds, onnx_embeds, atol, rtol
    )

    check_results([result], logger)
