"""Verify embed_tokens ONNX export against PyTorch reference."""

import logging
import pathlib

import numpy as np
import pytest
import torch
from helpers import skip_if_missing

from liquidonnx.lfm2_vl import MODELS, VISION_MODE_TILED
from liquidonnx.lfm2_vl.inference import get_onnx_dir, load_onnx_session
from liquidonnx.verify import check_results, compare_arrays, get_tolerances

logger = logging.getLogger(__name__)

PROMPTS = ["Hello, how are you?", "The quick brown fox", "Describe this image:"]


# pytorch_model outermost so same model runs consecutively (memory optimization)
@pytest.mark.parametrize("pytorch_model", MODELS.keys(), indirect=True)
@pytest.mark.parametrize("prompt", PROMPTS)
def test_embed_tokens(exports_dir: pathlib.Path, pytorch_model, prompt: str):
    size, model, processor = pytorch_model
    logger.info(f"Testing {size}: '{prompt}'")

    onnx_dir = get_onnx_dir(exports_dir, size, VISION_MODE_TILED)
    skip_if_missing(onnx_dir, "Export not found")
    skip_if_missing(onnx_dir / "onnx" / "embed_tokens.onnx", "embed_tokens not found")

    embed_tokens_sess = load_onnx_session(onnx_dir, "embed_tokens.onnx")

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
