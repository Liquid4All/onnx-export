"""Verify embed_tokens ONNX export against PyTorch reference."""

import pathlib

import numpy as np
import pytest
import torch

from liquidonnx.lfm2_vl import MODELS, VISION_MODES
from test_lfm2_vl.helpers import (
    ATOL,
    RTOL,
    skip_if_missing,
    get_vl_onnx_dir,
    load_onnx_session,
    compare_arrays,
)

PROMPTS = ["Hello, how are you?", "The quick brown fox", "Describe this image:"]


# pytorch_model outermost so same model runs consecutively (memory optimization)
@pytest.mark.parametrize("pytorch_model", MODELS.keys(), indirect=True)
@pytest.mark.parametrize("vision_mode", VISION_MODES)
@pytest.mark.parametrize("prompt", PROMPTS)
def test_embed_tokens(exports_dir: pathlib.Path, pytorch_model, vision_mode: str, prompt: str):
    size, model, processor = pytorch_model

    onnx_dir = get_vl_onnx_dir(exports_dir, size, vision_mode)
    skip_if_missing(onnx_dir, "Export not found")
    skip_if_missing(onnx_dir / "onnx" / "embed_tokens.onnx", "embed_tokens not found")

    embed_tokens_sess = load_onnx_session(onnx_dir, "embed_tokens.onnx")

    input_ids = processor.tokenizer.encode(prompt, return_tensors="pt")

    with torch.no_grad():
        pytorch_embeds = model.model.language_model.embed_tokens(input_ids).numpy()

    onnx_embeds = embed_tokens_sess.run(None, {
        "input_ids": input_ids.numpy().astype(np.int64),
    })[0]

    result = compare_arrays(
        f"embed_tokens: '{prompt[:20]}...'",
        pytorch_embeds, onnx_embeds, ATOL, RTOL
    )

    assert result.passed, f"{result.name}: max_diff={result.max_diff:.6f}, {result.details}"
