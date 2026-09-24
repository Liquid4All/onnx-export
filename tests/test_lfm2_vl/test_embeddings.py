"""
Verify the embedding model (token lookup + image feature scatter) against PyTorch.

Run with:
    uv run pytest tests/test_lfm2_vl/test_embeddings.py -v
    uv run pytest tests/test_lfm2_vl/test_embeddings.py -v -k "450M"
"""

import logging
import pathlib

import numpy as np
import pytest
import torch
from helpers import get_model_name, get_onnx_dir

from liquidonnx.embeddings import embed
from liquidonnx.lfm2_vl.preprocessing import get_image_token_id
from liquidonnx.session import load_onnx_session
from liquidonnx.verify import check_results, compare_arrays, get_tolerances

logger = logging.getLogger(__name__)

MODELS = [
    "LiquidAI/LFM2-VL-450M",
    "LiquidAI/LFM2-VL-1.6B",
    "LiquidAI/LFM2-VL-3B",
    "LiquidAI/LFM2.5-VL-1.6B",
]

PROMPTS = ["Hello, how are you?", "The quick brown fox", "Describe this image:"]

# (file, tolerance key): the fp32 table, and the fp16 one every quantized precision uses
EMBEDDINGS = [
    pytest.param("embeddings.onnx", None, id="fp32"),
    pytest.param("embeddings_fp16.onnx", "fp16", id="fp16"),
]


# pytorch_model outermost so same model runs consecutively (memory optimization)
@pytest.mark.parametrize("pytorch_model", MODELS, indirect=True)
@pytest.mark.parametrize("filename,tolerance", EMBEDDINGS)
@pytest.mark.parametrize("prompt", PROMPTS)
def test_embeddings(
    exports_dir: pathlib.Path, pytorch_model, filename: str, tolerance: str | None, prompt: str
):
    model_id, model, processor = pytorch_model
    model_name = get_model_name(model_id)
    logger.info(f"Testing {model_name}/{filename}: '{prompt}'")

    onnx_dir = get_onnx_dir(exports_dir, model_id)
    if not (onnx_dir / filename).exists():
        pytest.skip(f"{filename} not found in {onnx_dir}")

    session = load_onnx_session(onnx_dir / filename)

    # Two image placeholders in the middle of the prompt, as the processor writes them
    image_token_id = get_image_token_id(processor.tokenizer)
    text_ids = processor.tokenizer.encode(prompt)
    input_ids = np.array([text_ids[:2] + [image_token_id] * 2 + text_ids[2:]], dtype=np.int64)
    features = np.random.default_rng(0).standard_normal((2, session.get_inputs()[1].shape[1]))
    features = features.astype(np.float32)

    with torch.no_grad():
        expected = model.model.language_model.embed_tokens(torch.from_numpy(input_ids)).numpy()
    expected[0, 2:4] = features

    atol, rtol = get_tolerances(tolerance)
    results = [
        compare_arrays(
            f"{filename} text: '{prompt[:20]}...'",
            expected[:, :2],
            embed(session, input_ids[:, :2]),
            atol,
            rtol,
        ),
        compare_arrays(
            f"{filename} with images: '{prompt[:20]}...'",
            expected,
            embed(session, input_ids, features),
            atol,
            rtol,
        ),
    ]
    check_results(results, logger)
