"""LFM2-MoE test fixtures."""

import gc
import logging

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from liquidonnx.lfm2_moe import MODELS

logger = logging.getLogger(__name__)


def load_pytorch_model(model_path: str) -> tuple:
    """Load PyTorch model and tokenizer from HuggingFace."""
    logger.info(f"Loading PyTorch model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


@pytest.fixture(scope="module")
def model_tokenizer(request):
    """Load tokenizer for model size. Use with indirect=True.

    Returns (size, tokenizer) tuple.
    """
    size = request.param
    tokenizer = AutoTokenizer.from_pretrained(MODELS[size], trust_remote_code=True)
    return size, tokenizer


@pytest.fixture(scope="module")
def pytorch_model(request):
    """Load PyTorch model for current test group. Use with indirect=True.

    Returns (size, model, tokenizer) tuple.
    """
    size = request.param
    model, tokenizer = load_pytorch_model(MODELS[size])
    yield size, model, tokenizer

    logger.info(f"Cleaning up {size} model...")
    del model
    del tokenizer
    gc.collect()
