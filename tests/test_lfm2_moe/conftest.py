"""LFM2-MoE test fixtures."""

import gc
import logging

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def model_tokenizer(request):
    """Load tokenizer for model. Use with indirect=True.

    Returns (model_id, tokenizer) tuple.
    """
    model_id = request.param
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    return model_id, tokenizer


@pytest.fixture(scope="module")
def pytorch_model(request):
    """Load PyTorch model for current test group. Use with indirect=True.

    Returns (model_id, model, tokenizer) tuple.
    """
    model_id = request.param
    logger.info(f"Loading PyTorch model from {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    model.eval()
    yield model_id, model, tokenizer

    logger.info(f"Cleaning up {model_id} model...")
    del model
    del tokenizer
    gc.collect()
