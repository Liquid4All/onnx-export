"""LFM2 test fixtures."""

import gc
import logging
import pathlib

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

MODELS = {
    "350M": "LiquidAI/LFM2-350M",
    "1.2B": "LiquidAI/LFM2-1.2B",
}


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


@pytest.fixture(scope="session")
def exports_dir() -> pathlib.Path:
    """Base directory for ONNX exports."""
    return pathlib.Path(__file__).parent.parent.parent / "exports"
