"""LFM2-VL test fixtures."""

import gc
import logging
import pathlib

import pytest
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

logger = logging.getLogger(__name__)

ASSETS_DIR = pathlib.Path(__file__).parent / "assets"


def load_pytorch_model(model_id: str) -> tuple:
    """Load PyTorch model and processor from HuggingFace."""
    logger.info(f"Loading PyTorch model from {model_id}...")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    model.eval()
    return model, processor


@pytest.fixture(scope="module")
def model_processor(request):
    """Load processor for model. Use with indirect=True.

    Returns (model_id, processor) tuple.
    """
    model_id = request.param
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return model_id, processor


@pytest.fixture(scope="module")
def pytorch_model(request):
    """Load PyTorch model for current test group. Use with indirect=True.

    Returns (model_id, model, processor) tuple.
    """
    model_id = request.param
    model, processor = load_pytorch_model(model_id)
    yield model_id, model, processor

    logger.info(f"Cleaning up {model_id} model...")
    del model
    del processor
    gc.collect()


@pytest.fixture
def cardinal_image() -> pathlib.Path:
    return ASSETS_DIR / "cardinal.jpg"


@pytest.fixture
def bluejay_image() -> pathlib.Path:
    return ASSETS_DIR / "bluejay.jpg"
