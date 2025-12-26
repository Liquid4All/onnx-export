"""LFM2-VL test fixtures."""

import gc
import logging
import pathlib

import pytest
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from liquidonnx.lfm2_vl import MODELS

logger = logging.getLogger(__name__)

ASSETS_DIR = pathlib.Path(__file__).parent / "assets"


def load_pytorch_model(model_path: str) -> tuple:
    """Load PyTorch model and processor from HuggingFace."""
    logger.info(f"Loading PyTorch model from {model_path}...")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    model.eval()
    return model, processor


@pytest.fixture(scope="module")
def model_processor(request):
    """Load processor for model size. Use with indirect=True.

    Lighter alternative to pytorch_model when full model isn't needed.
    Returns (size, processor) tuple.
    """
    size = request.param
    processor = AutoProcessor.from_pretrained(MODELS[size], trust_remote_code=True)
    return size, processor


@pytest.fixture(scope="module")
def pytorch_model(request):
    """Load PyTorch model for current test group. Use with indirect=True.

    Returns (size, model, processor) tuple.
    """
    size = request.param
    model, processor = load_pytorch_model(MODELS[size])
    yield size, model, processor

    logger.info(f"Cleaning up {size} model...")
    del model
    del processor
    gc.collect()


@pytest.fixture
def cardinal_image() -> pathlib.Path:
    return ASSETS_DIR / "cardinal.jpg"


@pytest.fixture
def bluejay_image() -> pathlib.Path:
    return ASSETS_DIR / "bluejay.jpg"
