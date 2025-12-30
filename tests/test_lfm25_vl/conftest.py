"""LFM2.5-VL test fixtures.

Tests the LFM2-VL-1.6B-3102461 model variant.
"""

import gc
import logging
import pathlib

import pytest
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

logger = logging.getLogger(__name__)

MODEL_PATH = "./LFM2-VL-1.6B-3102461"
MODEL_NAME = "LFM2-VL-1.6B-3102461"

ASSETS_DIR = pathlib.Path(__file__).parent.parent / "test_lfm2_vl" / "assets"


def load_pytorch_model(model_path: str) -> tuple:
    """Load PyTorch model and processor."""
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
def pytorch_model(request):
    """Load PyTorch model for current test group."""
    model, processor = load_pytorch_model(MODEL_PATH)
    yield MODEL_NAME, model, processor

    logger.info(f"Cleaning up {MODEL_NAME} model...")
    del model
    del processor
    gc.collect()


@pytest.fixture
def cardinal_image() -> pathlib.Path:
    return ASSETS_DIR / "cardinal.jpg"


@pytest.fixture
def bluejay_image() -> pathlib.Path:
    return ASSETS_DIR / "bluejay.jpg"
