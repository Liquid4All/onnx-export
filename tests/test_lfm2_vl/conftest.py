"""LFM2-VL test fixtures."""

import gc
import logging
import pathlib

import pytest

from liquidonnx.lfm2_vl import MODELS
from test_lfm2_vl.helpers import load_pytorch_model

logger = logging.getLogger(__name__)

ASSETS_DIR = pathlib.Path(__file__).parent / "assets"


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
