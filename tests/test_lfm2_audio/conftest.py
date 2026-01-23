"""LFM2.5-Audio test fixtures."""

import gc
import logging
import pathlib

import pytest
import torch

logger = logging.getLogger(__name__)

AUDIO_MODEL_ID = "LiquidAI/LFM2.5-Audio-1.5B"


@pytest.fixture(scope="module")
def reference_model():
    """Load reference PyTorch audio model.

    Returns (model, processor) tuple.
    """
    from liquid_audio import LFM2AudioModel, LFM2AudioProcessor

    device = "cpu"
    dtype = torch.float32

    logger.info(f"Loading reference model from {AUDIO_MODEL_ID}...")
    model = LFM2AudioModel.from_pretrained(
        AUDIO_MODEL_ID,
        dtype=dtype,
        device=device,
    )
    model.eval()

    processor = LFM2AudioProcessor.from_pretrained(
        AUDIO_MODEL_ID,
        device=device,
    )
    # Fix device mismatch for audio_detokenizer
    processor.audio_detokenizer.to(device)

    yield model, processor

    logger.info("Cleaning up reference model...")
    del model
    del processor
    gc.collect()


@pytest.fixture(scope="module")
def onnx_model(exports_dir: pathlib.Path):
    """Load ONNX audio model (fp32).

    Returns LFM2AudioInference instance.
    """
    from liquidonnx.lfm2_audio.infer import LFM2AudioInference

    model_dir = exports_dir / "LFM2.5-Audio-1.5B-ONNX"

    if not model_dir.exists():
        pytest.skip(
            f"ONNX model not found: {model_dir}\n"
            f"Export with: uv run lfm2-audio-export {AUDIO_MODEL_ID}"
        )

    logger.info(f"Loading ONNX model from {model_dir}...")
    model = LFM2AudioInference(model_dir)

    yield model

    logger.info("Cleaning up ONNX model...")
    del model
    gc.collect()


@pytest.fixture(scope="module")
def audio_processor():
    """Load audio processor for decoding.

    Returns LFM2AudioProcessor instance.
    """
    from liquid_audio import LFM2AudioProcessor

    device = "cpu"
    processor = LFM2AudioProcessor.from_pretrained(
        AUDIO_MODEL_ID,
        device=device,
    )
    processor.audio_detokenizer.to(device)

    yield processor

    del processor
    gc.collect()


@pytest.fixture(scope="session")
def sample_audio_short(exports_dir: pathlib.Path) -> pathlib.Path:
    """Short sample audio file for testing."""
    # Navigate from exports/ to samples/
    base_dir = exports_dir.parent
    audio_path = base_dir / "samples" / "audio" / "woodworks_question.wav"
    if not audio_path.exists():
        pytest.skip(f"Sample audio not found: {audio_path}")
    return audio_path


@pytest.fixture(scope="session")
def sample_audio_long(exports_dir: pathlib.Path) -> pathlib.Path:
    """Longer sample audio file for testing."""
    base_dir = exports_dir.parent
    audio_path = base_dir / "samples" / "audio" / "fool_me_once_mono.wav"
    if not audio_path.exists():
        pytest.skip(f"Sample audio not found: {audio_path}")
    return audio_path
