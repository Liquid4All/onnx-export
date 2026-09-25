"""LFM2.5-Audio test fixtures."""

import gc
import logging
import pathlib

import pytest
import torch
from helpers import require_genai

from liquidonnx.lfm2_audio.export import bundle

logger = logging.getLogger(__name__)

AUDIO_MODEL_ID = "LiquidAI/LFM2.5-Audio-1.5B"
SAMPLES_DIR = pathlib.Path(__file__).parents[2] / "samples" / "audio"


@pytest.fixture(scope="module")
def reference_model():
    """(model, processor) of liquid-audio in fp32 on CPU."""
    from liquid_audio import LFM2AudioModel, LFM2AudioProcessor

    logger.info(f"Loading reference model from {AUDIO_MODEL_ID}...")
    model = LFM2AudioModel.from_pretrained(AUDIO_MODEL_ID, dtype=torch.float32, device="cpu")
    model.eval()
    processor = LFM2AudioProcessor.from_pretrained(AUDIO_MODEL_ID, device="cpu")
    yield model, processor

    del model, processor
    gc.collect()


@pytest.fixture(scope="module")
def audio_chat(exports_dir: pathlib.Path):
    """precision -> AudioChat on exports/LFM2.5-Audio-1.5B-ONNX, loaded once per module."""
    from liquidonnx.lfm2_audio.infer import AudioChat

    require_genai("lfm2_audio")
    model_dir = exports_dir / "LFM2.5-Audio-1.5B-ONNX"
    if not model_dir.exists():
        pytest.skip(
            f"{model_dir} not found; export with: uv run lfm2-audio-export {AUDIO_MODEL_ID}"
        )
    chats = {}

    def load(precision: str) -> AudioChat:
        if precision not in chats:
            missing = [
                f for f in bundle(precision).values() if not (model_dir / "onnx" / f).exists()
            ]
            if missing:
                pytest.skip(f"{precision} not exported: {', '.join(missing)}")
            chats[precision] = AudioChat(model_dir, precision)
        return chats[precision]

    yield load

    chats.clear()
    gc.collect()


@pytest.fixture(scope="session")
def sample_audio_short() -> pathlib.Path:
    return SAMPLES_DIR / "woodworks_question.wav"


@pytest.fixture(scope="session")
def sample_audio_long() -> pathlib.Path:
    return SAMPLES_DIR / "fool_me_once_mono.wav"
