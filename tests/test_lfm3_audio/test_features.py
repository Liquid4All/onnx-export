"""
The runtime's numpy log-mel features (session.log_mel) against the checkpoint's own feature
extractor (Lfm3AudioFeatureExtractor: torch.stft in float32). Needs the checkpoint's config and
remote code (HF cache; no weights).

Run with:
    uv run pytest tests/test_lfm3_audio/test_features.py -v
    uv run pytest tests/test_lfm3_audio/test_features.py -v \\
        --lfm3-audio-model LiquidAI/LFM3-Omni-2B
"""

import pathlib

import numpy as np
import pytest

from liquidonnx.lfm3_audio.config import load_config
from liquidonnx.lfm3_audio.session import load_audio, log_mel, mel_filters

SAMPLES = pathlib.Path(__file__).parents[2] / "samples" / "audio"


def chirp(num_samples: int) -> np.ndarray:
    """A chirp plus noise at 16 kHz."""
    t = np.arange(num_samples) / 16000
    noise = np.random.default_rng(7).standard_normal(num_samples)
    return (0.3 * np.sin(2 * np.pi * (200 * t + 900 * t**2)) + 0.05 * noise).astype(np.float32)


# Below the 400-sample window, exactly one window, one hop more, odd frame counts (T = 99, 36)
SYNTHETIC = {f"chirp_{n}": n for n in (1, 399, 400, 401, 560, 5921, 16000, 16001)}


@pytest.fixture(scope="module")
def extractor(reference_processor):
    return reference_processor.feature_extractor


@pytest.fixture(scope="module")
def mel(checkpoint):
    return load_config(checkpoint).mel


def rel(ours: np.ndarray, ref: np.ndarray) -> float:
    return float(np.linalg.norm(ours - ref) / np.linalg.norm(ref))


def test_mel_filters(extractor, mel):
    assert np.array_equal(mel_filters(mel), extractor.mel_filters.astype(np.float32))


@pytest.mark.parametrize("name", SYNTHETIC)
def test_synthetic(extractor, mel, name):
    wave = chirp(SYNTHETIC[name])
    ref = extractor(wave, sampling_rate=16000)["input_features"][0]
    ours = log_mel(wave, mel)
    assert ours.shape == ref.shape == (128, mel.num_frames(len(wave)))
    assert rel(ours, ref) < 1e-6


def test_speech(extractor, mel):
    """Speech with near-silent stretches: the reference's float32 STFT is itself ~1e-5 (rel) off
    a float64 one there, so speech is held to 2e-5; the synthetic clips meet 1e-6."""
    wave = load_audio(SAMPLES / "woodworks_question.wav", 16000)
    ref = extractor(wave, sampling_rate=16000)["input_features"][0]
    ours = log_mel(wave, mel)
    assert ours.shape == ref.shape
    assert rel(ours, ref) < 2e-5
