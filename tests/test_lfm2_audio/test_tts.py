"""
TTS on the LFM2.5-Audio export through onnxruntime-genai.

Every precision samples speech with the model card's settings; at fp32, greedy audio codes must
equal liquid-audio's frame for frame.

Run with:
    uv run pytest tests/test_lfm2_audio/test_tts.py -v
    uv run pytest tests/test_lfm2_audio/test_tts.py -v -k "q4"
"""

import logging

import numpy as np
import pytest

from liquidonnx.compare.audio import reference_answer
from liquidonnx.lfm2_audio.export import PRECISIONS as DERIVED
from liquidonnx.lfm2_audio.infer import HOP_LENGTH, SYSTEM_PROMPTS, single_turn

logger = logging.getLogger(__name__)

PRECISIONS = ["fp32", *DERIVED]
PROMPTS = ["Hello, how are you today?", "The quick brown fox jumps over the lazy dog."]
SAMPLES_PER_FRAME = 6 * HOP_LENGTH  # the detokenizer upsamples each frame 6x


def speak(chat, text: str, **sampling):
    answer = chat.answer("tts", *single_turn(text), random_seed=42, **sampling)
    logger.info(f"  {text!r}: {len(answer.codes)} frames")
    return answer


@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize("prompt", PROMPTS)
def test_tts_generation(audio_chat, precision: str, prompt: str):
    chat = audio_chat(precision)
    answer = speak(chat, prompt)

    assert len(answer.codes) >= 5
    assert answer.codes.min() >= 0 and answer.codes.max() < 2048
    wave = chat.detokenizer(answer.codes)
    assert wave.shape == (len(answer.codes) * SAMPLES_PER_FRAME,)
    assert np.isfinite(wave).all()


@pytest.mark.parametrize("precision", PRECISIONS)
def test_tts_is_seeded(audio_chat, precision: str):
    chat = audio_chat(precision)
    np.testing.assert_array_equal(speak(chat, "Hello").codes, speak(chat, "Hello").codes)


def test_tts_length_consistent_across_precisions(audio_chat):
    frames = {p: len(speak(audio_chat(p), "Hello world").codes) for p in PRECISIONS}
    logger.info(f"  frames per precision: {frames}")
    assert min(frames.values()) / max(frames.values()) > 0.5


@pytest.mark.parametrize("prompt", PROMPTS)
def test_tts_matches_reference(reference_model, audio_chat, prompt: str):
    """fp32 with greedy audio codes, frame for frame against liquid-audio."""
    _, frames, _ = reference_answer(
        *reference_model, SYSTEM_PROMPTS["tts"], prompt, None, False, max_new=160
    )
    answer = speak(
        audio_chat("fp32"), prompt, max_new_tokens=160, audio_temperature=0.0, audio_top_k=1
    )
    np.testing.assert_array_equal(answer.codes, frames)
