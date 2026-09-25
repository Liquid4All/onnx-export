"""
ASR on the LFM2.5-Audio export through onnxruntime-genai, at every precision.

Run with:
    uv run pytest tests/test_lfm2_audio/test_asr.py -v
    uv run pytest tests/test_lfm2_audio/test_asr.py -v -k "fp16"
"""

import logging
import pathlib

import pytest

from liquidonnx.lfm2_audio.export import PRECISIONS as DERIVED
from liquidonnx.lfm2_audio.infer import single_turn

logger = logging.getLogger(__name__)

PRECISIONS = ["fp32", *DERIVED]

ASR_KEYWORDS = {
    "fool_me_once_mono.wav": ["tennessee", "texas", "fool"],
    "woodworks_question.wav": ["woodwork", "slogan", "tagline"],
}


def transcribe(chat, clip: pathlib.Path) -> str:
    answer = chat.answer("asr", *single_turn(None, clip))
    assert not len(answer.codes), "ASR answered in speech"
    logger.info(f"  {clip.name}: {answer.text}")
    return answer.text


def assert_keywords(clip: pathlib.Path, text: str):
    found = [keyword for keyword in ASR_KEYWORDS[clip.name] if keyword in text.lower()]
    assert found, f"none of {ASR_KEYWORDS[clip.name]} in {text!r}"


@pytest.mark.parametrize("precision", PRECISIONS)
def test_asr_short_audio(audio_chat, sample_audio_short: pathlib.Path, precision: str):
    text = transcribe(audio_chat(precision), sample_audio_short)
    assert len(text.split()) >= 3
    assert_keywords(sample_audio_short, text)


@pytest.mark.parametrize("precision", PRECISIONS)
def test_asr_long_audio(audio_chat, sample_audio_long: pathlib.Path, precision: str):
    text = transcribe(audio_chat(precision), sample_audio_long)
    assert len(text) > 20
    assert_keywords(sample_audio_long, text)
