"""
Interleaved text and speech on the LFM2.5-Audio export through onnxruntime-genai.

Every precision samples speech with the model card's settings; at fp32, greedy audio codes must
reproduce liquid-audio's text and frames exactly.

Run with:
    uv run pytest tests/test_lfm2_audio/test_interleaved.py -v
    uv run pytest tests/test_lfm2_audio/test_interleaved.py -v -k "fp32"
"""

import logging
import pathlib

import numpy as np
import pytest

from liquidonnx.compare.audio import reference_answer
from liquidonnx.lfm2_audio.export import PRECISIONS as DERIVED
from liquidonnx.lfm2_audio.infer import SYSTEM_PROMPTS, single_turn, user_content

logger = logging.getLogger(__name__)

PRECISIONS = ["fp32", *DERIVED]
MAX_NEW_TOKENS = 160


def interleave(chat, turns, audios, **sampling):
    answer = chat.answer("interleaved", turns, audios, MAX_NEW_TOKENS, random_seed=42, **sampling)
    logger.info(f"  {len(answer.codes)} frames: {answer.text}")
    return answer


@pytest.mark.parametrize("precision", PRECISIONS)
def test_interleaved_audio_input(audio_chat, sample_audio_short: pathlib.Path, precision: str):
    answer = interleave(audio_chat(precision), *single_turn(None, sample_audio_short))
    assert answer.text.strip()
    assert len(answer.codes) > 0
    assert answer.codes.max() < 2048


@pytest.mark.parametrize("precision", PRECISIONS)
def test_interleaved_text_input(audio_chat, precision: str):
    answer = interleave(audio_chat(precision), *single_turn("What are the three primary colors?"))
    assert answer.text.strip()
    assert len(answer.codes) > 0


def test_interleaved_multi_turn(audio_chat, sample_audio_short: pathlib.Path):
    """A second turn after a spoken question, with the first answer in the conversation."""
    chat = audio_chat("fp32")
    turns, audios = single_turn(None, sample_audio_short)
    first = interleave(chat, turns, audios)
    turns += [("assistant", first.text), ("user", user_content(None, "Make it shorter."))]
    second = interleave(chat, turns, audios)
    assert second.text.strip()
    assert len(second.codes) > 0


def test_interleaved_matches_reference(
    reference_model, audio_chat, sample_audio_short: pathlib.Path
):
    """fp32 with greedy audio codes: the same text and frames as liquid-audio."""
    model, processor = reference_model
    tokens, frames, _ = reference_answer(
        model,
        processor,
        SYSTEM_PROMPTS["interleaved"],
        None,
        sample_audio_short,
        True,
        MAX_NEW_TOKENS,
    )
    answer = interleave(
        audio_chat("fp32"),
        *single_turn(None, sample_audio_short),
        audio_temperature=0.0,
        audio_top_k=1,
    )
    assert answer.text == processor.text.decode(tokens, skip_special_tokens=True)
    np.testing.assert_array_equal(answer.codes, frames)
