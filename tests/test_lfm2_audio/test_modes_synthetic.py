"""lfm2-audio-infer's pipeline on a synthetic export (no checkpoint download).

Every ONNX graph, the tokenizer and genai_config.json are real; only the weights are random. The
mode switching itself (audio start, interleaving by count, end of audio) runs in
onnxruntime-genai and is tested there; these tests check what this repository adds around it.

    uv run pytest tests/test_lfm2_audio/test_modes_synthetic.py -v
"""

import pathlib

import numpy as np
import pytest
from helpers import require_genai

from liquidonnx.lfm2_audio.infer import (
    AUDIO_MARKER,
    HOP_LENGTH,
    NUM_CODEBOOKS,
    AudioChat,
    Detokenizer,
    chat_prompt,
    default_precision,
    single_turn,
    write_wav,
)

from .synthetic import build_model_dir
from .synthetic import write_wav as write_tone

SAMPLES_PER_FRAME = 6 * HOP_LENGTH


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory) -> pathlib.Path:
    return build_model_dir(tmp_path_factory.mktemp("export"))


@pytest.fixture(scope="module")
def chat(model_dir) -> AudioChat:
    require_genai("lfm2_audio")
    return AudioChat(model_dir)


@pytest.fixture(scope="module")
def tone(tmp_path_factory) -> pathlib.Path:
    return write_tone(tmp_path_factory.mktemp("wav") / "tone.wav")


def test_chat_prompt():
    turns, audios = single_turn("and this", "clip.wav")
    assert chat_prompt("Perform ASR.", turns) == (
        "<|startoftext|><|im_start|>system\nPerform ASR.<|im_end|>\n"
        "<|im_start|>user\n<|audio|>and this<|im_end|>\n<|im_start|>assistant\n"
    )
    assert audios == ["clip.wav"]
    assert chat_prompt(None, [("user", "hi"), ("assistant", "hello"), ("user", "bye")]) == (
        "<|startoftext|><|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\nhello<|im_end|>\n"
        "<|im_start|>user\nbye<|im_end|>\n<|im_start|>assistant\n"
    )


def test_default_precision(model_dir):
    assert default_precision(model_dir) == "fp32"


def test_detokenizer_waveform(model_dir, tmp_path):
    codes = np.random.default_rng(0).integers(0, 2049, (3, NUM_CODEBOOKS))
    wave = Detokenizer(model_dir / "onnx" / "audio_detokenizer.onnx")(codes)
    assert wave.shape == (3 * SAMPLES_PER_FRAME,)
    assert np.isfinite(wave).all()

    write_wav(str(tmp_path / "out.wav"), wave)
    assert (tmp_path / "out.wav").stat().st_size > 2 * len(wave)


@pytest.mark.parametrize(
    "mode,text,audio",
    [
        ("text", "hello", False),
        ("asr", None, True),
        ("tts", "hello", False),
        ("interleaved", None, True),
        ("interleaved", "hello", False),
    ],
)
def test_modes_run(chat, tone, mode: str, text: str | None, audio: bool):
    answer = chat.answer(mode, *single_turn(text, tone if audio else None), max_new_tokens=24)
    assert isinstance(answer.text, str)
    assert answer.codes.shape[1] == NUM_CODEBOOKS
    if len(answer.codes):
        assert answer.codes.min() >= 0 and answer.codes.max() <= 2048
        assert chat.detokenizer(answer.codes).shape == (len(answer.codes) * SAMPLES_PER_FRAME,)


def test_multi_turn_resends_the_conversation(chat, tone, monkeypatch):
    prompts, processor = [], chat.processor

    def record(prompt, **kwargs):
        prompts.append(prompt)
        return processor(prompt, **kwargs)

    monkeypatch.setattr(chat, "processor", record)
    turns, audios = single_turn(None, tone)
    first = chat.answer("interleaved", turns, audios, max_new_tokens=8)
    turns += [("assistant", first.text), ("user", "again")]
    chat.answer("interleaved", turns, audios, max_new_tokens=8)

    assert prompts[1].startswith(prompts[0] + first.text + "<|im_end|>")
    assert prompts[1].count(AUDIO_MARKER) == len(audios) == 1
