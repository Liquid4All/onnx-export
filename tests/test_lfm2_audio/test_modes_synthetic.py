"""ASR / TTS / interleaved pipeline tests on a synthetic export (no checkpoint download).

Every ONNX graph, the embedding binaries, the mel front-end and the tokenizer are real;
only the weights are random. Sampling is scripted where the test needs to steer the
orchestration (mode switches, end-of-audio, EOS) - the graphs still run for every step,
so shapes, KV caches and feed-back paths are exercised end to end.

    uv run pytest tests/test_lfm2_audio/test_modes_synthetic.py -v
"""

import pathlib

import numpy as np
import pytest

from liquidonnx.lfm2_audio.infer import LFM2AudioInference

from .synthetic import SPECIAL_TOKENS, VOCAB, build_model_dir, char_id, write_wav

AUDIO_START = SPECIAL_TOKENS["<|audio_start|>"]
TEXT_END = SPECIAL_TOKENS["<|text_end|>"]
IM_END = SPECIAL_TOKENS["<|im_end|>"]
EOA = LFM2AudioInference.END_OF_AUDIO_TOKEN
CODEBOOKS = 8
FRAME_SAMPLES = 6 * 320  # detokenizer upsample × ISTFT hop


class ScriptedSampler:
    """Drop-in for LFM2AudioInference._sample driven by fixed token scripts.

    Text logits have VOCAB entries, depthformer logits 2049; the two scripts are
    consumed independently. A script that runs out repeats its last value.
    """

    def __init__(self, text: list[int], audio_frames: list[list[int]]):
        self.text = list(text)
        self.audio = [code for frame in audio_frames for code in frame]
        self.text_i = 0
        self.audio_i = 0

    def __call__(self, logits, temperature, top_p=None, top_k=None) -> int:
        if len(logits) == EOA + 1:
            code = self.audio[min(self.audio_i, len(self.audio) - 1)]
            self.audio_i += 1
            return code
        assert len(logits) == VOCAB
        token = self.text[min(self.text_i, len(self.text) - 1)]
        self.text_i += 1
        return token


def frame(code: int) -> list[int]:
    return [code] * CODEBOOKS


def end_frame() -> list[int]:
    return [EOA] + [0] * (CODEBOOKS - 1)


def ids(text: str) -> list[int]:
    return [char_id(c) for c in text]


# === Fixtures ===


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory) -> pathlib.Path:
    return build_model_dir(tmp_path_factory.mktemp("export"))


@pytest.fixture(scope="module")
def model(model_dir) -> LFM2AudioInference:
    return LFM2AudioInference(model_dir)


@pytest.fixture(autouse=True)
def fresh_conversation(model):
    model.reset()
    yield
    model.reset()


@pytest.fixture(scope="module")
def wav(tmp_path_factory) -> pathlib.Path:
    return write_wav(tmp_path_factory.mktemp("wav") / "tone.wav")


# === ASR ===


def test_asr_greedy_returns_text(model, wav):
    text = model.transcribe(str(wav), max_new_tokens=5, temperature=0)
    assert isinstance(text, str)


def test_asr_stops_at_im_end(model, wav, monkeypatch):
    monkeypatch.setattr(model, "_sample", ScriptedSampler(ids("hi") + [IM_END], []))
    assert model.transcribe(str(wav), max_new_tokens=10, temperature=0.7) == "hi"


# === TTS ===


def test_tts_stops_at_end_of_audio(model, monkeypatch):
    frames = [frame(11), frame(12), frame(13), end_frame()]
    monkeypatch.setattr(model, "_sample", ScriptedSampler([AUDIO_START], frames))

    codes = model.synthesize("hello", max_new_tokens=20)

    assert [list(c) for c in codes] == frames[:3]
    waveform = model.decode_audio(np.stack(codes))
    assert waveform.shape == (3 * FRAME_SAMPLES,)
    assert np.isfinite(waveform).all()
    assert np.abs(waveform).max() <= 0.9


def test_tts_budget_counts_text_and_audio(model, monkeypatch):
    """max_new_tokens covers the audio_start token plus every frame, like the reference."""
    monkeypatch.setattr(model, "_sample", ScriptedSampler([AUDIO_START], [frame(5)]))
    codes = model.synthesize("hello", max_new_tokens=4)
    assert len(codes) == 3


def test_tts_forces_audio_when_text_never_yields(model, monkeypatch):
    monkeypatch.setattr(model, "_sample", ScriptedSampler(ids("x"), [frame(5)]))
    codes = model.synthesize("hello", max_new_tokens=6)
    # 6 text tokens exhaust the budget, audio_start is forced, no frames fit
    assert codes == []


# === Interleaved ===


def test_interleaved_alternates_by_count(model, monkeypatch):
    """N_TEXT tokens, then N_AUDIO frames, then text again."""
    n_text, n_audio = model.INTERLEAVED_N_TEXT, model.INTERLEAVED_N_AUDIO
    monkeypatch.setattr(model, "_sample", ScriptedSampler(ids("a"), [frame(5)]))

    text, codes = model.generate_interleaved_from_text("hi", max_new_tokens=2 * n_text + n_audio)

    assert text == "a" * (2 * n_text)
    assert len(codes) == n_audio
    assert all((c == 5).all() for c in codes)


def test_interleaved_text_end_forces_audio_then_eoa_returns_to_text(model, monkeypatch):
    script = ScriptedSampler(ids("ab") + [TEXT_END, IM_END], [frame(9), frame(9), end_frame()])
    monkeypatch.setattr(model, "_sample", script)

    text, codes = model.generate_interleaved_from_text("hi", max_new_tokens=50)

    assert text == "ab"
    assert len(codes) == 2
    waveform = model.decode_audio(np.stack(codes))
    assert waveform.shape == (2 * FRAME_SAMPLES,)
    assert np.isfinite(waveform).all()


def test_interleaved_multi_turn_keeps_cache(model, monkeypatch):
    monkeypatch.setattr(model, "_sample", ScriptedSampler(ids("ok") + [IM_END], []))

    assert model.cache is None
    model.generate_interleaved_from_text("first")
    first_len = model.cache_seq_len
    assert model.cache is not None and first_len > 0

    model.generate_interleaved_from_text("second")
    assert model.cache_seq_len > first_len

    model.reset()
    assert model.cache is None and model.cache_seq_len == 0


def test_interleaved_from_audio(model, wav, monkeypatch):
    monkeypatch.setattr(model, "_sample", ScriptedSampler(ids("b") + [IM_END], []))
    text, codes = model.generate_interleaved_from_audio(str(wav), text_prompt="and text")
    assert text == "b"
    assert codes == []
    assert model.cache_seq_len > 0


def test_generate_interleaved_stateless(model, monkeypatch):
    script = ScriptedSampler([AUDIO_START, IM_END], [frame(3), end_frame()])
    monkeypatch.setattr(model, "_sample", script)

    text, codes = model.generate_interleaved("hi", max_new_tokens=10, audio_temperature=1.0)

    assert text == ""  # audio_start is a special token
    assert len(codes) == 1 and (codes[0] == 3).all()
    assert model.cache is None  # this entry point does not touch conversation state
