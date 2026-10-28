"""
lfm3-audio-infer and lfm3-omni-infer (no download): the system prompt of each mode
(infer.conversation of build_parser's arguments), by default and with --language and --voice, and
main() on the CPU on the tiny exports (conftest.tiny_exports): ASR, TTS, S2S and chat, and an image
question. The default prompts are the model cards'.

Run with:
    uv run pytest tests/test_lfm3_audio/test_infer.py -v
"""

import json
import pathlib
import sys

import pytest
import scipy.io.wavfile

from liquidonnx.lfm3.session import load_tokenizer
from liquidonnx.lfm3_audio import infer as audio_infer
from liquidonnx.lfm3_audio import session as audio_session
from liquidonnx.lfm3_audio.session import BUNDLES, AudioSampling, ONNXAudioModel
from liquidonnx.lfm3_omni import infer as omni_infer
from liquidonnx.lfm3_omni.session import BUNDLES as OMNI_BUNDLES

from .remote import SPEECH_SAMPLING
from .tiny import ASSETS

# (mode, flags, system prompt)
PROMPTS = [
    ("asr", [], "Perform ASR in English."),
    ("asr", ["--language", "French"], "Perform ASR in French."),
    ("asr", ["--language", "any"], "Perform ASR."),
    ("tts", [], "Perform TTS in American English using a female voice."),
    (
        "tts",
        ["--language", "British English", "--voice", "male"],
        "Perform TTS in British English using a male voice.",
    ),
    ("s2s", ["--language", "French"], "Respond with interleaved text and audio."),
    ("chat", ["--language", "French"], None),
]


@pytest.mark.parametrize("cli", [audio_infer, omni_infer], ids=["audio", "omni"])
@pytest.mark.parametrize(("mode", "flags", "system"), PROMPTS)
def test_system_prompt(cli, mode, flags, system):
    """Without --language (None, the CLIs' default) asr and tts take their own default; s2s and
    chat ignore it."""
    argv = ["--model", "export", "--mode", mode, "--prompt", "Hello.", *flags]
    messages, _ = audio_infer.conversation(cli.build_parser().parse_args(argv), 0)
    systems = [message["content"] for message in messages if message["role"] == "system"]
    assert systems == ([system] if system else [])


# === main() on the tiny exports ===

QUESTION = pathlib.Path(__file__).parents[2] / "samples" / "audio" / "woodworks_question.wav"
CARDINAL = pathlib.Path(__file__).parents[1] / "test_lfm2_vl" / "assets" / "cardinal.jpg"
AUDIO, IMAGE, AUDIO_OUT, HIDDEN = 125031, 125030, 125032, 64
# mode -> the prompt lfm3-audio-infer sends after <|startoftext|>, {clip} standing for the clip's
# <|audio|> tokens
SENT = {
    "asr": "<|im_start|>system\nPerform ASR in English.<|im_end|>\n<|im_start|>user\n{clip}"
    "<|im_end|>\n<|im_start|>assistant\n",
    "tts": "<|im_start|>system\nPerform TTS in American English using a female voice.<|im_end|>\n"
    "<|im_start|>user\nHello.<|im_end|>\n<|im_start|>assistant\n<|audio_start|>",
    "s2s": "<|im_start|>system\nRespond with interleaved text and audio.<|im_end|>\n"
    "<|im_start|>user\n{clip}<|im_end|>\n<|im_start|>assistant\n<|mixed_start|>",
    "chat": "<|im_start|>user\n{clip}<|im_end|>\n<|im_start|>assistant\n<think></think>",
}


@pytest.fixture
def runs(monkeypatch):
    """The session files each main() loads and its generate_ids calls (ids, audio features,
    keyword arguments, Generation)."""
    recorded = {"loaded": [], "generations": []}
    load_session, generate_ids = audio_session.load_session, ONNXAudioModel.generate_ids

    def record_load(path, **kwargs):
        recorded["loaded"].append(path.name)
        return load_session(path, **kwargs)

    def record_generation(model, ids, audio_features=None, **kwargs):
        generation = generate_ids(model, ids, audio_features, **kwargs)
        recorded["generations"].append((ids, audio_features, kwargs, generation))
        return generation

    monkeypatch.setattr(audio_session, "load_session", record_load)
    monkeypatch.setattr(ONNXAudioModel, "generate_ids", record_generation)
    return recorded


def release_sampling(release: str) -> AudioSampling:
    config = json.loads((ASSETS / release / "generation_config.json").read_text())
    return AudioSampling(*(config[name] for name in SPEECH_SAMPLING))


@pytest.mark.parametrize("mode", SENT)
def test_audio_cli(tiny_exports, runs, monkeypatch, capsys, tmp_path, mode):
    """lfm3-audio-infer --cpu without --precision loads the first exported bundle the CPU prefers
    (q4), prompts with the mode's system prompt (its default language; none for chat) and opening,
    samples speech with the export's generation_config.json and answers in --max-tokens slots: text
    on stdout, and for tts the speech in --output, a 24 kHz WAV. chat, and asr with --suppress-mixed
    and --suppress-audio-out, forbid <|mixed_start|> and <|audio_start|>."""
    export = tiny_exports("lfm3-audio-export", "LFM3-Audio-2B")
    output = tmp_path / "speech.wav"
    argv = ["lfm3-audio-infer", "--model", str(export), "--mode", mode, "--cpu"]
    argv += ["--max-tokens", "3", "--seed", "0"]
    if mode == "tts":
        argv += ["--prompt", "Hello.", "--output", str(output)]
    else:
        argv += ["--audio", str(QUESTION)]
    if mode == "asr":
        argv += ["--suppress-mixed", "--suppress-audio-out"]
    monkeypatch.setattr(sys, "argv", argv)
    audio_infer.main()

    assert runs["loaded"] == list(BUNDLES["q4"].values())
    [(ids, features, kwargs, generation)] = runs["generations"]
    tokenizer = load_tokenizer(export)
    clip = ids.count(AUDIO)
    assert tokenizer.decode(ids) == "<|startoftext|>" + SENT[mode].format(clip="<|audio|>" * clip)
    assert kwargs["audio_sampling"] == release_sampling("LFM3-Audio-2B")
    assert kwargs["suppress_mixed"] == kwargs["suppress_audio_out"] == (mode in ("asr", "chat"))
    assert len(generation.tokens) == 3
    if mode == "tts":
        frames = len(generation.audio_codes)
        rate, wave = scipy.io.wavfile.read(output)
        assert features is None and frames and rate == 24000 and len(wave) == frames * 1920
    else:
        assert clip and features.shape == (clip, HIDDEN)
        text = [token for token in generation.tokens if token != AUDIO_OUT]
        answer = tokenizer.decode(text, skip_special_tokens=True)
        assert answer.strip() and answer in capsys.readouterr().out


def test_omni_cli_answers_about_an_image(tiny_exports, runs, monkeypatch, capsys):
    """lfm3-omni-infer --cpu without --precision loads the first exported bundle the CPU prefers
    (q4, with the int8 vision encoder), puts the image, resized to at most --max-image-tokens
    tokens, before the text of a chat turn (no system prompt, an empty think block, text only)
    and prints the answer."""
    export = tiny_exports("lfm3-omni-export", "LFM3-Omni-2B")
    argv = ["lfm3-omni-infer", "--model", str(export), "--images", str(CARDINAL)]
    argv += ["--prompt", "What bird is this?", "--cpu", "--max-tokens", "3"]
    monkeypatch.setattr(sys, "argv", [*argv, "--max-image-tokens", "16"])
    omni_infer.main()

    assert runs["loaded"] == list(OMNI_BUNDLES["q4"].values())
    [(ids, features, kwargs, generation)] = runs["generations"]
    image_tokens = ids.count(IMAGE)
    assert 0 < image_tokens <= 16 and kwargs["image_features"].shape == (image_tokens, HIDDEN)
    tokenizer = load_tokenizer(export)
    assert tokenizer.decode(ids) == (
        "<|startoftext|><|im_start|>user\n<|vision_start|>"
        + "<|image|>" * image_tokens
        + "<|vision_end|>What bird is this?<|im_end|>\n<|im_start|>assistant\n<think></think>"
    )
    assert features is None and kwargs["suppress_mixed"] and kwargs["suppress_audio_out"]
    assert kwargs["audio_sampling"] == release_sampling("LFM3-Omni-2B")
    assert len(generation.tokens) == 3
    answer = tokenizer.decode(generation.tokens, skip_special_tokens=True)
    assert answer.strip() and answer in capsys.readouterr().out
