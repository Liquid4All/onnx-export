"""
The system prompt of each lfm3-audio-infer and lfm3-omni-infer mode (infer.conversation), by
default and with --language and --voice (no download). The defaults are the model cards' prompts.

Run with:
    uv run pytest tests/test_lfm3_audio/test_infer.py -v
"""

import argparse

import pytest

from liquidonnx.lfm3_audio.infer import conversation

# (mode, flags, system prompt)
PROMPTS = [
    ("asr", {}, "Perform ASR in English."),
    ("asr", {"language": "French"}, "Perform ASR in French."),
    ("asr", {"language": "any"}, "Perform ASR."),
    ("tts", {}, "Perform TTS in American English using a female voice."),
    (
        "tts",
        {"language": "British English", "voice": "male"},
        "Perform TTS in British English using a male voice.",
    ),
    ("s2s", {"language": "French"}, "Respond with interleaved text and audio."),
    ("chat", {"language": "French"}, None),
]


@pytest.mark.parametrize(("mode", "flags", "system"), PROMPTS)
def test_system_prompt(mode, flags, system):
    """Without --language (None, the CLIs' default) asr and tts take their own default; s2s and
    chat ignore it."""
    args = {"mode": mode, "prompt": "Hello.", "language": None, "voice": "female"} | flags
    messages, _ = conversation(argparse.Namespace(**args), 0)
    systems = [message["content"] for message in messages if message["role"] == "system"]
    assert systems == ([system] if system else [])
