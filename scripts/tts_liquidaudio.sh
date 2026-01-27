#!/bin/bash
set -e
set -x
mkdir -p output

SYSTEM_PROMPT="Perform TTS. Use the UK female voice."

uv run scripts/tts_liquidaudio.py \
    "Don't ask what you can do for your country. Ask what your country can do for you." \
    --system "$SYSTEM_PROMPT" \
    --output output/tts_liquidaudio.wav
