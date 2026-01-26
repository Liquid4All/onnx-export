#!/bin/bash
set -e
set -x
mkdir -p output

SYSTEM_PROMPT="Perform TTS. Use the UK female voice."

uv run lfm2-audio-infer exports/LFM2.5-Audio-1.5B-ONNX \
    --mode tts \
    --prompt "Don't ask what you can do for your country. Ask what your country can do for you." \
    --system "$SYSTEM_PROMPT" \
    --output output/tts_onnx.wav
