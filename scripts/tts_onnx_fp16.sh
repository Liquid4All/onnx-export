#!/bin/bash
set -e
set -x
mkdir -p output
uv run lfm2-audio-infer exports/LFM2.5-Audio-1.5B-ONNX \
    --mode tts \
    --precision fp16 \
    --prompt "Don't ask what you can do for your country. Ask what your country can do for you." \
    --output output/tts_onnx_fp16.wav
