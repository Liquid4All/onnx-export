#!/bin/bash
# Generate interleaved audio using liquid-audio with detokenizer (ONNX-compatible)
set -e
set -x
mkdir -p output
uv run scripts/interleaved_liquidaudio.py samples/audio/woodworks_question.wav \
    --decoder detokenizer \
    --output output/interleaved_liquidaudio.wav \
    --save-codes output/interleaved_liquidaudio_codes.npy
