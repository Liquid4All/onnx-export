#!/bin/bash
# Generate interleaved audio using liquid-audio with mimi decoder (official demo style)
set -e
set -x
mkdir -p output
uv run scripts/interleaved_liquidaudio.py samples/audio/woodworks_question.wav \
    --decoder mimi \
    --output output/interleaved_liquidaudio_mimi.wav \
    --save-codes output/interleaved_liquidaudio_mimi_codes.npy
