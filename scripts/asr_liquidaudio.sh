#!/bin/bash
set -e
set -x
mkdir -p output
uv run scripts/asr_liquidaudio.py samples/audio/fool_me_once_mono.wav \
    | tee output/asr_liquidaudio.txt
