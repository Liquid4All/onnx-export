#!/bin/bash
set -e
set -x
mkdir -p output
uv run lfm2-audio-infer exports/LFM2.5-Audio-1.5B-ONNX \
    --mode asr \
    --audio samples/audio/fool_me_once_mono.wav \
    | tee output/asr_onnx.txt
