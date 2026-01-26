#!/bin/bash
set -e
set -x
mkdir -p output
uv run lfm2-audio-infer exports/LFM2.5-Audio-1.5B-ONNX \
    --mode interleaved \
    --precision q4 \
    --audio samples/audio/woodworks_question.wav \
    --output output/interleaved_onnx_q4.wav \
    --save-codes output/interleaved_onnx_q4_codes.npy \
    --seed 42
