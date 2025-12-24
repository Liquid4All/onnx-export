#!/usr/bin/env python3
"""
Benchmark script for LFM2-VL ONNX models.

Usage:
    lfm2-vl-bench --model LFM2-VL-1.6B-ONNX

TODO: Implement benchmarking for LFM2-VL vision-language models.
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Benchmark LFM2-VL ONNX models")
    parser.add_argument("--model", required=True, help="Path to ONNX model directory")
    parser.add_argument("--image", default=None, help="Test image path")
    parser.add_argument("--iterations", type=int, default=10, help="Number of iterations")
    args = parser.parse_args()

    print("LFM2-VL benchmarking is not yet implemented.")
    print(f"Model: {args.model}")


if __name__ == "__main__":
    main()
