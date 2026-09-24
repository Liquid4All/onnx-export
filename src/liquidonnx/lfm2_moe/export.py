#!/usr/bin/env python3
"""
Export LFM2-MoE models to ONNX for onnxruntime-genai.

Same pipeline as lfm2-export (see liquidonnx.lfm2.export); the experts become QMoE int4 (q4*)
or int8 (q8) and the routers stay fp32.

Usage:
    uv run lfm2-moe-export LiquidAI/LFM2.5-8B-A1B --precision
    uv run lfm2-moe-export LiquidAI/LFM2.5-8B-A1B --precision q4 q8
"""

from liquidonnx.lfm2.export import MOE_PRECISIONS
from liquidonnx.lfm2.export import main as export_main


def main():
    export_main(MOE_PRECISIONS, description="Export LFM2-MoE models to ONNX for onnxruntime-genai")


if __name__ == "__main__":
    main()
