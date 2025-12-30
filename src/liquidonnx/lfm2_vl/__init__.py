"""LFM2-VL vision-language model ONNX export and inference."""

MODELS = {
    "450M": "LiquidAI/LFM2-VL-450M",
    "1.6B": "LiquidAI/LFM2-VL-1.6B",
    "3B": "LiquidAI/LFM2-VL-3B",
}

VISION_MODE_TILED = "tiled"
VISION_MODE_CONV2D = "conv2d"
