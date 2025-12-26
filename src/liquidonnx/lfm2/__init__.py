"""LFM2 text model ONNX export and inference."""

from .builder import LFM2Builder, LFM2Config, export_model

MODELS = {
    "350M": "LiquidAI/LFM2-350M",
    "700M": "LiquidAI/LFM2-700M",
    "1.2B": "LiquidAI/LFM2-1.2B",
    "2.6B": "LiquidAI/LFM2-2.6B",
}

__all__ = [
    "MODELS",
    "LFM2Config",
    "LFM2Builder",
    "export_model",
]
