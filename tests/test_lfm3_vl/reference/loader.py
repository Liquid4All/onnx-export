"""
The LFM3-VL reference: a checkpoint's own modeling code, run with trust_remote_code.

The other files of this directory are LiquidAI/LFM3-VL-2B's remote code at revision f83d2b0,
vendored unchanged: tests/test_lfm3_vl/tiny.py copies them into the tiny checkpoint's folder, which
loads them like any checkpoint, so the tests need no download.
"""

import pathlib

REFERENCE_DIR = pathlib.Path(__file__).parent


def load_reference(checkpoint: str | pathlib.Path):
    """float32 Lfm3VlForConditionalGeneration of a checkpoint directory, every tensor loaded."""
    import torch
    from transformers import AutoModelForImageTextToText

    model, info = AutoModelForImageTextToText.from_pretrained(
        str(checkpoint), dtype=torch.float32, trust_remote_code=True, output_loading_info=True
    )
    problems = {k: v for k, v in info.items() if v}
    if problems:
        raise ValueError(f"reference loaded {checkpoint} with {problems}")
    layer_types = [
        type(layer.conv if hasattr(layer, "conv") else layer.self_attn).__name__
        for layer in model.model.language_model.layers
    ]
    if not any(name.endswith("ShortConv") for name in layer_types):
        raise ValueError(f"reference built no conv layers: {layer_types}")
    return model.eval()
