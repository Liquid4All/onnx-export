"""
LFM2-VL Builder for ONNX export.

This package contains builder classes for exporting LFM2-VL models to ONNX:

- config.py: SigLIP2Config, LFM2VLConfig
- vision_builder.py: VisionEmbedBuilder
- embed_tokens_builder.py: EmbedTokensBuilder

The LFM2-VL model exports as three ONNX models:
- embed_tokens.onnx: Token embedding lookup (input_ids -> inputs_embeds)
- embed_images.onnx: SigLIP2 vision encoder + MLP projector (fused)
- decoder.onnx: LFM2 language model backbone (takes inputs_embeds, not input_ids)

Vision Input Formats:
- Tiled (-T): Input [batch, num_patches, 768] with pre-extracted patches
- Conv2d (-C): Input [batch, 3, H, W] with raw normalized image
"""

from liquidonnx.lfm2_vl.builder.config import LFM2VLConfig, SigLIP2Config
from liquidonnx.lfm2_vl.builder.embed_tokens_builder import EmbedTokensBuilder
from liquidonnx.lfm2_vl.builder.vision_builder import VisionEmbedBuilder

__all__ = [
    "SigLIP2Config",
    "LFM2VLConfig",
    "VisionEmbedBuilder",
    "EmbedTokensBuilder",
]
