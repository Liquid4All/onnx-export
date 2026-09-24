"""
LFM2-VL vision encoder builder.

- config.py: SigLIP2Config, LFM2VLConfig
- vision_builder.py: VisionEmbedBuilder (SigLIP2 NaViT tower + MLP projector, fused)

The decoder comes from the onnxruntime-genai model builder and the embedding model from
liquidonnx.embeddings.

Vision Input Formats:
- Tiled (-T): Input [batch, num_patches, 768] with pre-extracted patches
- Conv2d (-C): Input [batch, 3, H, W] with raw normalized image
"""

from liquidonnx.lfm2_vl.builder.config import LFM2VLConfig, SigLIP2Config
from liquidonnx.lfm2_vl.builder.vision_builder import VisionEmbedBuilder

__all__ = [
    "SigLIP2Config",
    "LFM2VLConfig",
    "VisionEmbedBuilder",
]
