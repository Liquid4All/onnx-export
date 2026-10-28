"""LFM3-VL's transformers classes (`native_lfm3`): its config, vision tower, picture and video processors."""

from __future__ import annotations

from .native_lfm3 import Lfm3Model  # noqa: F401  (registered: the VL builds its text model by config)

try:
    from transformers import Lfm3VlConfig, Lfm3VlForConditionalGeneration, Lfm3VlProcessor
except ImportError:
    from transformers import AutoConfig, AutoImageProcessor, AutoModel, AutoProcessor, AutoVideoProcessor

    from .configuration_lfm3_vl import Lfm3VlConfig, Lfm3VlVisionConfig
    from .image_processing_lfm3_vl import Lfm3VlImageProcessor
    from .modeling_lfm3_vl import Lfm3VlForConditionalGeneration, Lfm3VlVisionModel
    from .processing_lfm3_vl import Lfm3VlProcessor
    from .video_processing_lfm3_vl import Lfm3VlVideoProcessor

    for config in (Lfm3VlConfig, Lfm3VlVisionConfig):
        AutoConfig.register(config.model_type, config, exist_ok=True)
    AutoModel.register(Lfm3VlVisionConfig, Lfm3VlVisionModel, exist_ok=True)  # the tower, built by config
    AutoImageProcessor.register(Lfm3VlConfig, fast_image_processor_class=Lfm3VlImageProcessor, exist_ok=True)
    AutoVideoProcessor.register(Lfm3VlConfig, Lfm3VlVideoProcessor, exist_ok=True)
    AutoProcessor.register(Lfm3VlConfig, Lfm3VlProcessor, exist_ok=True)

__all__ = ["Lfm3VlConfig", "Lfm3VlForConditionalGeneration", "Lfm3VlProcessor"]
