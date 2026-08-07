"""LFM2-VL vision-language model ONNX export and inference."""

VISION_MODE_TILED = "tiled"
VISION_MODE_CONV2D = "conv2d"

# Public inference API. Imported after the constants above: infer.py (and the
# modules it pulls in) import the constants back from this package, which works
# only because they are already bound when the submodule import starts.
from liquidonnx.lfm2_vl.infer import (  # noqa: E402
    VLModelInference,
    resolve_precision_files,
)

__all__ = [
    "VISION_MODE_CONV2D",
    "VISION_MODE_TILED",
    "VLModelInference",
    "resolve_precision_files",
]
