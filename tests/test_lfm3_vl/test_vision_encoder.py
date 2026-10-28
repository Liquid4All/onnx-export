"""
The exported vision encoders against the reference's get_image_features on real images.

Run with:
    uv run --no-sync pytest tests/test_lfm3_vl/test_vision_encoder.py -v --device cuda  # README 2
    uv run --no-sync pytest tests/test_lfm3_vl/test_vision_encoder.py -v -k q8 --device cuda --ep cpu
"""

import logging

import numpy as np
import pytest
import torch

from liquidonnx.lfm3_vl.session import run_vision
from liquidonnx.session import load_onnx_session

from .conftest import ASSETS

logger = logging.getLogger(__name__)

IMAGES = ["cardinal.jpg", "bluejay.jpg", "wide.jpg", "tall.jpg"]
# (mean, 1st percentile) token cosine against the float32 reference. A few tokens per image fall
# lower; test_decoder holds what that costs in logits. Only the q4 bundle takes the int8 encoder,
# whose int4 decoder's error is far larger.
FLOORS = {"fp32": (0.99999, 0.9999), "fp16": (0.9998, 0.998), "q8": (0.997, 0.94)}
FILES = {
    "fp32": "vision_encoder.onnx",
    "fp16": "vision_encoder_fp16.onnx",
    "q8": "vision_encoder_q8.onnx",
}


@pytest.mark.parametrize("precision", list(FLOORS))
@pytest.mark.parametrize("image", IMAGES)
def test_vision_features(export_dir, reference, reference_processor, device, ep, precision, image):
    from PIL import Image

    inputs = reference_processor.image_processor(
        images=[Image.open(ASSETS / image).convert("RGB")], return_tensors="pt"
    )
    with torch.no_grad():
        expected = reference.get_image_features(
            inputs["pixel_values"].to(device), inputs["image_grid_thw"].to(device)
        )
    expected = torch.cat(expected.pooler_output).float().cpu().numpy()

    session = load_onnx_session(export_dir / "onnx" / FILES[precision], ep=ep, tf32=False)
    got = run_vision(
        session, inputs["pixel_values"].numpy(), inputs["image_grid_thw"].numpy(), merge=2
    )
    cosine = (
        (got * expected).sum(-1) / np.linalg.norm(got, axis=-1) / np.linalg.norm(expected, axis=-1)
    )
    logger.info(
        f"  {image} {precision} grid {inputs['image_grid_thw'][0].tolist()}: worst cos {cosine.min():.6f}, "
        f"p1 {np.percentile(cosine, 1):.6f}, mean {cosine.mean():.6f}, max|diff| {np.abs(got - expected).max():.2e}"
    )
    mean_floor, p1_floor = FLOORS[precision]
    assert cosine.mean() > mean_floor
    assert np.percentile(cosine, 1) > p1_floor
