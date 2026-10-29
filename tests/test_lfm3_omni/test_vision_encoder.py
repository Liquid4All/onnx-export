"""
The exported vision encoders (LFM3-VL's graph built from LFM3-Omni's tower) against the reference's
get_image_features and get_video_features, on test_lfm3_vl's images and a short video.

Run with:
    uv run --no-sync pytest tests/test_lfm3_omni/test_vision_encoder.py -v --device cuda
    uv run pytest tests/test_lfm3_omni/test_vision_encoder.py -v
"""

import logging

import numpy as np
import pytest
import torch
from helpers import expect_known_failure

from liquidonnx.lfm3_vl.session import run_vision
from liquidonnx.session import load_onnx_session

from ..test_lfm3_audio.test_decoder import cosine
from ..test_lfm3_vl.test_vision_encoder import FILES, FLOORS, IMAGES
from .conftest import load_images, video_frames

logger = logging.getLogger(__name__)

# LFM3-VL's floors, except fp16's 1st percentile: Omni's tower ends in large activations (up to
# ~350 after its last layer), where fp16 math loses a few tokens, in fp16 PyTorch as in the graph.
# The fp16 bundle's image answers keep the reference's tokens (test_decoder, test_generation).
FLOORS = {**FLOORS, "fp16": (FLOORS["fp16"][0], 0.997)}
# (model, test, parameters' id, host or "*") -> the evidence of a known failure
# (helpers.expect_known_failure)
KNOWN_FAILURES = {
    ("LiquidAI/LFM3-Omni-2B", "test_vision_features", "wide.jpg-fp16", h): (
        "the fp16 vision encoder on the CPU EP: 1st-percentile token cosine 0.9958 to 0.9960 on "
        "wide.jpg, under 0.997, where CUDA measures 0.9980 (README 3.1e)"
    )
    for h in ("cpu-x86_64", "cpu-arm64")
}


@pytest.mark.parametrize("precision", list(FLOORS))
@pytest.mark.parametrize("media", [*IMAGES, "video"])
def test_vision_features(
    request, model_id, export_dir, reference, reference_processor, device, ep, precision, media
):
    """Token cosine (mean, 1st percentile) over LFM3-VL's floors."""
    expect_known_failure(request, KNOWN_FAILURES, model_id, ep)
    if media == "video":
        frames, metadata = video_frames()
        inputs = reference_processor.video_processor(
            videos=[frames], video_metadata=[metadata], return_tensors="pt"
        )
        pixels, grids = inputs["pixel_values_videos"], inputs["video_grid_thw"]
    else:
        inputs = reference_processor.image_processor(
            images=load_images([media]), return_tensors="pt"
        )
        pixels, grids = inputs["pixel_values"], inputs["image_grid_thw"]
    with torch.no_grad():
        expected = reference.model.get_image_features(pixels.to(device), grids.to(device))
    expected = expected.float().cpu().numpy()

    session = load_onnx_session(export_dir / "onnx" / FILES[precision], ep=ep, tf32=False)
    got = run_vision(session, pixels.numpy(), grids.numpy(), merge=2)
    similarity = cosine(got, expected)
    logger.info(
        f"  {media} {precision} grid {grids.tolist()}: worst cos {similarity.min():.6f}, "
        f"p1 {np.percentile(similarity, 1):.6f}, mean {similarity.mean():.6f}, "
        f"max|diff| {np.abs(got - expected).max():.2e}"
    )
    mean_floor, p1_floor = FLOORS[precision]
    assert similarity.mean() > mean_floor
    assert np.percentile(similarity, 1) > p1_floor
