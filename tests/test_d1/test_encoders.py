"""
The exported vision and audio encoders of a d1 checkpoint against its reference's
get_image_features and get_audio_features, on the inputs its engine's media reader makes (the
picture capped at d1_vision_max_pixels, audio resampled to 16 kHz), with the bounds of the LFM3-VL
and LFM3-Audio encoder tests.

Run with:
    uv run --no-sync pytest tests/test_d1/test_encoders.py -v --d1-model LiquidAI/d1-2B --device cuda
"""

import logging
import pathlib
import sys

import numpy as np
import pytest
import torch

from liquidonnx.lfm3_vl.session import run_vision
from liquidonnx.session import load_onnx_session

from ..test_lfm3_audio.test_audio_encoder import BOUNDS as AUDIO_BOUNDS
from ..test_lfm3_audio.test_audio_encoder import token_metrics
from ..test_lfm3_vl.test_vision_encoder import FILES as VISION_FILES
from ..test_lfm3_vl.test_vision_encoder import FLOORS as VISION_FLOORS
from .conftest import beeps, cats

logger = logging.getLogger(__name__)

SAMPLES = pathlib.Path(__file__).parents[2] / "samples" / "audio"
VL_ASSETS = pathlib.Path(__file__).parents[1] / "test_lfm2_vl" / "assets"
# name -> (file, size): the README's picture (None: cats()), a tall one, and one past
# d1_vision_max_pixels (4.9 MP), which the reader caps before the processor
IMAGES = {
    "cats": (None, None),
    "tall": (VL_ASSETS / "tall.jpg", None),
    "capped": (VL_ASSETS / "cardinal.jpg", (2560, 1920)),
}
CLIPS = {"beeps": None, "speech": SAMPLES / "woodworks_question.wav"}
AUDIO_FILES = {"fp32": "audio_encoder.onnx", "fp16": "audio_encoder_fp16.onnx"}


def encode(reference, item) -> dict:
    """The processor outputs of one media item (a PIL image or a media dict), as the engine's
    media reader makes them."""
    media = sys.modules[type(reference.media).__module__]
    _, [inputs] = reference.media.encode([media.as_media(item)])
    return inputs


@pytest.mark.parametrize("precision", list(VISION_FLOORS))
@pytest.mark.parametrize("image", list(IMAGES))
def test_vision_features(export_dir, reference, device, ep, precision, image):
    from PIL import Image

    path, size = IMAGES[image]
    picture = Image.open(path or cats())
    inputs = encode(reference, picture.resize(size) if size else picture)
    with torch.no_grad():
        expected = reference.model.get_image_features(
            inputs["pixel_values"].to(device), inputs["image_grid_thw"].to(device)
        )
    # LFM3-VL's code returns a BaseModelOutputWithPooling of per-image tensors, LFM3-Omni's one
    expected = getattr(expected, "pooler_output", expected)
    if isinstance(expected, (list, tuple)):
        expected = torch.cat(expected)
    expected = expected.float().cpu().numpy()
    session = load_onnx_session(export_dir / "onnx" / VISION_FILES[precision], ep=ep, tf32=False)
    got = run_vision(
        session, inputs["pixel_values"].numpy(), inputs["image_grid_thw"].numpy(), merge=2
    )
    _, cosine = token_metrics(got, expected)
    logger.info(
        f"  {image} {precision} grid {inputs['image_grid_thw'][0].tolist()}: worst cos "
        f"{cosine.min():.6f}, p1 {np.percentile(cosine, 1):.6f}, mean {cosine.mean():.6f}, "
        f"max|diff| {np.abs(got - expected).max():.2e}"
    )
    mean_floor, p1_floor = VISION_FLOORS[precision]
    assert cosine.mean() > mean_floor
    assert np.percentile(cosine, 1) > p1_floor


@pytest.mark.parametrize("precision", list(AUDIO_FILES))
@pytest.mark.parametrize("clip", list(CLIPS))
def test_audio_features(model_id, export_dir, reference, device, ep, precision, clip):
    if "omni" not in model_id.lower():
        pytest.skip("d1-Omni only")
    path = CLIPS[clip]
    item = {"type": "audio", "audio": beeps(), "sampling_rate": 16000}
    if path is not None:
        item = {"type": "audio", "path": str(path)}
    inputs = encode(reference, item)
    with torch.no_grad():
        expected = reference.model.get_audio_features(
            inputs["input_features"].to(device), inputs["feature_attention_mask"].to(device)
        )
    expected = expected.float().cpu().numpy()
    session = load_onnx_session(export_dir / "onnx" / AUDIO_FILES[precision], ep=ep, tf32=False)
    got = session.run(None, {"input_features": inputs["input_features"].numpy()})[0]
    assert got.shape == expected.shape
    rel, cosine = token_metrics(got, expected)
    logger.info(
        f"  {clip} {precision} ({len(got)} tokens): rel {rel:.2e}, worst cos {cosine.min():.6f}, "
        f"p1 {np.percentile(cosine, 1):.6f}, max|diff| {np.abs(got - expected).max():.2e}"
    )
    max_rel, p1_floor = AUDIO_BOUNDS[precision]
    assert rel < max_rel
    assert np.percentile(cosine, 1) > p1_floor
