"""
LFM3-VL bundles on long multimodal prompts against the reference (opt-in: --run-long):

- a 10 s video at 2 fps (20 frames of 832 x 1472 drifting across the four test images; 10 temporal
  patches of 52 x 92, about 12K prompt tokens), without and with thinking;
- the four test images in one turn (about 3.5K tokens).

Each is scored as the short prompts are: the vision encoders' video features (test_video.py's
floors), the answer KL of an uncached prefill (test_decoder.py's bounds and confident-token rule)
and a cached greedy decode in the reference's top 5 (test_decode.py).

Run with:
    uv run --no-sync pytest tests/test_lfm3_vl/test_long.py -v --device cuda --run-long
"""

import logging

import numpy as np
import pytest
import torch

from liquidonnx.lfm3_vl.session import run_vision
from liquidonnx.session import load_onnx_session

from .conftest import ASSETS
from .test_decode import assert_cached_decode
from .test_decoder import MAX_KL, assert_answer_logits, reference_answer, reference_inputs
from .test_vision_encoder import FILES, FLOORS, IMAGES

logger = logging.getLogger(__name__)

pytestmark = pytest.mark.long

FRAME = (832, 1472)  # 26 x 46 merged patches of 32 pixels
# name -> (images, video, question, enable_thinking, answer tokens)
PROMPTS = {
    "video": ([], True, "Describe what happens in this video.", False, 48),
    "video-thinking": (
        [],
        True,
        "What is the main object in this video and what is it doing?",
        True,
        64,
    ),
    "four-images": (
        IMAGES,
        False,
        "Describe each of the four images in one short sentence, in order.",
        False,
        64,
    ),
}


def video_frames() -> np.ndarray:
    """[20, 832, 1472, 3] uint8: five frames per test image, each image resized to cover
    1600 x 960 and cropped at a corner drifting down and right."""
    from PIL import Image

    frames = []
    height, width = FRAME
    for name in IMAGES:
        image = Image.open(ASSETS / name).convert("RGB")
        scale = max(1600 / image.width, 960 / image.height)
        size = (round(image.width * scale), round(image.height * scale))
        pixels = np.asarray(image.resize(size, Image.BICUBIC))
        dy, dx = (pixels.shape[0] - height) // 4, (pixels.shape[1] - width) // 4
        frames += [pixels[i * dy : i * dy + height, i * dx : i * dx + width] for i in range(5)]
    return np.stack(frames)


def video_inputs(reference_processor, question: str, enable_thinking: bool):
    from transformers.video_utils import VideoMetadata

    frames = video_frames()
    content = [{"type": "video"}, {"type": "text", "text": question}]
    text = reference_processor.apply_chat_template(
        [{"role": "user", "content": content}],
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=enable_thinking,
    )
    metadata = VideoMetadata(
        total_num_frames=len(frames), fps=2.0, frames_indices=list(range(len(frames)))
    )
    return reference_processor(
        text=text, videos=[frames], video_metadata=[metadata], return_tensors="pt"
    )


@pytest.fixture(scope="module")
def long_prompts(reference, reference_processor, device):
    """name -> (processor inputs, the reference's greedy answer, its logits), once per prompt."""
    built = {}

    def get(name: str):
        if name not in built:
            images, video, question, enable_thinking, answer = PROMPTS[name]
            if video:
                inputs = video_inputs(reference_processor, question, enable_thinking)
            else:
                inputs = reference_inputs(reference_processor, images, question, enable_thinking)
            grids = {
                k: inputs[k].tolist() for k in ("image_grid_thw", "video_grid_thw") if k in inputs
            }
            logger.info(f"  {name}: {inputs['input_ids'].shape[1]} prompt tokens, grids {grids}")
            built[name] = (inputs, *reference_answer(reference, inputs, device, answer))
            logger.info(f"  reference: {reference_processor.tokenizer.decode(built[name][1])!r}")
        return built[name]

    return get


@pytest.mark.parametrize("precision", list(FLOORS))
def test_long_video_features(export_dir, reference, reference_processor, device, ep, precision):
    video = video_inputs(reference_processor, PROMPTS["video"][2], False)
    pixel_values, grids = video["pixel_values_videos"], video["video_grid_thw"]
    with torch.no_grad():
        expected = reference.get_video_features(pixel_values.to(device), grids.to(device))
    expected = torch.cat(getattr(expected, "pooler_output", expected)).float().cpu().numpy()

    session = load_onnx_session(export_dir / "onnx" / FILES[precision], ep=ep, tf32=False)
    got = run_vision(session, pixel_values.numpy(), grids.numpy(), merge=2)
    cosine = (
        (got * expected).sum(-1) / np.linalg.norm(got, axis=-1) / np.linalg.norm(expected, axis=-1)
    )
    logger.info(
        f"  {precision} grid {grids[0].tolist()}, {len(cosine)} tokens: worst cos "
        f"{cosine.min():.6f}, p1 {np.percentile(cosine, 1):.6f}, mean {cosine.mean():.6f}"
    )
    mean_floor, p1_floor = FLOORS[precision]
    assert cosine.mean() > mean_floor
    assert np.percentile(cosine, 1) > p1_floor


@pytest.mark.parametrize("precision", list(MAX_KL))
@pytest.mark.parametrize("name", list(PROMPTS))
def test_long_prefill_logits(long_prompts, onnx_models, precision, name):
    inputs, answer, expected = long_prompts(name)
    assert_answer_logits(onnx_models(precision), precision, inputs, answer, expected)


@pytest.mark.parametrize("precision", list(MAX_KL))
@pytest.mark.parametrize("name", list(PROMPTS))
def test_long_cached_decode(
    long_prompts, reference, reference_processor, onnx_models, device, precision, name
):
    inputs, answer, _ = long_prompts(name)
    assert_cached_decode(
        reference,
        reference_processor.tokenizer,
        onnx_models(precision),
        inputs,
        device,
        precision,
        len(answer),
    )
