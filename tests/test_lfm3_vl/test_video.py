"""
A video prompt through the exported graphs, against the reference: the checkpoint's own video
processor on 8 frames (a pan across cardinal.jpg at 2 fps, 4 temporal patches of 30 x 40), the
vision encoders' features against get_video_features with test_vision_encoder.py's floors, and the
decoder's answer KL with test_decoder.py's bounds.

Run with:
    uv run --no-sync pytest tests/test_lfm3_vl/test_video.py -v --device cuda  # README 2
    uv run --no-sync pytest tests/test_lfm3_vl/test_video.py -v -k "q4 and not q4f16" --device cuda --ep cpu
"""

import logging
import re

import numpy as np
import pytest
import torch

from liquidonnx.lfm3_vl.session import run_vision
from liquidonnx.session import initialize_cache, load_onnx_session

from .conftest import ASSETS
from .test_decoder import ANSWER, MAX_KL, confident, log_softmax
from .test_vision_encoder import FILES, FLOORS

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def video(reference_processor):
    """Processor inputs of a question about the frames."""
    from PIL import Image
    from transformers.video_utils import VideoMetadata

    image = np.asarray(Image.open(ASSETS / "cardinal.jpg").convert("RGB"))  # [960, 1280, 3]
    frames = np.stack([image[240:720, x : x + 640] for x in range(0, 640, 80)])  # [8, 480, 640, 3]
    content = [{"type": "video"}, {"type": "text", "text": "Describe this video."}]
    text = reference_processor.apply_chat_template(
        [{"role": "user", "content": content}],
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )
    metadata = VideoMetadata(total_num_frames=len(frames), fps=2.0, frames_indices=list(range(8)))
    return reference_processor(
        text=text, videos=[frames], video_metadata=[metadata], return_tensors="pt"
    )


@pytest.fixture(scope="module")
def video_answer(reference, video, device):
    """The reference's greedy answer and its logits [len(answer), vocab] at the positions that
    predict it, teacher-forced."""
    ids = video["input_ids"].to(device)
    visual = {k: video[k].to(device) for k in ("pixel_values_videos", "video_grid_thw")}
    with torch.no_grad():
        answer = reference.generate(
            input_ids=ids, **visual, max_new_tokens=ANSWER, do_sample=False
        )[0, ids.shape[1] :]
        sequence = torch.cat([ids[0], answer[:-1]])[None]
        logits = reference(
            input_ids=sequence, logits_to_keep=len(answer), use_cache=False, **visual
        ).logits[0]
    return answer.tolist(), logits.float().cpu().numpy()


def test_video_markers(reference_processor, video):
    """The layout the graphs are given (README 3.1c): the template's <|vision_start|><|video|>
    <|vision_end|> keeps its pair, and the processor puts a timestamp and another pair around each
    temporal patch, as Lfm3OmniProcessor does (tests/test_lfm3_omni/test_decoder.py)."""
    text = reference_processor.tokenizer.decode(video["input_ids"][0])
    text = re.sub(r"(<\|video\|>)+", "<|video|>", re.sub(r"<\d+\.\d seconds>", "<t>", text))
    group = "<t><|vision_start|><|video|><|vision_end|>"
    patches = int(video["video_grid_thw"][0, 0])
    logger.info(f"  {patches} temporal patches: {text!r}")
    assert f"<|vision_start|>{group * patches}<|vision_end|>" in text


@pytest.mark.parametrize("precision", list(FLOORS))
def test_video_features(export_dir, reference, video, device, ep, precision):
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
        f"  {precision} grid {grids[0].tolist()}: worst cos {cosine.min():.6f}, "
        f"p1 {np.percentile(cosine, 1):.6f}, mean {cosine.mean():.6f}, "
        f"max|diff| {np.abs(got - expected).max():.2e}"
    )
    mean_floor, p1_floor = FLOORS[precision]
    assert cosine.mean() > mean_floor
    assert np.percentile(cosine, 1) > p1_floor


@pytest.mark.parametrize("precision", list(MAX_KL))
def test_video_prefill_logits(video, video_answer, onnx_models, precision):
    answer, expected = video_answer
    model = onnx_models(precision)
    features = run_vision(
        model.vision, video["pixel_values_videos"].numpy(), video["video_grid_thw"].numpy(), 2
    )
    ids = video["input_ids"][0].tolist() + answer[:-1]
    no_images = np.zeros((0, model.hidden), dtype=np.float32)
    feed = model.decoder_feed(ids, no_images, 0, len(answer), initialize_cache(model.decoder))
    feed["video_features"] = features
    got = model.decoder.run(["logits"], feed)[0][0]

    lp, lq = log_softmax(expected), log_softmax(got)
    kl = (np.exp(lp) * (lp - lq)).sum(-1)
    sure = confident(lp)
    same_top = expected.argmax(-1) == got.argmax(-1)
    logger.info(
        f"  {precision} {len(ids) - len(answer) + 1} prompt + {len(answer)} answer tokens: KL mean "
        f"{kl.mean():.2e} max {kl.max():.2e}, top-1 {same_top.sum()}/{len(answer)}, "
        f"{same_top[sure].sum()}/{sure.sum()} confident"
    )
    assert kl.mean() < MAX_KL[precision]
    if not precision.startswith("q4"):
        assert same_top[sure].all()
