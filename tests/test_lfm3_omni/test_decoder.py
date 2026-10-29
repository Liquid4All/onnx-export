"""
End-to-end prefill of each bundle (its vision and speech encoders, then its decoder through
ONNXOmniModel.prefill_outputs) against the float32 reference on image, video and image + speech
prompts, scored where generation reads logits: the last prompt position and the reference's greedy
text answer (generation_mode="text", <|mixed_start|> and <|audio_start|> suppressed),
teacher-forced in one uncached run. The speech-only, text and speech-history prompts are
tests/test_lfm3_audio/test_decoder.py's, run with --lfm3-audio-model LiquidAI/LFM3-Omni-2B.

Run with:
    uv run --no-sync pytest tests/test_lfm3_omni/test_decoder.py -v --device cuda
    uv run pytest tests/test_lfm3_omni/test_decoder.py -v
"""

import logging
import re
from dataclasses import dataclass

import numpy as np
import pytest
import torch
from helpers import expect_known_failure

from liquidonnx.lfm3_audio.session import load_audio
from liquidonnx.lfm3_vl.session import run_vision

from ..test_lfm3_audio.test_decoder import (
    CLIPS,
    CONFIDENT,
    MAX_KL,
    SAMPLES,
    kl_divergence,
    log_softmax,
)
from .conftest import load_images, video_frames

logger = logging.getLogger(__name__)

ANSWER = 24
# name -> (images, video, speech clip, question)
PROMPTS = {
    "cardinal": (["cardinal.jpg"], False, None, "What bird is this? Answer briefly."),
    "two-birds": (["cardinal.jpg", "bluejay.jpg"], False, None, "What birds are these?"),
    "video": ([], True, None, "What bird is in this video, and what does the camera do?"),
    "image-speech": (["bluejay.jpg"], False, CLIPS[0], "Describe the image, then answer the question."),
}  # fmt: skip


# MAX_KL and CONFIDENT are test_lfm3_audio's (LFM3-VL's). On the CPU EP q8 and q8f16 measure
# higher than on CUDA: their MatMulNBits accuracy_level 4 quantizes the activations to int8 on the
# CPU EP, which CUDA doesn't; set to 0 (--matmulnbits-accuracy-level), the same graphs measure as
# on CUDA.

# (model, test, parameters' id, host or "*") -> the evidence of a known failure
# (helpers.expect_known_failure)
OMNI = "LiquidAI/LFM3-Omni-2B"
TWO_BIRDS = (
    "the fp16 vision encoder on two images, as fp16 PyTorch's tower: fp16 KL 2.2e-4 to 2.7e-4, over "
    "1e-4, and with int8 weights q8 and q8f16 5.9e-4 to 2.0e-3, over 5e-4 (README 3.1e)"
)
CPU_INT8 = (
    "the CPU EP's int8 activations: KL 8.2e-4 to 9.6e-4 on the image-and-speech prompt, over 5e-4, "
    "where CUDA and --matmulnbits-accuracy-level 0 measure 2.0e-4 to 2.7e-4 (README 3.1e)"
)
KNOWN_FAILURES = {
    **{(OMNI, "test_prefill", f"{p}-two-birds", "*"): TWO_BIRDS for p in ("fp16", "q8", "q8f16")},
    **{
        (OMNI, "test_prefill", f"{p}-image-speech", h): CPU_INT8
        for p in ("q8", "q8f16")
        for h in ("cpu-x86_64", "cpu-arm64")
    },
}


@dataclass
class Case:
    ids: list[int]
    inputs: dict  # the processor's media tensors (pixel_values, input_features, ...)
    wave: np.ndarray | None
    answer: list[int]
    logits: np.ndarray  # [len(answer), V] at the positions that predict the answer


def processor_inputs(processor, name: str) -> tuple[dict, np.ndarray | None]:
    """(processor outputs of the prompt, its 16 kHz clip)"""
    images, video, clip, question = PROMPTS[name]
    content = [{"type": "image"}] * len(images) + [{"type": "video"}] * video
    content += [{"type": "audio"}] * (clip is not None) + [{"type": "text", "text": question}]
    text = processor.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
    )
    kwargs = {}
    if images:
        kwargs["images"] = load_images(images)
    if video:
        frames, metadata = video_frames()
        kwargs |= {"videos": [frames], "video_metadata": [metadata]}
    wave = None if clip is None else load_audio(SAMPLES / clip, 16000)
    if wave is not None:
        kwargs |= {"audio": [wave], "sampling_rate": 16000}
    return processor(text=text, return_tensors="pt", **kwargs), wave


def test_video_markers(reference_processor):
    """The layout the graphs are given (README 3.1c): the processor replaces the template's
    <|vision_start|><|video|><|vision_end|> with one pair around the video and, inside it, a
    timestamp and a pair per frame group, as tests/test_lfm3_vl/test_video.py finds for LFM3-VL."""
    inputs, _ = processor_inputs(reference_processor, "video")
    text = reference_processor.tokenizer.decode(inputs["input_ids"][0])
    text = re.sub(r"(<\|video\|>)+", "<|video|>", re.sub(r"<\d+\.\d seconds>", "<t>", text))
    group = "<t><|vision_start|><|video|><|vision_end|>"
    groups = int(inputs["video_grid_thw"][0, 0])
    logger.info(f"  {groups} frame groups: {text!r}")
    assert f"<|vision_start|>{group * groups}<|vision_end|>" in text


def media(inputs: dict, device) -> dict:
    return {k: v.to(device) for k, v in inputs.items() if k not in ("input_ids", "attention_mask")}


def greedy_text(reference, inputs: dict, device) -> list[int]:
    ids = inputs["input_ids"].to(device)
    out = reference.generate(
        ids,
        **media(inputs, device),
        max_new_tokens=ANSWER,
        generation_mode="text",
        suppress_mixed=True,
        suppress_audio_out=True,
    )
    return out.sequences[0, ids.shape[1] : int(out.sequence_lengths[0])].tolist()


@pytest.fixture(scope="module")
def cases(reference, reference_processor, device):
    """name -> Case of PROMPTS[name] with the reference's answer and logits (built once)."""
    built = {}

    def get(name: str) -> Case:
        if name not in built:
            inputs, wave = processor_inputs(reference_processor, name)
            ids = inputs["input_ids"][0].tolist()
            with torch.no_grad():
                answer = greedy_text(reference, inputs, device)
                sequence = torch.tensor([ids + answer[:-1]], device=device)
                hidden = reference.model(
                    input_ids=sequence, **media(inputs, device), use_cache=False
                ).last_hidden_state[0, len(ids) - 1 :]
                logits = reference.lm_head(hidden).float().cpu().numpy()
            built[name] = Case(ids, inputs, wave, answer, logits)
        return built[name]

    return get


def bundle_features(model, case: Case) -> dict:
    """The bundle's encoders' features of the case's media, as decoder inputs."""
    features = {}
    for pixels, grid, name in (
        ("pixel_values", "image_grid_thw", "image_features"),
        ("pixel_values_videos", "video_grid_thw", "video_features"),
    ):
        if pixels in case.inputs:
            features[name] = run_vision(
                model.vision_encoder,
                case.inputs[pixels].numpy(),
                case.inputs[grid].numpy(),
                model.vision_config.merge_size,
            )
    if case.wave is not None:
        features["audio_features"] = model.encode_audio([case.wave])
    return features


@pytest.mark.parametrize("name", list(PROMPTS))
@pytest.mark.parametrize("precision", list(MAX_KL))
def test_prefill(request, model_id, ep, cases, onnx_models, precision, name):
    expect_known_failure(request, KNOWN_FAILURES, model_id, ep)
    case = cases(name)
    model = onnx_models(precision)
    got, _ = model.prefill_outputs(
        case.ids + case.answer[:-1],
        logits_to_keep=len(case.answer),
        **bundle_features(model, case),
    )
    expected = case.logits

    kl = kl_divergence(expected, got)
    lp = log_softmax(expected)
    confident = -(np.exp(lp) * lp).sum(-1) < CONFIDENT
    same_top = expected.argmax(-1) == got.argmax(-1)
    logger.info(
        f"  {precision} {name}: {len(case.ids)} prompt + {len(case.answer)} answer tokens: KL "
        f"mean {kl.mean():.2e} max {kl.max():.2e}, top-1 {same_top.sum()}/{len(case.answer)}, "
        f"{same_top[confident].sum()}/{confident.sum()} confident"
    )
    assert kl.mean() < MAX_KL[precision]
    if not precision.startswith("q4"):
        assert same_top[confident].all()
