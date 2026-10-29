"""
End-to-end prefill: ONNX vision encoder + decoder against the float32 reference on real prompts,
scored where generation reads logits: the last prompt position and the reference's greedy answer,
teacher-forced, in one uncached forward. Earlier prompt positions are not: on image prompts many
take <|image|> as input, where nothing is generated, and one such position can outweigh all the
others.

Run with:
    uv run --no-sync pytest tests/test_lfm3_vl/test_decoder.py -v --device cuda  # README 2
    uv run --no-sync pytest tests/test_lfm3_vl/test_decoder.py -v -k "q4 and not q4f16" --device cuda --ep cpu
"""

import logging

import numpy as np
import pytest
import torch
from helpers import expect_known_failure

from liquidonnx.lfm3_vl.session import run_vision

from .conftest import chat

logger = logging.getLogger(__name__)

ANSWER = 24
# (images, question, enable_thinking): the template opens <think> by default, as lfm3-vl-infer
# does without --no-think; the answer of the thinking prompt is its first reasoning tokens
PROMPTS = [
    (["cardinal.jpg"], "What bird is this? Answer briefly.", False),
    (["cardinal.jpg", "bluejay.jpg"], "What birds are these?", False),
    ([], "Write a haiku about the sea.", False),
    (["cardinal.jpg"], "What bird is this, and how can you tell?", True),
]
PROMPT_IDS = ["cardinal", "two-birds", "text", "cardinal-thinking"]
# Mean KL(reference || ONNX) over the scored positions, with the reference on CUDA with TF32 off
# or on the CPU. The bounds were set at about 4 times bf16 PyTorch's KL, except fp32's.
MAX_KL = {"fp32": 1e-6, "fp16": 1e-4, "q8": 5e-4, "q8f16": 5e-4, "q4": 0.1, "q4f16": 0.1}
# Positions where the reference is confident (entropy below 1 nat, top token ahead of the second
# by 0.1 or more) must keep its top token, except in q4. Entropy alone admits two-way ties, such
# as two tokens at 0.50 each, which an exact bundle and bf16 PyTorch break either way.
CONFIDENT = 1.0
MARGIN = 0.1
# (model, test, parameters' id, host or "*") -> the evidence of a known failure
# (helpers.expect_known_failure)
HAIKU = "the haiku, on which LFM3-VL-2B is very confident, so rounding moves more probability"
KNOWN_FAILURES = {
    **{
        ("LiquidAI/LFM3-VL-2B", "test_prefill_logits", f"text-{p}", "*"): (
            f"{HAIKU}: int8 weights give KL 6.7e-4 to 6.8e-4 on CUDA and 1.5e-3 to 2.4e-3 with the "
            "CPU EP's int8 activations, over 5e-4; bf16 PyTorch is over 5e-4 too (README 3.1c)"
        )
        for p in ("q8", "q8f16")
    },
    **{
        ("LiquidAI/LFM3-VL-2B", "test_prefill_logits", f"text-{p}", "*"): (
            f"{HAIKU}: int4 weights give KL 0.145 to 0.162, over 0.1 (README 3.1c)"
        )
        for p in ("q4", "q4f16")
    },
}


def log_softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64) - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


def confident(lp: np.ndarray) -> np.ndarray:
    """Positions of log-probabilities [positions, vocab] where the top token is clear."""
    p = np.exp(lp)
    top2 = np.sort(p, -1)[:, -2:]
    return (-(p * lp).sum(-1) < CONFIDENT) & (top2[:, 1] - top2[:, 0] >= MARGIN)


def reference_inputs(reference_processor, images, question, enable_thinking=False):
    messages, pil = chat(images, question)
    text = reference_processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False, enable_thinking=enable_thinking
    )
    return reference_processor(images=pil or None, text=text, return_tensors="pt")


VISUAL = ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw")


def reference_answer(reference, inputs, device, max_new_tokens: int):
    """The reference's greedy answer of up to max_new_tokens to processor inputs (images and
    videos) and its logits [len(answer), vocab] at the positions that predict it, teacher-forced
    in one uncached forward."""
    ids = inputs["input_ids"].to(device)
    visual = {k: inputs[k].to(device) for k in VISUAL if k in inputs}
    with torch.no_grad():
        generated = reference.generate(
            input_ids=ids, **visual, max_new_tokens=max_new_tokens, do_sample=False
        )
        answer = generated[0, ids.shape[1] :]
        # [prompt + answer[:-1]] → logits of the last len(answer) positions
        sequence = torch.cat([ids[0], answer[:-1]])[None]
        logits = reference(
            input_ids=sequence, logits_to_keep=len(answer), use_cache=False, **visual
        ).logits[0]
    return answer.tolist(), logits.float().cpu().numpy()


def onnx_features(model, inputs) -> dict[str, np.ndarray]:
    """image_features and video_features of processor inputs through the bundle's encoder."""
    features = {}
    for name, pixels, grids in [
        ("image_features", "pixel_values", "image_grid_thw"),
        ("video_features", "pixel_values_videos", "video_grid_thw"),
    ]:
        features[name] = (
            run_vision(model.vision, inputs[pixels].numpy(), inputs[grids].numpy(), merge=2)
            if pixels in inputs
            else np.zeros((0, model.hidden), dtype=np.float32)
        )
    return features


def assert_answer_logits(model, precision, inputs, answer, expected):
    """The bundle's uncached prefill over prompt + answer against the reference's answer logits:
    mean KL under MAX_KL, and the confident positions' top token kept (but in q4)."""
    from liquidonnx.session import initialize_cache

    features = onnx_features(model, inputs)
    ids = inputs["input_ids"][0].tolist()
    model.check_features(ids, features["image_features"])
    feed = model.decoder_feed(
        ids + answer[:-1],
        features["image_features"],
        0,
        len(answer),
        initialize_cache(model.decoder),
    )
    feed["video_features"] = features["video_features"]
    got = model.decoder.run(["logits"], feed)[0][0]

    lp, lq = log_softmax(expected), log_softmax(got)
    kl = (np.exp(lp) * (lp - lq)).sum(-1)
    sure = confident(lp)
    same_top = expected.argmax(-1) == got.argmax(-1)
    logger.info(
        f"  {precision} {len(ids)} prompt + {len(answer)} answer tokens: KL mean {kl.mean():.2e} "
        f"max {kl.max():.2e} (at answer token {kl.argmax()}), top-1 {same_top.sum()}/{len(answer)}, "
        f"{same_top[sure].sum()}/{sure.sum()} confident, "
        f"max|diff| {np.abs(expected - got).max():.2e}"
    )
    assert kl.mean() < MAX_KL[precision]
    if not precision.startswith("q4"):
        assert same_top[sure].all()


@pytest.fixture(scope="module")
def reference_answers(reference, reference_processor, device):
    """(images, question, enable_thinking) -> (processor inputs, the reference's greedy answer of
    up to ANSWER tokens, its logits [len(answer), vocab]), once per prompt."""
    answers = {}

    def get(images: list[str], question: str, enable_thinking: bool = False):
        key = (*images, question, enable_thinking)
        if key not in answers:
            inputs = reference_inputs(reference_processor, images, question, enable_thinking)
            answers[key] = (inputs, *reference_answer(reference, inputs, device, ANSWER))
        return answers[key]

    return get


@pytest.mark.parametrize("precision", list(MAX_KL))
@pytest.mark.parametrize("images,question,enable_thinking", PROMPTS, ids=PROMPT_IDS)
def test_prefill_logits(
    request,
    model_id,
    ep,
    reference_answers,
    onnx_models,
    precision,
    images,
    question,
    enable_thinking,
):
    expect_known_failure(request, KNOWN_FAILURES, model_id, ep)
    inputs, answer, expected = reference_answers(images, question, enable_thinking)
    assert_answer_logits(onnx_models(precision), precision, inputs, answer, expected)
