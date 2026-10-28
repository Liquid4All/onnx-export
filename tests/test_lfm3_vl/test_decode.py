"""
Cached greedy decode of the ONNX bundles against one teacher-forced reference forward: fp32 must
pick the reference's tokens, the others a token from its top 5 at every step.

Run with:
    uv run --no-sync pytest tests/test_lfm3_vl/test_decode.py -v --device cuda  # README 2
    uv run --no-sync pytest tests/test_lfm3_vl/test_decode.py -v -k "q4 and not q4f16" --device cuda --ep cpu
"""

import logging

import numpy as np
import pytest
import torch

from liquidonnx.lfm3.session import CachedDecoder

from .test_decoder import VISUAL, onnx_features, reference_inputs

logger = logging.getLogger(__name__)

STEPS = 24
# (images, question, enable_thinking)
PROMPTS = [
    (["cardinal.jpg"], "Describe this bird and its surroundings.", False),
    # > 256 tokens of text: every decode step of the sliding-window layers drops a key
    (
        [],
        "Summarize: "
        + "Liquid Foundation Models are a class of foundation models built by Liquid AI. " * 30,
        False,
    ),
    (["cardinal.jpg"], "What bird is this, and how can you tell?", True),
]


def greedy_ids(model, ids: list[int], features: dict[str, np.ndarray], steps: int) -> list[int]:
    """steps greedy tokens (past the end of turn) with the bundle's caches; the first step takes
    the image and video features."""
    capacity = len(ids) + steps if model.share_kv else None
    decoder = CachedDecoder(model.decoder, capacity)
    none = np.zeros((0, model.hidden), dtype=np.float32)
    tokens = []
    for _ in range(steps):
        logits = decoder.step(ids, **features, logits_to_keep=np.array([1], dtype=np.int64))
        tokens.append(int(np.argmax(logits[0, -1])))
        ids, features = [tokens[-1]], {"image_features": none, "video_features": none}
    return tokens


def assert_cached_decode(reference, tokenizer, model, inputs, device, precision, steps):
    ids = inputs["input_ids"][0].tolist()
    tokens = greedy_ids(model, ids, onnx_features(model, inputs), steps)

    visual = {k: v.to(device) for k, v in inputs.items() if k in VISUAL}
    sequence = torch.tensor([ids + tokens[:-1]], device=device)
    with torch.no_grad():
        logits = reference(
            input_ids=sequence, logits_to_keep=len(tokens), use_cache=False, **visual
        ).logits[0]
    top5 = torch.topk(logits.float(), 5, dim=-1).indices.cpu().numpy()
    in_top5 = [token in row for token, row in zip(tokens, top5, strict=True)]
    logger.info(
        f"  {precision} {len(ids)} prompt tokens: {tokenizer.decode(tokens)!r}, top-1 agreement "
        f"{np.mean([t == row[0] for t, row in zip(tokens, top5, strict=True)]):.0%}"
    )
    if precision == "fp32":
        assert tokens == top5[:, 0].tolist()
    else:
        assert all(in_top5), (
            f"tokens outside the reference top 5 at steps {np.flatnonzero(~np.array(in_top5))}"
        )


@pytest.mark.parametrize("precision", ["fp32", "fp16", "q8", "q8f16", "q4", "q4f16"])
@pytest.mark.parametrize(
    "images,question,enable_thinking", PROMPTS, ids=["cardinal", "long-text", "cardinal-thinking"]
)
def test_cached_decode(
    reference,
    reference_processor,
    onnx_models,
    device,
    precision,
    images,
    question,
    enable_thinking,
):
    inputs = reference_inputs(reference_processor, images, question, enable_thinking)
    assert_cached_decode(
        reference,
        reference_processor.tokenizer,
        onnx_models(precision),
        inputs,
        device,
        precision,
        STEPS,
    )
