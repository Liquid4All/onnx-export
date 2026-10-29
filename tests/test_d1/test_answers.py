"""
End-to-end answers of each bundle (d1.session: the checkpoint's system_one_batch on onnxruntime)
against the float32 reference's (its own D1Model engine) on the README's requests.

Every discrete part of an answer must match: a choice, a score level, an estimate's value and
interval, a read's text, span, number and date, a box count. Probabilities (P(yes), choice and
score probabilities, found, confidence, presence) must be within PROBABILITY[precision], box
edges within EDGE[precision]. A near-tie of the reference (NEAR_TIES), whose margin to its decision
threshold is within the precision's probability bound, may fall on the other side; its
probabilities are still bounded.

Run with:
    uv run --no-sync pytest tests/test_d1/test_answers.py -v --d1-model LiquidAI/d1-2B --device cuda
"""

import logging

import pytest
import torch
from helpers import expect_known_failure

from .conftest import requests

logger = logging.getLogger(__name__)

NAMES = ["ticket", "invoice", "parcels", "picture", "clip", "speech"]
# Keys whose values are the answer itself, compared exactly
EXACT = {"choice", "choices", "score", "number", "interval", "text", "span", "path", "date"}
EDGES = {"top", "bottom", "left", "right"}
PROBABILITY = {"fp32": 1e-3, "fp16": 1e-2, "q8": 3e-2, "q8f16": 3e-2, "q4": 0.1, "q4f16": 0.1}
EDGE = {"fp32": 2e-3, "fp16": 5e-3, "q8": 1e-2, "q8f16": 1e-2, "q4": 3e-2, "q4f16": 3e-2}
# (model, request, question) -> the float32 reference's margin to the threshold its answer is read
# at. d1-Omni-2B's ticket team is billing 0.505 / shipping 0.478 (bfloat16: 0.500 / 0.482; shipping
# with set_size=1). Its clip's beeps interval ends at 4 because P(beeps <= 3) is 0.8958, under the
# p90 level of 0.9354 that calibration.json's estimate table gives (a bundle over it answers 3).
NEAR_TIES = {
    ("LiquidAI/d1-Omni-2B", "ticket", "team"): 0.027,
    ("LiquidAI/d1-Omni-2B", "clip", "beeps"): 0.040,
}
# (model, test, parameters' id, host or "*") -> the evidence of a known failure
# (helpers.expect_known_failure)
SPEECH_ANSWER = (
    "the spoken question's P(speech), 0.574: the int8 weights move it by 0.023 to 0.024 on CUDA, "
    "and with the CPU EP's int8 activations by 0.040 (q8, x86-64) and 0.042 (q8f16, arm64), over "
    "3e-2 (README 3.1f)"
)
KNOWN_FAILURES = {
    ("LiquidAI/d1-Omni-2B", "test_answers", "q8-speech", "cpu-x86_64"): SPEECH_ANSWER,
    ("LiquidAI/d1-Omni-2B", "test_answers", "q8f16-speech", "cpu-arm64"): SPEECH_ANSWER,
}


def differences(expected, got, probability: float, edge: float, path=()):
    """(path, expected, got) of every part of got that differs from expected beyond the bounds."""
    if isinstance(expected, dict):
        if not isinstance(got, dict) or expected.keys() != got.keys():
            yield path, expected, got
            return
        for key in expected:
            yield from differences(expected[key], got[key], probability, edge, (*path, key))
    elif isinstance(expected, list):
        if not isinstance(got, list) or len(expected) != len(got):
            yield path, expected, got
            return
        for index, (a, b) in enumerate(zip(expected, got, strict=True)):
            yield from differences(a, b, probability, edge, (*path, index))
    elif isinstance(expected, float) and not EXACT & set(path):
        bound = edge if path[-1] in EDGES else probability
        if not isinstance(got, float) or abs(expected - got) > bound:
            yield path, expected, got
    elif expected != got:
        yield path, expected, got


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("precision", list(PROBABILITY))
def test_answers(request, model_id, ep, reference, onnx_engines, precision, name):
    expect_known_failure(request, KNOWN_FAILURES, model_id, ep)
    batch = requests(model_id).get(name)
    if batch is None:
        pytest.skip("audio is d1-Omni's")
    with torch.inference_mode():
        expected = reference.system_one_batch(batch)
    got = onnx_engines(precision).system_one_batch(batch)
    found = []
    for want, have in zip(expected, got, strict=True):
        for question, answer in want["answers"].items():
            mine = have["answers"][question]
            diffs = list(differences(answer, mine, PROBABILITY[precision], EDGE[precision]))
            drift = max(
                (
                    abs(a - b)
                    for _, a, b in differences(answer, mine, 0.0, 0.0)
                    if isinstance(a, float) and isinstance(b, float)
                ),
                default=0.0,
            )
            tie = NEAR_TIES.get((model_id, name, question), 1.0) < PROBABILITY[precision]
            logger.info(f"  {precision} {name} {question}: max |diff| {drift:.1e}, {mine}")
            found += [(question, *d) for d in diffs if not (tie and EXACT & set(d[0]))]
        assert want["usage"] == have["usage"]
    assert not found, found
