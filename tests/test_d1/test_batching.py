"""
d1.session's batched forwards against its own one row a forward: every request of the README in
one system_one_batch call under a token budget (right-padded rows of different lengths, their
pictures and clips beside text rows), held to the rounding of a padded row (BOUNDS). No reference
model.

Run with:
    uv run pytest tests/test_d1/test_batching.py -v --d1-model LiquidAI/d1-2B -k q4
    uv run --no-sync pytest tests/test_d1/test_batching.py -v --d1-model LiquidAI/d1-Omni-2B --device cuda
"""

import logging

import pytest

from .conftest import requests
from .test_answers import PROBABILITY, differences

logger = logging.getLogger(__name__)

TOKEN_BUDGET = 65536  # SystemOne's default: every request of the README in one forward
# A row right-padded to its batch's width rounds differently from the row alone. On CUDA that is
# float rounding: 4e-6 at most with float32 activations and 4e-3 with fp16 ones (d1-Omni-2B's
# spoken question), measured on an A40. On the CPU EP it was 0 on x86-64 for every precision; the
# int8 activations of the q bundles there get test_answers.py's bounds.
FLOAT = {"fp32": 1e-4, "fp16": 1e-2}
BOUNDS = {
    "cuda": FLOAT | {"q4": 1e-4, "q8": 1e-4, "q4f16": 1e-2, "q8f16": 1e-2},
    "cpu": FLOAT | {p: PROBABILITY[p] for p in ("q4", "q8", "q4f16", "q8f16")},
}


def largest(expected, got) -> float:
    """The largest difference of two answers' floats, of the same structure."""
    if isinstance(expected, dict):
        return max((largest(expected[k], got[k]) for k in expected), default=0.0)
    if isinstance(expected, list):
        return max((largest(a, b) for a, b in zip(expected, got, strict=True)), default=0.0)
    return abs(expected - got) if isinstance(expected, float) else 0.0


@pytest.mark.parametrize("precision", list(PROBABILITY))
def test_batched_answers(monkeypatch, model_id, ep, onnx_engines, precision):
    batch = [request for group in requests(model_id).values() for request in group]
    # one engine, its token_budget set after the single rows: in a whole run of tests/test_d1 a
    # second one would sit on the GPU beside the first and the references
    engine = onnx_engines(precision)
    expected = engine.system_one_batch(batch)
    forwards = []
    slot_logits = type(engine)._slot_logits

    def spy(self, seqs, rows, *args):
        forwards.append([len(seqs[b][2]) for b in rows])
        return slot_logits(self, seqs, rows, *args)

    # on the class: an engine that holds its own bound method is freed by the cycle collector
    # only, and its sessions' GPU memory with it, some precisions later
    monkeypatch.setattr(type(engine), "_slot_logits", spy)
    monkeypatch.setattr(engine, "token_budget", TOKEN_BUDGET)
    got = engine.system_one_batch(batch)
    logger.info(f"  {precision} {len(batch)} requests, rows per forward {forwards}")
    assert len(forwards) == 1 and len(set(forwards[0])) > 1
    bound = BOUNDS[ep][precision]
    found = list(differences(expected, got, bound, bound))
    assert not found
    logger.info(f"  {precision} max |diff| {largest(expected, got):.1e}, bound {bound:.0e}")
