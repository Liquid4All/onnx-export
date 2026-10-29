"""
Each bundle's forwards against the float32 reference at the answer slots, on the README's requests:
every forward the ONNX engine runs (OrtSystemOne._slot_logits: encoders and decoder) is replayed
through the reference engine's SystemOne._slot_logits on the same packed rows and processor outputs.
Scored by KL(reference || ONNX): the code slots' distributions over the vocabulary, which the
readouts take the option tokens of, and the read slots' over the state positions (pointer reads).
On CUDA the bfloat16 reference is scored the same way and logged: the floor the bounds sit
against.

Run with:
    uv run --no-sync pytest tests/test_d1/test_decoder.py -v --d1-model LiquidAI/d1-2B --device cuda
    uv run --no-sync pytest tests/test_d1/test_decoder.py -v --device cuda --ep cpu -k q8
"""

import logging

import pytest
import torch
from helpers import NearBound, expect_known_failure

from ..test_lfm3_vl.test_decoder import MAX_KL
from .conftest import requests

logger = logging.getLogger(__name__)

NAMES = ["ticket", "invoice", "parcels", "picture", "clip", "speech"]
# (model, test, parameters' id, host or "*") -> the evidence of a known failure
# (helpers.expect_known_failure)
D1, D1_OMNI = "LiquidAI/d1-2B", "LiquidAI/d1-Omni-2B"
D1_ARM64 = (
    "the CPU EP's int8 activations on arm64: code-slot KL 5.3e-4 to 6.7e-4, over 5e-4, where CUDA "
    "measures 4.9e-5 to 1.2e-4 (README 3.1f)"
)
BOX_EDGES = (
    "d1-Omni-2B's box edges, 1000-way distributions sensitive in this checkpoint: code-slot KL "
    "9.4e-4 (CUDA) to 3.8e-3 (CPU EP), over 5e-4 and under bf16 PyTorch's 5.6e-3; the answers hold "
    "(README 3.1f)"
)
SPOKEN = (
    "the int8 rounding of the weights at the spoken question's speech slot: code-slot KL 9.4e-4 to "
    "1.7e-2, over 5e-4, where bf16 PyTorch measures 1.5e-5 (README 3.1f)"
)
OMNI_CPU = (
    "the CPU EP's int8 activations: code-slot KL 5.7e-4 to 1.6e-3, over 5e-4, where CUDA measures "
    "3.6e-5 to 7.4e-5 (README 3.1f)"
)
KNOWN_FAILURES = {
    (D1, "test_slot_logits", "q8-ticket", "cpu-arm64"): D1_ARM64,
    (D1, "test_slot_logits", "q8-picture", "cpu-arm64"): NearBound(D1_ARM64),
    (D1, "test_slot_logits", "q8f16-picture", "cpu-arm64"): D1_ARM64,
    (D1, "test_slot_logits", "q8f16-picture", "cpu-x86_64"): NearBound(
        "the CPU EP's int8 activations on x86-64: code-slot KL 4.9e-4, at the 5e-4 bound, where "
        "arm64 measures 6.1e-4 (README 3.1f)"
    ),
    (D1_OMNI, "test_slot_logits", "fp16-picture", "cuda"): NearBound(
        "d1-Omni-2B's box edges, 1000-way distributions sensitive in this checkpoint: fp16 on CUDA "
        "at code-slot KL 1.03e-4, bound 1e-4, under bf16 PyTorch's 5.6e-3 (README 3.1f)"
    ),
    **{(D1_OMNI, "test_slot_logits", f"{p}-picture", "*"): BOX_EDGES for p in ("q8", "q8f16")},
    **{(D1_OMNI, "test_slot_logits", f"{p}-speech", "*"): SPOKEN for p in ("q8", "q8f16")},
    (D1_OMNI, "test_slot_logits", "q8-ticket", "cpu-x86_64"): OMNI_CPU,
    (D1_OMNI, "test_slot_logits", "q8-ticket", "cpu-arm64"): OMNI_CPU,
    (D1_OMNI, "test_slot_logits", "q8-clip", "cpu-x86_64"): OMNI_CPU,
}


def forwards(engine, batch: list[tuple]) -> list[tuple[tuple, tuple]]:
    """(arguments, outputs) of each _slot_logits call of engine.system_one_batch(batch)."""
    calls = []
    run = engine._slot_logits

    def spy(*args):
        out = run(*args)
        calls.append((args, out))
        return out

    engine._slot_logits = spy
    try:
        engine.system_one_batch(batch)
    finally:
        del engine._slot_logits
    return calls


def kl(expected: torch.Tensor, got: torch.Tensor) -> torch.Tensor:
    """KL(softmax(expected) || softmax(got)) per row, in float64; -inf logits are masked keys."""
    lp, lq = expected.double().log_softmax(-1), got.double().log_softmax(-1)
    p = lp.exp()
    return torch.where(p > 0, p * (lp - lq), 0.0).sum(-1)


def scores(expected: tuple, got: tuple) -> dict[str, torch.Tensor]:
    """KL per slot of one forward's _slot_logits outputs, of the kinds it has: "code" slots
    (log-probabilities over the vocabulary) and "read" slots (position logits)."""
    out = {}
    for kind, a, b in (("code", expected[0], got[0]), ("read", expected[1], got[1])):
        if a is not None and len(a):
            out[kind] = kl(a.float().cpu(), b.float().cpu())
    return out


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("precision", list(MAX_KL))
def test_slot_logits(
    request, model_id, ep, reference, reference_bf16, onnx_engines, precision, name
):
    expect_known_failure(request, KNOWN_FAILURES, model_id, ep)
    batch = requests(model_id).get(name)
    if batch is None:
        pytest.skip("audio is d1-Omni's")
    for args, got in forwards(onnx_engines(precision), batch):
        with torch.inference_mode():
            expected = reference._slot_logits(*args)
            floor = {}
            if reference_bf16 is not None:
                floor = scores(expected, reference_bf16._slot_logits(*args))
        found = scores(expected, got)
        seqs, [row], *_ = args
        for kind, values in found.items():
            bf16 = f", bf16 PyTorch {floor[kind].mean():.2e}" if kind in floor else ""
            logger.info(
                f"  {precision} {name} ({len(seqs[row][2])} tokens), {len(values)} {kind} slots: "
                f"KL {values.mean():.2e} max {values.max():.2e}{bf16}"
            )
        assert all(values.mean() < MAX_KL[precision] for values in found.values())
