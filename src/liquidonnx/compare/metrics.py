"""Scores of an export against its reference model."""

import logging

import numpy as np

from liquidonnx.session import decoder_inputs, initialize_cache, update_cache

logger = logging.getLogger(__name__)


def log_softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


def logit_metrics(reference: np.ndarray, actual: np.ndarray) -> dict:
    """KL(reference || actual) per position, top-1 agreement and max |Δlogit| over [N, V] rows."""
    lp, lq = log_softmax(reference), log_softmax(actual)
    kl = (np.exp(lp) * (lp - lq)).sum(-1)
    return {
        "kl_mean": float(kl.mean()),
        "kl_max": float(kl.max()),
        "top1": float((reference.argmax(-1) == actual.argmax(-1)).mean()),
        "max_abs": float(np.abs(reference - actual).max()),
        "n": int(reference.shape[0]),
    }


def merge_metrics(items: list[dict]) -> dict:
    """logit_metrics of several prompts, weighted by their positions."""
    n = sum(m["n"] for m in items)
    return {
        "kl_mean": sum(m["kl_mean"] * m["n"] for m in items) / n,
        "kl_max": max(m["kl_max"] for m in items),
        "top1": sum(m["top1"] * m["n"] for m in items) / n,
        "max_abs": max(m["max_abs"] for m in items),
        "n": n,
    }


def sequence_match(reference: list[int], actual: list[int], eos: set[int]) -> dict:
    """Compare continuations; a trailing stop token is ignored (onnxruntime-genai drops it)."""
    reference = reference[:-1] if reference and reference[-1] in eos else reference
    actual = actual[:-1] if actual and actual[-1] in eos else actual
    prefix = 0
    for a, b in zip(reference, actual, strict=False):
        if a != b:
            break
        prefix += 1
    return {"exact": reference == actual, "prefix": prefix, "ref_len": len(reference)}


def merge_sequences(items: list[dict]) -> dict:
    return {
        "exact": sum(m["exact"] for m in items),
        "of": len(items),
        "prefix_frac": float(np.mean([m["prefix"] / max(m["ref_len"], 1) for m in items])),
    }


def greedy(session, prompt: np.ndarray, max_new: int, eos: set[int], embed=None) -> list[int]:
    """Greedy continuation on plain onnxruntime; embed maps [1, S] ids to the decoder input."""
    cache, outputs = initialize_cache(session), session.get_outputs()
    inputs = embed(prompt[None]) if embed else prompt[None]
    tokens, past = [], 0
    for _ in range(max_new):
        result = session.run(None, decoder_inputs(inputs, cache, past))
        update_cache(cache, result, outputs)
        past += inputs.shape[1]
        tokens.append(int(result[0][0, -1].argmax()))
        if tokens[-1] in eos:
            break
        inputs = np.array([[tokens[-1]]], dtype=np.int64)
        inputs = embed(inputs) if embed else inputs
    return tokens


def guarded(fn, *args) -> dict:
    """Score one part, recording a failure instead of aborting the whole comparison."""
    try:
        return fn(*args)
    except Exception as e:
        logger.exception(f"{fn.__name__} failed")
        return {"error": str(e)[:400]}
