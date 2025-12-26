"""
Compare local ONNX exports against onnx-community versions.

Both are compared against PyTorch reference to show which is closer.

Run with:
    pytest tests/test_lfm2/test_community.py -v
    pytest tests/test_lfm2/test_community.py -v -k "350M and q4"

Set ONNX_COMMUNITY_DIR environment variable to the directory containing community models:
    export ONNX_COMMUNITY_DIR=/path/to/onnx-community
"""

import logging
import os
import pathlib

import numpy as np
import pytest
import torch
from helpers import skip_if_missing

from liquidonnx.lfm2 import MODELS
from liquidonnx.lfm2.generate import get_onnx_dir
from liquidonnx.session import get_onnx_file, load_onnx_session

logger = logging.getLogger(__name__)

QUANT_CONFIGS = [
    pytest.param(None, id="fp32"),
    pytest.param(4, id="q4"),
]

PROMPTS = ["Hello, how are", "The sky is", "1 + 1 ="]


@pytest.fixture(scope="session")
def community_dir() -> pathlib.Path:
    """Base directory for onnx-community models."""
    env_dir = os.environ.get("ONNX_COMMUNITY_DIR")
    if env_dir:
        return pathlib.Path(env_dir)
    return pathlib.Path.home() / "workplace" / "models" / "onnx-community"


def get_community_onnx_dir(community_dir: pathlib.Path, size: str) -> pathlib.Path:
    """Get onnx-community model directory."""
    return community_dir / f"LFM2-{size}-ONNX" / "onnx"


def get_community_onnx_file(onnx_dir: pathlib.Path, bits: int | None) -> pathlib.Path:
    """Get onnx-community model file."""
    if bits is None:
        return onnx_dir / "model.onnx"
    return onnx_dir / f"model_q{bits}.onnx"


def bits_to_str(bits: int | None) -> str:
    return f"q{bits}" if bits else "fp32"


def compute_metrics(expected: np.ndarray, actual: np.ndarray) -> dict:
    """Compute comparison metrics between two logit arrays."""
    diff = np.abs(expected - actual)

    exp_last = expected[0, -1]
    act_last = actual[0, -1]

    exp_top5 = np.argsort(exp_last)[-5:][::-1]
    act_top5 = np.argsort(act_last)[-5:][::-1]

    top1_match = exp_top5[0] == act_top5[0]
    top5_overlap = len(set(exp_top5) & set(act_top5))

    return {
        "max_diff": float(diff.max()),
        "mean_diff": float(diff.mean()),
        "top1_match": top1_match,
        "top5_overlap": top5_overlap,
        "expected_top5": exp_top5.tolist(),
        "actual_top5": act_top5.tolist(),
    }


@pytest.mark.parametrize("pytorch_model", MODELS.keys(), indirect=True)
@pytest.mark.parametrize("bits", QUANT_CONFIGS)
@pytest.mark.parametrize("prompt", PROMPTS)
def test_community_comparison(
    exports_dir: pathlib.Path,
    community_dir: pathlib.Path,
    pytorch_model,
    bits: int | None,
    prompt: str,
):
    """Compare local and community ONNX exports against PyTorch reference."""
    size, model, tokenizer = pytorch_model
    quant_str = bits_to_str(bits)
    logger.info(f"Comparing {size}/{quant_str}: '{prompt}'")

    # Check local export exists
    local_onnx_dir = get_onnx_dir(exports_dir, size)
    skip_if_missing(local_onnx_dir, "Local export not found")

    local_onnx_file = get_onnx_file(local_onnx_dir, bits)
    skip_if_missing(local_onnx_file, f"Local ONNX file not found: {local_onnx_file.name}")

    # Check community export exists
    community_onnx_dir = get_community_onnx_dir(community_dir, size)
    skip_if_missing(community_onnx_dir, f"Community export not found: {community_onnx_dir}")

    community_onnx_file = get_community_onnx_file(community_onnx_dir, bits)
    skip_if_missing(community_onnx_file, f"Community ONNX file not found: {community_onnx_file}")

    # Load ONNX models
    local_sess = load_onnx_session(local_onnx_file)
    community_sess = load_onnx_session(community_onnx_file)

    # Prepare inputs
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    seq_len = input_ids.shape[1]
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(seq_len).unsqueeze(0)

    logger.info(f"  Input: seq_len={seq_len}, tokens={input_ids[0].tolist()}")

    # Run PyTorch inference
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        pytorch_logits = outputs.logits.numpy()

    # Prepare ONNX inputs
    available_inputs = {
        "input_ids": input_ids.numpy().astype(np.int64),
        "attention_mask": attention_mask.numpy().astype(np.int64),
        "position_ids": position_ids.numpy().astype(np.int64),
    }

    # Build inputs for local model
    local_inputs = {}
    for inp in local_sess.get_inputs():
        if inp.name in available_inputs:
            local_inputs[inp.name] = available_inputs[inp.name]
        else:
            shape = [d if isinstance(d, int) else 1 for d in inp.shape]
            local_inputs[inp.name] = np.zeros(shape, dtype=np.float32)

    # Build inputs for community model
    community_inputs = {}
    for inp in community_sess.get_inputs():
        if inp.name in available_inputs:
            community_inputs[inp.name] = available_inputs[inp.name]
        else:
            shape = [d if isinstance(d, int) else 1 for d in inp.shape]
            community_inputs[inp.name] = np.zeros(shape, dtype=np.float32)

    # Run ONNX inference
    local_logits = local_sess.run(None, local_inputs)[0]
    community_logits = community_sess.run(None, community_inputs)[0]

    # Compare both against PyTorch
    local_metrics = compute_metrics(pytorch_logits, local_logits)
    community_metrics = compute_metrics(pytorch_logits, community_logits)

    # Log comparison results
    logger.info(f"  PyTorch top-5: {local_metrics['expected_top5']}")
    logger.info(
        f"  Local vs PyTorch:     max_diff={local_metrics['max_diff']:.4f}, "
        f"mean_diff={local_metrics['mean_diff']:.4f}, "
        f"top1={'✓' if local_metrics['top1_match'] else '✗'}, "
        f"top5={local_metrics['top5_overlap']}/5"
    )
    logger.info(
        f"  Community vs PyTorch: max_diff={community_metrics['max_diff']:.4f}, "
        f"mean_diff={community_metrics['mean_diff']:.4f}, "
        f"top1={'✓' if community_metrics['top1_match'] else '✗'}, "
        f"top5={community_metrics['top5_overlap']}/5"
    )

    # Determine winner
    if local_metrics["max_diff"] < community_metrics["max_diff"]:
        winner = "LOCAL"
    elif community_metrics["max_diff"] < local_metrics["max_diff"]:
        winner = "COMMUNITY"
    else:
        winner = "TIE"
    logger.info(f"  Winner: {winner} (lower max_diff)")

    # Assert both produce reasonable results (top-1 match with PyTorch)
    min_overlap = 4 if bits is None else 3

    assert local_metrics["top5_overlap"] >= min_overlap, (
        f"Local top-5 overlap too low: {local_metrics['top5_overlap']}/5"
    )
    assert community_metrics["top5_overlap"] >= min_overlap, (
        f"Community top-5 overlap too low: {community_metrics['top5_overlap']}/5"
    )
