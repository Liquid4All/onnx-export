"""
Compare local ONNX exports against onnx-community versions for LFM2-MoE.

Both are compared against PyTorch reference to show which is closer.

Run with:
    uv run pytest tests/test_lfm2_moe/test_community.py -v
    uv run pytest tests/test_lfm2_moe/test_community.py -v -k "fp32"

Set ONNX_COMMUNITY_DIR environment variable to the directory containing community models:
    export ONNX_COMMUNITY_DIR=/path/to/onnx-community
"""

import logging
import pathlib

import numpy as np
import pytest
import torch
from helpers import get_community_moe_onnx_dir, get_community_moe_onnx_file, skip_if_missing

from liquidonnx.lfm2_moe import MODELS
from liquidonnx.session import get_onnx_file, load_onnx_session

logger = logging.getLogger(__name__)


def get_onnx_dir(exports_dir: pathlib.Path, size: str) -> pathlib.Path:
    """Get ONNX directory for a model size."""
    return exports_dir / f"LFM2-MoE-{size}-ONNX" / "onnx"


QUANT_CONFIGS = [
    pytest.param(None, id="fp32"),
    pytest.param("fp16", id="fp16"),
    pytest.param("q4", id="q4"),
    pytest.param("q4f16", id="q4f16"),
]

PROMPTS = ["Hello, how are", "The sky is", "1 + 1 ="]


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
@pytest.mark.parametrize("precision", QUANT_CONFIGS)
@pytest.mark.parametrize("prompt", PROMPTS)
def test_community_comparison(
    exports_dir: pathlib.Path,
    community_dir: pathlib.Path,
    pytorch_model,
    precision: str | None,
    prompt: str,
):
    """Compare local and community ONNX exports against PyTorch reference."""
    size, model, tokenizer = pytorch_model
    logger.info(f"Comparing {size}/{precision or 'fp32'}: '{prompt}'")

    # Check local export exists
    local_onnx_dir = get_onnx_dir(exports_dir, size)
    skip_if_missing(local_onnx_dir, "Local export not found")

    local_onnx_file = get_onnx_file(local_onnx_dir, precision)
    skip_if_missing(local_onnx_file, f"Local ONNX file not found: {local_onnx_file.name}")

    # Check community export exists
    community_onnx_dir = get_community_moe_onnx_dir(community_dir, size)
    skip_if_missing(community_onnx_dir, f"Community export not found: {community_onnx_dir}")

    community_onnx_file = get_community_moe_onnx_file(community_onnx_dir, precision)
    skip_if_missing(community_onnx_file, f"Community ONNX file not found: {community_onnx_file}")

    # Load ONNX models
    local_sess = load_onnx_session(local_onnx_file)
    try:
        community_sess = load_onnx_session(community_onnx_file)
    except Exception as e:
        pytest.skip(f"Community model failed to load: {e}")

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

    def build_inputs_for_session(sess, available_inputs, use_fp16: bool = False):
        """Build inputs dict for an ONNX session, handling dtype requirements."""
        inputs = {}
        for inp in sess.get_inputs():
            if inp.name in available_inputs:
                inputs[inp.name] = available_inputs[inp.name]
            else:
                # For KV cache inputs, check expected dtype
                expected_dtype = inp.type
                if "float16" in expected_dtype and use_fp16:
                    dtype = np.float16
                else:
                    dtype = np.float32
                shape = [d if isinstance(d, int) else 1 for d in inp.shape]
                inputs[inp.name] = np.zeros(shape, dtype=dtype)
        return inputs

    use_fp16 = precision in ("fp16", "q4f16")
    local_inputs = build_inputs_for_session(local_sess, available_inputs, use_fp16)
    community_inputs = build_inputs_for_session(community_sess, available_inputs, use_fp16)

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

    # Assert both produce reasonable results (top-5 match with PyTorch)
    # Q4/Q4F16 have more aggressive quantization, so lower threshold
    min_overlap = 4 if precision in (None, "fp16") else 2

    assert local_metrics["top5_overlap"] >= min_overlap, (
        f"Local top-5 overlap too low: {local_metrics['top5_overlap']}/5"
    )
    assert community_metrics["top5_overlap"] >= min_overlap, (
        f"Community top-5 overlap too low: {community_metrics['top5_overlap']}/5"
    )
