"""
Verify LFM3 decoder ONNX export against PyTorch reference.

Single-step logit comparison between PyTorch and ONNX models. A model whose remote code the
installed transformers can't build is skipped (see conftest.load_reference).

Run with (--lfm3-model: LiquidAI/LFM3-1.7B unless given):
    uv run pytest tests/test_lfm3/test_decoder.py -v -k q4
    uv run --no-sync pytest tests/test_lfm3/test_decoder.py -v --device cuda  # README 2
"""

import logging
import pathlib

import numpy as np
import pytest
import torch
from helpers import expect_known_failure, get_model_name, get_onnx_dir

from liquidonnx.lfm3.session import get_onnx_file, load_session
from liquidonnx.session import initialize_cache
from liquidonnx.verify import check_results, compare_arrays, compare_top_k, get_tolerances

logger = logging.getLogger(__name__)

PROMPTS = [
    "Hello, how are",
    "The sky is",
    "1 + 1 =",
    # Long prompt: > 256 tokens so sliding-window attention (window=256) is
    # actually exercised. Without this, every SWA layer behaves identically to
    # full-attention because the cache is always inside the window.
    pytest.param(
        "Liquid Foundation Models are a class of foundation models built by Liquid AI. " * 60,
        id="long",
    ),
]

# fp32 graph vs float32 PyTorch: accumulation order only. An eps/4 final norm would move them by
# over 1e-3.
FP32_MAX_ABS_DIFF = 5e-4
# Top-5 overlap at the last position; fp32 must match exactly. On the long prompt the reference
# puts 0.99+ on one token and ranks 2-6 are near ties: there fp16 can keep 4 of 5, and q4f32
# (round-to-nearest int4, no int8 layers) 2 of 5.
MIN_TOP5_OVERLAP = {None: 5, "fp16": 4, "q4f32": 2}

QUANT_CONFIGS = [
    pytest.param(None, ["arrays", "top_k"], id="fp32"),
    pytest.param("fp16", ["top_k"], id="fp16"),
    pytest.param("q4", ["top_k"], id="q4"),
    pytest.param("q4f32", ["top_k"], id="q4f32"),
    pytest.param("q4f16", ["top_k"], id="q4f16"),
    pytest.param("q8", ["top_k"], id="q8"),
    pytest.param("q8f16", ["top_k"], id="q8f16"),
]

# (model, test, parameters' id, host or "*") -> the evidence of a known failure
# (helpers.expect_known_failure)
ARM64_FP32 = (
    "the CPU EP's float32 arithmetic on arm64: fp32 logits 7.9e-4 to 2.5e-3 from PyTorch's, over "
    "5e-4, where x86-64 and CUDA stay under 3.1e-4; the answers' KL stays below 1e-9 (README 3.1b)"
)
KNOWN_FAILURES = {
    ("LiquidAI/LFM3-1.7B", "test_decoder", f"{prompt}-fp32", "cpu-arm64"): ARM64_FP32
    for prompt in ("Hello, how are", "The sky is", "1 + 1 =", "long")
}


@pytest.mark.parametrize("precision,checks", QUANT_CONFIGS)
@pytest.mark.parametrize("prompt", PROMPTS)
def test_decoder(
    request,
    exports_dir: pathlib.Path,
    ep: str,
    pytorch_model,
    precision: str | None,
    checks: list[str],
    prompt: str,
):
    model_id, model, tokenizer = pytorch_model
    expect_known_failure(request, KNOWN_FAILURES, model_id, ep)
    model_name = get_model_name(model_id)
    logger.info(f"Testing {model_name}/{precision or 'fp32'}: '{prompt}'")

    onnx_dir = get_onnx_dir(exports_dir, model_id)
    onnx_file = get_onnx_file(onnx_dir, precision)

    if not onnx_file.exists():
        precision_arg = f" --precision {precision}" if precision else ""
        pytest.fail(
            f"ONNX file not found: {onnx_file}\n"
            f"Export with: uv run lfm3-export {model_id}{precision_arg}"
        )

    onnx_sess = load_session(onnx_file, ep=ep, tf32=False)

    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    seq_len = input_ids.shape[1]
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(seq_len).unsqueeze(0)
    logger.info(f"  Input: seq_len={seq_len}, tokens={input_ids[0].tolist()}")

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids.to(model.device),
            attention_mask=attention_mask.to(model.device),
            position_ids=position_ids.to(model.device),
        )
        pytorch_logits = outputs.logits.cpu().numpy()
    logger.info(f"  PyTorch logits: shape={pytorch_logits.shape}")

    onnx_inputs = {
        "logits_to_keep": np.array([0], dtype=np.int64),
        "input_ids": input_ids.numpy().astype(np.int64),
        "attention_mask": attention_mask.numpy().astype(np.int64),
        "position_ids": position_ids.numpy().astype(np.int64),
        **initialize_cache(onnx_sess),
    }

    onnx_logits = onnx_sess.run(None, onnx_inputs)[0]
    logger.info(f"  ONNX logits: shape={onnx_logits.shape}")

    results = []
    if "arrays" in checks:
        atol, rtol = get_tolerances(precision)
        results.append(
            compare_arrays(f"decoder: '{prompt[:20]}...'", pytorch_logits, onnx_logits, atol, rtol)
        )
        if precision is None:
            max_diff = float(np.abs(pytorch_logits - onnx_logits).max())
            logger.info(f"  fp32 max |logit diff| = {max_diff:.2e}")
            assert max_diff < FP32_MAX_ABS_DIFF, f"fp32 logits differ by {max_diff:.2e}"
    if "top_k" in checks:
        results.append(
            compare_top_k(
                f"top-5: '{prompt[:20]}...'",
                pytorch_logits,
                onnx_logits,
                min_overlap=MIN_TOP5_OVERLAP.get(precision, 3),
            )
        )

    check_results(results, logger)
