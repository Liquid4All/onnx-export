"""
Verify decoder ONNX export against PyTorch reference for LFM2-MoE.

Tests single-step logit comparison between PyTorch and ONNX models.

Run with:
    uv run pytest tests/test_lfm2_moe/test_decoder.py -v
    uv run pytest tests/test_lfm2_moe/test_decoder.py -v -k "LFM2.5-8B-A1B and q4"
"""

import logging
import pathlib

import numpy as np
import pytest
import torch
from helpers import get_model_name, get_onnx_dir

from liquidonnx.session import get_onnx_file, load_onnx_session
from liquidonnx.verify import check_results, compare_arrays, compare_top_k, get_tolerances

logger = logging.getLogger(__name__)

# HuggingFace model IDs to test
MODELS = [
    "LiquidAI/LFM2-8B-A1B",
    "LiquidAI/LFM2.5-8B-A1B",
]

PROMPTS = ["Hello, how are", "The sky is", "1 + 1 ="]

QUANT_CONFIGS = [
    pytest.param(None, ["arrays", "top_k"], id="fp32"),
    pytest.param("fp16", ["top_k"], id="fp16"),
    pytest.param("q4", ["top_k"], id="q4"),
    pytest.param("q4f16", ["top_k"], id="q4f16"),
    pytest.param("q8", ["top_k"], id="q8"),
]


@pytest.mark.parametrize("pytorch_model", MODELS, indirect=True)
@pytest.mark.parametrize("precision,checks", QUANT_CONFIGS)
@pytest.mark.parametrize("prompt", PROMPTS)
def test_decoder(
    exports_dir: pathlib.Path,
    pytorch_model,
    precision: str | None,
    checks: list[str],
    prompt: str,
):
    """Test single-step decoder logits against PyTorch."""
    model_id, model, tokenizer = pytorch_model
    model_name = get_model_name(model_id)
    logger.info(f"Testing {model_name}/{precision or 'fp32'}: '{prompt}'")

    onnx_dir = get_onnx_dir(exports_dir, model_id)
    onnx_file = get_onnx_file(onnx_dir, precision)

    if not onnx_file.exists():
        precision_arg = f" --precision {precision}" if precision else ""
        pytest.skip(
            f"ONNX file not found: {onnx_file}\n"
            f"Export with: uv run lfm2-moe-export {model_id}{precision_arg}"
        )

    try:
        onnx_sess = load_onnx_session(onnx_file)
    except Exception as e:
        pytest.skip(f"ONNX model failed to load (may need CUDA for {precision}): {e}")

    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    seq_len = input_ids.shape[1]
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(seq_len).unsqueeze(0)
    logger.info(f"  Input: seq_len={seq_len}, tokens={input_ids[0].tolist()}")

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        pytorch_logits = outputs.logits.numpy()
    logger.info(f"  PyTorch logits: shape={pytorch_logits.shape}")

    available_inputs = {
        "input_ids": input_ids.numpy().astype(np.int64),
        "attention_mask": attention_mask.numpy().astype(np.int64),
        "position_ids": position_ids.numpy().astype(np.int64),
    }

    # Build inputs dict based on what the session actually expects
    onnx_inputs = {}
    for inp in onnx_sess.get_inputs():
        if inp.name in available_inputs:
            onnx_inputs[inp.name] = available_inputs[inp.name]
        else:
            # KV cache and other optional inputs - initialize with zeros
            # FP16 models expect float16 inputs for KV cache
            expected_dtype = inp.type
            if "float16" in expected_dtype:
                dtype = np.float16
            else:
                dtype = np.float32
            shape = [d if isinstance(d, int) else 1 for d in inp.shape]
            onnx_inputs[inp.name] = np.zeros(shape, dtype=dtype)

    onnx_logits = onnx_sess.run(None, onnx_inputs)[0]
    logger.info(f"  ONNX logits: shape={onnx_logits.shape}")

    results = []
    if "arrays" in checks:
        atol, rtol = get_tolerances(precision)
        results.append(
            compare_arrays(f"decoder: '{prompt[:20]}...'", pytorch_logits, onnx_logits, atol, rtol)
        )
    if "top_k" in checks:
        # Q4 has more aggressive quantization, so lower threshold
        min_overlap = 5 if precision in (None, "fp16") else 2
        results.append(
            compare_top_k(
                f"top-5: '{prompt[:20]}...'", pytorch_logits, onnx_logits, min_overlap=min_overlap
            )
        )

    check_results(results, logger)
