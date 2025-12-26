"""
Verify decoder ONNX export against PyTorch reference.

Tests single-step logit comparison between PyTorch and ONNX models.

Run with:
    pytest tests/test_lfm2/test_decoder.py -v
    pytest tests/test_lfm2/test_decoder.py -v -k "350M and q4"
"""

import logging
import pathlib

import numpy as np
import pytest
import torch
from helpers import skip_if_missing

from liquidonnx.lfm2 import MODELS
from liquidonnx.lfm2.generate import get_onnx_dir
from liquidonnx.quantize import bits_to_str
from liquidonnx.session import get_onnx_file, load_onnx_session
from liquidonnx.verify import check_results, compare_arrays, compare_top_k, get_tolerances

logger = logging.getLogger(__name__)

PROMPTS = ["Hello, how are", "The sky is", "1 + 1 ="]

QUANT_CONFIGS = [
    pytest.param(None, ["arrays", "top_k"], id="fp32"),
    pytest.param(4, ["top_k"], id="q4"),
    pytest.param(8, ["top_k"], id="q8"),
]


@pytest.mark.parametrize("pytorch_model", MODELS.keys(), indirect=True)
@pytest.mark.parametrize("bits,checks", QUANT_CONFIGS)
@pytest.mark.parametrize("prompt", PROMPTS)
def test_decoder(
    exports_dir: pathlib.Path,
    pytorch_model,
    bits: int | None,
    checks: list[str],
    prompt: str,
):
    """Test single-step decoder logits against PyTorch."""
    size, model, tokenizer = pytorch_model
    logger.info(f"Testing {size}/{bits_to_str(bits)}: '{prompt}'")

    onnx_dir = get_onnx_dir(exports_dir, size)
    skip_if_missing(onnx_dir, "Export not found")

    onnx_file = get_onnx_file(onnx_dir, bits)
    skip_if_missing(onnx_file, f"ONNX file not found: {onnx_file.name}")

    onnx_sess = load_onnx_session(onnx_file)

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

    onnx_inputs = {
        "input_ids": input_ids.numpy().astype(np.int64),
        "attention_mask": attention_mask.numpy().astype(np.int64),
        "position_ids": position_ids.numpy().astype(np.int64),
    }

    for inp in onnx_sess.get_inputs():
        if inp.name not in onnx_inputs:
            shape = [d if isinstance(d, int) else 1 for d in inp.shape]
            onnx_inputs[inp.name] = np.zeros(shape, dtype=np.float32)

    onnx_logits = onnx_sess.run(None, onnx_inputs)[0]
    logger.info(f"  ONNX logits: shape={onnx_logits.shape}")

    results = []
    if "arrays" in checks:
        atol, rtol = get_tolerances(bits)
        results.append(
            compare_arrays(f"decoder: '{prompt[:20]}...'", pytorch_logits, onnx_logits, atol, rtol)
        )
    if "top_k" in checks:
        # Quantized models may have slight logit differences causing token reordering
        min_overlap = 3 if bits else 5
        results.append(
            compare_top_k(
                f"top-5: '{prompt[:20]}...'", pytorch_logits, onnx_logits, min_overlap=min_overlap
            )
        )

    check_results(results, logger)
