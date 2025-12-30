"""
Verify decoder ONNX export against PyTorch reference for LFM2.5-VL.

Run with:
    uv run pytest tests/test_lfm25_vl/test_decoder.py -v
    uv run pytest tests/test_lfm25_vl/test_decoder.py -v -k "q4"
"""

import logging
import pathlib

import numpy as np
import pytest
import torch
from helpers import skip_if_missing

from liquidonnx.session import get_onnx_file, load_onnx_session
from liquidonnx.verify import check_results, compare_arrays, compare_top_k, get_tolerances

logger = logging.getLogger(__name__)

MODEL_NAME = "LFM2-VL-1.6B-3102461"


def get_onnx_dir(exports_dir: pathlib.Path) -> pathlib.Path:
    """Get ONNX directory for the LFM2.5-VL model."""
    return exports_dir / f"{MODEL_NAME}-ONNX" / "onnx"


PROMPTS = ["Hello, how are", "The image shows", "I can see"]

QUANT_CONFIGS = [
    pytest.param(None, ["arrays", "top_k"], id="fp32"),
    pytest.param("fp16", ["arrays", "top_k"], id="fp16"),
    pytest.param("q4", ["top_k"], id="q4"),
    pytest.param("q8", ["arrays", "top_k"], id="q8"),
]


@pytest.mark.parametrize("decoder_type,checks", QUANT_CONFIGS)
@pytest.mark.parametrize("prompt", PROMPTS)
def test_decoder(
    exports_dir: pathlib.Path,
    pytorch_model,
    decoder_type: str | None,
    checks: list[str],
    prompt: str,
):
    model_name, model, processor = pytorch_model
    logger.info(f"Testing {model_name}/{decoder_type or 'fp32'}: '{prompt}'")

    onnx_dir = get_onnx_dir(exports_dir)
    skip_if_missing(onnx_dir, "Export not found")

    decoder_file = get_onnx_file(onnx_dir, decoder_type, "decoder")
    skip_if_missing(decoder_file, "Decoder not found")
    embed_tokens_sess = load_onnx_session(onnx_dir / "embed_tokens.onnx")
    decoder_sess = load_onnx_session(decoder_file)

    input_ids = processor.tokenizer.encode(prompt, return_tensors="pt")
    seq_len = input_ids.shape[1]
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(seq_len).unsqueeze(0)
    logger.info(f"  Input: seq_len={seq_len}, tokens={input_ids[0].tolist()}")

    with torch.no_grad():
        inputs_embeds = model.model.language_model.embed_tokens(input_ids)
        lm_outputs = model.model.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        pytorch_logits = model.lm_head(lm_outputs.last_hidden_state).numpy()
    logger.info(f"  PyTorch logits: shape={pytorch_logits.shape}")

    onnx_embeds = embed_tokens_sess.run(
        None,
        {
            "input_ids": input_ids.numpy().astype(np.int64),
        },
    )[0]

    onnx_inputs = {
        "inputs_embeds": onnx_embeds.astype(np.float32),
        "attention_mask": attention_mask.numpy().astype(np.int64),
        "position_ids": position_ids.numpy().astype(np.int64),
    }

    for inp in decoder_sess.get_inputs():
        if inp.name not in onnx_inputs:
            shape = [d if isinstance(d, int) else 1 for d in inp.shape]
            onnx_inputs[inp.name] = np.zeros(shape, dtype=np.float32)

    onnx_logits = decoder_sess.run(None, onnx_inputs)[0]
    logger.info(f"  ONNX logits: shape={onnx_logits.shape}")

    results = []
    if "arrays" in checks:
        atol, rtol = get_tolerances(decoder_type)
        results.append(
            compare_arrays(f"decoder: '{prompt[:20]}...'", pytorch_logits, onnx_logits, atol, rtol)
        )
    if "top_k" in checks:
        min_overlap = 5 if decoder_type in (None, "fp16") else 3
        results.append(
            compare_top_k(
                f"top-5: '{prompt[:20]}...'", pytorch_logits, onnx_logits, min_overlap=min_overlap
            )
        )

    check_results(results, logger)
