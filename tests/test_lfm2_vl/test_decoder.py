"""Verify decoder ONNX export against PyTorch reference."""

import logging
import pathlib

import numpy as np
import pytest
import torch
from helpers import skip_if_missing

from liquidonnx.lfm2_vl import MODELS, VISION_MODE_TILED
from liquidonnx.lfm2_vl.generate import get_onnx_dir
from liquidonnx.quantize import bits_to_str
from liquidonnx.session import get_onnx_file, load_onnx_session
from liquidonnx.verify import check_results, compare_arrays, compare_top_k, get_tolerances

logger = logging.getLogger(__name__)

PROMPTS = ["Hello, how are", "The image shows", "I can see"]

QUANT_CONFIGS = [
    pytest.param(None, ["arrays", "top_k"], id="fp32"),
    pytest.param(4, ["top_k"], id="q4"),
    pytest.param(8, ["arrays", "top_k"], id="q8"),
]


# pytorch_model outermost so same model runs consecutively (memory optimization)
@pytest.mark.parametrize("pytorch_model", MODELS.keys(), indirect=True)
@pytest.mark.parametrize("decoder_bits,checks", QUANT_CONFIGS)
@pytest.mark.parametrize("prompt", PROMPTS)
def test_decoder(
    exports_dir: pathlib.Path,
    pytorch_model,
    decoder_bits: int | None,
    checks: list[str],
    prompt: str,
):
    size, model, processor = pytorch_model
    logger.info(f"Testing {size}/{bits_to_str(decoder_bits)}: '{prompt}'")

    onnx_dir = get_onnx_dir(exports_dir, size, VISION_MODE_TILED)
    skip_if_missing(onnx_dir, "Export not found")

    decoder_file = get_onnx_file(onnx_dir, decoder_bits, "decoder")
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
        atol, rtol = get_tolerances(decoder_bits)
        results.append(
            compare_arrays(f"decoder: '{prompt[:20]}...'", pytorch_logits, onnx_logits, atol, rtol)
        )
    if "top_k" in checks:
        min_overlap = 3 if decoder_bits else 5
        results.append(
            compare_top_k(
                f"top-5: '{prompt[:20]}...'", pytorch_logits, onnx_logits, min_overlap=min_overlap
            )
        )

    check_results(results, logger)
