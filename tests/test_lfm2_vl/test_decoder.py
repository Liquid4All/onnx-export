"""
Verify decoder ONNX export against PyTorch reference.

Run with:
    uv run pytest tests/test_lfm2_vl/test_decoder.py -v
    uv run pytest tests/test_lfm2_vl/test_decoder.py -v -k "450M and q4"
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
    "LiquidAI/LFM2-VL-450M",
    "LiquidAI/LFM2-VL-1.6B",
    "LiquidAI/LFM2-VL-3B",
]

PROMPTS = ["Hello, how are", "The image shows", "I can see"]

QUANT_CONFIGS = [
    pytest.param(None, ["arrays", "top_k"], id="fp32"),
    pytest.param("fp16", ["arrays", "top_k"], id="fp16"),
    pytest.param("q4", ["top_k"], id="q4"),
    # arrays check ok: embed_tokens.onnx stays fp32, reducing quantization error
    pytest.param("q8", ["arrays", "top_k"], id="q8"),
]


# pytorch_model outermost so same model runs consecutively (memory optimization)
@pytest.mark.parametrize("pytorch_model", MODELS, indirect=True)
@pytest.mark.parametrize("decoder_type,checks", QUANT_CONFIGS)
@pytest.mark.parametrize("prompt", PROMPTS)
def test_decoder(
    exports_dir: pathlib.Path,
    pytorch_model,
    decoder_type: str | None,
    checks: list[str],
    prompt: str,
):
    model_id, model, processor = pytorch_model
    model_name = get_model_name(model_id)
    logger.info(f"Testing {model_name}/{decoder_type or 'fp32'}: '{prompt}'")

    onnx_dir = get_onnx_dir(exports_dir, model_id)
    if not onnx_dir.exists():
        pytest.skip(f"Export not found: {onnx_dir}")

    decoder_file = get_onnx_file(onnx_dir, decoder_type, "decoder")
    if not decoder_file.exists():
        pytest.skip(f"Decoder not found: {decoder_file}")

    embed_tokens_file = onnx_dir / "embed_tokens.onnx"
    if not embed_tokens_file.exists():
        pytest.skip(f"embed_tokens.onnx not found: {embed_tokens_file}")

    embed_tokens_sess = load_onnx_session(embed_tokens_file)
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

    # Build inputs based on what the decoder actually expects
    decoder_input_names = {inp.name for inp in decoder_sess.get_inputs()}
    onnx_inputs = {
        "inputs_embeds": onnx_embeds.astype(np.float32),
        "attention_mask": attention_mask.numpy().astype(np.int64),
    }
    if "position_ids" in decoder_input_names:
        onnx_inputs["position_ids"] = position_ids.numpy().astype(np.int64)

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
