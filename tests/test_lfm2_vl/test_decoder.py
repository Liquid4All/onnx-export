"""
Verify decoder ONNX export against PyTorch reference.

Run with:
    uv run pytest tests/test_lfm2_vl/test_decoder.py -v
    uv run pytest tests/test_lfm2_vl/test_decoder.py -v -k "450M and q4"
"""

import logging
import pathlib

import pytest
import torch
from helpers import get_model_name, get_onnx_dir

from liquidonnx.embeddings import embed
from liquidonnx.lfm2_vl.export import bundle
from liquidonnx.session import decoder_inputs, initialize_cache, load_onnx_session
from liquidonnx.verify import check_results, compare_arrays, compare_top_k, get_tolerances

logger = logging.getLogger(__name__)

# HuggingFace model IDs to test
MODELS = [
    "LiquidAI/LFM2-VL-450M",
    "LiquidAI/LFM2-VL-1.6B",
    "LiquidAI/LFM2-VL-3B",
    "LiquidAI/LFM2.5-VL-1.6B",
]

PROMPTS = ["Hello, how are", "The image shows", "I can see"]

QUANT_CONFIGS = [
    pytest.param(None, ["arrays", "top_k"], id="fp32"),
    pytest.param("fp16", ["arrays", "top_k"], id="fp16"),
    pytest.param("q4", ["top_k"], id="q4"),
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

    files = bundle(decoder_type or "fp32")
    for name in ("decoder", "embedding"):
        if not (onnx_dir / files[name]).exists():
            pytest.skip(f"{files[name]} not found in {onnx_dir}")

    embeddings_sess = load_onnx_session(onnx_dir / files["embedding"])
    decoder_sess = load_onnx_session(onnx_dir / files["decoder"])

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

    onnx_embeds = embed(embeddings_sess, input_ids.numpy())
    onnx_inputs = decoder_inputs(
        decoder_sess, onnx_embeds, initialize_cache(decoder_sess), past_len=0
    )
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
