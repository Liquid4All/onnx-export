"""Verify decoder ONNX export against PyTorch reference."""

import pathlib

import numpy as np
import pytest
import torch

from liquidonnx.lfm2_vl import MODELS, VISION_MODES
from test_lfm2_vl.helpers import (
    DECODER_BITS,
    VerificationResult,
    skip_if_missing,
    get_onnx_file,
    get_vl_onnx_dir,
    get_tolerances,
    load_pytorch_model,
    load_onnx_session,
    compare_arrays,
)

PROMPTS = ["Hello, how are", "The image shows", "I can see"]


def compare_top_k(name: str, expected: np.ndarray, actual: np.ndarray, k: int = 5) -> VerificationResult:
    exp_logits = expected[0, -1]
    act_logits = actual[0, -1]

    exp_top_k = np.argsort(exp_logits)[-k:][::-1]
    act_top_k = np.argsort(act_logits)[-k:][::-1]

    top1_match = exp_top_k[0] == act_top_k[0]
    top_k_overlap = len(set(exp_top_k) & set(act_top_k))

    return VerificationResult(
        name=name, passed=top1_match,
        max_diff=0.0 if top1_match else 1.0,
        mean_diff=1.0 - (top_k_overlap / k),
        correlation=top_k_overlap / k,
        details=f"Top-1 match: {top1_match}, Top-{k} overlap: {top_k_overlap}/{k}, "
                f"Expected: {exp_top_k.tolist()}, Actual: {act_top_k.tolist()}"
    )


@pytest.mark.parametrize("prompt", PROMPTS)
@pytest.mark.parametrize("decoder_bits", DECODER_BITS)
@pytest.mark.parametrize("vision_mode", VISION_MODES)
@pytest.mark.parametrize("size", MODELS.keys())
def test_decoder(
    exports_dir: pathlib.Path,
    size: str,
    vision_mode: str,
    decoder_bits: int | None,
    prompt: str,
):
    onnx_dir = get_vl_onnx_dir(exports_dir, size, vision_mode)
    skip_if_missing(onnx_dir, "Export not found")

    decoder_file = get_onnx_file(onnx_dir, "decoder", decoder_bits)
    skip_if_missing(decoder_file, "Decoder not found")

    model, processor = load_pytorch_model(MODELS[size])
    embed_tokens_sess = load_onnx_session(onnx_dir, "embed_tokens.onnx")
    decoder_sess = load_onnx_session(onnx_dir, decoder_file.name)

    input_ids = processor.tokenizer.encode(prompt, return_tensors="pt")
    seq_len = input_ids.shape[1]
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(seq_len).unsqueeze(0)

    with torch.no_grad():
        inputs_embeds = model.model.language_model.embed_tokens(input_ids)
        lm_outputs = model.model.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        pytorch_logits = model.lm_head(lm_outputs.last_hidden_state).numpy()

    onnx_embeds = embed_tokens_sess.run(None, {
        "input_ids": input_ids.numpy().astype(np.int64),
    })[0]

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

    atol, rtol = get_tolerances(decoder_bits)
    results = (
        compare_arrays(f"decoder: '{prompt[:20]}...'",
                       pytorch_logits, onnx_logits, atol, rtol),
        compare_top_k(f"top-5: '{prompt[:20]}...'",
                      pytorch_logits, onnx_logits),
    )

    for r in results:
        assert r.passed, f"{r.name}: max_diff={r.max_diff:.6f}, {r.details}"
