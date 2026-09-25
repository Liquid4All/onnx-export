"""
Multi-turn coherence of LFM2-MoE exports on onnxruntime-genai against PyTorch.

Each side continues its own conversation; the score is the mean cosine of the logits that chose
each answer token.

Run with:
    uv run pytest tests/test_lfm2_moe/test_coherence.py -v
    uv run pytest tests/test_lfm2_moe/test_coherence.py -v -k "LFM2.5-8B-A1B and q4"
"""

import logging
import pathlib

import pytest
from helpers import get_export_dir, require_genai, text_coherence

from liquidonnx.genai_runtime import load_model
from liquidonnx.lfm2.export import genai_files, model_file

logger = logging.getLogger(__name__)

MODELS = [
    "LiquidAI/LFM2-8B-A1B",
    "LiquidAI/LFM2.5-8B-A1B",
]
PRECISIONS = ["fp32", "q4", "q8"]
MAX_NEW_TOKENS = 20
SIMILARITY_THRESHOLD_FP32 = 0.95
SIMILARITY_THRESHOLD_QUANT = 0.70

PROMPTS = [
    "My name is Sarah and I work as a software engineer. Can you remember this?",
    "What is my name?",
    "What is my profession?",
]


@pytest.mark.parametrize("pytorch_model", MODELS, indirect=True)
@pytest.mark.parametrize("precision", PRECISIONS)
def test_coherence(exports_dir: pathlib.Path, pytorch_model, precision: str):
    require_genai("lfm2_moe")
    model_id, model, tokenizer = pytorch_model
    export = get_export_dir(exports_dir, model_id)
    if not (export / "onnx" / model_file(precision)).exists():
        pytest.skip(
            f"{precision} not exported: uv run lfm2-moe-export {model_id} --precision {precision}"
        )

    genai_model = load_model(export, genai_files(precision))
    similarity = text_coherence(model, tokenizer, genai_model, PROMPTS, MAX_NEW_TOKENS)

    threshold = SIMILARITY_THRESHOLD_FP32 if precision == "fp32" else SIMILARITY_THRESHOLD_QUANT
    assert similarity > threshold, f"similarity {similarity:.4f} <= {threshold}"
