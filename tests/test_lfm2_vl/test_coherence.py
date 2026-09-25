"""
Multi-turn coherence of LFM2-VL exports on onnxruntime-genai against PyTorch, with images.

Each side continues its own conversation; the score is the mean cosine of the logits that chose
each answer token. onnxruntime-genai resizes each image once, so PyTorch runs with
do_image_splitting=False.

Run with:
    uv run pytest tests/test_lfm2_vl/test_coherence.py -v
    uv run pytest tests/test_lfm2_vl/test_coherence.py -v -k "LFM2.5-VL-1.6B and q4"
"""

import logging
import pathlib

import numpy as np
import onnxruntime_genai as og
import pytest
from helpers import generate_pytorch, generate_with_logits, get_export_dir, require_genai
from PIL import Image

from liquidonnx.genai_runtime import load_model
from liquidonnx.lfm2_vl.export import bundle, genai_files
from liquidonnx.verify import compare_logits_similarity

logger = logging.getLogger(__name__)

MODELS = [
    "LiquidAI/LFM2-VL-450M",
    "LiquidAI/LFM2-VL-1.6B",
    "LiquidAI/LFM2-VL-3B",
    "LiquidAI/LFM2.5-VL-1.6B",
]
PRECISIONS = ["fp32", "fp16", "q4", "q8"]
MAX_NEW_TOKENS = 20
SIMILARITY_THRESHOLD_FP32 = 0.75
SIMILARITY_THRESHOLD_QUANT = 0.7

SINGLE_IMAGE_PROMPTS = [
    "What do you see in this image? Describe the main elements.",
    "What colors are present in the image?",
    "Can you identify any shapes or patterns?",
]
MULTI_IMAGE_PROMPTS = [
    "Which one most important thing do you see on each image? Be concise and exact.",
    "What are the similarities between these images?",
    "What are the differences between these images?",
]
SCENARIOS = [
    pytest.param("single", SINGLE_IMAGE_PROMPTS, id="single"),
    pytest.param("multi", MULTI_IMAGE_PROMPTS, id="multi"),
]


def run_multi_turn_coherence(model, processor, genai_model, images: list[pathlib.Path], prompts):
    """The images go with the first prompt and stay in the conversation."""
    genai_processor = genai_model.create_multimodal_processor()
    pil = [Image.open(path).convert("RGB") for path in images]
    conversations = {"pytorch": [], "genai": []}
    similarities = []
    for turn, prompt in enumerate(prompts, 1):
        answers = {}
        for side, messages in conversations.items():
            if turn == 1:
                content = [*({"type": "image"} for _ in images), {"type": "text", "text": prompt}]
                messages.append({"role": "user", "content": content})
            else:
                messages.append({"role": "user", "content": prompt})
            text = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            if side == "pytorch":
                inputs = processor(
                    text=text, images=pil or None, return_tensors="pt", do_image_splitting=False
                )
                answers[side] = generate_pytorch(model, dict(inputs), MAX_NEW_TOKENS)
            else:
                paths = [str(path) for path in images]
                inputs = genai_processor(text, images=og.Images.open(*paths) if paths else None)
                answers[side] = generate_with_logits(genai_model, inputs, MAX_NEW_TOKENS)
            response = processor.tokenizer.decode(answers[side][0], skip_special_tokens=True)
            messages.append({"role": "assistant", "content": response})

        similarities.append(compare_logits_similarity(answers["pytorch"][1], answers["genai"][1]))
        logger.info(f"  Turn {turn}: similarity={similarities[-1]:.4f}")
        for side, messages in conversations.items():
            logger.info(f"    {side}: {messages[-1]['content'][:80]}")
    return float(np.mean(similarities))


# pytorch_model outermost so same model runs consecutively (memory optimization)
@pytest.mark.parametrize("pytorch_model", MODELS, indirect=True)
@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize("scenario,prompts", SCENARIOS)
def test_coherence(
    exports_dir: pathlib.Path,
    cardinal_image: pathlib.Path,
    bluejay_image: pathlib.Path,
    pytorch_model,
    precision: str,
    scenario: str,
    prompts: list[str],
):
    require_genai("lfm2_vl")
    model_id, model, processor = pytorch_model
    export = get_export_dir(exports_dir, model_id)
    missing = [f for f in bundle(precision).values() if not (export / "onnx" / f).exists()]
    if missing:
        pytest.skip(f"{precision} not exported in {export}: {', '.join(missing)}")

    images = [cardinal_image] if scenario == "single" else [cardinal_image, bluejay_image]
    genai_model = load_model(export, genai_files(precision))
    similarity = run_multi_turn_coherence(model, processor, genai_model, images, prompts)

    is_float = precision in ("fp32", "fp16")
    threshold = SIMILARITY_THRESHOLD_FP32 if is_float else SIMILARITY_THRESHOLD_QUANT
    assert similarity > threshold, f"similarity {similarity:.4f} <= {threshold}"
