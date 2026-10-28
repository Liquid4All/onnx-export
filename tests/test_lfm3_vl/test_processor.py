"""
Preprocessing of lfm3-vl-infer (stock Qwen2VLImageProcessor, chat template, image-token
expansion) against the checkpoint's own processor on its tokenizer (no weights needed).

Run with:
    uv run pytest tests/test_lfm3_vl/test_processor.py -v
"""

import numpy as np
import pytest

from liquidonnx.lfm3_vl.session import ONNXVLModel, preprocess_images

from .conftest import chat

CASES = [
    (["cardinal.jpg"], "What bird is this?"),
    (["wide.jpg", "tall.jpg"], "Compare the two images."),
    (["bluejay.jpg"], "Describe it."),
    ([], "Write a haiku about the sea."),
]


@pytest.mark.parametrize("images,question", CASES, ids=["cardinal", "wide+tall", "bluejay", "text"])
@pytest.mark.parametrize("thinking", [True, False], ids=["think", "no-think"])
def test_matches_reference_processor(
    checkpoint_files, reference_processor, images, question, thinking
):
    messages, pil = chat(images, question)
    text = reference_processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False, enable_thinking=thinking
    )
    expected = reference_processor(images=pil or None, text=text, return_tensors="np")

    model = ONNXVLModel(checkpoint_files)
    model.load_preprocessing()
    grids = np.zeros((0, 3), dtype=np.int64)
    if pil:
        pixel_values, grids = preprocess_images(model.processor, pil)
        np.testing.assert_array_equal(grids, expected["image_grid_thw"])
        np.testing.assert_array_equal(pixel_values, expected["pixel_values"].astype(np.float32))
    ids = model.prompt_ids(messages, grids, enable_thinking=thinking)
    assert ids == expected["input_ids"][0].tolist()
