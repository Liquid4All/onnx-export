"""
Verify vision encoder ONNX export against PyTorch reference.

Note: Only tests tiled format. Conv2d format uses different preprocessing
(our preprocess_conv2d vs HuggingFace processor) which causes numerical
differences. Coherence tests verify conv2d works end-to-end.

Run with:
    uv run pytest tests/test_lfm2_vl/test_vision_encoder.py -v
    uv run pytest tests/test_lfm2_vl/test_vision_encoder.py -v -k "450M and q4"
"""

import logging
import pathlib

import numpy as np
import pytest
import torch
from helpers import get_model_name, get_onnx_dir
from PIL import Image

from liquidonnx.lfm2_vl.preprocessing import pad_to_square
from liquidonnx.session import get_onnx_file, load_onnx_session
from liquidonnx.verify import check_results, compare_arrays, compare_correlation, get_tolerances

logger = logging.getLogger(__name__)

# HuggingFace model IDs to test
MODELS = [
    "LiquidAI/LFM2-VL-450M",
    "LiquidAI/LFM2-VL-1.6B",
    "LiquidAI/LFM2-VL-3B",
    "LiquidAI/LFM2.5-VL-1.6B",
]

VISION_CORRELATION_THRESHOLD = 0.89

QUANT_CONFIGS = [
    pytest.param(None, ["arrays"], id="fp32"),
    pytest.param("fp16", ["arrays"], id="fp16"),
    pytest.param("q4", ["correlation"], id="q4"),
    pytest.param("q8", ["arrays"], id="q8"),
]


def get_pytorch_vision_embeddings(model, processor, image):
    """Get vision embeddings from PyTorch model."""
    image = pad_to_square(image)

    inputs = processor.image_processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"]
    pixel_attention_mask = inputs["pixel_attention_mask"]
    spatial_shapes = inputs["spatial_shapes"]

    with torch.no_grad():
        vision_outputs = model.model.vision_tower(
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
        ).last_hidden_state

        pytorch_embeddings = []
        for tile_idx in range(pixel_values.shape[0]):
            feature = vision_outputs[tile_idx]
            h, w = spatial_shapes[tile_idx].tolist()
            feature = feature[: h * w].reshape(1, h, w, -1)
            proj_out = model.model.multi_modal_projector(feature)
            proj_out = proj_out.reshape(-1, proj_out.shape[-1])
            pytorch_embeddings.append(proj_out)

    return pytorch_embeddings, inputs


def verify_vision_tiled(embed_images_sess, inputs, pytorch_embeddings, checks, vision_type):
    """Verify tiled vision encoder outputs."""
    pixel_values = inputs["pixel_values"]
    pixel_attention_mask = inputs["pixel_attention_mask"]
    spatial_shapes = inputs["spatial_shapes"]

    onnx_outputs = embed_images_sess.run(
        None,
        {
            "pixel_values": pixel_values.numpy().astype(np.float32),
            "pixel_attention_mask": pixel_attention_mask.numpy().astype(np.int64),
            "spatial_shapes": spatial_shapes.numpy().astype(np.int64),
        },
    )
    # Output is 2D [total_tokens, hidden] after Compress
    onnx_embeddings = onnx_outputs[0]

    # Concatenate PyTorch embeddings for comparison
    pytorch_concat = torch.cat(pytorch_embeddings, dim=0).numpy()
    min_tokens = min(pytorch_concat.shape[0], onnx_embeddings.shape[0])

    atol, rtol = get_tolerances(vision_type)
    results = []

    if "arrays" in checks:
        results.append(
            compare_arrays(
                "vision_embeddings",
                pytorch_concat[:min_tokens],
                onnx_embeddings[:min_tokens],
                atol,
                rtol,
            )
        )
    if "correlation" in checks:
        results.append(
            compare_correlation(
                "vision_embeddings_corr",
                pytorch_concat[:min_tokens],
                onnx_embeddings[:min_tokens],
                threshold=VISION_CORRELATION_THRESHOLD,
            )
        )

    return results


# pytorch_model outermost so same model runs consecutively (memory optimization)
# Only tests tiled format (conv2d has different preprocessing, verified via coherence tests)
@pytest.mark.parametrize("pytorch_model", MODELS, indirect=True)
@pytest.mark.parametrize("vision_type,checks", QUANT_CONFIGS)
def test_vision_encoder(
    exports_dir: pathlib.Path,
    cardinal_image: pathlib.Path,
    pytorch_model,
    vision_type: str | None,
    checks: list[str],
):
    model_id, model, processor = pytorch_model
    model_name = get_model_name(model_id)
    logger.info(f"Testing vision encoder {model_name}/{vision_type or 'fp32'}")

    onnx_dir = get_onnx_dir(exports_dir, model_id)
    if not onnx_dir.exists():
        pytest.skip(f"Export not found: {onnx_dir}")

    embed_images_file = get_onnx_file(onnx_dir, vision_type, "embed_images")
    if not embed_images_file.exists():
        pytest.skip(f"Vision encoder not found: {embed_images_file}")

    embed_images_sess = load_onnx_session(embed_images_file)
    image = Image.open(cardinal_image).convert("RGB")

    pytorch_embeddings, inputs = get_pytorch_vision_embeddings(model, processor, image)
    results = verify_vision_tiled(
        embed_images_sess, inputs, pytorch_embeddings, checks, vision_type
    )

    check_results(results, logger)
