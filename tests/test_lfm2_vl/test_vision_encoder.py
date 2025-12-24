"""Verify vision encoder ONNX export against PyTorch reference.

Note: Only tests tiled format. Conv2d format uses different preprocessing
(our preprocess_conv2d vs HuggingFace processor) which causes numerical
differences. Coherence tests verify conv2d works end-to-end.
"""

import logging
import pathlib

import numpy as np
import pytest
import torch
from PIL import Image

from liquidonnx.lfm2_vl import MODELS, VISION_MODE_TILED
from test_lfm2_vl.helpers import (
    bits_to_str,
    skip_if_missing,
    get_onnx_file,
    get_vl_onnx_dir,
    get_tolerances,
    load_onnx_session,
    compare_arrays,
    compare_correlation,
)

logger = logging.getLogger(__name__)

VISION_CORRELATION_THRESHOLD = 0.89

VISION_CONFIGS = [
    pytest.param(None, ["arrays"], id="fp32"),
    pytest.param(4, ["correlation"], id="q4"),
    pytest.param(8, ["arrays"], id="q8"),
]


def pad_to_square(image):
    """Pad image to square (matches ONNX preprocessing)."""
    w, h = image.size
    if w == h:
        return image
    max_dim = max(w, h)
    square_img = Image.new('RGB', (max_dim, max_dim), (0, 0, 0))
    paste_x = (max_dim - w) // 2
    paste_y = (max_dim - h) // 2
    square_img.paste(image, (paste_x, paste_y))
    return square_img


def get_pytorch_vision_embeddings(model, processor, image, apply_pad_to_square=True):
    """Get vision embeddings from PyTorch model.

    Args:
        apply_pad_to_square: Pad to square to match ONNX preprocessing
    """
    if apply_pad_to_square:
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
            feature = feature[:h * w].reshape(1, h, w, -1)
            proj_out = model.model.multi_modal_projector(feature)
            proj_out = proj_out.reshape(-1, proj_out.shape[-1])
            pytorch_embeddings.append(proj_out)

    return pytorch_embeddings, inputs


def verify_vision_tiled(embed_images_sess, inputs, pytorch_embeddings, checks, vision_bits):
    """Verify tiled vision encoder outputs."""
    pixel_values = inputs["pixel_values"]
    pixel_attention_mask = inputs["pixel_attention_mask"]

    onnx_outputs = embed_images_sess.run(None, {
        "pixel_values": pixel_values.numpy().astype(np.float32),
        "patch_attention_mask": pixel_attention_mask.numpy().astype(np.int64),
    })
    onnx_embeddings = onnx_outputs[0]

    atol, rtol = get_tolerances(vision_bits)
    results = []
    for tile_idx, pytorch_tile in enumerate(pytorch_embeddings):
        pytorch_np = pytorch_tile.numpy()
        onnx_tile = onnx_embeddings[tile_idx]
        min_tokens = min(pytorch_np.shape[0], onnx_tile.shape[0])

        if "arrays" in checks:
            results.append(compare_arrays(
                f"vision_tile_{tile_idx}",
                pytorch_np[:min_tokens], onnx_tile[:min_tokens],
                atol, rtol
            ))
        if "correlation" in checks:
            results.append(compare_correlation(
                f"vision_tile_{tile_idx}_corr",
                pytorch_np[:min_tokens], onnx_tile[:min_tokens],
                threshold=VISION_CORRELATION_THRESHOLD,
            ))

    return results


# pytorch_model outermost so same model runs consecutively (memory optimization)
# Only tests tiled format (conv2d has different preprocessing, verified via coherence tests)
@pytest.mark.parametrize("pytorch_model", MODELS.keys(), indirect=True)
@pytest.mark.parametrize("vision_bits,checks", VISION_CONFIGS)
def test_vision_encoder(
    exports_dir: pathlib.Path,
    cardinal_image: pathlib.Path,
    pytorch_model,
    vision_bits: int | None,
    checks: list[str],
):
    size, model, processor = pytorch_model
    logger.info(f"Testing vision encoder {size}/tiled/{bits_to_str(vision_bits)}")

    onnx_dir = get_vl_onnx_dir(exports_dir, size, VISION_MODE_TILED)
    skip_if_missing(onnx_dir, "Export not found")

    embed_images_file = get_onnx_file(onnx_dir, "embed_images", vision_bits)
    skip_if_missing(embed_images_file, "Vision encoder not found")

    embed_images_sess = load_onnx_session(onnx_dir, embed_images_file.name)
    image = Image.open(cardinal_image).convert("RGB")

    pytorch_embeddings, inputs = get_pytorch_vision_embeddings(model, processor, image)
    results = verify_vision_tiled(embed_images_sess, inputs, pytorch_embeddings, checks, vision_bits)

    for r in results:
        logger.info(f"  {r.name}: {'PASS' if r.passed else 'FAIL'} "
                    f"max_diff={r.max_diff:.4f} corr={r.correlation:.4f}")
        assert r.passed, f"{r.name}: max_diff={r.max_diff:.6f}, corr={r.correlation:.4f}, {r.details}"
