"""Verify vision encoder ONNX export against PyTorch reference."""

import pathlib

import numpy as np
import pytest
import torch
from PIL import Image

from liquidonnx.lfm2_vl import MODELS, VISION_MODES, VISION_MODE_CONV2D
from liquidonnx.lfm2_vl.preprocessing import detect_vision_format, preprocess_conv2d
from test_lfm2_vl.helpers import (
    VISION_BITS,
    skip_if_missing,
    get_onnx_file,
    get_vl_onnx_dir,
    get_tolerances,
    load_pytorch_model,
    load_onnx_session,
    compare_arrays,
)


def get_pytorch_vision_embeddings(model, processor, image):
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


def verify_vision_conv2d(embed_images_sess, image, pytorch_embeddings, atol, rtol):
    onnx_pixel_values, spatial_h, spatial_w = preprocess_conv2d(image)
    onnx_outputs = embed_images_sess.run(None, {
        "pixel_values": onnx_pixel_values,
        "spatial_h": np.array(spatial_h, dtype=np.int64),
        "spatial_w": np.array(spatial_w, dtype=np.int64),
    })
    onnx_embeddings = onnx_outputs[0]

    pytorch_concat = torch.cat(pytorch_embeddings, dim=0).numpy()
    onnx_flat = onnx_embeddings[0]
    min_tokens = min(pytorch_concat.shape[0], onnx_flat.shape[0])

    return [compare_arrays(
        "vision_conv2d",
        pytorch_concat[:min_tokens], onnx_flat[:min_tokens],
        atol, rtol
    )]


def verify_vision_tiled(embed_images_sess, inputs, pytorch_embeddings, atol, rtol):
    pixel_values = inputs["pixel_values"]
    pixel_attention_mask = inputs["pixel_attention_mask"]

    onnx_outputs = embed_images_sess.run(None, {
        "pixel_values": pixel_values.numpy().astype(np.float32),
        "patch_attention_mask": pixel_attention_mask.numpy().astype(np.int64),
    })
    onnx_embeddings = onnx_outputs[0]

    results = []
    for tile_idx, pytorch_tile in enumerate(pytorch_embeddings):
        pytorch_np = pytorch_tile.numpy()
        onnx_tile = onnx_embeddings[tile_idx]
        min_tokens = min(pytorch_np.shape[0], onnx_tile.shape[0])

        results.append(compare_arrays(
            f"vision_tile_{tile_idx}",
            pytorch_np[:min_tokens], onnx_tile[:min_tokens],
            atol, rtol
        ))
    return results


@pytest.mark.parametrize("vision_bits", VISION_BITS)
@pytest.mark.parametrize("vision_mode", VISION_MODES)
@pytest.mark.parametrize("size", MODELS.keys())
def test_vision_encoder(
    exports_dir: pathlib.Path,
    cardinal_image: pathlib.Path,
    size: str,
    vision_mode: str,
    vision_bits: int | None,
):
    onnx_dir = get_vl_onnx_dir(exports_dir, size, vision_mode)
    skip_if_missing(onnx_dir, "Export not found")

    embed_images_file = get_onnx_file(onnx_dir, "embed_images", vision_bits)
    skip_if_missing(embed_images_file, "Vision encoder not found")

    model, processor = load_pytorch_model(MODELS[size])
    embed_images_sess = load_onnx_session(onnx_dir, embed_images_file.name)
    image = Image.open(cardinal_image).convert("RGB")

    vision_format = detect_vision_format(embed_images_sess)
    pytorch_embeddings, inputs = get_pytorch_vision_embeddings(model, processor, image)

    atol, rtol = get_tolerances(vision_bits)
    if vision_format == VISION_MODE_CONV2D:
        results = verify_vision_conv2d(embed_images_sess, image, pytorch_embeddings, atol, rtol)
    else:
        results = verify_vision_tiled(embed_images_sess, inputs, pytorch_embeddings, atol, rtol)

    for r in results:
        assert r.passed, f"{r.name}: max_diff={r.max_diff:.6f}, {r.details}"
