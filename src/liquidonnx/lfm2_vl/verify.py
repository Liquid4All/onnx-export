"""
Numerical verification for LFM2-VL ONNX exports.

Verifies:
1. Token embeddings (embed_tokens.onnx)
2. Vision encoder + projector (embed_images.onnx)
3. Decoder/backbone (decoder.onnx)
"""

import logging
import os
from dataclasses import dataclass

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from liquidonnx.lfm2_vl.preprocessing import detect_vision_format, preprocess_conv2d

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    name: str
    passed: bool
    max_diff: float
    mean_diff: float
    correlation: float
    details: str = ""


def load_pytorch_model(model_path: str) -> tuple:
    """Load PyTorch VL model for reference.

    Returns (model, processor) tuple.
    """
    logger.info(f"Loading PyTorch model from {model_path}...")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    model.eval()
    return model, processor


def load_onnx_sessions(onnx_dir: str) -> tuple:
    """Load ONNX sessions from directory.

    Returns (embed_tokens_sess, embed_images_sess, decoder_sess) tuple.
    """
    onnx_path = os.path.join(onnx_dir, "onnx")

    def load_session(filename: str):
        path = os.path.join(onnx_path, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{filename} not found in {onnx_path}")
        logger.info(f"Loading {filename}...")
        return ort.InferenceSession(path, providers=["CPUExecutionProvider"])

    return (
        load_session("embed_tokens.onnx"),
        load_session("embed_images.onnx"),
        load_session("decoder.onnx"),
    )


def load_image(image_path: str):
    logger.info(f"Loading image from {image_path}")
    return Image.open(image_path).convert("RGB")


def compare_arrays(name: str, expected: np.ndarray, actual: np.ndarray,
                   atol: float, rtol: float) -> VerificationResult:
    if expected.shape != actual.shape:
        return VerificationResult(
            name=name, passed=False,
            max_diff=float('inf'), mean_diff=float('inf'), correlation=0.0,
            details=f"Shape mismatch: {expected.shape} vs {actual.shape}"
        )

    diff = np.abs(expected - actual)
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())
    correlation = float(np.corrcoef(expected.flatten(), actual.flatten())[0, 1])
    passed = np.allclose(expected, actual, atol=atol, rtol=rtol)

    return VerificationResult(
        name=name, passed=passed,
        max_diff=max_diff, mean_diff=mean_diff, correlation=correlation,
    )


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


def _verify_embed_prompt(model, processor, embed_tokens_sess, prompt, atol, rtol):
    input_ids = processor.tokenizer.encode(prompt, return_tensors="pt")

    with torch.no_grad():
        pytorch_embeds = model.model.language_model.embed_tokens(input_ids).numpy()

    onnx_embeds = embed_tokens_sess.run(None, {
        "input_ids": input_ids.numpy().astype(np.int64),
    })[0]

    return compare_arrays(
        f"embed_tokens: '{prompt[:20]}...'",
        pytorch_embeds, onnx_embeds, atol, rtol
    )


def verify_embed_tokens(model, processor, embed_tokens_sess,
                        atol: float, rtol: float) -> list[VerificationResult]:
    """Verify embed_tokens (token embedding lookup)."""
    prompts = ["Hello, how are you?", "The quick brown fox", "Describe this image:"]
    logger.info("Verifying embed_tokens...")
    return [_verify_embed_prompt(model, processor, embed_tokens_sess, p, atol, rtol)
            for p in prompts]


def _get_pytorch_vision_embeddings(model, processor, image):
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


def _verify_vision_conv2d(embed_images_sess, image, pytorch_embeddings, atol, rtol):
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


def _verify_vision_tiled(embed_images_sess, inputs, pytorch_embeddings, atol, rtol):
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


def verify_vision_encoder(model, processor, embed_images_sess, image,
                          atol: float, rtol: float) -> list[VerificationResult]:
    """Verify vision encoder + projector outputs."""
    logger.info("Verifying vision encoder...")

    vision_format = detect_vision_format(embed_images_sess)
    logger.info(f"Detected vision format: {vision_format}")

    pytorch_embeddings, inputs = _get_pytorch_vision_embeddings(model, processor, image)

    if vision_format == "conv2d":
        return _verify_vision_conv2d(embed_images_sess, image, pytorch_embeddings, atol, rtol)
    return _verify_vision_tiled(embed_images_sess, inputs, pytorch_embeddings, atol, rtol)


def _verify_decoder_prompt(model, processor, embed_tokens_sess, decoder_sess,
                           prompt, atol, rtol):
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

    return (
        compare_arrays(f"decoder: '{prompt[:20]}...'",
                       pytorch_logits, onnx_logits, atol, rtol),
        compare_top_k(f"top-5: '{prompt[:20]}...'",
                      pytorch_logits, onnx_logits),
    )


def verify_decoder(model, processor, embed_tokens_sess, decoder_sess,
                   atol: float, rtol: float) -> list[VerificationResult]:
    prompts = ["Hello, how are", "The image shows", "I can see"]
    logger.info("Verifying decoder...")
    return [r for p in prompts
            for r in _verify_decoder_prompt(model, processor, embed_tokens_sess,
                                            decoder_sess, p, atol, rtol)]


def verify_onnx(model, processor, onnx_dir: str, image_path: str | None = None,
                atol: float = 1e-3, rtol: float = 1e-2) -> list[VerificationResult]:
    """Verify ONNX export against PyTorch reference.

    Returns list of VerificationResult.
    """
    embed_tokens_sess, embed_images_sess, decoder_sess = load_onnx_sessions(onnx_dir)
    image = load_image(image_path)

    return [
        *verify_embed_tokens(model, processor, embed_tokens_sess, atol, rtol),
        *verify_vision_encoder(model, processor, embed_images_sess, image, atol, rtol),
        *verify_decoder(model, processor, embed_tokens_sess, decoder_sess, atol, rtol),
    ]


def print_results(results: list[VerificationResult]) -> bool:
    """Print verification results. Returns True if all passed."""
    passed = sum(1 for r in results if r.passed)
    total = len(results)

    for result in results:
        status = "PASS" if result.passed else "FAIL"
        logger.info(f"{status} {result.name}")
        logger.info(f"  Max diff: {result.max_diff:.6f}, Mean diff: {result.mean_diff:.6f}, Correlation: {result.correlation:.6f}")
        if result.details:
            logger.info(f"  Details: {result.details}")

    logger.info(f"Summary: {passed}/{total} checks passed")
    return passed == total
