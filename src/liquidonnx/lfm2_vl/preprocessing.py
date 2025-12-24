"""
Common utilities for LFM2-VL ONNX inference.

This module provides shared functionality for vision-language model inference,
matching the PyTorch reference implementation in HuggingFace transformers.

Supports two vision input formats:
- Tiled (-T): Pre-extracted patches [B, num_patches, patch_dim]
- Conv2d (-C): Raw image [B, 3, H, W] with spatial dimensions
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from liquidonnx.lfm2_vl import VISION_MODE_TILED, VISION_MODE_CONV2D


@dataclass
class VLConfig:
    """Configuration for VL model inference.

    Matches PyTorch LFM2-VL configuration parameters.
    """
    # Vision encoder
    patch_size: int = 16
    num_channels: int = 3

    # Projector
    downsample_factor: int = 2  # n_merge

    # Token bounds
    min_image_tokens: int = 64
    max_image_tokens: int = 256

    # Tiling (for do_image_splitting=True)
    tile_size: int = 512
    min_tiles: int = 2
    max_tiles: int = 10
    use_thumbnail: bool = True

    # Normalization (SigLIP2 uses ImageNet mean/std)
    image_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    image_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    @classmethod
    def from_hf_config(cls, config) -> "VLConfig":
        """Create VLConfig from HuggingFace config."""
        return cls(
            patch_size=getattr(config, 'encoder_patch_size', 16),
            downsample_factor=getattr(config, 'downsample_factor', 2),
            min_image_tokens=getattr(config, 'min_image_tokens', 64),
            max_image_tokens=getattr(config, 'max_image_tokens', 256),
            tile_size=getattr(config, 'tile_size', 512),
            min_tiles=getattr(config, 'min_tiles', 2),
            max_tiles=getattr(config, 'max_tiles', 10),
            use_thumbnail=getattr(config, 'use_thumbnail', True),
        )


def detect_vision_format(session) -> str:
    """Detect vision input format from ONNX session.

    Args:
        session: ONNX InferenceSession for embed_images model

    Returns:
        VISION_MODE_CONV2D if model expects raw image input with spatial dims
        VISION_MODE_TILED if model expects pre-extracted patches
    """
    input_names = {inp.name for inp in session.get_inputs()}
    # Conv2d format has spatial_h and spatial_w inputs
    if "spatial_h" in input_names:
        return VISION_MODE_CONV2D
    return VISION_MODE_TILED


def round_by_factor(number: float, factor: int) -> int:
    """Round number to nearest multiple of factor.

    Matches PyTorch: round(number / factor) * factor
    """
    return round(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    config: Optional[VLConfig] = None,
    patch_size: int = 16,
    downsample_factor: int = 2,
    min_image_tokens: int = 64,
    max_image_tokens: int = 256,
) -> Tuple[int, int]:
    """Compute target size for image, matching PyTorch smart_resize exactly.

    Rescales the image so that:
    1. Both dimensions are divisible by patch_size * downsample_factor (32)
    2. Total tokens within [min_image_tokens, max_image_tokens]
    3. Aspect ratio is maintained as closely as possible

    Args:
        height: Original image height
        width: Original image width
        config: Optional VLConfig (overrides other params if provided)
        patch_size: Vision encoder patch size (default 16)
        downsample_factor: Projector downsample factor / n_merge (default 2)
        min_image_tokens: Minimum output tokens (default 64)
        max_image_tokens: Maximum output tokens (default 256)

    Returns:
        (new_width, new_height) - note: width first to match PyTorch
    """
    if config is not None:
        patch_size = config.patch_size
        downsample_factor = config.downsample_factor
        min_image_tokens = config.min_image_tokens
        max_image_tokens = config.max_image_tokens

    total_factor = patch_size * downsample_factor  # 32

    # Pixel bounds based on token limits
    # Each output token represents (patch_size * downsample_factor)^2 pixels
    smart_resize_min_pixels = min_image_tokens * (patch_size ** 2) * (downsample_factor ** 2)
    smart_resize_max_pixels = max_image_tokens * (patch_size ** 2) * (downsample_factor ** 2)

    # Round to nearest multiple of total_factor
    h_bar = max(total_factor, round_by_factor(height, total_factor))
    w_bar = max(total_factor, round_by_factor(width, total_factor))

    # Scale if outside bounds
    if h_bar * w_bar > smart_resize_max_pixels:
        # Scale down - use floor to stay within max
        beta = math.sqrt((height * width) / smart_resize_max_pixels)
        h_bar = max(total_factor, math.floor(height / beta / total_factor) * total_factor)
        w_bar = max(total_factor, math.floor(width / beta / total_factor) * total_factor)
    elif h_bar * w_bar < smart_resize_min_pixels:
        # Scale up - use ceil to reach min
        beta = math.sqrt(smart_resize_min_pixels / (height * width))
        h_bar = math.ceil(height * beta / total_factor) * total_factor
        w_bar = math.ceil(width * beta / total_factor) * total_factor

    # Return (width, height) to match PyTorch convention
    return w_bar, h_bar


def preprocess_conv2d(
    image: Image.Image,
    config: Optional[VLConfig] = None,
    patch_size: int = 16,
    downsample_factor: int = 2,
    min_image_tokens: int = 64,
    max_image_tokens: int = 256,
    image_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    image_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> Tuple[np.ndarray, int, int]:
    """Preprocess image for conv2d format ONNX model.

    Matches PyTorch preprocessing:
    1. smart_resize to target dimensions
    2. Normalize with ImageNet mean/std
    3. Return spatial dimensions AFTER n_merge (for projector output)

    Args:
        image: PIL Image to preprocess
        config: Optional VLConfig (overrides other params if provided)
        patch_size: Vision encoder patch size
        downsample_factor: Projector n_merge value
        min_image_tokens: Minimum output tokens
        max_image_tokens: Maximum output tokens
        image_mean: Normalization mean (ImageNet default)
        image_std: Normalization std (ImageNet default)

    Returns:
        (pixel_values, spatial_h, spatial_w) where:
        - pixel_values: [1, 3, H, W] normalized float32 array
        - spatial_h: height AFTER n_merge (projector output rows)
        - spatial_w: width AFTER n_merge (projector output cols)
    """
    if config is not None:
        patch_size = config.patch_size
        downsample_factor = config.downsample_factor
        min_image_tokens = config.min_image_tokens
        max_image_tokens = config.max_image_tokens
        image_mean = config.image_mean
        image_std = config.image_std

    w, h = image.size

    # Compute target size using smart_resize
    new_w, new_h = smart_resize(
        height=h,
        width=w,
        patch_size=patch_size,
        downsample_factor=downsample_factor,
        min_image_tokens=min_image_tokens,
        max_image_tokens=max_image_tokens,
    )

    # Resize with bilinear interpolation (matches PyTorch default)
    image_resized = image.resize((new_w, new_h), Image.BILINEAR)

    # Compute spatial dimensions AFTER n_merge (for projector output)
    spatial_h = new_h // patch_size // downsample_factor
    spatial_w = new_w // patch_size // downsample_factor

    # Convert to float and normalize with ImageNet mean/std
    pixels = np.array(image_resized).astype(np.float32) / 255.0

    # Normalize: (pixel - mean) / std
    mean = np.array(image_mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(image_std, dtype=np.float32).reshape(1, 1, 3)
    pixels = (pixels - mean) / std

    # Convert to [1, 3, H, W] format
    pixels = pixels.transpose(2, 0, 1)[np.newaxis, ...]

    return pixels.astype(np.float32), spatial_h, spatial_w


def preprocess_tiled(
    image: Image.Image,
    processor,
    do_image_splitting: bool = False,
    pad_to_square: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Preprocess image for tiled format ONNX model.

    Uses HuggingFace processor for patch extraction, matching PyTorch exactly.

    Args:
        image: PIL Image to preprocess
        processor: HuggingFace processor with image_processor
        do_image_splitting: Whether to split large images into tiles
        pad_to_square: If True, pad non-square images to square first
                      (recommended for ONNX models that assume square input)

    Returns:
        (pixel_values, patch_attention_mask, spatial_shapes) where:
        - pixel_values: [num_tiles, num_patches, patch_dim] float32 array
        - patch_attention_mask: [num_tiles, num_patches] int64 array
        - spatial_shapes: [num_tiles, 2] int64 array with (H, W) per tile
    """
    # Optionally pad to square for ONNX compatibility
    if pad_to_square:
        w, h = image.size
        if w != h:
            max_dim = max(w, h)
            square_img = Image.new('RGB', (max_dim, max_dim), (0, 0, 0))
            paste_x = (max_dim - w) // 2
            paste_y = (max_dim - h) // 2
            square_img.paste(image, (paste_x, paste_y))
            image = square_img

    # Use processor's image_processor for patch extraction
    inputs = processor.image_processor(
        images=image,
        return_tensors="pt",
        do_image_splitting=do_image_splitting,
    )

    pixel_values = inputs["pixel_values"].numpy().astype(np.float32)
    patch_attention_mask = inputs["pixel_attention_mask"].numpy().astype(np.int64)
    spatial_shapes = inputs["spatial_shapes"].numpy().astype(np.int64)

    return pixel_values, patch_attention_mask, spatial_shapes


def get_image_embeddings(
    session,
    images: Union[Image.Image, List[Image.Image]],
    vision_format: Optional[str] = None,
    processor=None,
    config: Optional[VLConfig] = None,
    do_image_splitting: bool = False,
    pad_to_square: bool = True,
) -> List[np.ndarray]:
    """Get image embeddings from ONNX vision encoder.

    Unified function that handles both tiled and conv2d formats.

    Args:
        session: ONNX InferenceSession for embed_images model
        images: Single image or list of images to encode
        vision_format: "tiled" or "conv2d" (auto-detected if None)
        processor: HuggingFace processor (required for tiled format)
        config: VLConfig for preprocessing parameters
        do_image_splitting: Whether to split large images into tiles
        pad_to_square: For tiled format, whether to pad to square

    Returns:
        List of embeddings, one per image. Each is [num_tokens, hidden_dim].
    """
    if vision_format is None:
        vision_format = detect_vision_format(session)

    if isinstance(images, Image.Image):
        images = [images]

    embeddings = []

    for image in images:
        if vision_format == VISION_MODE_CONV2D:
            # Conv2d format: preprocess and pass spatial dims
            pixel_values, spatial_h, spatial_w = preprocess_conv2d(image, config=config)

            outputs = session.run(
                None,
                {
                    "pixel_values": pixel_values,
                    "spatial_h": np.array(spatial_h, dtype=np.int64),
                    "spatial_w": np.array(spatial_w, dtype=np.int64),
                },
            )
            # Output: [1, num_tokens, hidden_dim]
            img_embeds = outputs[0][0]  # [num_tokens, hidden_dim]

        else:
            # Tiled format: use processor for patch extraction
            if processor is None:
                raise ValueError("processor is required for tiled format")

            pixel_values, patch_attention_mask, spatial_shapes = preprocess_tiled(
                image,
                processor,
                do_image_splitting=do_image_splitting,
                pad_to_square=pad_to_square,
            )

            outputs = session.run(
                None,
                {
                    "pixel_values": pixel_values,
                    "patch_attention_mask": patch_attention_mask,
                },
            )
            # Output: [num_tiles, num_tokens_per_tile, hidden_dim]
            # Flatten across tiles
            onnx_embeds = outputs[0]
            num_tiles, tokens_per_tile, hidden = onnx_embeds.shape
            img_embeds = onnx_embeds.reshape(-1, hidden)

        embeddings.append(img_embeds)

    return embeddings


def build_inputs_embeds(
    text_embeds: np.ndarray,
    image_embeds_list: List[np.ndarray],
    image_token_id: int,
    input_ids: np.ndarray,
) -> np.ndarray:
    """Build inputs_embeds by replacing image tokens with image embeddings.

    For use with expanded token sequences where each <image> token
    corresponds to exactly one patch embedding.

    Args:
        text_embeds: [seq_len, hidden] text embeddings
        image_embeds_list: List of [num_patches, hidden] per image
        image_token_id: Token ID for <image> placeholder
        input_ids: [1, seq_len] input token IDs

    Returns:
        [1, seq_len, hidden] combined embeddings
    """
    # Find all <image> token positions
    image_positions = np.where(input_ids[0] == image_token_id)[0]

    # Total patches across all images
    total_patches = sum(embeds.shape[0] for embeds in image_embeds_list)

    if len(image_positions) != total_patches:
        print(f"  [WARNING] Mismatch: {len(image_positions)} <image> tokens vs {total_patches} patches")

    # Replace each <image> token with corresponding patch embedding
    result = text_embeds.copy()
    patch_idx = 0
    for img_embeds in image_embeds_list:
        num_patches = img_embeds.shape[0]
        for local_idx in range(num_patches):
            if patch_idx < len(image_positions):
                pos = image_positions[patch_idx]
                result[pos] = img_embeds[local_idx]
            patch_idx += 1

    return result[np.newaxis, ...].astype(np.float32)
