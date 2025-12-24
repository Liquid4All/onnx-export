"""
LFM2-VL vision-language model ONNX export and inference.
"""

import pathlib

# Predefined models
MODELS = {
    "450M": "LiquidAI/LFM2-VL-450M",
    "1.6B": "LiquidAI/LFM2-VL-1.6B",
    "3B": "LiquidAI/LFM2-VL-3B",
}

# Vision input formats
FORMATS = ["tiled", "conv2d"]

# Assets directory
ASSETS_DIR = pathlib.Path(__file__).parent / "assets"

# Test images
TEST_IMAGES = {
    "cardinal": ASSETS_DIR / "cardinal.jpg",
    "bluejay": ASSETS_DIR / "bluejay.jpg",
}

# Test prompts for single image
SINGLE_IMAGE_PROMPTS = [
    "What do you see in this image? Describe the main elements.",
    "What colors are present in the image?",
    "Can you identify any shapes or patterns?",
    "Based on what you described, what type of image is this?",
    "If I wanted to recreate this image, what would I need?",
]

# Test prompts for multiple images
MULTI_IMAGE_PROMPTS = [
    "Which one most important thing do you see on each image? Be concise and exact.",
    "What are the similarities between these images?",
    "What are the differences between these images?",
    "Which image do you prefer and why?",
]
