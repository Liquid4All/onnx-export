#!/usr/bin/env python3
"""
Verify LFM2-VL ONNX exports against PyTorch reference.

Usage:
    # Verify all models (both formats by default)
    lfm2-vl-verify --sizes all

    # Verify specific sizes
    lfm2-vl-verify --sizes 450M 1.6B

    # Verify only tiled format
    lfm2-vl-verify --sizes 450M --tiled

    # Verify only conv2d format
    lfm2-vl-verify --sizes 450M --conv2d

    # Custom tolerances
    lfm2-vl-verify --sizes 450M --atol 0.1 --rtol 0.1

    # Use specific image for vision tests
    lfm2-vl-verify --sizes 450M --image cardinal.jpg
"""

import argparse
import logging
import pathlib

from liquidonnx.lfm2_vl import MODELS, FORMATS, TEST_IMAGES
from liquidonnx.lfm2_vl.verify import load_pytorch_model, verify_onnx, print_results

logger = logging.getLogger(__name__)


def get_output_dir(size: str, fmt: str, output_base: pathlib.Path) -> pathlib.Path:
    """Get output directory for a model."""
    return output_base / "exports" / f"LFM2-VL-{size}-ONNX-{fmt}"


def main():
    parser = argparse.ArgumentParser(
        description="Verify LFM2-VL ONNX exports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--sizes",
        nargs="+",
        required=True,
        help="Model sizes: 450M, 1.6B, 3B, or 'all'",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("."),
        help="Directory containing exported models (default: current directory)",
    )
    parser.add_argument(
        "--tiled",
        action="store_true",
        help="Verify tiled format [B, N, 768] (HuggingFace style)",
    )
    parser.add_argument(
        "--conv2d",
        action="store_true",
        help="Verify conv2d format [B, 3, H, W] (llama.cpp style)",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=str(TEST_IMAGES["cardinal"]),
        help="Test image path for vision verification (default: cardinal.jpg)",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-3,
        help="Absolute tolerance (default: 1e-3)",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-2,
        help="Relative tolerance (default: 1e-2)",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    formats = [f for f in FORMATS if getattr(args, f)] or FORMATS
    sizes = list(MODELS.keys()) if "all" in args.sizes else args.sizes

    for s in sizes:
        if s not in MODELS:
            parser.error(f"Unknown size: {s}. Available: {', '.join(MODELS.keys())}")

    # Cache PyTorch models (expensive to load)
    models: dict[str, tuple] = {}
    passed = 0
    total = 0

    for fmt in formats:
        for size in sizes:
            total += 1
            onnx_dir = get_output_dir(size, fmt, args.output_dir)

            if not onnx_dir.exists():
                logger.warning(f"Skipping {size} ({fmt}): {onnx_dir} not found")
                continue

            logger.info(f"Verifying {size} ({fmt})...")

            if size not in models:
                models[size] = load_pytorch_model(MODELS[size])

            model, processor = models[size]
            results = verify_onnx(model, processor, str(onnx_dir), args.image, args.atol, args.rtol)

            if print_results(results):
                passed += 1

    logger.info(f"Total: {passed}/{total} passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    exit(main())
