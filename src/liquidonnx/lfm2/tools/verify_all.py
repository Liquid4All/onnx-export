#!/usr/bin/env python3
"""
Verify all LFM2 ONNX quantized models against PyTorch.

Compares Builder and Community quantized versions (Q4 and Q8) to identify which
produces outputs closer to the original PyTorch model.

Usage:
    lfm2-verify-all
    lfm2-verify-all --models 350M 1.2B
    lfm2-verify-all --quant q4       # Only Q4
    lfm2-verify-all --quant q8       # Only Q8
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

from liquidonnx.lfm2.verify import NumericalVerifier

logger = logging.getLogger(__name__)

# ============================================================================
# MODEL PATHS - Edit these if your paths differ
# ============================================================================

PYTORCH_MODELS = {
    "350M": "LiquidAI/LFM2-350M",
    "700M": "LiquidAI/LFM2-700M",
    "1.2B": "LiquidAI/LFM2-1.2B",
    "2.6B": "LiquidAI/LFM2-2.6B",
}

BUILDER_Q4_MODELS = {
    "350M": "LFM2-350M-ONNX-builder-Q4-fp32head/onnx/model.onnx",
    "700M": "LFM2-700M-ONNX-builder-Q4-fp32head/onnx/model.onnx",
    "1.2B": "LFM2-1.2B-ONNX-builder-Q4-fp32head/onnx/model.onnx",
    "2.6B": "LFM2-2.6B-ONNX-builder-Q4-fp32head/onnx/model.onnx",
}

BUILDER_Q8_MODELS = {
    "350M": "LFM2-350M-ONNX-builder-Q8-fp32head/onnx/model.onnx",
    "700M": "LFM2-700M-ONNX-builder-Q8-fp32head/onnx/model.onnx",
    "1.2B": "LFM2-1.2B-ONNX-builder-Q8-fp32head/onnx/model.onnx",
    "2.6B": "LFM2-2.6B-ONNX-builder-Q8-fp32head/onnx/model.onnx",
}

COMMUNITY_Q4_MODELS = {
    "350M": "/Users/ykhrustalev/workplace/models/onnx-community/LFM2-350M-ONNX/onnx/model_q4.onnx",
    "700M": "/Users/ykhrustalev/workplace/models/onnx-community/LFM2-700M-ONNX/onnx/model_q4.onnx",
    "1.2B": "/Users/ykhrustalev/workplace/models/onnx-community/LFM2-1.2B-ONNX/onnx/model_q4.onnx",
    "2.6B": "/Users/ykhrustalev/workplace/models/onnx-community/LFM2-2.6B-ONNX/onnx/model_q4.onnx",
}

COMMUNITY_Q8_MODELS = {
    "350M": "/Users/ykhrustalev/workplace/models/onnx-community/LFM2-350M-ONNX/onnx/model_q8.onnx",
    "700M": "/Users/ykhrustalev/workplace/models/onnx-community/LFM2-700M-ONNX/onnx/model_q8.onnx",
    "1.2B": "/Users/ykhrustalev/workplace/models/onnx-community/LFM2-1.2B-ONNX/onnx/model_q8.onnx",
    "2.6B": "/Users/ykhrustalev/workplace/models/onnx-community/LFM2-2.6B-ONNX/onnx/model_q8.onnx",
}


# ============================================================================


def get_model_size_mb(onnx_path: str) -> float:
    """Get total model size in MB (including external data)."""
    path = Path(onnx_path)
    if not path.exists():
        return 0.0

    total = path.stat().st_size

    # Check for external data files (different naming conventions)
    data_path = Path(str(path) + ".data")
    if data_path.exists():
        total += data_path.stat().st_size
    else:
        data_path = path.parent / (path.stem + ".onnx_data")
        if data_path.exists():
            total += data_path.stat().st_size

    return total / (1024 * 1024)


@dataclass
class ComparisonResult:
    """Result of comparing a quantized model against PyTorch."""
    model_size: str
    source: str  # "builder_q4", "builder_q8", "community_q4", "community_q8"
    quant_type: str  # "q4" or "q8"
    max_diff: float
    mean_diff: float
    top1_match: bool
    top5_overlap: int
    pytorch_top5: List[int]
    onnx_top5: List[int]
    file_size_mb: float = 0.0


class ModelComparator:
    """Compares quantized models against PyTorch ground truth."""

    def __init__(self):
        self.verifier = None
        self.current_model_path = None

    def load_pytorch_model(self, model_path: str):
        """Load PyTorch model for reference."""
        if self.current_model_path == model_path:
            return

        self.verifier = NumericalVerifier(model_path)
        self.verifier.load_pytorch_model()
        self.current_model_path = model_path

    def compare(
        self,
        model_size: str,
        pytorch_path: str,
        onnx_path: str,
        source: str,
        quant_type: str,
        prompt: str = "Hello, how are",
    ) -> ComparisonResult:
        """Compare a quantized model against PyTorch."""
        self.load_pytorch_model(pytorch_path)
        onnx_sess = self.verifier.load_onnx_model(onnx_path)

        inputs = self.verifier.prepare_inputs(prompt)
        pytorch_logits = self.verifier.run_pytorch(inputs)
        onnx_logits = self.verifier.run_onnx(onnx_sess, inputs)

        diff = np.abs(pytorch_logits - onnx_logits)
        max_diff = float(diff.max())
        mean_diff = float(diff.mean())

        pytorch_last = pytorch_logits[0, -1]
        onnx_last = onnx_logits[0, -1]

        pytorch_top5 = np.argsort(pytorch_last)[-5:][::-1].tolist()
        onnx_top5 = np.argsort(onnx_last)[-5:][::-1].tolist()

        top1_match = pytorch_top5[0] == onnx_top5[0]
        top5_overlap = len(set(pytorch_top5) & set(onnx_top5))

        file_size_mb = get_model_size_mb(onnx_path)

        return ComparisonResult(
            model_size=model_size,
            source=source,
            quant_type=quant_type,
            max_diff=max_diff,
            mean_diff=mean_diff,
            top1_match=top1_match,
            top5_overlap=top5_overlap,
            pytorch_top5=pytorch_top5,
            onnx_top5=onnx_top5,
            file_size_mb=file_size_mb,
        )


def print_results(results: List[ComparisonResult], quant_type: str):
    """Print comparison results as a table."""
    quant_label = quant_type.upper()

    print("\n" + "=" * 115)
    print(f"PYTORCH vs {quant_label} QUANTIZED MODELS COMPARISON")
    print("=" * 115)

    # Group by model size
    by_size = {}
    for r in results:
        if r.quant_type != quant_type:
            continue
        if r.model_size not in by_size:
            by_size[r.model_size] = {}
        by_size[r.model_size][r.source] = r

    # Print header
    print(f"\n{'Model':<12} | {'Builder ' + quant_label:<40} | {'Community ' + quant_label:<40} | Winner")
    print(f"{'':<12} | {'Size MB':<10} {'MaxDiff':<10} {'MeanDiff':<10} {'Top1':<8} | {'Size MB':<10} {'MaxDiff':<10} {'MeanDiff':<10} {'Top1':<8} |")
    print("-" * 115)

    for size in ["350M", "700M", "1.2B", "2.6B"]:
        if size not in by_size:
            continue

        builder = by_size[size].get("builder")
        community = by_size[size].get("community")

        if builder and community:
            if builder.max_diff < community.max_diff:
                winner = "BUILDER"
            elif community.max_diff < builder.max_diff:
                winner = "COMMUNITY"
            else:
                winner = "TIE"

            print(
                f"LFM2-{size:<6} | "
                f"{builder.file_size_mb:<10.1f} {builder.max_diff:<10.4f} {builder.mean_diff:<10.4f} {'Yes' if builder.top1_match else 'No':<8} | "
                f"{community.file_size_mb:<10.1f} {community.max_diff:<10.4f} {community.mean_diff:<10.4f} {'Yes' if community.top1_match else 'No':<8} | "
                f"{winner}"
            )
        elif builder:
            print(
                f"LFM2-{size:<6} | "
                f"{builder.file_size_mb:<10.1f} {builder.max_diff:<10.4f} {builder.mean_diff:<10.4f} {'Yes' if builder.top1_match else 'No':<8} | "
                f"{'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<8} | "
                f"BUILDER"
            )

    print("=" * 115)

    # Summary
    builder_wins = sum(
        1 for size, data in by_size.items()
        if "builder" in data and "community" in data
        and data["builder"].max_diff < data["community"].max_diff
    )
    community_wins = sum(
        1 for size, data in by_size.items()
        if "builder" in data and "community" in data
        and data["community"].max_diff < data["builder"].max_diff
    )
    print(f"\nSUMMARY ({quant_label}):")
    print(f"  Builder wins: {builder_wins}")
    print(f"  Community wins: {community_wins}")


def main():
    parser = argparse.ArgumentParser(description="Compare quantized models against PyTorch")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["350M", "700M", "1.2B", "2.6B"],
        default=["350M", "700M", "1.2B", "2.6B"],
        help="Model sizes to compare",
    )
    parser.add_argument(
        "--quant",
        nargs="+",
        choices=["q4", "q8"],
        default=["q4", "q8"],
        help="Quantization types to compare",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Hello, how are",
        help="Prompt for comparison",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    comparator = ModelComparator()
    results = []

    for size in args.models:
        print(f"\n{'='*60}")
        print(f"COMPARING LFM2-{size}")
        print(f"{'='*60}")

        pytorch_path = PYTORCH_MODELS[size]

        # Q4 comparison
        if "q4" in args.quant:
            builder_q4_path = BUILDER_Q4_MODELS[size]
            community_q4_path = COMMUNITY_Q4_MODELS[size]

            print(f"\n--- Builder Q4 vs PyTorch ---")
            try:
                result = comparator.compare(size, pytorch_path, builder_q4_path, "builder", "q4", args.prompt)
                results.append(result)
                print(f"  Size: {result.file_size_mb:.1f} MB")
                print(f"  Max diff: {result.max_diff:.4f}")
                print(f"  Mean diff: {result.mean_diff:.4f}")
                print(f"  Top-1 match: {result.top1_match}")
            except Exception as e:
                print(f"  ERROR: {e}")

            print(f"\n--- Community Q4 vs PyTorch ---")
            try:
                result = comparator.compare(size, pytorch_path, community_q4_path, "community", "q4", args.prompt)
                results.append(result)
                print(f"  Size: {result.file_size_mb:.1f} MB")
                print(f"  Max diff: {result.max_diff:.4f}")
                print(f"  Mean diff: {result.mean_diff:.4f}")
                print(f"  Top-1 match: {result.top1_match}")
            except Exception as e:
                print(f"  ERROR: {e}")

        # Q8 comparison
        if "q8" in args.quant:
            builder_q8_path = BUILDER_Q8_MODELS[size]
            community_q8_path = COMMUNITY_Q8_MODELS.get(size)

            print(f"\n--- Builder Q8 vs PyTorch ---")
            try:
                result = comparator.compare(size, pytorch_path, builder_q8_path, "builder", "q8", args.prompt)
                results.append(result)
                print(f"  Size: {result.file_size_mb:.1f} MB")
                print(f"  Max diff: {result.max_diff:.4f}")
                print(f"  Mean diff: {result.mean_diff:.4f}")
                print(f"  Top-1 match: {result.top1_match}")
            except Exception as e:
                print(f"  ERROR: {e}")

            if community_q8_path and Path(community_q8_path).exists():
                print(f"\n--- Community Q8 vs PyTorch ---")
                try:
                    result = comparator.compare(size, pytorch_path, community_q8_path, "community", "q8", args.prompt)
                    results.append(result)
                    print(f"  Size: {result.file_size_mb:.1f} MB")
                    print(f"  Max diff: {result.max_diff:.4f}")
                    print(f"  Mean diff: {result.mean_diff:.4f}")
                    print(f"  Top-1 match: {result.top1_match}")
                except Exception as e:
                    print(f"  ERROR: {e}")

    # Print final comparison tables
    if results:
        if "q4" in args.quant:
            print_results(results, "q4")
        if "q8" in args.quant:
            print_results(results, "q8")


if __name__ == "__main__":
    main()
