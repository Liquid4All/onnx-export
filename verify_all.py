#!/usr/bin/env python3
"""
Verify all LFM2 ONNX Q4 models against PyTorch.

Compares Builder Q4 and Community Q4 quantized versions to identify which
produces outputs closer to the original PyTorch model.

Usage:
    python verify_all.py
    python verify_all.py --models 350M 1.2B
"""

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

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

COMMUNITY_Q4_MODELS = {
    "350M": "/Users/ykhrustalev/workplace/models/onnx-community/LFM2-350M-ONNX/onnx/model_q4.onnx",
    "700M": "/Users/ykhrustalev/workplace/models/onnx-community/LFM2-700M-ONNX/onnx/model_q4.onnx",
    "1.2B": "/Users/ykhrustalev/workplace/models/onnx-community/LFM2-1.2B-ONNX/onnx/model_q4.onnx",
    "2.6B": "/Users/ykhrustalev/workplace/models/onnx-community/LFM2-2.6B-ONNX/onnx/model_q4.onnx",
}

# ============================================================================


def get_model_size_mb(onnx_path: str) -> float:
    """Get total model size in MB (including external data)."""
    path = Path(onnx_path)
    total = path.stat().st_size if path.exists() else 0

    # Check for external data files (different naming conventions)
    # model.onnx -> model.onnx.data (onnxruntime style)
    data_path = Path(str(path) + ".data")
    if data_path.exists():
        total += data_path.stat().st_size
    else:
        # model.onnx -> model.onnx_data (community style)
        # model_q4.onnx -> model_q4.onnx_data
        data_path = path.parent / (path.stem + ".onnx_data")
        if data_path.exists():
            total += data_path.stat().st_size

    return total / (1024 * 1024)  # Convert to MB


@dataclass
class ComparisonResult:
    """Result of comparing a Q4 model against PyTorch."""
    model_size: str
    source: str  # "builder" or "community"
    max_diff: float
    mean_diff: float
    top1_match: bool
    top5_overlap: int
    pytorch_top5: List[int]
    onnx_top5: List[int]
    file_size_mb: float = 0.0


class Q4Comparator:
    """Compares Q4 quantized models against PyTorch ground truth."""

    def __init__(self):
        self.tokenizer = None
        self.torch_model = None
        self.current_model_path = None

    def load_pytorch_model(self, model_path: str):
        """Load PyTorch model for reference."""
        if self.current_model_path == model_path:
            return  # Already loaded

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading PyTorch model: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.torch_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        self.torch_model.eval()
        self.current_model_path = model_path

    def load_onnx_model(self, onnx_path: str):
        """Load ONNX model."""
        import onnxruntime as ort

        logger.info(f"Loading ONNX model: {onnx_path}")
        return ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    def prepare_inputs(self, prompt: str) -> Dict[str, np.ndarray]:
        """Prepare input tensors."""
        input_ids = self.tokenizer.encode(prompt, return_tensors="np")
        seq_len = input_ids.shape[1]

        return {
            "input_ids": input_ids.astype(np.int64),
            "attention_mask": np.ones((1, seq_len), dtype=np.int64),
            "position_ids": np.arange(seq_len, dtype=np.int64).reshape(1, -1),
        }

    def run_pytorch(self, inputs: Dict[str, np.ndarray]) -> np.ndarray:
        """Run PyTorch model and return logits."""
        import torch

        with torch.no_grad():
            input_ids = torch.from_numpy(inputs["input_ids"])
            attention_mask = torch.from_numpy(inputs["attention_mask"])
            position_ids = torch.from_numpy(inputs["position_ids"])

            outputs = self.torch_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            return outputs.logits.numpy()

    def run_onnx(self, sess, inputs: Dict[str, np.ndarray]) -> np.ndarray:
        """Run ONNX model and return logits."""
        feed = {}
        for inp in sess.get_inputs():
            if inp.name in inputs:
                feed[inp.name] = inputs[inp.name]
            else:
                # Initialize cache to zeros
                shape = [d if isinstance(d, int) else 1 for d in inp.shape]
                feed[inp.name] = np.zeros(shape, dtype=np.float32)

        outputs = sess.run(None, feed)
        return outputs[0]

    def compare(
        self,
        model_size: str,
        pytorch_path: str,
        onnx_path: str,
        source: str,
        prompt: str = "Hello, how are",
    ) -> ComparisonResult:
        """Compare a single Q4 model against PyTorch."""
        self.load_pytorch_model(pytorch_path)
        onnx_sess = self.load_onnx_model(onnx_path)

        inputs = self.prepare_inputs(prompt)

        # Run both models
        pytorch_logits = self.run_pytorch(inputs)
        onnx_logits = self.run_onnx(onnx_sess, inputs)

        # Compute differences
        diff = np.abs(pytorch_logits - onnx_logits)
        max_diff = float(diff.max())
        mean_diff = float(diff.mean())

        # Top-k analysis (last token)
        pytorch_last = pytorch_logits[0, -1]
        onnx_last = onnx_logits[0, -1]

        pytorch_top5 = np.argsort(pytorch_last)[-5:][::-1].tolist()
        onnx_top5 = np.argsort(onnx_last)[-5:][::-1].tolist()

        top1_match = pytorch_top5[0] == onnx_top5[0]
        top5_overlap = len(set(pytorch_top5) & set(onnx_top5))

        # Get file size
        file_size_mb = get_model_size_mb(onnx_path)

        return ComparisonResult(
            model_size=model_size,
            source=source,
            max_diff=max_diff,
            mean_diff=mean_diff,
            top1_match=top1_match,
            top5_overlap=top5_overlap,
            pytorch_top5=pytorch_top5,
            onnx_top5=onnx_top5,
            file_size_mb=file_size_mb,
        )


def print_results(results: List[ComparisonResult]):
    """Print comparison results as a table."""
    print("\n" + "=" * 115)
    print("PYTORCH vs Q4 QUANTIZED MODELS COMPARISON")
    print("=" * 115)

    # Group by model size
    by_size = {}
    for r in results:
        if r.model_size not in by_size:
            by_size[r.model_size] = {}
        by_size[r.model_size][r.source] = r

    # Print header
    print(f"\n{'Model':<12} | {'Builder Q4':<40} | {'Community Q4':<40} | Winner")
    print(f"{'':<12} | {'Size MB':<10} {'MaxDiff':<10} {'MeanDiff':<10} {'Top1':<8} | {'Size MB':<10} {'MaxDiff':<10} {'MeanDiff':<10} {'Top1':<8} |")
    print("-" * 115)

    # Print each model
    for size in ["350M", "700M", "1.2B", "2.6B"]:
        if size not in by_size:
            continue

        builder = by_size[size].get("builder")
        community = by_size[size].get("community")

        if builder and community:
            # Determine winner (lower max_diff is better)
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

    print("=" * 115)

    # Summary
    print("\nSUMMARY:")
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
    print(f"  Builder Q4 wins: {builder_wins}")
    print(f"  Community Q4 wins: {community_wins}")


def main():
    parser = argparse.ArgumentParser(description="Compare Q4 models against PyTorch")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["350M", "700M", "1.2B", "2.6B"],
        default=["350M", "700M", "1.2B", "2.6B"],
        help="Model sizes to compare",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Hello, how are",
        help="Prompt for comparison",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    comparator = Q4Comparator()
    results = []

    for size in args.models:
        print(f"\n{'='*60}")
        print(f"COMPARING LFM2-{size}")
        print(f"{'='*60}")

        pytorch_path = PYTORCH_MODELS[size]
        builder_path = BUILDER_Q4_MODELS[size]
        community_path = COMMUNITY_Q4_MODELS[size]

        # Compare Builder Q4
        print(f"\n--- Builder Q4 vs PyTorch ---")
        try:
            result = comparator.compare(size, pytorch_path, builder_path, "builder", args.prompt)
            results.append(result)
            print(f"  Size: {result.file_size_mb:.1f} MB")
            print(f"  Max diff: {result.max_diff:.4f}")
            print(f"  Mean diff: {result.mean_diff:.4f}")
            print(f"  Top-1 match: {result.top1_match}")
        except Exception as e:
            print(f"  ERROR: {e}")

        # Compare Community Q4
        print(f"\n--- Community Q4 vs PyTorch ---")
        try:
            result = comparator.compare(size, pytorch_path, community_path, "community", args.prompt)
            results.append(result)
            print(f"  Size: {result.file_size_mb:.1f} MB")
            print(f"  Max diff: {result.max_diff:.4f}")
            print(f"  Mean diff: {result.mean_diff:.4f}")
            print(f"  Top-1 match: {result.top1_match}")
        except Exception as e:
            print(f"  ERROR: {e}")

    # Print final comparison table
    if results:
        print_results(results)


if __name__ == "__main__":
    main()
