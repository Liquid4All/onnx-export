#!/usr/bin/env python3
"""
Multi-turn coherence testing CLI for LFM2 ONNX quantized models.

Tests whether quantized models maintain coherent multi-turn conversations
compared to PyTorch ground truth.

Usage:
    # Test specific model and quantization
    uv run lfm2-coherence --models 1.2B --quant q4

    # Test multiple models and quantizations
    uv run lfm2-coherence --models 350M 1.2B --quant q4 q8

    # Verbose output with turn-by-turn details
    uv run lfm2-coherence --models 1.2B --quant q4 --verbose
"""

import argparse
import logging
import pathlib

from liquidonnx.lfm2.coherence import (
    DEFAULT_PROMPTS,
    MODELS,
    CoherenceResult,
    MultiTurnTester,
)

logger = logging.getLogger(__name__)


def get_onnx_path(exports_dir: pathlib.Path, size: str, quant: str) -> pathlib.Path:
    """Get ONNX model path for given size and quantization."""
    onnx_dir = exports_dir / f"LFM2-{size}-ONNX" / "onnx"

    if quant == "fp32":
        # Try model.onnx first, then decoder_fp32.onnx
        fp32_path = onnx_dir / "model.onnx"
        if fp32_path.exists():
            return fp32_path
        return onnx_dir / "decoder_fp32.onnx"

    return onnx_dir / f"model_{quant}.onnx"


def print_turn_results(result: CoherenceResult):
    """Print detailed turn-by-turn results."""
    print(f"\n{'=' * 80}")
    print(f"MULTI-TURN COHERENCE: LFM2-{result.model_size} {result.quant_type.upper()}")
    print(f"{'=' * 80}")

    for turn in result.turns:
        print(f"\n--- Turn {turn.turn} ---")
        print(f"Prompt: {turn.prompt}")
        pt_resp = turn.pytorch_response[:100]
        ox_resp = turn.onnx_response[:100]
        print(f"PyTorch: {pt_resp}{'...' if len(turn.pytorch_response) > 100 else ''}")
        print(f"ONNX:    {ox_resp}{'...' if len(turn.onnx_response) > 100 else ''}")
        print(f"Token Match: {turn.token_match_rate * 100:.1f}%")
        print(f"Semantic Sim: {turn.semantic_similarity:.4f}")
        print(f"Max Logit Diff: {turn.max_logit_diff:.4f}")

    print("\n--- Summary ---")
    print(f"Avg Token Match: {result.avg_token_match * 100:.1f}%")
    print(f"Avg Semantic Sim: {result.avg_semantic_sim:.4f}")
    print(f"Accumulated Error: {result.accumulated_error:.4f}")


def print_summary_table(results: list[CoherenceResult]):
    """Print summary table."""
    if not results:
        return

    print(f"\n{'=' * 90}")
    print("COHERENCE SUMMARY")
    print(f"{'=' * 90}")

    header = f"{'Model':<12} | {'Quant':<8} | {'Avg Token%':<12} | {'Avg Semantic':<12} | {'Accum Error':<12}"
    print(f"\n{header}")
    print("-" * 90)

    for r in sorted(results, key=lambda x: (x.model_size, x.quant_type)):
        print(
            f"LFM2-{r.model_size:<6} | {r.quant_type:<8} | "
            f"{r.avg_token_match * 100:<12.1f} | {r.avg_semantic_sim:<12.4f} | "
            f"{r.accumulated_error:<12.4f}"
        )

    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(
        description="Multi-turn coherence testing for LFM2 quantized models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS.keys()),
        default=["1.2B"],
        help="Model sizes to test",
    )
    parser.add_argument(
        "--quant",
        nargs="+",
        choices=["fp32", "q4", "q8"],
        default=["q4"],
        help="Quantization types to test",
    )
    parser.add_argument(
        "--exports-dir",
        type=pathlib.Path,
        default=pathlib.Path("exports"),
        help="Base directory for ONNX exports (default: ./exports)",
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=5,
        help="Number of conversation turns",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=20,
        help="Max tokens to generate per turn",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed turn-by-turn results",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    prompts = DEFAULT_PROMPTS[: args.turns]
    results = []

    for size in args.models:
        print(f"\n{'=' * 60}")
        print(f"TESTING LFM2-{size}")
        print(f"{'=' * 60}")

        pytorch_path = MODELS[size]
        tester = MultiTurnTester(pytorch_path, max_new_tokens=args.max_tokens)
        tester.load_pytorch()

        for quant in args.quant:
            onnx_path = get_onnx_path(args.exports_dir, size, quant)
            print(f"\n--- {quant.upper()} ---")

            if not onnx_path.exists():
                print(f"  SKIPPED: {onnx_path} not found")
                continue

            try:
                result = tester.test_coherence(size, str(onnx_path), quant, prompts)
                results.append(result)

                if args.verbose:
                    print_turn_results(result)
                else:
                    print(f"  Avg Token Match: {result.avg_token_match * 100:.1f}%")
                    print(f"  Avg Semantic Sim: {result.avg_semantic_sim:.4f}")
                    print(f"  Accumulated Error: {result.accumulated_error:.4f}")
            except Exception as e:
                logger.error(f"  ERROR: {e}")

    if results:
        print_summary_table(results)


if __name__ == "__main__":
    main()
