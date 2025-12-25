#!/usr/bin/env python3
"""
Numerical verification for LFM2 ONNX exports.

Compares outputs between:
1. PyTorch model (ground truth)
2. Exported ONNX model
3. Community ONNX model (if available)

Usage:
    python verify.py --model LFM2-1.2B --onnx LFM2-1.2B-ONNX-builder
    python verify.py --model LFM2-1.2B --onnx LFM2-1.2B-ONNX-builder --community LFM2-1.2B-ONNX-community
"""

import argparse
import logging
import os
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Result of a single verification check."""

    name: str
    passed: bool
    max_diff: float
    mean_diff: float
    correlation: float
    details: str = ""


class NumericalVerifier:
    """Verifies numerical correctness of ONNX exports."""

    def __init__(self, model_path: str, atol: float = 1e-4, rtol: float = 1e-3):
        self.model_path = model_path
        self.atol = atol
        self.rtol = rtol
        self.results: list[VerificationResult] = []

    def load_pytorch_model(self):
        """Load PyTorch model for reference."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading PyTorch model from {self.model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.torch_model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        self.torch_model.eval()

    def load_onnx_model(self, onnx_path: str):
        """Load ONNX model for testing.

        Accepts:
        - Direct path to .onnx file (e.g., model_q4.onnx)
        - Directory with onnx/model.onnx
        - Directory with model.onnx
        """
        import onnxruntime as ort

        if onnx_path.endswith(".onnx"):
            model_file = onnx_path
        else:
            model_file = os.path.join(onnx_path, "onnx", "model.onnx")
            if not os.path.exists(model_file):
                model_file = os.path.join(onnx_path, "model.onnx")

        if not os.path.exists(model_file):
            raise FileNotFoundError(f"ONNX model not found: {model_file}")

        logger.info(f"Loading ONNX model from {model_file}...")
        return ort.InferenceSession(model_file, providers=["CPUExecutionProvider"])

    def prepare_inputs(self, prompt: str = "Hello, how are") -> dict[str, np.ndarray]:
        """Prepare input tensors."""
        input_ids = self.tokenizer.encode(prompt, return_tensors="np")
        seq_len = input_ids.shape[1]

        return {
            "input_ids": input_ids.astype(np.int64),
            "attention_mask": np.ones((1, seq_len), dtype=np.int64),
            "position_ids": np.arange(seq_len, dtype=np.int64).reshape(1, -1),
        }

    def run_pytorch(self, inputs: dict[str, np.ndarray]) -> np.ndarray:
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

    def run_onnx(self, sess, inputs: dict[str, np.ndarray]) -> np.ndarray:
        """Run ONNX model and return logits."""
        # Get all input names
        [inp.name for inp in sess.get_inputs()]

        # Build feed dict with cache inputs initialized to zeros
        feed = {}
        for inp in sess.get_inputs():
            if inp.name in inputs:
                feed[inp.name] = inputs[inp.name]
            else:
                # Initialize cache to zeros
                shape = [d if isinstance(d, int) else 1 for d in inp.shape]
                feed[inp.name] = np.zeros(shape, dtype=np.float32)

        outputs = sess.run(None, feed)
        return outputs[0]  # logits

    def compare_arrays(
        self, name: str, expected: np.ndarray, actual: np.ndarray
    ) -> VerificationResult:
        """Compare two arrays and return verification result."""
        if expected.shape != actual.shape:
            return VerificationResult(
                name=name,
                passed=False,
                max_diff=float("inf"),
                mean_diff=float("inf"),
                correlation=0.0,
                details=f"Shape mismatch: {expected.shape} vs {actual.shape}",
            )

        diff = np.abs(expected - actual)
        max_diff = float(diff.max())
        mean_diff = float(diff.mean())

        # Correlation
        flat_exp = expected.flatten()
        flat_act = actual.flatten()
        correlation = float(np.corrcoef(flat_exp, flat_act)[0, 1])

        # Check if within tolerance
        passed = np.allclose(expected, actual, atol=self.atol, rtol=self.rtol)

        return VerificationResult(
            name=name,
            passed=passed,
            max_diff=max_diff,
            mean_diff=mean_diff,
            correlation=correlation,
        )

    def compare_top_k(
        self, name: str, expected: np.ndarray, actual: np.ndarray, k: int = 5
    ) -> VerificationResult:
        """Compare top-k predictions."""
        # Get last token logits
        exp_logits = expected[0, -1]
        act_logits = actual[0, -1]

        exp_top_k = np.argsort(exp_logits)[-k:][::-1]
        act_top_k = np.argsort(act_logits)[-k:][::-1]

        top1_match = exp_top_k[0] == act_top_k[0]
        top_k_overlap = len(set(exp_top_k) & set(act_top_k))

        return VerificationResult(
            name=name,
            passed=top1_match,
            max_diff=0.0 if top1_match else 1.0,
            mean_diff=1.0 - (top_k_overlap / k),
            correlation=top_k_overlap / k,
            details=f"Top-1 match: {top1_match}, Top-{k} overlap: {top_k_overlap}/{k}, "
            f"Expected: {exp_top_k.tolist()}, Actual: {act_top_k.tolist()}",
        )

    def verify_against_pytorch(
        self, onnx_path: str, prompts: list[str] = None
    ) -> list[VerificationResult]:
        """Verify ONNX model against PyTorch."""
        if prompts is None:
            prompts = [
                "Hello, how are",
                "The sky is",
                "1 + 1 =",
            ]

        self.load_pytorch_model()
        onnx_sess = self.load_onnx_model(onnx_path)

        results = []
        for prompt in prompts:
            logger.info(f"Testing prompt: '{prompt}'")
            inputs = self.prepare_inputs(prompt)

            # Run both models
            pytorch_logits = self.run_pytorch(inputs)
            onnx_logits = self.run_onnx(onnx_sess, inputs)

            # Compare logits
            result = self.compare_arrays(f"logits: '{prompt[:20]}...'", pytorch_logits, onnx_logits)
            results.append(result)

            # Compare top-k
            top_k_result = self.compare_top_k(
                f"top-5: '{prompt[:20]}...'", pytorch_logits, onnx_logits
            )
            results.append(top_k_result)

        self.results.extend(results)
        return results

    def verify_against_community(
        self, onnx_path: str, community_path: str, prompts: list[str] = None
    ) -> list[VerificationResult]:
        """Verify ONNX model against community version."""
        if prompts is None:
            prompts = ["Hello, how are"]

        self.load_pytorch_model()  # For tokenizer
        onnx_sess = self.load_onnx_model(onnx_path)
        community_sess = self.load_onnx_model(community_path)

        results = []
        for prompt in prompts:
            logger.info(f"Comparing with community: '{prompt}'")
            inputs = self.prepare_inputs(prompt)

            # Run both ONNX models
            our_logits = self.run_onnx(onnx_sess, inputs)
            community_logits = self.run_onnx(community_sess, inputs)

            # Compare
            result = self.compare_arrays(
                f"vs community: '{prompt[:20]}...'", community_logits, our_logits
            )
            results.append(result)

            top_k_result = self.compare_top_k(
                f"top-5 vs community: '{prompt[:20]}...'", community_logits, our_logits
            )
            results.append(top_k_result)

        self.results.extend(results)
        return results

    def test_generation(
        self, onnx_path: str, prompt: str = "Hello, how are", max_tokens: int = 10
    ) -> VerificationResult:
        """Test multi-step generation with cache updates."""
        self.load_pytorch_model()
        onnx_sess = self.load_onnx_model(onnx_path)

        input_ids = self.tokenizer.encode(prompt, return_tensors="np")[0].tolist()
        generated_pytorch = self._generate_pytorch(input_ids.copy(), max_tokens)
        generated_onnx = self._generate_onnx(onnx_sess, input_ids.copy(), max_tokens)

        match = generated_pytorch == generated_onnx
        text_pytorch = self.tokenizer.decode(generated_pytorch)
        text_onnx = self.tokenizer.decode(generated_onnx)

        result = VerificationResult(
            name="generation",
            passed=match,
            max_diff=0.0 if match else 1.0,
            mean_diff=0.0 if match else 1.0,
            correlation=1.0 if match else 0.0,
            details=f"PyTorch: '{text_pytorch}'\nONNX: '{text_onnx}'",
        )
        self.results.append(result)
        return result

    def _generate_pytorch(self, input_ids: list[int], max_tokens: int) -> list[int]:
        """Generate tokens with PyTorch model."""
        import torch

        generated = input_ids.copy()
        past_key_values = None

        for _ in range(max_tokens):
            with torch.no_grad():
                if past_key_values is None:
                    ids = torch.tensor([generated], dtype=torch.long)
                    pos = torch.arange(len(generated), dtype=torch.long).unsqueeze(0)
                else:
                    ids = torch.tensor([[generated[-1]]], dtype=torch.long)
                    pos = torch.tensor([[len(generated) - 1]], dtype=torch.long)

                outputs = self.torch_model(
                    input_ids=ids,
                    position_ids=pos,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values
                next_token = int(outputs.logits[0, -1].argmax())
                generated.append(next_token)

                if next_token == self.tokenizer.eos_token_id:
                    break

        return generated

    def _generate_onnx(self, sess, input_ids: list[int], max_tokens: int) -> list[int]:
        """Generate tokens with ONNX model."""
        generated = input_ids.copy()

        # Initialize caches
        cache = {}
        for inp in sess.get_inputs():
            if inp.name not in ["input_ids", "attention_mask", "position_ids"]:
                shape = [d if isinstance(d, int) else 1 for d in inp.shape]
                cache[inp.name] = np.zeros(shape, dtype=np.float32)

        outputs_info = sess.get_outputs()

        for step in range(max_tokens):
            cur_len = len(generated)

            if step == 0:
                ids = np.array([generated], dtype=np.int64)
                pos = np.arange(cur_len, dtype=np.int64).reshape(1, -1)
            else:
                ids = np.array([[generated[-1]]], dtype=np.int64)
                pos = np.array([[cur_len - 1]], dtype=np.int64)

            attn_mask = np.ones((1, cur_len), dtype=np.int64)

            feed = {"input_ids": ids, "attention_mask": attn_mask, "position_ids": pos}
            feed.update(cache)

            result = sess.run(None, feed)

            # Update caches
            for i, out_info in enumerate(outputs_info[1:], 1):
                out_name = out_info.name
                # Map present -> past
                if "present_conv" in out_name:
                    cache_name = out_name.replace("present_conv", "past_conv")
                elif "present." in out_name:
                    cache_name = out_name.replace("present.", "past_key_values.")
                else:
                    continue
                if cache_name in cache:
                    cache[cache_name] = result[i]

            next_token = int(np.argmax(result[0][0, -1]))
            generated.append(next_token)

            if next_token == self.tokenizer.eos_token_id:
                break

        return generated

    def print_report(self):
        """Print verification report."""
        print("\n" + "=" * 70)
        print("NUMERICAL VERIFICATION REPORT")
        print("=" * 70)

        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)

        for result in self.results:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            print(f"\n{status} {result.name}")
            print(f"  Max diff: {result.max_diff:.6f}")
            print(f"  Mean diff: {result.mean_diff:.6f}")
            print(f"  Correlation: {result.correlation:.6f}")
            if result.details:
                print(f"  Details: {result.details}")

        print("\n" + "=" * 70)
        print(f"SUMMARY: {passed}/{total} checks passed")
        print("=" * 70)

        return passed == total


def main():
    parser = argparse.ArgumentParser(description="Verify LFM2 ONNX export")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model path")
    parser.add_argument("--onnx", type=str, required=True, help="ONNX model directory")
    parser.add_argument("--community", type=str, help="Community ONNX model for comparison")
    parser.add_argument("--atol", type=float, default=1e-4, help="Absolute tolerance")
    parser.add_argument("--rtol", type=float, default=1e-3, help="Relative tolerance")
    parser.add_argument("--test-generation", action="store_true", help="Test multi-step generation")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    verifier = NumericalVerifier(args.model, atol=args.atol, rtol=args.rtol)

    # Verify against PyTorch
    logger.info("Verifying against PyTorch model...")
    verifier.verify_against_pytorch(args.onnx)

    # Verify against community if provided
    if args.community:
        logger.info("Verifying against community ONNX model...")
        verifier.verify_against_community(args.onnx, args.community)

    # Test generation
    if args.test_generation:
        logger.info("Testing multi-step generation...")
        verifier.test_generation(args.onnx)

    # Print report
    success = verifier.print_report()
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
