#!/usr/bin/env python3
"""
CPU inference for LFM2.5-Audio ONNX models.

This module provides text generation using the exported ONNX models.
For audio processing, the Conformer encoder export is pending - currently
only text-to-text generation is supported.

Usage:
    uv run lfm2-audio-infer /path/to/LFM2.5-Audio-1.5B-ONNX --prompt "Hello, world!"
    uv run lfm2-audio-infer /path/to/model --precision q4 --prompt "What is AI?"
"""

import argparse
import logging
import pathlib
import time

import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)


def get_onnx_files(model_dir: pathlib.Path, precision: str) -> dict[str, pathlib.Path]:
    """Get paths to ONNX model files for given precision."""
    onnx_dir = model_dir / "onnx"

    suffix = "" if precision == "fp32" else f"_{precision}"

    files = {
        "embed_tokens": onnx_dir / f"embed_tokens{suffix}.onnx",
        "decoder": onnx_dir / f"decoder{suffix}.onnx",
    }

    # Fall back to fp32 if requested precision not available
    for name, path in files.items():
        if not path.exists():
            fp32_path = onnx_dir / f"{name}.onnx"
            if fp32_path.exists():
                logger.warning(f"{path.name} not found, falling back to {fp32_path.name}")
                files[name] = fp32_path

    return files


def load_session(model_path: pathlib.Path) -> ort.InferenceSession:
    """Load ONNX model as inference session."""
    # CPU-only execution
    providers = ["CPUExecutionProvider"]

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    return ort.InferenceSession(str(model_path), sess_options, providers=providers)


class LFM2AudioInference:
    """ONNX inference for LFM2.5-Audio text generation."""

    def __init__(self, model_dir: pathlib.Path, precision: str = "fp32"):
        self.model_dir = model_dir
        self.precision = precision

        # Load tokenizer
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)

        # Load ONNX sessions
        files = get_onnx_files(model_dir, precision)
        logger.info(f"Loading embed_tokens from {files['embed_tokens'].name}...")
        self.embed_session = load_session(files["embed_tokens"])
        logger.info(f"Loading decoder from {files['decoder'].name}...")
        self.decoder_session = load_session(files["decoder"])

        # Get model config
        self._load_config()

    def _load_config(self):
        """Load model config from config.json."""
        import json

        config_path = self.model_dir / "config.json"
        with open(config_path) as f:
            config = json.load(f)

        lfm_config = config.get("lfm", {})
        self.hidden_size = lfm_config.get("hidden_size", 2048)
        self.num_layers = lfm_config.get("num_hidden_layers", 16)
        self.num_kv_heads = lfm_config.get("num_key_value_heads", 8)
        self.head_dim = self.hidden_size // lfm_config.get("num_attention_heads", 32)
        self.conv_L = lfm_config.get("conv_L_cache", 3)
        self.layer_types = lfm_config.get("layer_types", [])

    def _init_cache(self, batch_size: int = 1) -> dict[str, np.ndarray]:
        """Initialize KV cache for generation."""
        cache = {}

        for idx, layer_type in enumerate(self.layer_types):
            if layer_type == "conv":
                cache[f"past_conv.{idx}"] = np.zeros(
                    (batch_size, self.hidden_size, self.conv_L), dtype=np.float32
                )
            else:
                cache[f"past_key_values.{idx}.key"] = np.zeros(
                    (batch_size, self.num_kv_heads, 0, self.head_dim), dtype=np.float32
                )
                cache[f"past_key_values.{idx}.value"] = np.zeros(
                    (batch_size, self.num_kv_heads, 0, self.head_dim), dtype=np.float32
                )

        return cache

    def _update_cache(self, cache: dict, outputs: dict) -> dict:
        """Update cache with decoder outputs."""
        for key in cache:
            if key.startswith("past_conv."):
                idx = int(key.split(".")[1])
                cache[key] = outputs[f"present_conv.{idx}"]
            elif key.startswith("past_key_values."):
                parts = key.split(".")
                idx = int(parts[1])
                kv_type = parts[2]  # "key" or "value"
                cache[key] = outputs[f"present.{idx}.{kv_type}"]
        return cache

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """Generate text from prompt.

        Args:
            prompt: Input prompt text
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p (nucleus) sampling threshold

        Returns:
            Generated text
        """
        # Tokenize input
        input_ids = self.tokenizer.encode(prompt, return_tensors="np")
        batch_size, seq_len = input_ids.shape

        # Get embeddings
        embeds = self.embed_session.run(["inputs_embeds"], {"input_ids": input_ids})[0]

        # Initialize cache
        cache = self._init_cache(batch_size)

        # Prefill: process entire prompt
        attention_mask = np.ones((batch_size, seq_len), dtype=np.int64)

        decoder_inputs = {
            "inputs_embeds": embeds.astype(np.float32),
            "attention_mask": attention_mask,
            **cache,
        }

        decoder_outputs = self.decoder_session.run(None, decoder_inputs)

        # Parse outputs - first is logits, rest are cache updates
        output_names = [o.name for o in self.decoder_session.get_outputs()]
        outputs = dict(zip(output_names, decoder_outputs, strict=True))

        logits = outputs["logits"]
        cache = self._update_cache(cache, outputs)

        # Sample next token from last position
        next_logits = logits[0, -1, :]
        next_token = self._sample(next_logits, temperature, top_p)

        generated_tokens = [next_token]
        total_len = seq_len + 1

        # Generation loop
        start_time = time.time()

        for _ in range(max_new_tokens - 1):
            if next_token == self.tokenizer.eos_token_id:
                break

            # Get embedding for single token
            next_ids = np.array([[next_token]], dtype=np.int64)
            next_embeds = self.embed_session.run(["inputs_embeds"], {"input_ids": next_ids})[0]

            # Update attention mask
            attention_mask = np.ones((batch_size, total_len), dtype=np.int64)

            decoder_inputs = {
                "inputs_embeds": next_embeds.astype(np.float32),
                "attention_mask": attention_mask,
                **cache,
            }

            decoder_outputs = self.decoder_session.run(None, decoder_inputs)
            outputs = dict(zip(output_names, decoder_outputs, strict=True))

            logits = outputs["logits"]
            cache = self._update_cache(cache, outputs)

            # Sample next token
            next_logits = logits[0, -1, :]
            next_token = self._sample(next_logits, temperature, top_p)

            generated_tokens.append(next_token)
            total_len += 1

        elapsed = time.time() - start_time
        tokens_per_sec = len(generated_tokens) / elapsed if elapsed > 0 else 0
        logger.info(
            f"Generated {len(generated_tokens)} tokens in {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s)"
        )

        # Decode generated tokens
        output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return output_text

    def _sample(self, logits: np.ndarray, temperature: float, top_p: float) -> int:
        """Sample next token using temperature and top-p sampling."""
        if temperature == 0:
            return int(np.argmax(logits))

        # Apply temperature
        logits = logits / temperature

        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()

        # Top-p filtering
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumsum = np.cumsum(sorted_probs)

        # Find cutoff
        cutoff_idx = np.searchsorted(cumsum, top_p) + 1
        top_indices = sorted_indices[:cutoff_idx]
        top_probs = probs[top_indices]
        top_probs = top_probs / top_probs.sum()

        # Sample
        return int(np.random.choice(top_indices, p=top_probs))


def main():
    parser = argparse.ArgumentParser(
        description="LFM2.5-Audio ONNX inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "model_dir",
        type=pathlib.Path,
        help="Path to exported ONNX model directory",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The capital of France is",
        help="Input prompt for generation",
    )
    parser.add_argument(
        "--precision",
        choices=["fp32", "fp16", "q4", "q8"],
        default="fp32",
        help="Model precision to use",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling threshold",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    # Initialize model
    logger.info(f"Loading model from {args.model_dir}...")
    model = LFM2AudioInference(args.model_dir, precision=args.precision)

    # Generate
    logger.info(f"Prompt: {args.prompt}")
    logger.info("Generating...")

    output = model.generate(
        args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    print("\n" + "=" * 60)
    print(f"Input:  {args.prompt}")
    print(f"Output: {output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
