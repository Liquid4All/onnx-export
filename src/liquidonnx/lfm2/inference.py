#!/usr/bin/env python3
"""
ONNX inference script for LFM2 text models.

Usage:
    uv run inference.py --model LFM2-1.2B-ONNX-builder-Q4-fp32head
    uv run inference.py --model LFM2-1.2B-ONNX-builder-Q4-fp32head --prompt "Hello, how are you?"
    uv run inference.py --model LFM2-1.2B-ONNX-builder-Q4-fp32head --max-tokens 100
"""

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


class TextModelInference:
    """ONNX inference for LFM2 text models."""

    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.tokenizer = None
        self.session = None

    def load(self):
        """Load tokenizer and ONNX model."""
        print(f"Loading model from {self.model_path}...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path), trust_remote_code=True
        )

        # Load ONNX model
        onnx_path = self.model_path / "onnx" / "decoder.onnx"
        if not onnx_path.exists():
            # Try model.onnx for non-split models
            onnx_path = self.model_path / "onnx" / "model.onnx"

        print(f"Loading ONNX from {onnx_path}...")
        self.session = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )

        # Get model info
        self.input_names = {inp.name for inp in self.session.get_inputs()}
        print(f"Model loaded. Inputs: {list(self.input_names)[:5]}...")

    def _initialize_cache(self) -> dict:
        """Initialize KV cache tensors."""
        cache = {}
        for inp in self.session.get_inputs():
            name = inp.name
            if name in ["input_ids", "inputs_embeds", "attention_mask", "position_ids"]:
                continue
            # Initialize cache with zeros
            shape = [d if isinstance(d, int) else 1 for d in inp.shape]
            # For past_key_values, set sequence length to 0
            if "past" in name.lower():
                for i, d in enumerate(inp.shape):
                    if isinstance(d, str) and "sequence" in d.lower():
                        shape[i] = 0
            cache[name] = np.zeros(shape, dtype=np.float32)
        return cache

    def _update_cache(self, cache: dict, outputs: dict, output_names: list):
        """Update cache from model outputs."""
        for i, name in enumerate(output_names):
            if name.startswith("present"):
                # Map present -> past
                cache_name = name.replace("present", "past").replace(".", "_")
                if cache_name not in cache:
                    cache_name = name.replace("present.", "past_key_values.")
                if cache_name not in cache:
                    cache_name = name.replace("present_conv", "past_conv")
                if cache_name in cache:
                    cache[cache_name] = outputs[i]

    def generate(
        self,
        messages: list,
        max_new_tokens: int = 100,
        stream: bool = True,
    ) -> str:
        """Generate response for chat messages."""
        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Tokenize
        input_ids = np.array(
            [self.tokenizer.encode(prompt, add_special_tokens=False)], dtype=np.int64
        )

        # Initialize cache
        cache = self._initialize_cache()

        # Get output names
        output_names = [out.name for out in self.session.get_outputs()]

        seq_len = input_ids.shape[1]
        position_ids = np.arange(seq_len, dtype=np.int64).reshape(1, -1)

        generated_tokens = []
        cur_len = seq_len

        for step in range(max_new_tokens):
            # Build inputs
            if step == 0:
                ids = input_ids
                pos = position_ids
            else:
                ids = np.array([[generated_tokens[-1]]], dtype=np.int64)
                pos = np.array([[cur_len - 1]], dtype=np.int64)

            attn_mask = np.ones((1, cur_len), dtype=np.int64)

            feed = {"input_ids": ids, "attention_mask": attn_mask}
            if "position_ids" in self.input_names:
                feed["position_ids"] = pos
            feed.update(cache)

            # Run inference
            outputs = self.session.run(None, feed)
            logits = outputs[0][0, -1]

            # Greedy decoding
            next_token = int(np.argmax(logits))
            generated_tokens.append(next_token)

            # Update cache
            self._update_cache(cache, outputs, output_names)
            cur_len += 1

            # Stream output
            if stream:
                token_str = self.tokenizer.decode([next_token])
                print(token_str, end="", flush=True)

            # Check for EOS
            if next_token == self.tokenizer.eos_token_id:
                break

        if stream:
            print()

        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="ONNX inference for LFM2 text models")
    parser.add_argument("--model", required=True, help="Path to ONNX model directory")
    parser.add_argument("--prompt", default=None, help="Initial prompt (optional)")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming output")
    args = parser.parse_args()

    # Load model
    model = TextModelInference(args.model)
    model.load()

    print("\n" + "=" * 50)
    print("LFM2 Text Model - ONNX Inference")
    print("Type 'quit' or 'exit' to stop")
    print("=" * 50 + "\n")

    messages = []

    # Initial prompt if provided
    if args.prompt:
        messages.append({"role": "user", "content": args.prompt})
        print(f"User: {args.prompt}")
        print("Assistant: ", end="")
        response = model.generate(
            messages, max_new_tokens=args.max_tokens, stream=not args.no_stream
        )
        messages.append({"role": "assistant", "content": response})
        if args.no_stream:
            print(response)

    # Interactive loop
    while True:
        try:
            user_input = input("\nUser: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break
        if user_input.lower() == "clear":
            messages = []
            print("Chat history cleared.")
            continue

        messages.append({"role": "user", "content": user_input})
        print("Assistant: ", end="")
        response = model.generate(
            messages, max_new_tokens=args.max_tokens, stream=not args.no_stream
        )
        messages.append({"role": "assistant", "content": response})
        if args.no_stream:
            print(response)


if __name__ == "__main__":
    main()
