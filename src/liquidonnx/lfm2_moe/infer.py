#!/usr/bin/env python3
"""
ONNX inference script for LFM2-MoE models.

Usage:
    uv run lfm2-moe-infer --model exports/LFM2-MoE-8B-A1B-ONNX
    uv run lfm2-moe-infer --model exports/LFM2-MoE-8B-A1B-ONNX --prompt "Hello"
"""

import argparse
import logging
import pathlib

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from liquidonnx.session import initialize_cache, update_cache

logger = logging.getLogger(__name__)


class MoEModelInference:
    """ONNX inference for LFM2-MoE models."""

    def __init__(self, model_path: str):
        self.model_path = pathlib.Path(model_path)
        self.tokenizer = None
        self.session = None

    def load(self):
        """Load tokenizer and ONNX model."""
        logger.info(f"Loading model from {self.model_path}...")

        # Handle both directory and direct ONNX file paths
        if self.model_path.suffix == ".onnx":
            onnx_path = self.model_path
            # Tokenizer is in parent of onnx/ directory
            tokenizer_path = self.model_path.parent.parent
        else:
            tokenizer_path = self.model_path
            onnx_path = self.model_path / "onnx" / "model.onnx"

        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_path), trust_remote_code=True
        )

        logger.info(f"Loading ONNX from {onnx_path}...")
        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

        self.input_names = {inp.name for inp in self.session.get_inputs()}
        logger.info(f"Model loaded. Inputs: {list(self.input_names)[:5]}...")

    def generate(
        self,
        messages: list,
        max_new_tokens: int = 100,
        stream: bool = True,
    ) -> str:
        """Generate response for chat messages."""
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = np.array(
            [self.tokenizer.encode(prompt, add_special_tokens=False)], dtype=np.int64
        )

        cache = initialize_cache(self.session)
        output_infos = self.session.get_outputs()

        seq_len = input_ids.shape[1]

        generated_tokens = []
        cur_len = seq_len

        for step in range(max_new_tokens):
            if step == 0:
                ids = input_ids
            else:
                ids = np.array([[generated_tokens[-1]]], dtype=np.int64)

            attn_mask = np.ones((1, cur_len), dtype=np.int64)

            feed = {"input_ids": ids, "attention_mask": attn_mask}
            feed.update(cache)

            outputs = self.session.run(None, feed)
            logits = outputs[0][0, -1]

            next_token = int(np.argmax(logits))
            generated_tokens.append(next_token)

            update_cache(cache, outputs, output_infos)
            cur_len += 1

            if stream:
                token_str = self.tokenizer.decode([next_token])
                print(token_str, end="", flush=True)

            if next_token == self.tokenizer.eos_token_id:
                break

        if stream:
            print()

        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="ONNX inference for LFM2-MoE models")
    parser.add_argument("--model", required=True, help="Path to ONNX model directory")
    parser.add_argument("--prompt", default=None, help="Initial prompt (optional)")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming output")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    model = MoEModelInference(args.model)
    model.load()

    print("\n" + "=" * 50)
    print("LFM2-MoE Model - ONNX Inference")
    print("Type 'quit' or 'exit' to stop")
    print("=" * 50 + "\n")

    messages = []

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
