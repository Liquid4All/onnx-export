#!/usr/bin/env python3
"""
Chat with an LFM2 or LFM2-MoE export through onnxruntime-genai.

Usage:
    uv run lfm2-infer --model exports/LFM2.5-1.2B-Instruct-ONNX
    uv run lfm2-infer --model exports/LFM2.5-1.2B-Instruct-ONNX --precision q8 --prompt "Hello"
    uv run lfm2-moe-infer --model exports/LFM2.5-8B-A1B-ONNX
"""

import argparse
import logging
import pathlib

import numpy as np
import onnxruntime_genai as og

from liquidonnx.genai_runtime import TokenPrinter, add_runtime_arguments, generate, load_model
from liquidonnx.lfm2.export import ALL_PRECISIONS, genai_files


class TextChat:
    """A conversation with an LFM2 or LFM2-MoE export."""

    def __init__(self, model_dir: pathlib.Path, precision: str | None = None, ep: str = "cpu"):
        from transformers import AutoTokenizer

        self.model = load_model(model_dir, precision and genai_files(precision), ep)
        self.tokenizer = og.Tokenizer(self.model)
        # The chat template and prompt ids come from transformers, as for the reference model.
        self.hf_tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.messages: list[dict] = []

    def send(self, text: str, max_new_tokens: int = 100, stream: bool = True) -> str:
        self.messages.append({"role": "user", "content": text})
        prompt = self.hf_tokenizer.apply_chat_template(
            self.messages, tokenize=False, add_generation_prompt=True
        )
        ids = np.array(self.hf_tokenizer.encode(prompt, add_special_tokens=False))
        printer = TokenPrinter(self.tokenizer) if stream else None
        generator = generate(self.model, ids, max_new_tokens, printer)
        if stream:
            print()
        response = self.tokenizer.decode(generator.get_sequence(0)[len(ids) :])
        self.messages.append({"role": "assistant", "content": response})
        return response

    def clear(self):
        self.messages.clear()

    def command(self, line: str) -> bool:
        """Run line if it is a chat command other than quit and clear; True if it was."""
        return False


def run_chat_loop(chat, args: argparse.Namespace, title: str, commands: str = ""):
    """Interactive chat, starting with args.prompt.

    chat has send(), clear() and command() as TextChat does; commands describes its commands.
    """
    print("\n" + "=" * 50)
    print(f"{title} - onnxruntime-genai")
    print("Type 'quit' or 'exit' to stop, 'clear' to reset the conversation")
    if commands:
        print(commands)
    print("=" * 50 + "\n")

    def answer(text: str):
        print("Assistant: ", end="")
        response = chat.send(text, args.max_tokens, stream=not args.no_stream)
        if args.no_stream:
            print(response)

    if args.prompt:
        print(f"User: {args.prompt}")
        answer(args.prompt)

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
            chat.clear()
            print("Chat history cleared.")
            continue
        if not chat.command(user_input):
            answer(user_input)


def main():
    parser = argparse.ArgumentParser(
        description="Chat with an LFM2 or LFM2-MoE export through onnxruntime-genai"
    )
    parser.add_argument("--model", required=True, type=pathlib.Path, help="Export folder")
    add_runtime_arguments(parser, ALL_PRECISIONS)
    parser.add_argument("--prompt", default=None, help="Initial prompt (optional)")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming output")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    run_chat_loop(TextChat(args.model, args.precision, args.ep), args, "LFM2")


if __name__ == "__main__":
    main()
