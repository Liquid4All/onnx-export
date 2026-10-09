#!/usr/bin/env python3
"""
Chat with an LFM2 or LFM2-MoE export through onnxruntime-genai.

Usage:
    uv run lfm2-infer --model exports/LFM2.5-1.2B-Instruct-ONNX
    uv run lfm2-infer --model exports/LFM2.5-1.2B-Instruct-ONNX --precision q8 --prompt "Hello"
    uv run lfm2-moe-infer --model exports/LFM2.5-8B-A1B-ONNX
"""

import argparse
import copy
import functools
import json
import logging
import pathlib

import numpy as np
import onnxruntime_genai as og

from liquidonnx.genai_runtime import TokenPrinter, add_runtime_arguments, generate, load_model
from liquidonnx.lfm2.export import ALL_PRECISIONS, genai_files

# LFM2.5-2.6B and LFM2.5-8B-A1B think before they answer: greedy at q4, eight chat prompts took
# 52 to 2,447 tokens. An answer ends at the end-of-turn token; the cap stops one that loops.
MAX_NEW_TOKENS = 4096


def template_messages(messages: list[dict], join_text: bool) -> list[dict]:
    """A copy of OpenAI chat messages in the shape the LFM2 chat templates take.

    Every template needs tool-call arguments as a mapping, where OpenAI sends a JSON string;
    arguments that are not JSON raise ValueError. Content that is null or missing becomes "", as
    vLLM sends it: the LFM2.5-VL templates fail on null, and LFM2.5-VL-450M's leaves an assistant
    turn of tool calls without content unended. join_text makes a list of text parts one string,
    one part per line as vLLM joins them, for templates that would render the list as JSON
    (LFM2's); the LFM2.5 templates take the list.
    """
    messages = copy.deepcopy(messages)
    for message in messages:
        for call in message.get("tool_calls") or []:
            function = call.get("function", call)
            if isinstance(function.get("arguments"), str):
                function["arguments"] = json.loads(function["arguments"] or "{}")
        content = message.get("content")
        if content is None:
            message["content"] = ""
        elif join_text and isinstance(content, list):
            if all(part.get("type") == "text" for part in content):
                message["content"] = "\n".join(part["text"] for part in content)
    return messages


class TextChat:
    """A conversation with an LFM2 or LFM2-MoE export.

    answer() keeps no state, so one TextChat can serve independent requests; send() keeps the
    conversation of lfm2-infer.
    """

    def __init__(
        self,
        model_dir: pathlib.Path,
        precision: str | None = None,
        ep: str = "cpu",
        tf32: bool = True,
    ):
        from transformers import AutoTokenizer

        self.model = load_model(model_dir, precision and genai_files(precision), ep, tf32)
        self.tokenizer = og.Tokenizer(self.model)
        # The chat template and prompt ids come from transformers, as for the reference model.
        self.hf_tokenizer = AutoTokenizer.from_pretrained(model_dir)
        parts = [{"role": "user", "content": [{"type": "text", "text": ""}]}]
        self.join_text = '"type"' in self.hf_tokenizer.apply_chat_template(parts, tokenize=False)
        self.messages: list[dict] = []

    @functools.cached_property
    def raw_tokenizer(self) -> og.Tokenizer:
        """self.tokenizer, but decode keeps special tokens; made on first use (0.4 s for LFM2.5)."""
        tokenizer = og.Tokenizer(self.model)
        tokenizer.update_options(skip_special_tokens="false")
        return tokenizer

    def answer(
        self,
        messages: list[dict],
        max_new_tokens: int = MAX_NEW_TOKENS,
        tools: list[dict] | None = None,
        keep_special_tokens: bool = False,
        stream: bool = False,
    ) -> str:
        """The assistant's answer to messages, OpenAI chat messages, which it leaves as they are.

        Tool calls can come with their arguments as a JSON string and content as a list of text
        parts (see template_messages). tools, OpenAI function schemas, go to the chat template.
        The answer leaves out the tokens tokenizer.json marks special, as onnxruntime-genai
        decodes, unless keep_special_tokens. A tool-call parser needs them where
        <|tool_call_start|> and <|tool_call_end|> are special (LFM2-350M, -700M and -1.2B); the
        LFM2.5 checkpoints keep those, <think> and </think> either way. The end-of-turn token that
        ends the answer is never part of it. stream prints the answer as it comes.
        """
        prompt = self.hf_tokenizer.apply_chat_template(
            template_messages(messages, join_text=self.join_text),
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
        )
        ids = np.array(self.hf_tokenizer.encode(prompt, add_special_tokens=False))
        tokenizer = self.raw_tokenizer if keep_special_tokens else self.tokenizer
        printer = TokenPrinter(tokenizer) if stream else None
        generator = generate(self.model, ids, max_new_tokens, printer)
        if stream:
            print()
        return tokenizer.decode(generator.get_sequence(0)[len(ids) :])

    def send(self, text: str, max_new_tokens: int = MAX_NEW_TOKENS, stream: bool = True) -> str:
        self.messages.append({"role": "user", "content": text})
        response = self.answer(self.messages, max_new_tokens, stream=stream)
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
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=MAX_NEW_TOKENS,
        help=f"Max tokens per answer (default: {MAX_NEW_TOKENS})",
    )
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming output")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    run_chat_loop(TextChat(args.model, args.precision, args.ep, args.tf32), args, "LFM2")


if __name__ == "__main__":
    main()
