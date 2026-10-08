"""LFM2 / LFM2-MoE exports against the PyTorch model."""

import json
import logging
import pathlib
import time

import numpy as np
import onnxruntime_genai as og

from liquidonnx.compare.metrics import (
    greedy,
    guarded,
    logit_metrics,
    merge_metrics,
    merge_sequences,
    sequence_match,
)
from liquidonnx.genai_runtime import generate, load_model
from liquidonnx.lfm2.export import ALL_PRECISIONS, genai_files, model_file
from liquidonnx.quantize import get_total_model_size_mb
from liquidonnx.session import cached_outputs, load_onnx_session

logger = logging.getLogger(__name__)

PROMPTS = [
    "What is the capital of France? Answer in one sentence.",
    "Write a Python function that returns the n-th Fibonacci number.",
    "Explain in two sentences why the sky is blue.",
    "List three primary colors.",
    "Translate 'Good morning, how are you?' into Spanish and German.",
    "日本の首都はどこですか？",
    "A train travels 120 km in 1.5 hours. What is its average speed in km/h? Show the steps.",
    "Summarize the plot of Romeo and Juliet in three sentences.",
]


def precisions(export: pathlib.Path) -> list[str]:
    return [p for p in ("fp32", *ALL_PRECISIONS) if (export / "onnx" / model_file(p)).exists()]


def plain_greedy(model) -> None:
    """Make generate() answer with the argmax of the model's logits.

    With do_sample=False, generate() still applies the logit processors of the checkpoint's
    generation_config, such as LFM2.5-8B-A1B's repetition_penalty of 1.05. Its answers then
    disagree with the argmax of the logits the decoders are scored on, which caps every
    precision's greedy score. Only the special tokens are kept.
    """
    from transformers import GenerationConfig

    config = model.generation_config
    model.generation_config = GenerationConfig(
        bos_token_id=config.bos_token_id,
        eos_token_id=config.eos_token_id,
        pad_token_id=config.pad_token_id,
    )


def check_answer(answer: list[int], logits: np.ndarray, name: str) -> None:
    """Warn when a greedy answer leaves the argmax of its logits: the decoders' greedy answers
    follow the argmax, so they cannot match the reference past that token."""
    off = np.flatnonzero(logits.argmax(-1) != np.asarray(answer))
    if len(off):
        logger.warning(
            f"reference {name}: answer token {off[0]} of {len(answer)} is not the argmax of "
            "its logits"
        )


def reference(checkpoint: pathlib.Path, prompts: int, max_new: int, device: str) -> list[dict]:
    """Greedy answers of the fp32 PyTorch model and the logits that predict each answer token."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, dtype=torch.float32).to(device).eval()
    plain_greedy(model)
    items = []
    for i, text in enumerate(PROMPTS[:prompts]):
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False
        )
        ids = np.array(tokenizer.encode(rendered, add_special_tokens=False), np.int64)
        start = time.time()
        with torch.no_grad():
            out = model.generate(
                torch.from_numpy(ids)[None].to(device), max_new_tokens=max_new, do_sample=False
            )
            # A full forward pass, not generate()'s own logits: MoE routing depends on the batch
            # shape, and the ONNX decoder is scored against the model, not one run of it.
            logits = model(out).logits[0].float().cpu().numpy()
        answer, logits = out[0, len(ids) :].tolist(), logits[len(ids) - 1 : -1]
        logger.info(f"reference {len(ids)}+{len(answer)} tokens in {time.time() - start:.1f}s")
        check_answer(answer, logits, f"prompt {i}")
        items.append(
            {
                "rendered": rendered,
                "prompt": ids,
                "answer": answer,
                "logits": logits,
                "text": tokenizer.decode(answer, skip_special_tokens=True),
            }
        )
    return items


def eos_ids(export: pathlib.Path) -> set[int]:
    eos = json.loads((export / "genai_config.json").read_text())["model"]["eos_token_id"]
    return set(eos if isinstance(eos, list) else [eos])


def score_decoder(path: pathlib.Path, refs: list[dict], eos: set[int], ep: str) -> dict:
    session = load_onnx_session(path, ep, tf32=False)
    forced, answers = [], []
    for ref in refs:
        sequence = np.concatenate([ref["prompt"], ref["answer"]])[None]
        actual = cached_outputs(session, sequence[:, :-1], len(ref["prompt"]))["logits"]
        forced.append(logit_metrics(ref["logits"], actual))
        answer = greedy(session, ref["prompt"], len(ref["answer"]), eos)
        answers.append(sequence_match(ref["answer"], answer, eos))
    return {
        "teacher_forced": merge_metrics(forced),
        "greedy": merge_sequences(answers),
        "size_mb": get_total_model_size_mb(path),
    }


def score_genai(
    export: pathlib.Path, precision: str, refs: list[dict], eos: set[int], ep: str
) -> dict:
    model = load_model(export, genai_files(precision), ep, tf32=False)
    tokenizer = og.Tokenizer(model)
    answers, same_ids = [], 0
    for ref in refs:
        same_ids += tokenizer.encode(ref["rendered"]).tolist() == ref["prompt"].tolist()
        generator = generate(model, ref["prompt"], len(ref["answer"]))
        answer = generator.get_sequence(0)[len(ref["prompt"]) :].tolist()
        answers.append(sequence_match(ref["answer"], answer, eos))
    return {"greedy": merge_sequences(answers), "prompt_ids_equal": f"{same_ids}/{len(refs)}"}


def score(export: pathlib.Path, precision: str, refs: list[dict], genai: bool, ep: str) -> dict:
    eos = eos_ids(export)
    path = export / "onnx" / model_file(precision)
    row = {"decoder": guarded(score_decoder, path, refs, eos, ep)}
    if genai:
        row["genai"] = guarded(score_genai, export, precision, refs, eos, ep)
    return row
