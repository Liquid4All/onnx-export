"""LFM2-VL exports against the PyTorch model.

The reference runs with do_image_splitting=False, which is what onnxruntime-genai does: it resizes
each image once instead of tiling it.
"""

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
from liquidonnx.compare.text import eos_ids
from liquidonnx.embeddings import embed
from liquidonnx.genai_runtime import generate, load_model
from liquidonnx.lfm2_vl.export import PRECISIONS, bundle, genai_files
from liquidonnx.quantize import get_total_model_size_mb
from liquidonnx.session import cached_outputs, load_onnx_session

logger = logging.getLogger(__name__)

ASSETS = pathlib.Path(__file__).parents[3] / "tests/test_lfm2_vl/assets"
CASES = [
    (["cardinal.jpg"], "Describe this image in one sentence."),
    (["bluejay.jpg"], "What bird is this and what color is it?"),
    (["cardinal.jpg", "bluejay.jpg"], "Compare these two images."),
    ([], "What is the capital of France? Answer in one sentence."),
]
PIXEL_INPUTS = ("pixel_values", "pixel_attention_mask", "spatial_shapes")


def precisions(export: pathlib.Path) -> list[str]:
    onnx_dir = export / "onnx"
    return [
        p for p in ("fp32", *PRECISIONS) if all((onnx_dir / f).exists() for f in bundle(p).values())
    ]


def reference(checkpoint: pathlib.Path, max_new: int) -> list[dict]:
    """Greedy answers, their logits, the merged input embeddings and the image features."""
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(checkpoint)
    model = AutoModelForImageTextToText.from_pretrained(checkpoint, dtype=torch.float32).eval()
    captured = {}

    def grab(_module, _args, kwargs):
        captured["embeds"] = kwargs["inputs_embeds"].detach().float().numpy()[0]

    model.model.language_model.register_forward_pre_hook(grab, with_kwargs=True)
    items = []
    for images, text in CASES:
        content = [*({"type": "image"} for _ in images), {"type": "text", "text": text}]
        rendered = processor.apply_chat_template(
            [{"role": "user", "content": content}], add_generation_prompt=True, tokenize=False
        )
        pil = [Image.open(ASSETS / f).convert("RGB") for f in images]
        inputs = processor(
            text=rendered,
            return_tensors="pt",
            do_image_splitting=False,
            **({"images": pil} if pil else {}),
        )
        ids = inputs["input_ids"]
        start = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
            logits = model(**{**inputs, "input_ids": out, "attention_mask": torch.ones_like(out)})
        answer = out[0, ids.shape[1] :].tolist()
        item = {
            "images": images,
            "rendered": rendered,
            "prompt": ids[0].numpy(),
            "answer": answer,
            "embeds": captured["embeds"],  # [S, H]: text and image features, prompt + answer
            "logits": logits.logits[0, ids.shape[1] - 1 : -1].float().numpy(),
            "text": processor.tokenizer.decode(answer, skip_special_tokens=True),
        }
        if pil:
            with torch.no_grad():
                features = model.get_image_features(**{k: inputs[k] for k in PIXEL_INPUTS})
            features = getattr(features, "pooler_output", features)
            item["image_features"] = np.concatenate(
                [f.float().numpy().reshape(-1, f.shape[-1]) for f in features]
            )
            item["pixel_inputs"] = {k: inputs[k].numpy() for k in PIXEL_INPUTS}
        logger.info(f"reference {images}: {time.time() - start:.1f}s")
        items.append(item)
    return items


def score_decoder(onnx_dir: pathlib.Path, files: dict, refs: list[dict], eos: set[int]) -> dict:
    """The decoder fed the reference embeddings; greedy decode looks tokens up in embeddings."""
    session = load_onnx_session(onnx_dir / files["decoder"], ["CPUExecutionProvider"])
    embedding = load_onnx_session(onnx_dir / files["embedding"], ["CPUExecutionProvider"])
    forced, answers = [], []
    for ref in refs:
        n, embeds = len(ref["prompt"]), ref["embeds"][None]
        logits = cached_outputs(session, embeds[:, :-1], n)["logits"]
        forced.append(logit_metrics(ref["logits"], logits))

        def lookup(ids, embeds=embeds, n=n):
            return embeds[:, :n] if ids.shape[1] == n else embed(embedding, ids)

        answer = greedy(session, ref["prompt"], len(ref["answer"]), eos, lookup)
        answers.append(sequence_match(ref["answer"], answer, eos))
    return {
        "teacher_forced": merge_metrics(forced),
        "greedy": merge_sequences(answers),
        "size_mb": get_total_model_size_mb(onnx_dir / files["decoder"]),
    }


def score_vision(path: pathlib.Path, refs: list[dict]) -> dict:
    session = load_onnx_session(path, ["CPUExecutionProvider"])
    worst, cosines = 0.0, []
    for ref in refs:
        if "image_features" not in ref:
            continue
        feed = {
            k: v.astype(np.float32 if k == "pixel_values" else np.int64)
            for k, v in ref["pixel_inputs"].items()
        }
        want = ref["image_features"]
        got = session.run(None, feed)[0].reshape(-1, want.shape[-1])
        worst = max(worst, float(np.abs(got - want).max()))
        cosines.append(float((got * want).sum() / (np.linalg.norm(got) * np.linalg.norm(want))))
    return {"max_abs": worst, "cosine_min": min(cosines), "size_mb": get_total_model_size_mb(path)}


def score_embedding(path: pathlib.Path, refs: list[dict]) -> dict:
    """The embedding model given the reference image features, against the merged embeddings."""
    session = load_onnx_session(path, ["CPUExecutionProvider"])
    worst = 0.0
    for ref in refs:
        n = len(ref["prompt"])
        got = embed(session, ref["prompt"][None], ref.get("image_features"))[0]
        worst = max(worst, float(np.abs(got - ref["embeds"][:n]).max()))
    return {"max_abs": worst, "size_mb": get_total_model_size_mb(path)}


def score_genai(export: pathlib.Path, precision: str, refs: list[dict], eos: set[int]) -> dict:
    model = load_model(export, genai_files(precision))
    processor = model.create_multimodal_processor()
    answers, same_ids = [], 0
    for ref in refs:
        paths = [str(ASSETS / f) for f in ref["images"]]
        inputs = processor(ref["rendered"], images=og.Images.open(*paths) if paths else None)
        ids = inputs["input_ids"].as_numpy()[0]
        same_ids += ids.tolist() == ref["prompt"].tolist()
        answer = generate(model, inputs, len(ref["answer"])).get_sequence(0)[len(ids) :]
        answers.append(sequence_match(ref["answer"], answer.tolist(), eos))
    return {"greedy": merge_sequences(answers), "prompt_ids_equal": f"{same_ids}/{len(refs)}"}


def score(export: pathlib.Path, precision: str, refs: list[dict], genai: bool) -> dict:
    onnx_dir, files, eos = export / "onnx", bundle(precision), eos_ids(export)
    row = {
        "decoder": guarded(score_decoder, onnx_dir, files, refs, eos),
        "vision": guarded(score_vision, onnx_dir / files["vision"], refs),
        "embedding": guarded(score_embedding, onnx_dir / files["embedding"], refs),
    }
    if genai:
        row["genai"] = guarded(score_genai, export, precision, refs, eos)
    return row
