"""
Score an export against its reference model, precision by precision.

For every precision in the export folder:
- decoder: teacher-forced KL(reference || ONNX) over the reference answers, with the prompt
  prefilled in one call and each answer token fed through the caches after it; plus greedy
  answers on plain onnxruntime against the reference's
- vision encoder and embedding model (VL): largest difference to the reference features
- genai: greedy answers through onnxruntime-genai (audio: text and audio codes, frame by frame)

References come from PyTorch (text, MoE, VL) or liquid-audio (audio) in fp32, and are cached per
device in ~/.cache/liquidonnx/compare. --device cuda runs the references on the GPU with TF32 off,
and the ONNX sessions and onnxruntime-genai on the CUDA EP, also without TF32. Run from the
repository: the VL images and audio clips are tests/test_lfm2_vl/assets and samples/audio.

PyTorch answers greedily without the checkpoint's generation_config (LFM2.5-8B-A1B's
repetition_penalty), so each answer token is the argmax of the reference logits, as on onnxruntime.

Usage:
    uv run lfm2-compare text --model LiquidAI/LFM2.5-350M --export exports/LFM2.5-350M-ONNX
    uv run lfm2-compare moe --model LiquidAI/LFM2.5-8B-A1B --export exports/LFM2.5-8B-A1B-ONNX \\
        --prompts 4 --max-new 32 --device cuda
    uv run lfm2-compare vl --model LiquidAI/LFM2.5-VL-1.6B --export exports/LFM2.5-VL-1.6B-ONNX
    uv run lfm2-compare vl --model LiquidAI/LFM2.5-VL-1.6B --export exports/LFM2.5-VL-1.6B-ONNX \\
        --no-image-splitting
    uv run lfm2-compare audio --model LiquidAI/LFM2.5-Audio-1.5B \\
        --export exports/LFM2.5-Audio-1.5B-ONNX --precision fp32 q4

lfm2-compare wikitext scores the precisions on wikitext-2 against an fp32 reference instead, and
fails above per-model KLD ceilings (liquidonnx.compare.wikitext):
    uv run lfm2-compare wikitext --export exports/LFM2.5-350M-ONNX --reference ref.kld
"""

import argparse
import contextlib
import gc
import hashlib
import json
import logging
import os
import pathlib
import platform
import sys
import tempfile
import time

import numpy as np
import onnxruntime_genai as og

from liquidonnx.compare.metrics import guarded
from liquidonnx.genai_builder import cache_dir, resolve_checkpoint
from liquidonnx.genai_runtime import EXECUTION_PROVIDERS

logger = logging.getLogger(__name__)

MAX_NEW = {"text": 48, "moe": 48, "vl": 48, "audio": 160}


def load_family(family: str):
    if family in ("text", "moe"):
        from liquidonnx.compare import text as module
    elif family == "vl":
        from liquidonnx.compare import vl as module
    else:
        from liquidonnx.compare import audio as module
    return module


def checkpoint_revision(model: str, checkpoint: pathlib.Path) -> str:
    """The commit of a Hugging Face checkpoint; for a local one, its files' sizes and mtimes."""
    if not pathlib.Path(model).is_dir():
        return checkpoint.name
    files = sorted(f for f in checkpoint.rglob("*") if f.is_file())
    return repr(
        [
            (f.relative_to(checkpoint).as_posix(), f.stat().st_size, f.stat().st_mtime_ns)
            for f in files
        ]
    )


def cached_reference(
    model: str,
    checkpoint: pathlib.Path,
    family: str,
    prompts: int,
    max_new: int,
    device: str = "cpu",
    image_splitting: bool | None = None,
):
    """The reference answers, computed once per checkpoint revision, inputs, answer length and
    device; image_splitting (VL) overrides the checkpoint's do_image_splitting.

    Hugging Face checkpoints are keyed by commit, so a reference computed on one host can be
    copied to another. An unreadable cache file is recomputed and overwritten.
    """
    module = load_family(family)
    if family in ("text", "moe"):
        inputs = module.PROMPTS[:prompts]
    elif family == "vl":
        inputs = (module.CASES, image_splitting)
    else:
        inputs = module.CASES
    key = (checkpoint_revision(model, checkpoint), inputs, max_new)
    if family != "audio":
        key += ("argmax",)  # earlier answers took the checkpoint's repetition_penalty
    digest = hashlib.sha256(repr(key).encode()).hexdigest()[:12]
    tag = "" if device == "cpu" else f"-{device}"  # CPU references keep their pre-device names
    path = cache_dir() / "compare" / f"{pathlib.Path(model).name}-{family}{tag}-{digest}.npz"
    if path.exists():
        try:
            with np.load(path, allow_pickle=True) as cache:
                items = list(cache["items"])
        except Exception as e:  # a corrupt pickle can raise almost any type
            logger.warning(f"Recomputing the unreadable reference {path}: {e!r}")
        else:
            logger.info(f"Reference: {path}")
            return items

    logger.info(f"Running the reference model of {checkpoint} on {device}...")
    if device == "cuda":
        import torch

        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    if family in ("text", "moe"):
        items = module.reference(checkpoint, prompts, max_new, device)
    elif family == "vl":
        items = module.reference(checkpoint, max_new, device, image_splitting)
    else:
        items = module.reference(checkpoint, max_new, device)
    if device == "cuda":
        # torch keeps the freed reference model's memory cached; the ONNX sessions need it
        gc.collect()
        torch.cuda.empty_cache()
    save_reference(path, items)
    return items


@contextlib.contextmanager
def atomic_write(path: pathlib.Path):
    """A binary file renamed over path once the block completes, so no reader sees a partial
    file; on an error path stays as it was."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    tmp = pathlib.Path(name)
    try:
        with os.fdopen(fd, "wb") as f:
            yield f
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def save_reference(path: pathlib.Path, items: list) -> None:
    with atomic_write(path) as f:
        np.savez(f, items=np.array(items, dtype=object))


def _answers(result: dict) -> str:
    return f"{result['exact']}/{result['of']} ({result['prefix_frac']:.2f})"


def _genai_cell(family: str, genai: dict | None) -> str:
    if genai is None:
        return ""
    if "error" in genai:
        return "error"
    if family != "audio":
        return _answers(genai["greedy"])
    return "; ".join(
        f"{name}: {'=' if c['text']['exact'] else c['text']['prefix']}/{c['text']['ref_len']}, "
        f"{c['frames_equal']}/{c['frames']}"
        for name, c in genai.items()
    )


def report(results: dict) -> str:
    """Markdown tables of a results dict."""
    family = results["family"]
    lines = [f"### {results['model']} ({family}) on {results['host']} ({results['device']})", ""]
    if family == "audio":
        lines += [
            "| precision | KL mean | top-1 | max abs | hidden max abs | MB | genai: text, frames |",
            "|---|---|---|---|---|---|---|",
        ]
    else:
        lines += [
            "| precision | KL mean | KL max | top-1 | max abs | ORT greedy | genai greedy | MB |",
            "|---|---|---|---|---|---|---|---|",
        ]
    notes = []
    for precision, row in results["rows"].items():
        decoder, genai = row["decoder"], row.get("genai")
        if "error" in decoder:
            lines.append(f"| {precision} | error: {decoder['error'][:120]} |")
        elif family == "audio":
            forced = decoder["teacher_forced"]
            lines.append(
                f"| {precision} | {forced['kl_mean']:.2e} | {forced['top1']:.3f} "
                f"| {forced['max_abs']:.3f} | {decoder['hidden_max_abs']:.3f} "
                f"| {decoder['size_mb']:.0f} | {_genai_cell(family, genai)} |"
            )
        else:
            forced = decoder["teacher_forced"]
            lines.append(
                f"| {precision} | {forced['kl_mean']:.2e} | {forced['kl_max']:.2e} "
                f"| {forced['top1']:.3f} | {forced['max_abs']:.3f} | {_answers(decoder['greedy'])} "
                f"| {_genai_cell(family, genai)} | {decoder['size_mb']:.0f} |"
            )
        if genai and "error" in genai:
            notes.append(f"- {precision} genai: error: {genai['error'][:200]}")
        elif genai and "prompt_ids_equal" in genai:
            notes.append(f"- {precision} genai: prompt ids equal {genai['prompt_ids_equal']}")
        for part in ("vision", "embedding"):
            value = row.get(part)
            if value is None:
                continue
            if "error" in value:
                notes.append(f"- {precision} {part}: error: {value['error'][:120]}")
                continue
            cosine = f", min cosine {value['cosine_min']:.6f}" if "cosine_min" in value else ""
            notes.append(
                f"- {precision} {part}: max abs {value['max_abs']:.4f}{cosine}, "
                f"{value['size_mb']:.0f} MB"
            )
    return "\n".join(lines + ([""] + notes if notes else [])) + "\n"


def has_errors(value) -> bool:
    if isinstance(value, dict):
        return "error" in value or any(has_errors(v) for v in value.values())
    return False


def main():
    from liquidonnx.compare import wikitext

    parser = argparse.ArgumentParser(
        description="Score an export against its reference model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    commands = parser.add_subparsers(dest="family", required=True)
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--model", required=True, help="Reference checkpoint (HF ID or path)")
    shared.add_argument("--export", required=True, type=pathlib.Path, help="Export folder")
    shared.add_argument(
        "--precision", nargs="*", help="Precisions to score (default: every one exported)"
    )
    shared.add_argument("--prompts", type=int, default=8, help="Text prompts (text, moe)")
    shared.add_argument(
        "--max-new",
        type=int,
        help="Reference answer length (default: "
        f"{', '.join(f'{f} {n}' for f, n in MAX_NEW.items())})",
    )
    shared.add_argument("--no-genai", action="store_true", help="Skip onnxruntime-genai")
    shared.add_argument(
        "--device",
        choices=EXECUTION_PROVIDERS,
        default="cpu",
        help="Device of the reference, the ONNX sessions and onnxruntime-genai; cuda turns TF32 "
        "off (default: cpu)",
    )
    shared.add_argument(
        "--output",
        type=pathlib.Path,
        help="Results JSON; a markdown report is written next to it "
        "(default: compare-{export name}.json)",
    )
    for family in ("text", "moe", "vl", "audio"):
        family_parser = commands.add_parser(
            family, parents=[shared], help=f"{family} export against its model"
        )
        if family == "vl":
            family_parser.add_argument(
                "--image-splitting",
                action=argparse.BooleanOptionalAction,
                help="Split large images into tiles plus a thumbnail in the reference and genai "
                "(default: the checkpoint's do_image_splitting, on for LFM2-VL and LFM2.5-VL)",
            )
    wikitext_parser = commands.add_parser(
        "wikitext",
        help="wikitext-2 KLD gate (olive-recipes #638's protocol)",
        description=wikitext.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    wikitext.add_arguments(wikitext_parser)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    if args.family == "wikitext":
        wikitext.run(args, wikitext_parser)
        return

    module = load_family(args.family)
    available = module.precisions(args.export)
    precisions = args.precision or available
    missing = [p for p in precisions if p not in available]
    if missing:
        parser.error(f"{args.export} has no {', '.join(missing)}; it has {', '.join(available)}")

    checkpoint = resolve_checkpoint(args.model)
    max_new = args.max_new or MAX_NEW[args.family]
    options = {"image_splitting": args.image_splitting} if args.family == "vl" else {}
    refs = cached_reference(
        args.model, checkpoint, args.family, args.prompts, max_new, args.device, **options
    )

    results = {
        "model": args.model,
        "revision": checkpoint.name,
        "family": args.family,
        "host": platform.node().split(".")[0],
        "device": args.device,
        "export": str(args.export),
        "genai": og.__version__,
        "rows": {},
    }
    # genai loads a second onnxruntime that shares the CUDA provider library; once it holds a CUDA
    # model, new QMoE sessions of the Python module fail to prepack, so onnxruntime scores run first.
    rows = results["rows"]
    for precision in precisions:
        start = time.time()
        if args.family == "audio":
            rows[precision] = module.score(
                args.export, precision, refs, False, max_new, args.device
            )
        else:
            rows[precision] = module.score(args.export, precision, refs, False, args.device)
        rows[precision]["seconds"] = time.time() - start
    if not args.no_genai:
        last = max_new if args.family == "audio" else module.eos_ids(args.export)
        for precision in precisions:
            start = time.time()
            rows[precision]["genai"] = guarded(
                module.score_genai, args.export, precision, refs, last, args.device
            )
            rows[precision]["seconds"] += time.time() - start
    for precision, row in rows.items():
        logger.info(f"{precision}: {json.dumps(row, default=str)}")

    output = args.output or pathlib.Path(f"compare-{args.export.name}.json")
    output.write_text(json.dumps(results, indent=2, default=str))
    markdown = report(results)
    output.with_suffix(".md").write_text(markdown)
    logger.info(f"Results: {output}, {output.with_suffix('.md')}\n{markdown}")
    if has_errors(results["rows"]):
        sys.exit(1)
