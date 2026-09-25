"""
Score an export against its reference model, precision by precision.

For every precision in the export folder:
- decoder: teacher-forced KL(reference || ONNX) over the reference answers, with the prompt
  prefilled in one call and each answer token fed through the caches after it; plus greedy
  answers on plain onnxruntime against the reference's
- vision encoder and embedding model (VL): largest difference to the reference features
- genai: greedy answers through onnxruntime-genai (audio: text and audio codes, frame by frame)

References come from PyTorch (text, MoE, VL) or liquid-audio (audio), in fp32 on CPU, and are
cached in ~/.cache/liquidonnx/compare. Run from the repository: the VL images and audio clips are
tests/test_lfm2_vl/assets and samples/audio.

Usage:
    uv run lfm2-compare text --model LiquidAI/LFM2.5-350M --export exports/LFM2.5-350M-ONNX
    uv run lfm2-compare moe --model LiquidAI/LFM2.5-8B-A1B --export exports/LFM2.5-8B-A1B-ONNX \\
        --prompts 4 --max-new 32
    uv run lfm2-compare vl --model LiquidAI/LFM2.5-VL-1.6B --export exports/LFM2.5-VL-1.6B-ONNX
    uv run lfm2-compare audio --model LiquidAI/LFM2.5-Audio-1.5B \\
        --export exports/LFM2.5-Audio-1.5B-ONNX --precision fp32 q4
"""

import argparse
import hashlib
import json
import logging
import pathlib
import platform
import sys
import time

import numpy as np
import onnxruntime_genai as og

from liquidonnx.genai_builder import cache_dir, resolve_checkpoint

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


def cached_reference(model: str, checkpoint: pathlib.Path, family: str, prompts: int, max_new: int):
    """The reference answers, computed once per checkpoint revision, inputs and answer length.

    Hugging Face checkpoints are keyed by commit, so a reference computed on one host can be
    copied to another.
    """
    module = load_family(family)
    inputs = module.PROMPTS[:prompts] if family in ("text", "moe") else module.CASES
    key = repr((checkpoint_revision(model, checkpoint), inputs, max_new))
    digest = hashlib.sha256(key.encode()).hexdigest()[:12]
    path = cache_dir() / "compare" / f"{pathlib.Path(model).name}-{family}-{digest}.npz"
    if path.exists():
        logger.info(f"Reference: {path}")
        return list(np.load(path, allow_pickle=True)["items"])

    logger.info(f"Running the reference model of {checkpoint}...")
    if family in ("text", "moe"):
        items = module.reference(checkpoint, prompts, max_new)
    else:
        items = module.reference(checkpoint, max_new)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, items=np.array(items, dtype=object))
    return items


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
    lines = [f"### {results['model']} ({family}) on {results['host']}", ""]
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
    parser = argparse.ArgumentParser(
        description="Score an export against its reference model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("family", choices=["text", "moe", "vl", "audio"])
    parser.add_argument("--model", required=True, help="Reference checkpoint (HF ID or path)")
    parser.add_argument("--export", required=True, type=pathlib.Path, help="Export folder")
    parser.add_argument(
        "--precision", nargs="*", help="Precisions to score (default: every one exported)"
    )
    parser.add_argument("--prompts", type=int, default=8, help="Text prompts (text, moe)")
    parser.add_argument(
        "--max-new",
        type=int,
        help="Reference answer length (default: "
        f"{', '.join(f'{f} {n}' for f, n in MAX_NEW.items())})",
    )
    parser.add_argument("--no-genai", action="store_true", help="Skip onnxruntime-genai")
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        help="Results JSON; a markdown report is written next to it "
        "(default: compare-{export name}.json)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    module = load_family(args.family)
    available = module.precisions(args.export)
    precisions = args.precision or available
    missing = [p for p in precisions if p not in available]
    if missing:
        parser.error(f"{args.export} has no {', '.join(missing)}; it has {', '.join(available)}")

    checkpoint = resolve_checkpoint(args.model)
    max_new = args.max_new or MAX_NEW[args.family]
    refs = cached_reference(args.model, checkpoint, args.family, args.prompts, max_new)

    results = {
        "model": args.model,
        "revision": checkpoint.name,
        "family": args.family,
        "host": platform.node().split(".")[0],
        "export": str(args.export),
        "genai": og.__version__,
        "rows": {},
    }
    for precision in precisions:
        start = time.time()
        if args.family == "audio":
            row = module.score(args.export, precision, refs, not args.no_genai, max_new)
        else:
            row = module.score(args.export, precision, refs, not args.no_genai)
        row["seconds"] = time.time() - start
        results["rows"][precision] = row
        logger.info(f"{precision}: {json.dumps(row, default=str)}")

    output = args.output or pathlib.Path(f"compare-{args.export.name}.json")
    output.write_text(json.dumps(results, indent=2, default=str))
    markdown = report(results)
    output.with_suffix(".md").write_text(markdown)
    logger.info(f"Results: {output}, {output.with_suffix('.md')}\n{markdown}")
    if has_errors(results["rows"]):
        sys.exit(1)
