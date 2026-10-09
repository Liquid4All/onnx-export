"""
Wikitext-2 KL divergence of an export's precisions from an fp32 reference, with the protocol of
microsoft/olive-recipes#638, so the numbers line up with `llama-perplexity --kl-divergence` and
the recipes' published tables:

- the wikitext-2 test split as `llama-perplexity -c 512 --chunks 64` cuts it, with BOS in place of
  the first token of every chunk
- each chunk is prefilled in one call and its second half is scored: 64 x 255 = 16,320 tokens
- KLD from the reference over the reference's top 16 nats, as mean ± SE over the chunk means
  (tokens of one chunk are correlated), and how often the top tokens agree (same top)

The reference is a llama.cpp KL-divergence base file. --reference names one, such as #638's
make_ref.py writes; its .json sidecar gives the BOS, and without one the export's bos_token_id is
used if the stream starts with it. --model computes one instead: the Hugging Face model in fp32 on
--reference-device (cuda: TF32 off), cached in ~/.cache/liquidonnx/compare. Its token stream comes
from --tokens, any base file of the model (e.g. `llama-perplexity --kl-divergence-base` of a GGUF),
or from --text, wiki.test.raw tokenized as llama-perplexity does: BOS, then the text's tokens. The
Hugging Face tokenizers give exactly llama.cpp 4e7481175's stream for LFM2.5-350M, 1.2B-Instruct,
2.6B, 8B-A1B and VL-450M; for other models --tokens is the safe choice. Audio references need
liquid-audio, so audio exports take --reference only.

A precision fails, and the command exits with 1, when its KLD is above --max-kld or, without it,
above its ceiling in BASELINES for --device: the CPU ceilings cover q4 and q4f32, the CUDA ones q8,
fp16 and q4f16, and each device scores those by default. A precision without a ceiling on its
device is reported as "no ceiling", not as a pass. The ONNX sessions run a fixed number of
intra-op threads (--threads) on every machine, as the scores of MoE models depend on it.

Usage:
    uv run lfm2-compare wikitext --export exports/LFM2.5-350M-ONNX \\
        --reference ref/LFM2.5-350M/ref.kld
    uv run lfm2-compare wikitext --export exports/LFM2.5-350M-ONNX --model LiquidAI/LFM2.5-350M \\
        --text wikitext-2-raw/wiki.test.raw --reference-device cuda
    uv run lfm2-compare wikitext --export exports/LFM2.5-350M-ONNX \\
        --reference ref/LFM2.5-350M/ref.kld --device cuda
"""

import argparse
import gc
import hashlib
import json
import logging
import pathlib
import platform
import sys
import time
from dataclasses import dataclass

import numpy as np
import onnxruntime as ort

from liquidonnx.compare import atomic_write, checkpoint_revision, kld_base
from liquidonnx.compare.metrics import guarded
from liquidonnx.embeddings import embed
from liquidonnx.genai_builder import cache_dir, resolve_checkpoint
from liquidonnx.genai_runtime import EXECUTION_PROVIDERS
from liquidonnx.lfm2.export import model_file
from liquidonnx.lfm2_audio import export as audio_export
from liquidonnx.lfm2_vl import export as vl_export
from liquidonnx.quantize import get_total_model_size_mb
from liquidonnx.session import decoder_inputs, initialize_cache, load_onnx_session

logger = logging.getLogger(__name__)

PRECISIONS = ("fp32", "fp16", "q4", "q4f16", "q4f32", "q8")
# Each device scores the precisions it has ceilings for unless --precision says otherwise.
DEVICE_PRECISIONS = {"cpu": ("q4", "q4f32"), "cuda": ("q8", "fp16", "q4f16")}
CHUNKS = 64  # llama-perplexity --chunks
BATCH = 4  # chunks per forward pass of the reference model, as make_ref.py
# onnxruntime's QMoE CPU kernel sums the experts in an order that depends on the intra-op thread
# count, so MoE scores move with the machine's core count unless the count is fixed
THREADS = 13

# KLD ± SE per device against H100 fp32 references: on the CPU EP of the locked onnxruntime
# nightly, and on the H100's CUDA EP with TF32 off (onnxruntime-gpu 1.31.0). A row gates its own
# device only: q8 scores 1.5x to 3.7x higher on the CPU, so a CPU ceiling would let a CUDA q8 grow
# 1.7x to 4x unnoticed. A precision fails above KLD + 2 SE. The text, MoE and VL q4 rows, and the
# q4f16 rows converted from them, are the genai builder's k_quant build on the locked onnxruntime
# nightly, which lacks onnxruntime#32814: with it, q4 scores 0.467 / 0.106 / 0.146 / 0.187 on
# 350M / 1.2B / 2.6B / 8B. The other rows do not depend on #32814. The 8B-A1B q4 and q8 rows hold
# at THREADS (13) intra-op threads only (the CUDA EP leaves q8's int8 QMoE nodes to the CPU EP):
# at 12 threads q4 scores +0.17%, at 4 CUDA q8 -0.3%.
BASELINES = {
    "cpu": {
        "LFM2.5-350M": {"q4": (0.4896, 0.0120), "q4f32": (0.7908, 0.0182)},
        "LFM2.5-1.2B-Instruct": {"q4": (0.1537, 0.0131), "q4f32": (0.1584, 0.0107)},
        "LFM2.5-2.6B": {"q4": (0.1657, 0.0041), "q4f32": (0.2310, 0.0051)},
        "LFM2.5-8B-A1B": {"q4": (0.2103, 0.0060)},
        "LFM2.5-VL-450M": {"q4": (0.0643, 0.0014)},
        "LFM2.5-Audio-1.5B": {"q4": (0.03209, 0.00056)},
    },
    "cuda": {
        "LFM2.5-350M": {
            "q8": (0.002742, 0.0000619),
            "fp16": (4.372e-5, 9.6e-7),
            "q4f16": (0.4860, 0.0119),
        },
        "LFM2.5-1.2B-Instruct": {
            "q8": (0.0008429, 0.0000858),
            "fp16": (1.636e-5, 3.9e-7),
            "q4f16": (0.1524, 0.0134),
        },
        "LFM2.5-2.6B": {
            "q8": (0.0009020, 0.0000365),
            "fp16": (2.154e-5, 8.1e-7),
            "q4f16": (0.1638, 0.0040),
        },
        "LFM2.5-8B-A1B": {
            "q8": (0.01824, 0.00077),
            # not deterministic on CUDA (0.00513 to 0.00565 over 8 runs): the worst run
            "fp16": (0.00565, 0.000400),
            # with weights_prepacked=0 on the QMoE nodes: without it the CUDA EP misreads the experts
            "q4f16": (0.2022, 0.0053),
        },
        "LFM2.5-VL-450M": {"q8": (0.0003829, 0.0000093), "fp16": (1.315e-5, 2.6e-7)},
        "LFM2.5-Audio-1.5B": {"q8": (0.0001238, 0.0000018), "fp16": (5.444e-6, 1.12e-7)},
    },
}


@dataclass
class Reference:
    """A llama.cpp KL-divergence base file; bos replaces the first token of every chunk."""

    path: pathlib.Path
    n_ctx: int
    n_vocab: int
    tokens: np.ndarray  # [n_chunk, n_ctx]
    rows: np.ndarray  # [n_chunk, scored rows, row width]
    bos: int | None
    meta: dict

    def input_ids(self, chunk: int) -> np.ndarray:
        """[1, n_ctx] ids of a chunk, as llama-perplexity feeds them."""
        ids = self.tokens[chunk].astype(np.int64)[None].copy()
        if self.bos is not None:
            ids[0, 0] = self.bos
        return ids


def load_reference(path: pathlib.Path, bos_token_id: int | None) -> Reference:
    n_ctx, n_vocab, tokens, rows = kld_base.open_rows(path)
    sidecar = path.with_name(f"{path.name}.json")
    meta = json.loads(sidecar.read_text()) if sidecar.exists() else {}
    if "add_bos" in meta:
        bos = meta["bos"] if meta["add_bos"] else None
    else:
        # make_ref.py's rule: llama-perplexity added a BOS if the stream starts with one
        bos = bos_token_id if int(tokens[0, 0]) == bos_token_id else None
    return Reference(path, n_ctx, n_vocab, tokens, rows, bos, meta)


def text_tokens(tokenizer, path: pathlib.Path, n_ctx: int, n_chunk: int) -> np.ndarray:
    """llama-perplexity's stream of a text file: BOS, then the text's tokens, in chunks."""
    if tokenizer.bos_token_id is None:
        raise ValueError("the tokenizer has no BOS token; pass --tokens instead of --text")
    text = path.read_bytes().decode("utf-8")  # as read by llama.cpp: no newline translation
    ids = [tokenizer.bos_token_id, *tokenizer.encode(text, add_special_tokens=False)]
    if len(ids) < n_ctx * n_chunk:
        raise ValueError(f"{path} has {len(ids)} tokens, fewer than {n_chunk} chunks of {n_ctx}")
    return np.array(ids[: n_ctx * n_chunk], np.int32).reshape(n_chunk, n_ctx)


def compute_reference(
    checkpoint: pathlib.Path,
    tokens: np.ndarray,
    n_vocab: int | None,
    vl: bool,
    device: str,
    path: pathlib.Path,
) -> dict:
    """Write the fp32 model's log-probabilities on tokens to path, as #638's make_ref.py does.

    Returns the BOS fields of the sidecar. n_vocab None takes the model's vocabulary size.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

    bos = AutoTokenizer.from_pretrained(checkpoint).bos_token_id
    add_bos = int(tokens[0, 0]) == bos
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    cls = AutoModelForImageTextToText if vl else AutoModelForCausalLM
    model = cls.from_pretrained(checkpoint, dtype=torch.float32).to(device).eval()
    n_vocab = n_vocab or model.config.get_text_config().vocab_size
    n_ctx, first, start = tokens.shape[1], tokens.shape[1] // 2, time.time()
    logger.info(f"Reference of {checkpoint} on {device}: {len(tokens)} chunks of {n_ctx}")
    with atomic_write(path) as f, torch.inference_mode():
        kld_base.write_header(f, n_ctx, n_vocab, tokens)
        for begin in range(0, len(tokens), BATCH):
            ids = torch.from_numpy(tokens[begin : begin + BATCH].astype(np.int64))
            if add_bos:
                ids[:, 0] = bos
            logits = model(input_ids=ids.to(device)).logits[:, first : n_ctx - 1, :n_vocab]
            if logits.shape[-1] != n_vocab:
                raise ValueError(f"{checkpoint} has {logits.shape[-1]} logits, not {n_vocab}")
            for chunk in logits.float().cpu().numpy():
                f.write(kld_base.encode_rows(chunk, n_vocab).tobytes())
            logger.info(f"chunks {begin + len(ids)}/{len(tokens)} ({time.time() - start:.0f}s)")
    del model
    if device == "cuda":
        # torch keeps the freed model's memory cached; the ONNX sessions need it
        gc.collect()
        torch.cuda.empty_cache()
    return {"add_bos": add_bos, "bos": bos}


def cached_reference(
    model: str,
    checkpoint: pathlib.Path,
    tokens: np.ndarray,
    n_vocab: int | None,
    source: pathlib.Path,
    vl: bool,
    device: str,
) -> pathlib.Path:
    """The reference of a checkpoint on a token stream, computed once per revision and device."""
    stream = hashlib.sha256(tokens.tobytes()).hexdigest()
    key = repr((checkpoint_revision(model, checkpoint), n_vocab, tokens.shape, stream))
    digest = hashlib.sha256(key.encode()).hexdigest()[:12]
    tag = "" if device == "cpu" else f"-{device}"
    path = cache_dir() / "compare" / f"{pathlib.Path(model).name}-wikitext{tag}-{digest}.kld"
    sidecar = path.with_name(f"{path.name}.json")
    if path.exists() and sidecar.exists():
        try:
            kld_base.open_rows(path)
        except ValueError as e:
            logger.warning(f"Recomputing the unreadable reference {path}: {e}")
        else:
            logger.info(f"Reference: {path}")
            return path

    meta = {
        "model": model,
        "revision": checkpoint.name,
        "dtype": "float32",
        "device": device,
        "tokens": str(source),
    }
    meta.update(compute_reference(checkpoint, tokens, n_vocab, vl, device, path))
    with atomic_write(sidecar) as f:
        f.write(json.dumps(meta).encode())
    logger.info(f"Reference: {path}")
    return path


def export_files(export: pathlib.Path, config: dict, precision: str) -> dict[str, pathlib.Path]:
    """The decoder of a precision and, for VL and audio, the embedding model that feeds it."""
    if "embedding" not in config:
        return {"decoder": export / "onnx" / model_file(precision)}
    bundle = (audio_export if "speech" in config else vl_export).bundle(precision)
    return {part: export / "onnx" / bundle[part] for part in ("decoder", "embedding")}


def score(
    files: dict[str, pathlib.Path],
    reference: Reference,
    chunks: int,
    device: str,
    threads: int,
    dump: pathlib.Path | None,
) -> dict:
    """#638's eval_onnx.py statistics of one precision over the first chunks of the reference;
    dump receives the per-token values (.npz)."""
    decoder = load_onnx_session(files["decoder"], device, tf32=False, threads=threads)
    embedding = None
    if "embedding" in files:
        embedding = load_onnx_session(files["embedding"], device, tf32=False, threads=threads)
    n_ctx, n_vocab = reference.n_ctx, reference.n_vocab
    first, stats, start = n_ctx // 2, kld_base.KldStats(), time.time()
    for c in range(chunks):
        ids = reference.input_ids(c)
        inputs = embed(embedding, ids) if embedding else ids
        feed = decoder_inputs(inputs, initialize_cache(decoder), 0)
        logits = decoder.run(["logits"], feed)[0][0, first : n_ctx - 1, :n_vocab]
        base = kld_base.decode_rows(np.asarray(reference.rows[c]), n_vocab)
        stats.add(logits.astype(np.float32), base, reference.tokens[c, first + 1 :])
        if (c + 1) % 8 == 0 or c + 1 == chunks:
            logger.info(f"chunks {c + 1}/{chunks} ({time.time() - start:.0f}s)")
    if dump:
        dump.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dump, **stats.arrays())
    return {
        **stats.summary(),
        "seconds": time.time() - start,
        "size_mb": sum(get_total_model_size_mb(path) for path in files.values()),
    }


def ceiling(names: list[str], device: str, precision: str, max_kld: float | None) -> float | None:
    """--max-kld, else the first of names with a baseline for the precision on the device:
    KLD + 2 SE."""
    if max_kld is not None:
        return max_kld
    models = BASELINES.get(device, {})
    for name in names:
        if precision in models.get(name, {}):
            kld, se = models[name][precision]
            return kld + 2 * se
    return None


def report(results: dict) -> str:
    reference = results["reference"]
    lines = [
        f"### {results['export']}: wikitext-2 KLD on {results['host']} "
        f"({results['device']}, {results['threads']} threads)",
        "",
        f"Reference {reference['path']} ({reference.get('model', 'model unknown')}), "
        f"{reference['chunks']} chunks of {reference['n_ctx']} tokens",
        "",
        "| precision | KLD ± SE | same top | ceiling | MB | s | gate |",
        "|---|---|---|---|---|---|---|",
    ]
    for precision, row in results["rows"].items():
        if "error" in row:
            lines.append(f"| {precision} | error: {row['error'][:120]} | | | | | FAIL |")
            continue
        limit = "—" if row["ceiling"] is None else f"{row['ceiling']:.4g}"
        gate = {None: "no ceiling", True: "pass", False: "FAIL"}[row["pass"]]
        lines.append(
            f"| {precision} | {row['kld_mean']:.4g} ± {row['kld_mean_err_chunks']:.2g} "
            f"| {row['same_top_pct']:.2f}% | {limit} | {row['size_mb']:.0f} "
            f"| {row['seconds']:.0f} | {gate} |"
        )
    return "\n".join(lines) + "\n"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--export", required=True, type=pathlib.Path, help="Export folder")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--reference", type=pathlib.Path, help="llama.cpp KL-divergence base file to score against"
    )
    source.add_argument(
        "--model", help="Compute the reference from this checkpoint (HF ID or path)"
    )
    stream = parser.add_mutually_exclusive_group()
    stream.add_argument(
        "--tokens",
        type=pathlib.Path,
        help="With --model: llama.cpp KL-divergence base file whose token stream to score",
    )
    stream.add_argument(
        "--text",
        type=pathlib.Path,
        help="With --model: text to tokenize as llama-perplexity does (wiki.test.raw)",
    )
    parser.add_argument(
        "--ctx", type=int, default=512, help="With --text: tokens per chunk (default: 512)"
    )
    parser.add_argument(
        "--chunks", type=int, help=f"Score the first N chunks (default: all; {CHUNKS} with --text)"
    )
    defaults = "; ".join(f"{device}: {' '.join(p)}" for device, p in DEVICE_PRECISIONS.items())
    parser.add_argument(
        "--precision",
        nargs="+",
        choices=PRECISIONS,
        help=f"Precisions to score (default: the exported ones of {defaults})",
    )
    parser.add_argument(
        "--device",
        choices=EXECUTION_PROVIDERS,
        default="cpu",
        help="Execution provider of the ONNX sessions; cuda turns TF32 off (default: cpu)",
    )
    parser.add_argument(
        "--reference-device",
        choices=EXECUTION_PROVIDERS,
        help="Device of the fp32 model of --model; cuda turns TF32 off (default: --device)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=THREADS,
        help=f"Intra-op threads of the ONNX sessions; 0 is one per physical core (default: "
        f"{THREADS}, the count of the ceilings)",
    )
    parser.add_argument(
        "--max-kld",
        type=float,
        help="Fail any precision above this KLD (default: the model's ceilings on --device, if "
        "known)",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        help="Results JSON; a markdown report is written next to it "
        "(default: wikitext-{export name}.json)",
    )
    parser.add_argument(
        "--dump",
        type=pathlib.Path,
        help="Write each precision's per-token values to {export}.{precision}.{device}.npz here, "
        "for #638's compare.py",
    )


def run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.model and not (args.tokens or args.text):
        parser.error("--model needs --tokens or --text for the token stream")
    if args.reference and (args.tokens or args.text):
        parser.error("--tokens and --text go with --model; --reference has its own tokens")
    if args.reference and not args.reference.exists():
        parser.error(f"{args.reference} does not exist")

    config = json.loads((args.export / "genai_config.json").read_text())["model"]
    family = PRECISIONS
    if "embedding" in config:
        family = ("fp32", *(audio_export if "speech" in config else vl_export).PRECISIONS)
    exported = [
        p for p in family if all(f.exists() for f in export_files(args.export, config, p).values())
    ]
    if args.precision:
        precisions = args.precision
        missing = [p for p in precisions if p not in exported]
        if missing:
            parser.error(f"{args.export} has no {', '.join(missing)}; it has {', '.join(exported)}")
    else:
        precisions = [p for p in DEVICE_PRECISIONS[args.device] if p in exported]
        if not precisions:
            parser.error(
                f"{args.export} has none of {', '.join(DEVICE_PRECISIONS[args.device])} "
                f"({args.device}); pick precisions with --precision"
            )

    if args.model:
        if "speech" in config:
            parser.error("audio references need liquid-audio: pass --reference")
        checkpoint = resolve_checkpoint(args.model)
        if args.tokens:
            _, n_vocab, tokens, _ = kld_base.read_header(args.tokens)
            tokens = tokens[: args.chunks] if args.chunks else tokens
        else:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(checkpoint)
            tokens = text_tokens(tokenizer, args.text, args.ctx, args.chunks or CHUNKS)
            n_vocab = None
        path = cached_reference(
            args.model,
            checkpoint,
            tokens,
            n_vocab,
            args.tokens or args.text,
            "vision" in config,
            args.reference_device or args.device,
        )
    else:
        path = args.reference
    reference = load_reference(path, config.get("bos_token_id"))
    chunks = min(args.chunks or len(reference.tokens), len(reference.tokens))
    names = [pathlib.Path(reference.meta["model"]).name] if "model" in reference.meta else []
    names.append(args.export.name.removesuffix("-ONNX"))

    results = {
        "export": str(args.export),
        "host": platform.node().split(".")[0],
        "device": args.device,
        "threads": args.threads,
        "onnxruntime": ort.__version__,
        "reference": {
            **reference.meta,
            "path": str(path),
            "bos": reference.bos,
            "n_ctx": reference.n_ctx,
            "n_vocab": reference.n_vocab,
            "chunks": chunks,
        },
        "rows": {},
    }
    for precision in precisions:
        files = export_files(args.export, config, precision)
        dump = (
            args.dump / f"{args.export.name}.{precision}.{args.device}.npz" if args.dump else None
        )
        row = guarded(score, files, reference, chunks, args.device, args.threads, dump)
        if "error" not in row:
            row["ceiling"] = ceiling(names, args.device, precision, args.max_kld)
            row["pass"] = None if row["ceiling"] is None else row["kld_mean"] <= row["ceiling"]
            if row["ceiling"] is None:
                known = ", ".join(dict.fromkeys(names))
                logger.warning(f"{precision}: no {args.device} ceiling for {known}; not gated")
        results["rows"][precision] = row
        logger.info(f"{precision}: {json.dumps(row)}")

    output = args.output or pathlib.Path(f"wikitext-{args.export.name}.json")
    output.write_text(json.dumps(results, indent=2))
    markdown = report(results)
    output.with_suffix(".md").write_text(markdown)
    logger.info(f"Results: {output}, {output.with_suffix('.md')}\n{markdown}")
    if any("error" in row or row["pass"] is False for row in results["rows"].values()):
        sys.exit(1)
