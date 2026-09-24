"""
Build LFM2 decoders with the onnxruntime-genai model builder.

LFM2-MoE support landed in the builder after onnxruntime-genai 0.16.0, so the builder runs from
a pinned source checkout (fetched once into ~/.cache/liquidonnx) until a release carries it.
Set LIQUIDONNX_GENAI_BUILDER to a local `src/python/py/models` directory to use another copy.

The decoder is always built as fp32 for the CPU EP; every other precision is derived from it by
liquidonnx.quantize.
"""

import logging
import os
import pathlib
import shutil
import subprocess
import sys

logger = logging.getLogger(__name__)

GENAI_REPO = "https://github.com/microsoft/onnxruntime-genai.git"
GENAI_COMMIT = "9e83936110761f7c3f85d4c2db62db2bf5246969"
BUILDER_SUBDIR = "src/python/py/models"


def cache_dir() -> pathlib.Path:
    base = os.environ.get("XDG_CACHE_HOME") or pathlib.Path.home() / ".cache"
    return pathlib.Path(base) / "liquidonnx"


def builder_dir() -> pathlib.Path:
    """Directory holding the genai builder.py, fetching the pinned commit on first use."""
    override = os.environ.get("LIQUIDONNX_GENAI_BUILDER")
    if override:
        return pathlib.Path(override)

    checkout = cache_dir() / f"onnxruntime-genai-{GENAI_COMMIT[:12]}"
    models = checkout / BUILDER_SUBDIR
    if (models / "builder.py").exists():
        return models

    logger.info(f"Fetching onnxruntime-genai model builder @ {GENAI_COMMIT[:12]}...")
    staging = checkout.with_name(checkout.name + ".partial")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    for args in (
        ["init", "-q"],
        ["remote", "add", "origin", GENAI_REPO],
        ["sparse-checkout", "set", BUILDER_SUBDIR],
        ["fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", GENAI_COMMIT],
        ["checkout", "-q", "FETCH_HEAD"],
    ):
        subprocess.run(["git", "-C", str(staging), *args], check=True)
    staging.rename(checkout)
    return models


def resolve_checkpoint(model: str) -> pathlib.Path:
    """Local directory of a checkpoint given as a path or a Hugging Face model ID."""
    path = pathlib.Path(model)
    if path.is_dir():
        return path

    from huggingface_hub import snapshot_download

    return pathlib.Path(snapshot_download(model))


def build_decoder(
    model: str, output_dir: pathlib.Path, extra_options: dict[str, str] | None = None
) -> pathlib.Path:
    """Run the genai builder (fp32, CPU EP) and return the path of the generated model.onnx.

    output_dir also receives genai_config.json and the tokenizer files the builder writes.
    """
    checkpoint = resolve_checkpoint(model)
    cmd = [
        sys.executable,
        str(builder_dir() / "builder.py"),
        "-i",
        str(checkpoint),
        "-o",
        str(output_dir),
        "-p",
        "fp32",
        "-e",
        "cpu",
        "-c",
        str(cache_dir() / "genai-builder-cache"),
    ]
    if extra_options:
        cmd += ["--extra_options", *[f"{k}={v}" for k, v in extra_options.items()]]

    logger.info(f"Building decoder with the onnxruntime-genai builder: {checkpoint}")
    subprocess.run(cmd, check=True)
    return output_dir / "model.onnx"
