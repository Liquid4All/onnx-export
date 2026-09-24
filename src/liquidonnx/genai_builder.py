"""
Build LFM2 decoders (text, MoE, VL, audio) with the onnxruntime-genai model builder.

LFM2-MoE support landed in the builder after onnxruntime-genai 0.16.0, so the builder runs from
a pinned source checkout (fetched once into ~/.cache/liquidonnx) until a release carries it.
Set LIQUIDONNX_GENAI_BUILDER to a local `src/python/py/models` directory to use another copy.

The decoder is always built as fp32 for the CPU EP; every other precision is derived from it by
liquidonnx.quantize.
"""

import json
import logging
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

import onnx

from liquidonnx.quantize import get_total_model_size_mb, load_model, save_model

logger = logging.getLogger(__name__)

GENAI_REPO = "https://github.com/microsoft/onnxruntime-genai.git"
GENAI_COMMIT = "9e83936110761f7c3f85d4c2db62db2bf5246969"
BUILDER_SUBDIR = "src/python/py/models"
CHECKPOINT_FILES = ("config.json", "generation_config.json")


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
    checkout.parent.mkdir(parents=True, exist_ok=True)
    staging = pathlib.Path(tempfile.mkdtemp(dir=checkout.parent, prefix=f"{checkout.name}."))
    try:
        for args in (
            ["init", "-q"],
            ["remote", "add", "origin", GENAI_REPO],
            ["sparse-checkout", "set", BUILDER_SUBDIR],
            ["fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", GENAI_COMMIT],
            ["checkout", "-q", "FETCH_HEAD"],
        ):
            subprocess.run(["git", "-C", str(staging), *args], check=True)
        staging.rename(checkout)
    except OSError:
        # A concurrent export completed the same checkout first.
        if not (models / "builder.py").exists():
            raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
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


def pin_kv_head_size(model: onnx.ModelProto, head_size: int):
    """Replace the builder's symbolic `kv_cache_dim` with the head size on the cache I/O.

    genai leaves it symbolic so one graph can serve quantized KV caches; fixing it lets plain
    onnxruntime callers allocate empty caches from the graph signature.
    """
    for value in [*model.graph.input, *model.graph.output]:
        for dim in value.type.tensor_type.shape.dim:
            if dim.dim_param == "kv_cache_dim":
                dim.Clear()
                dim.dim_value = head_size


def export_decoder(
    model: str,
    output_dir: pathlib.Path,
    filename: str = "model.onnx",
    extra_options: dict[str, str] | None = None,
) -> pathlib.Path:
    """Build the fp32 decoder into output_dir/onnx/filename.

    output_dir also receives the builder's genai_config.json and tokenizer files, and the
    checkpoint's config.json and generation_config.json.
    """
    onnx_dir = output_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=output_dir, prefix=".genai-build-") as tmp:
        build_dir = pathlib.Path(tmp)
        built = build_decoder(model, build_dir, extra_options)

        genai_config = json.loads((build_dir / "genai_config.json").read_text())
        decoder = load_model(built)
        pin_kv_head_size(decoder, genai_config["model"]["decoder"]["head_size"])
        output_path = save_model(decoder, onnx_dir / filename)
        del decoder

        for f in build_dir.iterdir():
            if f.is_file() and not f.name.startswith("model.onnx"):
                shutil.copy2(f, output_dir / f.name)

    checkpoint = resolve_checkpoint(model)
    for name in CHECKPOINT_FILES:
        if (checkpoint / name).exists():
            shutil.copy2(checkpoint / name, output_dir / name)

    logger.info(f"Decoder saved to {output_path} ({get_total_model_size_mb(output_path):.1f} MB)")
    return output_path
