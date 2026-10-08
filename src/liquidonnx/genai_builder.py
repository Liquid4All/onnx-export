"""
Build LFM2 decoders (text, MoE, VL, audio) with the onnxruntime-genai model builder.

The builder runs from a source checkout of onnxruntime-genai main at the pinned GENAI_COMMIT,
fetched once into ~/.cache/liquidonnx. Set LIQUIDONNX_GENAI_BUILDER to a local
`src/python/py/models` directory to use another copy. The exports run on onnxruntime-genai 0.17.1
or later from PyPI, for all four families (lfm2, lfm2_moe, lfm2_vl, lfm2_audio); CI tests them on
the runtime built from main at GENAI_COMMIT.

Decoders are built for the CPU EP. The fp32 decoder and the quantized precisions in DECODER_PRESETS
are each their own builder run; fp16 and q4f16 are the fp32 and q4 decoders converted to fp16.
"""

import json
import logging
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass

import numpy as np
import onnx
from onnx import numpy_helper

from liquidonnx import remote_code_enabled
from liquidonnx.quantize import (
    DEFAULT_BLOCK_SIZE,
    convert_to_fp16,
    get_total_model_size_mb,
    load_model,
    rename_quantized_weights,
    save_model,
)
from liquidonnx.session import SESSION_CONFIG

logger = logging.getLogger(__name__)

GENAI_REPO = "https://github.com/microsoft/onnxruntime-genai.git"
GENAI_COMMIT = "957bdd8dc717e94c09fad3b17d22f10d33217fd2"
BUILDER_SUBDIR = "src/python/py/models"
CHECKPOINT_FILES = ("config.json", "generation_config.json")
EMBED_GATHER = "/model/embed_tokens/Gather"


@dataclass(frozen=True)
class DecoderPreset:
    precision: str  # builder -p
    options: dict[str, str]
    # Rename the tensors as liquidonnx.quantize named them when it derived this precision, so the
    # published decoder keeps its tensor names.
    legacy_names: bool = False
    # Gather a tied checkpoint's embeddings from the int8 LM head (gather_embeddings_from_lm_head).
    tie_int8_embeddings: bool = False


# === Decoder presets ===

# The builder ties only 4-bit embedding tables to the LM head and keeps a dense fp32 table next to
# an int8 one.
Q8 = DecoderPreset("int8", {"is_symmetric": "false"}, True, tie_int8_embeddings=True)
Q4F32 = DecoderPreset(
    "int4", {"nodes_to_exclude": "/lm_head/MatMul,/model/embed_tokens/Gather"}, True
)
# The olive-recipes cpu_int4 options: k_quant, with the LM head (and the tied embedding table it
# shares) and the layers most sensitive to int4 at int8.
Q4 = DecoderPreset(
    "int4",
    {"algo_config": "k_quant", "matmul_mixed_precision": "last_matmul:int8,mixed_layers:int8"},
)
# The VL and audio decoders take inputs_embeds, so their LM head shares no table: Q8_INT8_HEAD is
# Q8 without the tie, and Q4_INT8_HEAD is Q4F32 with an int8 head, the olive-recipes audio cpu_int4
# decoder (Q4's options are unmeasured on audio).
Q8_INT8_HEAD = DecoderPreset("int8", {"is_symmetric": "false"}, True)
Q4_INT8_HEAD = DecoderPreset("int4", {"matmul_mixed_precision": "last_matmul:int8"})
DECODER_PRESETS = {
    "lfm2": {"q4": Q4, "q4f32": Q4F32, "q8": Q8},
    "lfm2_moe": {"q4": Q4, "q4f32": Q4F32, "q8": Q8},
    "lfm2_vl": {"q4": Q4, "q8": Q8_INT8_HEAD},
    "lfm2_audio": {"q4": Q4_INT8_HEAD, "q8": Q8_INT8_HEAD},
}


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


def resolve_checkpoint(model: str, allow_patterns: list[str] | None = None) -> pathlib.Path:
    """Local directory of a checkpoint given as a path or a Hugging Face model ID.

    allow_patterns limits a download to the matching files; a local directory is used as is.
    """
    path = pathlib.Path(model)
    if path.is_dir():
        return path

    from huggingface_hub import snapshot_download

    return pathlib.Path(snapshot_download(model, allow_patterns=allow_patterns))


def build_decoder(
    model: str,
    output_dir: pathlib.Path,
    precision: str = "fp32",
    extra_options: dict[str, str] | None = None,
    filename: str = "model.onnx",
) -> pathlib.Path:
    """Run the genai builder (CPU EP) at a builder precision and return output_dir/filename.

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
        precision,
        "-e",
        "cpu",
        "-c",
        str(cache_dir() / "genai-builder-cache"),
    ]
    options = {
        "hf_remote": str(remote_code_enabled()).lower(),
        "filename": filename,
        **(extra_options or {}),
    }
    cmd += ["--extra_options", *[f"{k}={v}" for k, v in options.items()]]
    logger.info(f"Building the {precision} decoder with the genai builder: {checkpoint}")
    subprocess.run(cmd, check=True)
    return output_dir / filename


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


def mark_qmoe_weights_raw(model: onnx.ModelProto):
    """Set weights_prepacked=0 on every QMoE node.

    CPU builds store raw [E, N, K/pack] expert weights but omit the attribute, and CUDA's QMoE
    reads an omitted one as CUTLASS-prepacked weights; 0 makes it prepack them at load time.
    """
    for node in model.graph.node:
        if node.op_type == "QMoE" and all(a.name != "weights_prepacked" for a in node.attribute):
            node.attribute.append(onnx.helper.make_attribute("weights_prepacked", 0))


def ties_embeddings(checkpoint: pathlib.Path) -> bool:
    """Whether the checkpoint's token embedding is its LM head, as the builder reads its config."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=remote_code_enabled())
    return bool(getattr(config, "tie_word_embeddings", False))


def gather_embeddings_from_lm_head(model: onnx.ModelProto):
    """Gather the token embeddings of EMBED_GATHER from the int8 LM head; drop their fp32 table.

    The layout is the builder's for the int8 head of a 4-bit build: the head's [V, n_blocks,
    block_size] weights, reshaped to [V, n_blocks * block_size], feed a GatherBlockQuantized with
    the head's scales and zero points, and a Slice drops the block padding of the rows, if any.
    """
    graph = model.graph
    head = next(node for node in graph.node if "logits" in node.output)
    attrs = {a.name: onnx.helper.get_attribute_value(a) for a in head.attribute}
    if head.op_type != "MatMulNBits" or attrs["bits"] != 8:
        raise ValueError(f"{head.name} is not an int8 MatMulNBits LM head")
    index, gather = next((i, n) for i, n in enumerate(graph.node) if n.name == EMBED_GATHER)
    table = next(init for init in graph.initializer if init.name == gather.input[0])
    vocab, hidden, block_size = attrs["N"], attrs["K"], attrs["block_size"]
    if list(table.dims) != [vocab, hidden]:
        raise ValueError(f"{table.name} is {list(table.dims)}, the LM head [{vocab}, {hidden}]")

    base = EMBED_GATHER.rsplit("/", 1)[0]
    padded = -(-hidden // block_size) * block_size
    shape, weights = f"{base}/Reshape/shape", f"{base}/Reshape/output_0"
    constants = {shape: [vocab, padded]}
    gathered = gather.output[0] if padded == hidden else f"{base}/GatherBlockQuantized/output_0"
    nodes = [
        onnx.helper.make_node("Reshape", [head.input[1], shape], [weights], name=f"{base}/Reshape"),
        onnx.helper.make_node(
            "GatherBlockQuantized",
            [weights, gather.input[1], *head.input[2:4]],
            [gathered],
            name=f"{base}/GatherBlockQuantized",
            domain="com.microsoft",
            bits=8,
            block_size=block_size,
            gather_axis=0,
            quantize_axis=1,
        ),
    ]
    if padded != hidden:
        bounds = {
            f"{base}/Slice/{k}": [v] for k, v in (("starts", 0), ("ends", hidden), ("axes", -1))
        }
        constants |= bounds
        nodes.append(
            onnx.helper.make_node("Slice", [gathered, *bounds], gather.output, name=f"{base}/Slice")
        )

    graph.initializer.remove(table)
    graph.initializer.extend(
        numpy_helper.from_array(np.array(value, dtype=np.int64), name)
        for name, value in constants.items()
    )
    del graph.node[index]
    for offset, node in enumerate(nodes):
        graph.node.insert(index + offset, node)


def export_decoder(
    model: str,
    output_dir: pathlib.Path,
    filename: str = "model.onnx",
    extra_options: dict[str, str] | None = None,
    preset: DecoderPreset | None = None,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> pathlib.Path:
    """Build the fp32 decoder, or the one preset quantizes, into output_dir/onnx/filename.

    The fp32 build also writes the builder's genai_config.json and tokenizer files, and the
    checkpoint's config.json and generation_config.json, to output_dir.
    """
    onnx_dir = output_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    precision = preset.precision if preset else "fp32"
    options = dict(extra_options or {})
    if preset:
        options.update(preset.options, block_size=str(block_size), qmoe_block_size=str(block_size))

    with tempfile.TemporaryDirectory(dir=output_dir, prefix=".genai-build-") as tmp:
        build_dir = pathlib.Path(tmp)
        built = build_decoder(model, build_dir, precision, options, filename)

        genai_config = json.loads((build_dir / "genai_config.json").read_text())
        decoder = load_model(built)
        pin_kv_head_size(decoder, genai_config["model"]["decoder"]["head_size"])
        if preset and preset.legacy_names:
            rename_quantized_weights(decoder)
        if preset and preset.tie_int8_embeddings and ties_embeddings(resolve_checkpoint(model)):
            gather_embeddings_from_lm_head(decoder)
        mark_qmoe_weights_raw(decoder)
        output_path = save_model(decoder, onnx_dir / filename)
        del decoder

        if not preset:
            # genai passes unknown session_options keys to AddConfigEntry, and the multimodal
            # graphs without session_options of their own reuse the decoder's.
            session_options = genai_config["model"]["decoder"].setdefault("session_options", {})
            session_options.update(SESSION_CONFIG)
            (build_dir / "genai_config.json").write_text(json.dumps(genai_config, indent=4))
            for f in build_dir.iterdir():
                if f.is_file() and not f.name.startswith(filename):
                    shutil.copy2(f, output_dir / f.name)

            checkpoint = resolve_checkpoint(model)
            for name in CHECKPOINT_FILES:
                if (checkpoint / name).exists():
                    shutil.copy2(checkpoint / name, output_dir / name)

    logger.info(f"Decoder saved to {output_path} ({get_total_model_size_mb(output_path):.1f} MB)")
    return output_path


def export_precision(
    model: str,
    output_dir: pathlib.Path,
    family: str,
    precision: str,
    name: str = "model",
    extra_options: dict[str, str] | None = None,
    block_size: int = DEFAULT_BLOCK_SIZE,
    reuse_q4: bool = False,
) -> pathlib.Path:
    """Write output_dir/onnx/{name}_{precision}.onnx.

    A precision in DECODER_PRESETS[family] is its own builder run, with the family's builder
    options in extra_options as for fp32. fp16 converts onnx/{name}.onnx; q4f16 converts
    onnx/{name}_q4.onnx when reuse_q4 says the caller has just built it, and a fresh q4 otherwise.
    """
    onnx_dir = output_dir / "onnx"
    output_path = onnx_dir / f"{name}_{precision}.onnx"
    if precision == "fp16":
        return convert_to_fp16(onnx_dir / f"{name}.onnx", output_path)
    if precision == "q4f16":
        if reuse_q4:
            return convert_to_fp16(onnx_dir / f"{name}_q4.onnx", output_path)
        with tempfile.TemporaryDirectory(dir=output_dir, prefix=".q4-") as tmp:
            q4 = export_precision(
                model, pathlib.Path(tmp), family, "q4", name, extra_options, block_size
            )
            return convert_to_fp16(q4, output_path)

    preset = DECODER_PRESETS[family][precision]
    return export_decoder(model, output_dir, output_path.name, extra_options, preset, block_size)
