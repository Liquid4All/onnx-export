"""Write the tiny LFM3-family exports, feeds and fp16 references the browser tests load
(lfm3.spec.js), from seeded random checkpoints (no download).

    family      graphs                                                   precisions
    lfm3        model (LFM3-1.7B's LFM3Builder graph)                    fp32 fp16 q8 q8f16 q4 q4f16 q4f32
    lfm3_vl     decoder, vision_encoder                                  the export's (session.BUNDLES)
    lfm3_audio  decoder, audio_encoder, depth_decoder, codec_decoder
    lfm3_omni   lfm3_audio's graphs and vision_encoder
    d1          decoder, vision_encoder
    d1_omni     decoder, vision_encoder, audio_encoder
    probe       gqa_window: one GroupQueryAttention with local_window_size 8      fp32 fp16
                matmul_k{2048,7680}: one MatMul at the decoders' reduction        fp32 fp16 q4 q4f16
                lengths, plain and as the q4 decoders' int4 MatMulNBits

The tiny decoders have conv, full-attention and sliding-window layers (window 8). A prefill is
longer than the window, with image, video and audio spans between text and d1's set attention over
three questions; a cached decode step's past (24) is three windows. gqa_window isolates the window:
each query attends to the oldest key it may see (tests/test_lfm3_audio/test_gqa_window.py). The
tiny graphs' MatMuls reduce over 64 to 192 values, which hides the cost of accumulating in fp16 at
the real models' widths: the matmul probes measure it.

The prefills over 31 positions and the vision encoders' 8x12 grid come again in 28 positions
(prefill28) and 4x6 patches (grid4x6), in the precisions without fp16 activations: lfm3.spec.js
runs those on Chromium's SwiftShader adapter too, where it skips fp16 and a full attention over 32
or more positions.

Each case's feeds go into {family}/cases/ as one little-endian blob, described in manifest.json,
whose "nodes" lists each graph's nodes that read float tensors (float_nodes) for checks.js.
The browser compares the fp32 graphs, the q8 vision encoder and q4f32 with the wasm EP; the
fp16-activation graphs (fp16, q4f16, q8f16), the q4 and q8 decoders, whose GatherBlockQuantized
the wasm EP of onnxruntime-web 1.30.0 has no kernel for, and the probes, with native onnxruntime's
CPU EP (cpu_reference), whose outputs are written next to the feeds.

    uv run python tests/web/export_lfm3.py            # -> tests/web/.lfm3
    uv run python tests/web/export_lfm3.py /some/dir
"""

import json
import logging
import pathlib
import shutil
import sys
import tempfile
import zlib

import numpy as np
import onnx

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from liquidonnx.d1 import export as d1_export  # noqa: E402
from liquidonnx.lfm3 import export as text_export  # noqa: E402
from liquidonnx.lfm3.builder import LFM3Builder  # noqa: E402
from liquidonnx.lfm3_audio import export as audio_export  # noqa: E402
from liquidonnx.lfm3_audio.config import DEFAULT_TOKENS, load_config  # noqa: E402
from liquidonnx.lfm3_audio.weights import load_weights  # noqa: E402
from liquidonnx.lfm3_omni import export as omni_export  # noqa: E402
from liquidonnx.lfm3_vl import export as vl_export  # noqa: E402
from liquidonnx.quantize import load_model, save_model  # noqa: E402
from liquidonnx.session import session_options  # noqa: E402
from tests.test_d1.test_config import d1_config  # noqa: E402
from tests.test_lfm3_audio import tiny as audio_tiny  # noqa: E402
from tests.test_lfm3_audio.test_gqa_window import gqa_model  # noqa: E402
from tests.test_lfm3_vl import tiny as vl_tiny  # noqa: E402

logger = logging.getLogger(__name__)

FAMILIES = ("lfm3", "lfm3_vl", "lfm3_audio", "lfm3_omni", "d1", "d1_omni", "probe")
EXPORTS = {
    "lfm3_vl": vl_export,
    "lfm3_audio": audio_export,
    "lfm3_omni": omni_export,
    "d1": d1_export,
    "d1_omni": d1_export,
}
# The exports' default; the tiny decoders' 64-wide int4 MatMuls need an even number of blocks
BLOCK_SIZE = 32
WINDOW = vl_tiny.TINY_CONFIG["sliding_window"]
PAST = 3 * WINDOW
# The LFM3 decoders' hidden and MLP widths
ACCUMULATION_K = (2048, 7680)
SUFFIXES = ("fp16", "q4", "q4f16", "q4f32", "q8", "q8f16")
FP16_ACTIVATIONS = ("fp16", "q4f16", "q8f16")
# graph -> its cases; the decode step only where the graph has caches, prefill28 where the family
# has a 28-position row
CASES = {
    "model": ["prefill", "decode"],
    "decoder": ["prefill", "prefill28", "decode"],
    "gqa_window": ["prefill", "decode"],
    "vision_encoder": ["grid8x12", "grid4x6"],
    "audio_encoder": ["frames200"],
    "depth_decoder": ["batch2"],
    "codec_decoder": ["frames7"],
}

TINY = vl_tiny.TINY_CONFIG
BOS, IMAGE, VIDEO = TINY["bos_token_id"], TINY["image_token_id"], TINY["video_token_id"]
START, END = TINY["vision_start_token_id"], TINY["vision_end_token_id"]
AUDIO, AUDIO_OUT, AUDIO_START = (
    DEFAULT_TOKENS[f"audio{kind}_token_id"] for kind in ("", "_out", "_start")
)


def text(n: int, seed: int) -> list[int]:
    return np.random.default_rng(seed).integers(1000, 120000, n).tolist()


def image(n: int, token: int = IMAGE) -> list[int]:
    return [START, *[token] * n, END]


# Prefill rows: feature spans between text, longer than the window
PREFILL = {
    "lfm3": [BOS, *text(23, 0)],
    "lfm3_vl": [BOS, *image(4), *text(5, 1), *image(6), *text(3, 2), *image(4, VIDEO), *text(4, 3)],
    "lfm3_audio": [BOS, *[AUDIO] * 5, *text(4, 1), *[AUDIO] * 3, *text(3, 2), AUDIO_START]
    + [*[AUDIO_OUT] * 3, *text(2, 3), *[AUDIO_OUT] * 2, *text(3, 4)],
    "lfm3_omni": [BOS, *image(4), *text(3, 1), *[AUDIO] * 5, *text(3, 2), *image(4, VIDEO)]
    + [*text(2, 3), *[AUDIO] * 3, *text(2, 4), AUDIO_START, *[AUDIO_OUT] * 3, *text(2, 5)],
}
# The rows above over 31 positions, shortened to 28 with each of their spans
PREFILL28 = {
    "lfm3_vl": [BOS, *image(4), *text(2, 1), *image(6), *text(2, 2), *image(4, VIDEO), *text(3, 3)],
    "lfm3_omni": [BOS, *image(4), *text(1, 1), *[AUDIO] * 3, *text(1, 2), *image(4, VIDEO)]
    + [*text(1, 3), *[AUDIO] * 3, *text(1, 4), AUDIO_START, *[AUDIO_OUT] * 3, *text(1, 5)],
}
# d1's state rows; QUESTIONS follow them (packed_row of tests/test_d1/test_tiny_export.py). 21
# state tokens and 36 in all (D1_STATE28: 13 and 28): S % 4 == 0 with the state's end inside a
# group of 4, where an int32 Equal / Not / And set mask comes out wrong on WebGPU (README 4.6)
D1_STATE = {
    "d1": [BOS, *text(4, 1), *image(4), *text(2, 2), *image(4), *text(2, 3)],
    "d1_omni": [BOS, *text(5, 1), *image(4), *text(2, 2), *[AUDIO] * 5, *text(2, 3)],
}
D1_STATE28 = {
    "d1": [BOS, *text(1, 1), *image(3), *text(1, 2), *image(2), *text(1, 3)],
    "d1_omni": [BOS, *text(1, 1), *image(3), *text(1, 2), *[AUDIO] * 4, *text(1, 3)],
}
QUESTIONS = (4, 6, 5)


def d1_row(state: list[int]) -> tuple[list[int], list[int], list[int]]:
    """(ids, question ids, slots): each question's last token a slot, the first one's third too."""
    ids, question, slots = list(state), [-1] * len(state), [len(state) + 2]
    for k, n in enumerate(QUESTIONS, 1):
        ids += text(n, 10 + k)
        question += [k] * n
        slots.append(len(ids) - 1)
    return ids, question, slots


# === Checkpoints and exports ===


def write_d1_checkpoint(base: pathlib.Path, directory: pathlib.Path):
    shutil.copytree(base, directory)
    config = json.loads((base / "config.json").read_text())
    (directory / "config.json").write_text(json.dumps(d1_config(config), indent=2))


def export_text(audio: pathlib.Path, scratch: pathlib.Path, onnx_dir: pathlib.Path):
    """LFM3 text: the tiny LFM3-Audio checkpoint's language model (an lfm3 text config), built and
    derived as lfm3-export does."""
    from safetensors.numpy import save_file

    config = load_config(audio)
    weights = load_weights(audio, "language", config)
    del weights["codebook_embedding.weight"]
    checkpoint = scratch / "lfm3"
    checkpoint.mkdir()
    save_file(weights, str(checkpoint / "model.safetensors"))
    save_model(LFM3Builder(config.language).build(checkpoint), onnx_dir / "model.onnx")
    text_export.do_fp16(onnx_dir)
    for bits in (4, 8):
        text_export.derive_quantized(
            onnx_dir / "model.onnx",
            onnx_dir / f"model_q{bits}.onnx",
            bits,
            BLOCK_SIZE,
            keep=True,
            fp16_activations=True,
        )
    text_export.do_quantize(
        onnx_dir,
        bits=4,
        exclude_lm_head=True,
        block_size=BLOCK_SIZE,
        symmetric=True,
        output_name="model_q4f32.onnx",
    )


def export_probe(onnx_dir: pathlib.Path):
    """gqa_window.onnx and gqa_window_fp16.onnx: test_gqa_window's GroupQueryAttention with a past
    of any length. matmul_k{K}.onnx: x [1, 16, K] @ w [K, 256], w ~ N(0, 1/K), and its fp16
    (lfm3_vl.export's side-graph conversion), q4 (the q4 decoders' k_quant int4) and q4f16
    versions."""
    from liquidonnx.lfm3.export import convert_quantized_to_fp16
    from liquidonnx.quantize import convert_to_fp16, quantize_matmuls

    onnx_dir.mkdir(parents=True)
    for elem_type, name in ((onnx.TensorProto.FLOAT, ""), (onnx.TensorProto.FLOAT16, "_fp16")):
        model = gqa_model(WINDOW, elem_type)
        for past in model.graph.input[3:5]:
            past.type.tensor_type.shape.dim[2].dim_param = "past_sequence_length"
        onnx.save(model, str(onnx_dir / f"gqa_window{name}.onnx"))
    for K in ACCUMULATION_K:
        weight = np.random.default_rng(K).standard_normal((K, 256)) / np.sqrt(K)
        graph = onnx.helper.make_graph(
            [onnx.helper.make_node("MatMul", ["x", "w"], ["y"], name="/matmul/MatMul")],
            "matmul",
            [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 16, K])],
            [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 16, 256])],
            [onnx.numpy_helper.from_array(weight.astype(np.float32), "w")],
        )
        model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 21)])
        model.ir_version = 10
        path = save_model(model, onnx_dir / f"matmul_k{K}.onnx")
        convert_to_fp16(path, onnx_dir / f"{path.stem}_fp16.onnx", keep_io=True)
        q4 = quantize_matmuls(load_model(path), bits=4, block_size=BLOCK_SIZE, algorithm="k_quant")
        q4_path = save_model(q4, onnx_dir / f"{path.stem}_q4.onnx")
        convert_quantized_to_fp16(q4_path, onnx_dir / f"{path.stem}_q4f16.onnx")


# === Cases ===


def probe_feeds(case: str, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Key scores falling by 16 a position, so query i reads the value of the oldest key in its
    window, v[max(0, i - WINDOW + 1)] (exact in fp16 up to 128 positions)."""
    S, past = (1, PAST) if case == "decode" else (2 * WINDOW + 7, 0)
    total = past + S
    key = np.zeros((1, total, 64))
    key[0, :, 0] = -16 * np.arange(total)
    value = rng.standard_normal((1, total, 64))
    query = np.zeros((1, S, 64))
    query[..., 0] = 1
    return {
        "query": query,
        "key": key[:, past:],
        "value": value[:, past:],
        "past_key": key[:, None, :past],
        "past_value": value[:, None, :past],
        "seqlens_k": np.array([total - 1]),
        "total_sequence_length": np.array(total),
    }


def decoder_feeds(graph: onnx.GraphProto, family: str, case: str, rng: np.random.Generator):
    """Feeds by input name: a prefill of the family's row (d1: a packed row), or one decode step
    after PAST cached positions (a speech frame where the graph takes audio_out_codes)."""
    names = {i.name for i in graph.input}
    past, question, slots = 0, None, None
    if family.startswith("d1"):
        ids, question, slots = d1_row((D1_STATE28 if case == "prefill28" else D1_STATE)[family])
    elif case == "decode":
        past, ids = PAST, [AUDIO_OUT] if "audio_out_codes" in names else [5000]
    else:
        ids = (PREFILL28 if case == "prefill28" else PREFILL)[family]
    rows = {"image_features": IMAGE, "video_features": VIDEO, "audio_features": AUDIO}
    feeds = {}
    for inp in graph.input:
        name = inp.name
        dims = [d.dim_value or d.dim_param for d in inp.type.tensor_type.shape.dim]
        if name == "input_ids":
            value = np.array([ids])
        elif name == "attention_mask":
            value = np.ones((1, past + len(ids)))
        elif name == "position_ids":
            value = np.arange(past, past + len(ids))[None]
        elif name == "logits_to_keep":
            value = np.array([1 if past else 0])
        elif name in rows:
            value = rng.standard_normal((ids.count(rows[name]), dims[1]))
        elif name == "audio_out_codes":
            value = rng.integers(0, 2017, (ids.count(AUDIO_OUT), dims[1]))
        elif name == "question_ids":
            value = np.array([question])
        elif name == "slot_positions":
            value = np.array(slots)
        elif name.startswith("past_conv"):
            value = rng.standard_normal((1, *dims[1:])) * bool(past)
        elif name.startswith("past_key_values"):
            value = rng.standard_normal((1, dims[1], past, dims[3]))
        else:
            raise KeyError(f"no feed for decoder input {name}")
        feeds[name] = value
    return feeds


def side_feeds(case: str, rng: np.random.Generator) -> dict[str, np.ndarray]:
    if case == "grid8x12":
        # [h/m, w/m, m, m, patch_dim]: 24 image tokens
        return {"pixel_values": rng.uniform(-1, 1, (4, 6, 2, 2, 1536))}
    if case == "grid4x6":
        # 6 image tokens
        return {"pixel_values": rng.uniform(-1, 1, (2, 3, 2, 2, 1536))}
    if case == "frames200":
        # 25 encoder tokens, three windows
        return {"input_features": rng.standard_normal((1, 128, 200))}
    if case == "frames7":
        # 42 positions, five windows
        return {"audio_codes": rng.integers(0, 2016, (1, 12, 7))}
    masked = np.ones((2, 12), dtype=bool)
    masked[0, :4] = False  # after the first MaskGIT pass, and a random mask
    masked[1] = rng.random(12) < 0.5
    return {
        "hidden_states": rng.standard_normal((2, 64)),
        "codes": rng.integers(0, 2017, (2, 12)),
        "masked": masked,
    }


def typed(graph: onnx.GraphProto, arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """arrays in the dtypes the graph's inputs declare."""
    return {
        i.name: arrays[i.name].astype(
            onnx.helper.tensor_dtype_to_np_dtype(i.type.tensor_type.elem_type)
        )
        for i in graph.input
    }


def full_attention_queries(graph: onnx.GraphProto, feeds: dict[str, np.ndarray]) -> int:
    """The query positions of the case's attention without a window (GroupQueryAttention at
    local_window_size -1, MultiHeadAttention): its tokens or patches, 0 if the graph has none."""
    # shortcut: an attention within its window takes the same WebGPU path (lfm3.spec.js), but no
    # case runs 32 positions within a window; count those too if one does
    windows = [
        next((a.i for a in node.attribute if a.name == "local_window_size"), -1)
        for node in graph.node
        if node.op_type in ("GroupQueryAttention", "MultiHeadAttention")
    ]
    if -1 not in windows:
        return 0
    if "input_ids" in feeds:
        return feeds["input_ids"].shape[1]
    return int(np.prod(feeds["pixel_values"].shape[:4]))


def cpu_reference(path: pathlib.Path):
    """The graph on onnxruntime's CPU EP with every MatMulNBits at accuracy_level 0: float
    activations, as the WebGPU EP computes them on Apple GPUs (other adapters quantize them to
    int8 at the exported level 4, DP4A). At level 4 the CPU EP quantizes the activations to int8,
    which moves the tiny q4 and q8 decoders' logits by up to 4.7 (of 47)."""
    import onnxruntime as ort

    model = onnx.load(str(path))
    for node in model.graph.node:
        for attribute in node.attribute:
            if node.op_type == "MatMulNBits" and attribute.name == "accuracy_level":
                attribute.i = 0
    options = session_options()
    options.intra_op_num_threads = 1
    return ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def float_nodes(model: onnx.ModelProto) -> list[str]:
    """The nodes reading a tensor that isn't int64, int32 or bool, by ONNX type inference on the
    graph alone, which checks.js keeps off the CPU EP. A tensor inference can't type (anything
    after a com.microsoft op) counts as float."""
    graph = onnx.shape_inference.infer_shapes(model, strict_mode=False).graph
    values = [*graph.input, *graph.output, *graph.value_info]
    types = {v.name: v.type.tensor_type.elem_type for v in values}
    types |= {i.name: i.data_type for i in graph.initializer}
    for node in graph.node:
        if node.op_type == "Constant":
            types[node.output[0]] = next(a.t.data_type for a in node.attribute if a.name == "value")
    plumbing = (onnx.TensorProto.INT64, onnx.TensorProto.INT32, onnx.TensorProto.BOOL)
    return [
        node.name
        for node in graph.node
        if any(types.get(name) not in plumbing for name in node.input if name)
    ]


def write_tensors(path: pathlib.Path, arrays: dict[str, np.ndarray], root: pathlib.Path) -> dict:
    """arrays as one blob, and its manifest entry: {file, tensors: [{name, type, dims, offset,
    bytes}]}; numpy's dtype names are onnxruntime-web's."""
    tensors, offset = [], 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        for name, array in arrays.items():
            data = np.ascontiguousarray(array).tobytes()
            tensors.append(
                {
                    "name": name,
                    "type": str(array.dtype),
                    "dims": list(array.shape),
                    "offset": offset,
                    "bytes": len(data),
                }
            )
            f.write(data)
            offset += len(data)
    return {"file": str(path.relative_to(root)), "tensors": tensors}


def write_cases(out_dir: pathlib.Path, family: str, nodes: dict) -> list[dict]:
    """The manifest entries of every graph and case of a family, with their feeds, and native
    onnxruntime's outputs where the wasm EP is no reference; each graph's float_nodes go into
    nodes."""
    onnx_dir = out_dir / family / "onnx"
    entries = []
    for path in sorted(onnx_dir.glob("*.onnx")):
        graph_name, _, suffix = path.stem.rpartition("_")
        graph_name, precision = (graph_name, suffix) if suffix in SUFFIXES else (path.stem, "fp32")
        model = onnx.load(str(path), load_external_data=False)
        graph = model.graph
        nodes[str(path.relative_to(out_dir))] = float_nodes(model)
        ops = {node.op_type for node in graph.node}
        cached = any(i.name.startswith("past") for i in graph.input)
        short = family in PREFILL28 | D1_STATE28
        for case in CASES.get(graph_name, ["run"]):
            if (case == "decode" and not cached) or (case == "prefill28" and not short):
                continue
            # sized for SwiftShader, which has no shader-f16
            if case in ("prefill28", "grid4x6") and precision in FP16_ACTIVATIONS:
                continue
            # one seed per case: every precision of a graph gets the same values
            rng = np.random.default_rng(zlib.crc32(f"{family}/{graph_name}/{case}".encode()))
            if graph_name == "gqa_window":
                feeds = probe_feeds(case, rng)
            elif graph_name.startswith("matmul"):
                shape = [d.dim_value for d in graph.input[0].type.tensor_type.shape.dim]
                feeds = {"x": rng.standard_normal(shape)}
            elif graph_name in ("model", "decoder"):
                feeds = decoder_feeds(graph, family, case, rng)
            else:
                feeds = side_feeds(case, rng)
            feeds = typed(graph, feeds)
            stem = out_dir / family / "cases" / f"{path.stem}.{case}"
            expected = None
            if family == "probe" or precision in FP16_ACTIVATIONS or "GatherBlockQuantized" in ops:
                session = cpu_reference(path)
                outputs = session.run(None, feeds)
                names = [o.name for o in session.get_outputs()]
                expected = write_tensors(
                    stem.with_name(f"{stem.name}.expected.bin"),
                    dict(zip(names, outputs, strict=True)),
                    out_dir,
                )
            entries.append(
                {
                    "id": f"{family} {graph_name} {precision} {case}",
                    "family": family,
                    "graph": graph_name,
                    "precision": precision,
                    "case": case,
                    "model": str(path.relative_to(out_dir)),
                    "external_data": [p.name for p in onnx_dir.glob(f"{path.stem}.onnx_data*")],
                    "feeds": write_tensors(
                        stem.with_name(f"{stem.name}.feeds.bin"), feeds, out_dir
                    ),
                    "expected": expected,
                    "full_attention_queries": full_attention_queries(graph, feeds),
                }
            )
    return entries


def check_probe(out_dir: pathlib.Path):
    """The probe discriminates the window: on the CPU EP each query reads its oldest visible key's
    value, which no wider or narrower window gives."""
    path = out_dir / "probe" / "onnx" / "gqa_window.onnx"
    graph = onnx.load(str(path)).graph
    session = cpu_reference(path)
    for case in CASES["gqa_window"]:
        feeds = typed(graph, probe_feeds(case, np.random.default_rng(0)))
        values = np.concatenate([feeds["past_value"][:, 0], feeds["value"]], axis=1)[0]
        past = feeds["past_key"].shape[2]
        oldest = np.maximum(0, past + np.arange(feeds["query"].shape[1]) - WINDOW + 1)
        output = session.run(["output"], feeds)[0][0]
        np.testing.assert_allclose(output, values[oldest], atol=1e-5)


def main(out_dir: pathlib.Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    with tempfile.TemporaryDirectory() as scratch:
        scratch = pathlib.Path(scratch)
        vl = vl_tiny.write_tiny_checkpoint(scratch / "lfm3_vl")
        audio = audio_tiny.write_tiny_checkpoint(scratch / "lfm3_audio")
        omni = audio_tiny.write_omni_checkpoint(audio, scratch / "lfm3_omni")
        checkpoints = {"lfm3_vl": vl, "lfm3_audio": audio, "lfm3_omni": omni}
        for family, base in (("d1", "lfm3_vl"), ("d1_omni", "lfm3_omni")):
            checkpoints[family] = scratch / family
            write_d1_checkpoint(checkpoints[base], checkpoints[family])

        export_text(audio, scratch, out_dir / "lfm3" / "onnx")
        for family, module in EXPORTS.items():
            onnx_dir = out_dir / family / "onnx"
            module.export_fp32(checkpoints[family], onnx_dir)
            module.derive(onnx_dir, list(module.PRECISIONS), block_size=BLOCK_SIZE)
    export_probe(out_dir / "probe" / "onnx")
    check_probe(out_dir)

    nodes = {}
    cases = [entry for family in FAMILIES for entry in write_cases(out_dir, family, nodes)]
    manifest = {"cases": cases, "nodes": nodes}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    logger.info(f"{len(cases)} cases written to {out_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    target = (
        pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(__file__).parent / ".lfm3"
    )
    main(target)
