"""
The LFM3-Audio decoder graph of the tiny checkpoint against the vendored reference (no download):
speech features and speech frames scattered into the prompt, the final-norm hidden states next to
the logits, cached frame steps, the fp16 and quantized graphs of the LFM3 derive path, and the q4
recipe (LFM3's) that lfm3-audio-export and lfm3-omni-export give it and the tiny LFM3-Omni
checkpoint, with the checkpoint's generation_config.json copied.

Run with:
    uv run pytest tests/test_lfm3_audio/test_tiny_decoder.py -v
"""

import numpy as np
import onnx
import pytest
import torch

from liquidonnx.lfm3.export import convert_to_fp16, derive_quantized
from liquidonnx.lfm3.session import CachedDecoder
from liquidonnx.lfm3_audio.config import load_config
from liquidonnx.lfm3_audio.decoder import KEEP_IO_TYPES, LFM3AudioDecoderBuilder
from liquidonnx.quantize import save_model
from liquidonnx.session import initialize_cache, load_onnx_session

from ..test_lfm3_vl.test_tiny_export import assert_one_int8_table, assert_q4_recipe, attributes
from .tiny import ASSETS

BOS, AUDIO, AUDIO_OUT, AUDIO_START, EOS_CODE = 124894, 125031, 125032, 125012, 2016
HIDDEN, CODEBOOKS = 64, 12
NO_FEATURES = np.zeros((0, HIDDEN), dtype=np.float32)
NO_CODES = np.zeros((0, CODEBOOKS), dtype=np.int64)
DERIVED = ["decoder_fp16.onnx", "decoder_q8.onnx", "decoder_q8f16.onnx", "decoder_q4.onnx"]
DERIVED += ["decoder_q4f16.onnx"]


@pytest.fixture(scope="module")
def onnx_dir(tiny_checkpoint, tmp_path_factory):
    """decoder.onnx of the tiny checkpoint and its fp16, q8, q8f16, q4 and q4f16 versions."""
    onnx_dir = tmp_path_factory.mktemp("lfm3_audio_tiny_decoder")
    decoder = onnx_dir / "decoder.onnx"
    save_model(
        LFM3AudioDecoderBuilder(load_config(tiny_checkpoint)).build(tiny_checkpoint), decoder
    )
    convert_to_fp16(decoder, onnx_dir / "decoder_fp16.onnx", keep_io_types=KEEP_IO_TYPES)
    for bits in (4, 8):
        derive_quantized(decoder, onnx_dir / f"decoder_q{bits}.onnx", bits, 32, True, True)
    return onnx_dir


@pytest.fixture(scope="module")
def sessions(onnx_dir):
    return {path.name: load_onnx_session(path) for path in sorted(onnx_dir.glob("*.onnx"))}


def mel(frames: int, seed: int = 0) -> torch.Tensor:
    return torch.randn(1, 128, frames, generator=torch.Generator().manual_seed(seed))


def codes(frames: int, seed: int = 0) -> np.ndarray:
    """frames rows of random codes, the last one an end-of-audio frame."""
    rows = np.random.default_rng(seed).integers(0, EOS_CODE, (frames, CODEBOOKS))
    rows[-1] = EOS_CODE
    return rows


def prompt(audio_tokens: int, frames: int) -> list[int]:
    # past the tiny model's sliding window of 8
    text = np.random.default_rng(1).integers(1000, 120000, 8).tolist()
    return [BOS, *[AUDIO] * audio_tokens, *text[:4], AUDIO_START, *[AUDIO_OUT] * frames, *text[4:]]


def reference_outputs(reference, ids, features=None, frames=None):
    """(logits [S, V], final-norm hidden [S, H]) of an uncached reference forward."""
    with torch.no_grad():
        hidden = reference.model(
            input_ids=torch.tensor([ids]),
            input_features=features,
            audio_out_codes=None if frames is None else torch.from_numpy(frames),
            use_cache=False,
        ).last_hidden_state
        return reference.lm_head(hidden)[0].numpy(), hidden[0].numpy()


def audio_features(reference, features: torch.Tensor) -> np.ndarray:
    mask = torch.ones(features.shape[0], features.shape[-1], dtype=torch.bool)
    with torch.no_grad():
        return reference.model.get_audio_features(features, mask).numpy()


def run(session, ids, features=NO_FEATURES, frames=NO_CODES, keep=0):
    """(logits [K, V], hidden_states [K, H]) of one uncached decoder run."""
    feed = {
        "input_ids": np.array([ids], dtype=np.int64),
        "attention_mask": np.ones((1, len(ids)), dtype=np.int64),
        "position_ids": np.arange(len(ids), dtype=np.int64)[None, :],
        "audio_features": features,
        "audio_out_codes": frames,
        "logits_to_keep": np.array([keep], dtype=np.int64),
        **initialize_cache(session),
    }
    logits, hidden = session.run(["logits", "hidden_states"], feed)
    return logits[0], hidden[0]


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a * b).sum(-1) / np.linalg.norm(a, axis=-1) / np.linalg.norm(b, axis=-1)


def test_graph_io(tiny_checkpoint, sessions):
    session = sessions["decoder.onnx"]
    inputs = [i.name for i in session.get_inputs()]
    assert inputs[:6] == [
        "input_ids",
        "attention_mask",
        "position_ids",
        "audio_features",
        "audio_out_codes",
        "logits_to_keep",
    ]
    assert [o.name for o in session.get_outputs()][:2] == ["logits", "hidden_states"]
    # lfm3_omni: image and video features before the speech inputs
    omni = LFM3AudioDecoderBuilder(load_config(tiny_checkpoint), 125030, 125029)
    graph = omni.build(tiny_checkpoint).graph
    assert [i.name for i in graph.input][3:7] == [
        "image_features",
        "video_features",
        "audio_features",
        "audio_out_codes",
    ]


def test_prefill_matches_reference(sessions, tiny_reference):
    features, frames = mel(20), codes(3)
    ids = prompt(3, len(frames))
    expected_logits, expected_hidden = reference_outputs(tiny_reference, ids, features, frames)
    got_logits, got_hidden = run(
        sessions["decoder.onnx"], ids, audio_features(tiny_reference, features), frames
    )
    np.testing.assert_allclose(got_logits, expected_logits, rtol=1e-4, atol=2e-3)
    np.testing.assert_allclose(got_hidden, expected_hidden, rtol=1e-4, atol=1e-4)
    # logits_to_keep slices the hidden states too
    last_logits, last_hidden = run(
        sessions["decoder.onnx"], ids, audio_features(tiny_reference, features), frames, keep=2
    )
    np.testing.assert_array_equal(last_hidden, got_hidden[-2:])
    np.testing.assert_array_equal(last_logits, got_logits[-2:])


def test_cached_frame_steps_match_reference(sessions, tiny_reference):
    """A speech prompt prefilled, then one step per frame (<|audio_out|> with its codes, the last an
    end-of-audio frame) and a text step: each step's hidden state is the teacher-forced
    reference's at that position."""
    features, frames = mel(12, seed=2), codes(4, seed=2)
    head = [BOS, *[AUDIO] * 2, 5000, 6000, AUDIO_START]
    ids = [*head, *[AUDIO_OUT] * len(frames), 7000]
    expected_logits, expected_hidden = reference_outputs(tiny_reference, ids, features, frames)

    decoder = CachedDecoder(sessions["decoder.onnx"])
    keep = {"logits_to_keep": np.array([1], dtype=np.int64)}
    steps = [(head, audio_features(tiny_reference, features), NO_CODES)]
    steps += [([AUDIO_OUT], NO_FEATURES, frame[None]) for frame in frames]
    steps += [([7000], NO_FEATURES, NO_CODES)]
    position = -1
    for step_ids, step_features, step_codes in steps:
        logits = decoder.step(
            step_ids, audio_features=step_features, audio_out_codes=step_codes, **keep
        )
        position += len(step_ids)
        hidden = decoder.last_outputs["hidden_states"].numpy()
        np.testing.assert_allclose(hidden[0, -1], expected_hidden[position], rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(logits[0, -1], expected_logits[position], rtol=1e-4, atol=2e-3)


# Random weights 64 wide quantize far worse than trained ones: these floors catch broken wiring,
# the real checkpoints' tests hold the quality bars (as tests/test_lfm3_vl/test_tiny_export.py).
@pytest.mark.parametrize(
    "decoder_file,min_floor,mean_floor",
    [
        ("decoder_fp16.onnx", 0.9999, 0.9999),
        ("decoder_q8.onnx", 0.98, 0.99),
        ("decoder_q8f16.onnx", 0.98, 0.99),
        ("decoder_q4.onnx", 0.6, 0.85),
        ("decoder_q4f16.onnx", 0.6, 0.85),
    ],
)
def test_derived_decoders_track_fp32(sessions, tiny_reference, decoder_file, min_floor, mean_floor):
    features, frames = mel(20), codes(3)
    ids = prompt(3, len(frames))
    inputs = (ids, audio_features(tiny_reference, features), frames)
    expected = run(sessions["decoder.onnx"], *inputs)
    got = run(sessions[decoder_file], *inputs)
    for e, g in zip(expected, got, strict=True):
        assert g.dtype == np.float32
        similarity = cosine(g, e)
        assert similarity.min() > min_floor and similarity.mean() > mean_floor


@pytest.mark.parametrize("decoder_file", ["decoder.onnx", *DERIVED])
def test_codebook_table_stays_a_float_gather(onnx_dir, decoder_file):
    """The quantized decoders store the token table once, as the int8 head's weights; the speech
    frames' codebook table stays a float Gather (fp16 in the fp16-activation graphs)."""
    graph = onnx.load(str(onnx_dir / decoder_file), load_external_data=False).graph
    tables = {i.name: i for i in graph.initializer}
    (gather,) = [n for n in graph.node if n.name == "/model/audio_out_embed/Gather"]
    table = tables[gather.input[0]]
    assert list(table.dims) == [CODEBOOKS * 2017, HIDDEN]
    float16 = "f16" in decoder_file or decoder_file == "decoder_fp16.onnx"
    assert table.data_type == (onnx.TensorProto.FLOAT16 if float16 else onnx.TensorProto.FLOAT)
    quantized = [n for n in graph.node if n.op_type == "GatherBlockQuantized"]
    assert len(quantized) == (1 if "_q" in decoder_file else 0)
    outputs = {o.name: o.type.tensor_type.elem_type for o in graph.output}
    assert outputs["hidden_states"] == outputs["logits"] == onnx.TensorProto.FLOAT


@pytest.mark.parametrize("decoder_file", ["decoder_q8.onnx", "decoder_q8f16.onnx"])
def test_q8_decoders_store_one_int8_table(onnx_dir, decoder_file):
    graph = assert_one_int8_table(onnx_dir / decoder_file)
    assert {attributes(n)["bits"] for n in graph.node if n.op_type == "MatMulNBits"} == {8}


# (exporter, release): the tiny checkpoint in the release's layout and with its files
# (conftest.tiny_exports)
EXPORTS = [
    ("lfm3-audio-export", "LFM3-Audio-2B"),
    ("lfm3-audio-export", "LFM3-Omni-2B"),
    ("lfm3-omni-export", "LFM3-Omni-2B"),
]
EXPORT_IDS = ["audio", "audio-export-of-omni", "omni"]


@pytest.mark.parametrize(("cli", "release"), EXPORTS, ids=EXPORT_IDS)
def test_q4_recipe(tiny_exports, cli, release):
    """main() end to end: the q4 and q4f16 decoders of the tiny LFM3-Audio and LFM3-Omni checkpoints
    keep LFM3's recipe from either CLI; only d1-export gives the gate/up one
    (tests/test_d1/test_tiny_export.py)."""
    onnx_dir = tiny_exports(cli, release) / "onnx"
    for decoder in ("decoder_q4.onnx", "decoder_q4f16.onnx"):
        assert_q4_recipe(onnx_dir / decoder, onnx_dir / "decoder.onnx")


@pytest.mark.parametrize(("cli", "release"), EXPORTS, ids=EXPORT_IDS)
def test_generation_config_is_copied(tiny_exports, cli, release):
    """The checkpoint's generation_config.json, whose speech sampling lfm3-audio-infer and
    lfm3-omni-infer take (infer.speech_sampling), byte for byte."""
    name = "generation_config.json"
    copied = (tiny_exports(cli, release) / name).read_bytes()
    assert copied == (ASSETS / release / name).read_bytes()
