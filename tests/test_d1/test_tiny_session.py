"""
d1.session end to end on the tiny d1 export (no download): OrtSystemOne on the CPU EP under the
checkpoint's own host code, answering the model cards' picture questions about a random picture
and their invoice's reads, and batched rows against one row a forward.

reference/ holds the files of LiquidAI/d1-2B at revision 97c5704 (d1.config.REVISIONS) that the
session reads from an export folder, where d1-export copies them: d1.session.HOST_MODULES and the
modules they import, native_lfm3.py and native_lfm3_vl.py (the AutoConfig of its config.json, which
registers the processor classes), processor_config.json, tokenizer_config.json,
chat_template.jinja and calibration.json, under this repository's license (LFM Open License v1.0).
They are vendored unchanged but for the timing and hardware notes cut from attention.py,
depthwise.py, lfm3.py and runner.py, each headed by a line saying so. The LFM3-VL code that
native_lfm3_vl.py imports is the tiny checkpoint's (LFM3-VL-2B's, tests/test_lfm3_vl/reference),
and tokenizer.json is generated: a byte-level BPE on LFM3's pre-tokenizer with the special tokens
at their ids, and yes, no and every number up to 999 one token, as the readouts score them (box
edges as numbers: d1_numbers). The weights are random: the answers are checked for their shapes
and ranges, and batched against unbatched, not for their values.

Run with:
    uv run pytest tests/test_d1/test_tiny_session.py -v
"""

import json
import pathlib
import shutil

import numpy as np
import pytest
from PIL import Image
from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, pre_tokenizers, processors

from liquidonnx.d1.session import load

from .conftest import INVOICE, PICTURE_QUESTIONS, TICKET
from .test_answers import differences

REFERENCE = pathlib.Path(__file__).parent / "reference"
# LFM3's pre-tokenizer split, as in the checkpoints' tokenizer.json
PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+|\s+(?!\S)|\s+"
)
# The special tokens of the prompts, the chat template and tokenizer_config.json
SPECIAL = {
    "<|pad|>": 124893,
    "<|startoftext|>": 124894,
    "<|im_start|>": 124899,
    "<|im_end|>": 124900,
    "<|vision_start|>": 125009,
    "<|vision_end|>": 125010,
    "<|video|>": 125029,
    "<|image|>": 125030,
}
THINK = {"<think>": 124901, "</think>": 124902}


def write_tokenizer(path: pathlib.Path):
    vocab = {c: i for i, c in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    for word in [*map(str, range(1000)), "yes", "no"]:
        vocab.setdefault(word, len(vocab))
    # ignore_merges: a word of the vocabulary is one token without merges
    tokenizer = Tokenizer(models.BPE(vocab | SPECIAL | THINK, [], ignore_merges=True))
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(PATTERN), "isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    tokenizer.add_special_tokens([AddedToken(t, special=True, normalized=False) for t in SPECIAL])
    tokenizer.add_tokens([AddedToken(t, normalized=False) for t in THINK])
    tokenizer.save(str(path))


@pytest.fixture(scope="module")
def export_with_host_code(tiny_export):
    """The tiny export with reference/'s files, the tokenizer and d1-2B's AutoConfig."""
    shutil.copytree(REFERENCE, tiny_export, dirs_exist_ok=True)
    write_tokenizer(tiny_export / "tokenizer.json")
    config = json.loads((tiny_export / "config.json").read_text())
    config["auto_map"]["AutoConfig"] = "native_lfm3_vl.Lfm3VlConfig"
    (tiny_export / "config.json").write_text(json.dumps(config, indent=2))
    return tiny_export


def test_system_one_batch(export_with_host_code):
    """The picture through the processor and the vision encoder, the reads through the pointer, in
    one system_one_batch: every answer of its question's type, boxes inside the picture, an estimate
    on its grid, a read's span its text in the state."""
    model = load(export_with_host_code, ep="cpu")
    pixels = np.random.default_rng(0).integers(0, 256, (240, 320, 3), dtype=np.uint8)
    state, reads = INVOICE
    picture, invoice = model.system_one_batch(
        [(None, PICTURE_QUESTIONS, [Image.fromarray(pixels)]), (state, reads, None)]
    )
    for out, questions in ((picture, PICTURE_QUESTIONS), (invoice, reads)):
        types = {name: answer["type"] for name, answer in out["answers"].items()}
        assert types == {name: question["type"] for name, question in questions.items()}
        assert out["usage"]["input_tokens"] > 0 and out["usage"]["output_tokens"] == 0

    cats, remote, every_cat = (picture["answers"][name] for name in PICTURE_QUESTIONS)
    low, high = PICTURE_QUESTIONS["cats"]["range"]
    p10, p90 = cats["interval"]
    # the estimate's grid is the range in steps of 1
    assert p10 <= cats["number"] <= p90 and {p10, cats["number"], p90} <= set(range(low, high + 1))
    presence = every_cat["presence"]
    assert len(presence) == PICTURE_QUESTIONS["every_cat"]["max"]
    assert presence == sorted(presence, reverse=True) and all(0 <= p <= 1 for p in presence)
    assert every_cat["count"] == len(every_cat["boxes"]) == sum(p >= 0.5 for p in presence)
    assert (remote["box"] is None) == (remote["found"] < 0.5) and 0 <= remote["found"] <= 1
    for answer in every_cat["boxes"] + [remote] * (remote["box"] is not None):
        assert answer["found"] >= 0.5 and 0 <= answer["confidence"] <= 1
        box = answer["box"]
        assert 0 <= box["top"] < box["bottom"] <= 1 and 0 <= box["left"] < box["right"] <= 1

    for read in invoice["answers"].values():
        span = read["span"]
        assert 0 <= read["found"] <= 1 and (span is None or read["found"] >= 0.5)
        assert span is None or read["text"] == state[span[0] : span[1]]


# Batched against one row a forward. Rows of a forward are independent, but a row right-padded to
# its batch's width rounds differently from the row alone. Measured over 12 pairs of pictures on
# arm64: 5e-13 (fp32), 3e-10 (fp16), 7e-6 (q4) and 2e-6 (q8). The pictures are asked their number
# only: a box edge of the random weights is read off a flat distribution, where the int8
# activations of q4 and q8 on the CPU EP move it by up to 7e-2.
BATCHED = {"fp32": 1e-5, "fp16": 1e-4, "q4": 1e-3, "q8": 1e-3}
# token_budget -> rows of each forward, shortest first: the pictures, the invoice, the ticket and
# the long invoice in one forward, and the first three then the last two
BUDGETS = {65536: [5], 2500: [3, 2]}


@pytest.mark.parametrize("budget", list(BUDGETS))
@pytest.mark.parametrize("precision", list(BATCHED))
def test_batched_rows(monkeypatch, export_with_host_code, precision, budget):
    """Rows of different lengths right-padded into a forward (token_budget) give the answers of
    one row a forward: two pictures, whose rows are the shortest (their features are the first of
    the batch, their slots under the most padding), a ticket's options, and an invoice's reads
    over a short and a long state, whose pointer rows are then the third and the last of a
    forward, or the third of one and the second of the next."""
    rng = np.random.default_rng(0)
    pictures = [
        rng.integers(0, 256, (*size, 3), dtype=np.uint8) for size in ((240, 320), (160, 200))
    ]
    cats = {
        "cat": {"type": "noul", "instructions": "Is there a cat in the picture?"},
        "cats": PICTURE_QUESTIONS["cats"],
    }
    state, reads = INVOICE
    batch = [
        (None, cats, [Image.fromarray(pictures[0])]),
        (*TICKET, None),
        (None, cats, [Image.fromarray(pictures[1])]),
        (state, reads, None),
        ("\n".join([state] * 3), reads, None),
    ]
    expected = load(export_with_host_code, precision, ep="cpu").system_one_batch(batch)
    model = load(export_with_host_code, precision, ep="cpu", token_budget=budget)
    forwards, pointer_rows = [], []
    slot_logits = type(model)._slot_logits

    def spy(self, seqs, rows, slots, pointer):
        forwards.append(len(rows))
        pointer_rows.append(pointer and pointer["rows"].tolist())
        return slot_logits(self, seqs, rows, slots, pointer)

    monkeypatch.setattr(type(model), "_slot_logits", spy)
    got = model.system_one_batch(batch)
    assert forwards == BUDGETS[budget]
    assert pointer_rows == ([[2, 4]] if budget == 65536 else [[2], [1]])
    bound = BATCHED[precision]
    assert not list(differences(expected, got, bound, bound))


def test_mask_limit(monkeypatch, export_with_host_code):
    """A batch whose set mask (rows x longest row²) would pass MASK_ELEMENTS is split, shortest
    rows first, and a token_budget under 1 is refused."""
    from liquidonnx.d1 import session

    model = load(export_with_host_code, "fp32", ep="cpu", token_budget=65536)
    seqs = [(i, 0, [0] * n) for i, n in enumerate((300, 200, 700))]  # (request, _, ids)
    assert model._batches(seqs) == [[1, 0, 2]]
    # room for the two shortest rows at the second's width, not for the third beside them
    monkeypatch.setattr(session, "MASK_ELEMENTS", 2 * 300**2)
    assert model._batches(seqs) == [[1, 0], [2]]
    with pytest.raises(ValueError, match="token_budget"):
        load(export_with_host_code, "fp32", ep="cpu", token_budget=0)
