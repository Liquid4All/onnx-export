"""d1 fixtures: the tiny checkpoint and its export, built once a session; the checkpoint of
--d1-model at its pinned revision with its own remote code as the reference (D1Model's SystemOne
engine: AutoModel, trust_remote_code, float32, TF32 off), its export under --exports-dir run by
d1.session, and the README's requests."""

import hashlib
import json
import pathlib
import urllib.request

import numpy as np
import pytest

from liquidonnx.compare import atomic_write
from liquidonnx.d1.config import REVISIONS
from liquidonnx.d1.export import CHECKPOINT_PATTERNS, copy_checkpoint_files, derive, export_fp32

from ..test_lfm3_vl.tiny import write_tiny_checkpoint
from .test_config import d1_config

# The d1 model cards' picture (two cats, COCO val2017), downloaded on first use: not in the repo
CATS_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
CATS_SHA256 = "dea9e7ef97386345f7cff32f9055da4982da5471c48d575146c796ab4563b04e"


def cats() -> pathlib.Path:
    """CATS_URL's picture, kept in ~/.cache/liquidonnx once its sha256 is CATS_SHA256 (the URL is
    plain http)."""
    path = pathlib.Path.home() / ".cache" / "liquidonnx" / "000000039769.jpg"
    if not path.exists():
        with urllib.request.urlopen(CATS_URL) as response, atomic_write(path) as f:
            data = response.read()
            digest = hashlib.sha256(data).hexdigest()
            assert digest == CATS_SHA256, f"{CATS_URL}: sha256 {digest}, not {CATS_SHA256}"
            f.write(data)
    return path


# The checkpoint README's examples
TEAM = {
    "type": "choice",
    "instructions": "Which team should handle this?",
    "criteria": {
        "billing": "Charges, refunds, invoices",
        "shipping": "Delivery and parcels",
        "technical": "App or site faults",
    },
}
TICKET = (
    "I was charged twice and my parcel never arrived. Please refund one charge.",
    {
        "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
        "team": TEAM,
        "topics": {
            "type": "choice",
            "multiple": True,
            "instructions": "Which topics does the message raise?",
            "criteria": {
                "billing": "Payments and charges",
                "delivery": "Shipping and parcels",
                "login": "Signing in to the account",
            },
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this ticket?",
            "criteria": ["low", "medium", "high"],
        },
    },
)
INVOICE = (
    "ACME SA\nInvoice 2026-117\nTotal due: 1 234,50 EUR\nPay before 03/04/2026 (day/month/year)",
    {
        "payee": {"type": "text", "instructions": "Who issued the invoice?"},
        "total": {"type": "number", "instructions": "How much is due?"},
        "due": {"type": "date", "instructions": "By when must it be paid?"},
        "vat": {"type": "number", "instructions": "How much VAT is charged?"},
    },
)
PICTURE_QUESTIONS = {
    "cats": {
        "type": "number",
        "instructions": "How many cats are there?",
        "range": [0, 10],
        "step": 1,
    },
    "remote": {"type": "box", "instructions": "Where is the remote control on the left?"},
    "every_cat": {"type": "boxes", "instructions": "Every cat.", "max": 4},
}
CLIP_QUESTIONS = {
    "beeps": {
        "type": "number",
        "instructions": "How many beeps are there?",
        "range": [0, 10],
        "step": 1,
    },
    "speech": {"type": "noul", "instructions": "Does someone speak in the recording?"},
}
PARCELS = ["Where is my parcel? It was due Monday.", "The app crashes when I open settings."]
# A spoken question (samples/d1/clip.json, d1-infer's sample): speech beside the README's beeps
SPEECH = (
    pathlib.Path(__file__).parents[2] / "samples" / "audio" / "woodworks_question.wav",
    {
        "speech": {"type": "noul", "instructions": "Does someone speak in the recording?"},
        "question": {"type": "noul", "instructions": "Does the speaker ask a question?"},
    },
)


def beeps(rate: int = 16000) -> np.ndarray:
    """4 s of faint noise with three 0.2 s beeps at 880 Hz (d1-Omni-2B's model card's clip)."""
    clip = np.random.default_rng(0).normal(0, 0.003, 4 * rate).astype(np.float32)
    tone = 0.3 * np.sin(2 * np.pi * 880 * np.arange(3200) / rate)
    for start in (0.5, 1.5, 2.5):
        clip[int(start * rate) : int(start * rate) + 3200] += tone
    return clip


def requests(model_id: str) -> dict[str, list[tuple]]:
    """name -> (state, questions, media) requests of one system_one_batch call; the clips for
    d1-Omni only."""
    from PIL import Image

    out = {
        "ticket": [(*TICKET, None)],
        "invoice": [(*INVOICE, None)],
        "parcels": [(text, {"team": TEAM}, None) for text in PARCELS],
        "picture": [(None, PICTURE_QUESTIONS, [Image.open(cats())])],
    }
    if "omni" in model_id.lower():
        clip = {"type": "audio", "audio": beeps(), "sampling_rate": 16000}
        out["clip"] = [(None, CLIP_QUESTIONS, [clip])]
        out["speech"] = [(None, SPEECH[1], [{"type": "audio", "path": str(SPEECH[0])}])]
    return out


@pytest.fixture(scope="session")
def tiny_checkpoint(tmp_path_factory):
    directory = write_tiny_checkpoint(tmp_path_factory.mktemp("d1_tiny"))
    config = json.loads((directory / "config.json").read_text())
    (directory / "config.json").write_text(json.dumps(d1_config(config), indent=2))
    return directory


@pytest.fixture(scope="session")
def tiny_export(tiny_checkpoint, tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("d1_tiny_export")
    onnx_dir = output_dir / "onnx"
    onnx_dir.mkdir()
    export_fp32(tiny_checkpoint, onnx_dir)
    copy_checkpoint_files(tiny_checkpoint, output_dir)
    derive(onnx_dir, ["fp16", "q4", "q8", "q4f16", "q8f16"], block_size=32)
    return output_dir


@pytest.fixture(scope="session")
def model_id(request) -> str:
    return request.config.getoption("--d1-model")


@pytest.fixture(scope="session")
def checkpoint(model_id) -> pathlib.Path:
    """The checkpoint at its pinned revision; HF_HUB_OFFLINE=1 reads the cache."""
    from huggingface_hub import snapshot_download

    repo, revision = model_id, REVISIONS.get(model_id)
    return pathlib.Path(
        snapshot_download(repo, revision=revision, allow_patterns=CHECKPOINT_PATTERNS)
    )


@pytest.fixture(scope="session")
def export_dir(exports_dir, model_id) -> pathlib.Path:
    from helpers import get_export_dir

    path = get_export_dir(exports_dir, model_id)
    if not (path / "onnx" / "decoder.onnx").exists():
        pytest.fail(f"No export in {path}\nExport with: uv run d1-export {model_id} --precision")
    return path


def load_reference(checkpoint: pathlib.Path, device: str, dtype: str):
    """D1Model of the checkpoint's own code on device, in float32 or bfloat16."""
    import torch
    from transformers import AutoModel

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = AutoModel.from_pretrained(
        str(checkpoint), trust_remote_code=True, dtype=getattr(torch, dtype)
    )
    return model.to(device).eval()


@pytest.fixture(scope="session")
def reference(checkpoint, device):
    """The float32 reference's engine (SystemOne)."""
    return load_reference(checkpoint, device, "float32").engine


@pytest.fixture(scope="session")
def reference_bf16(checkpoint, device):
    """The bfloat16 reference's engine on CUDA: the floor of the bounds; None on the CPU."""
    return load_reference(checkpoint, device, "bfloat16").engine if device == "cuda" else None


@pytest.fixture(scope="session")
def onnx_engines(export_dir, ep):
    """precision -> d1.session OrtSystemOne on --ep, one precision at a time (the tests run
    precision by precision) to bound memory."""
    from liquidonnx.d1.session import load

    engines = {}

    def get(precision: str):
        if precision not in engines:
            engines.clear()
            engines[precision] = load(export_dir, precision, ep, tf32=False)
        return engines[precision]

    return get
