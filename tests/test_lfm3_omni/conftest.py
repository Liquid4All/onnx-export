"""LFM3-Omni fixtures: --lfm3-omni-model at its validated revision, its own remote code as the
float32 reference (test_lfm3_audio.remote.load_reference), and its export under --exports-dir.

The speech side of the export runs tests/test_lfm3_audio with
--lfm3-audio-model LiquidAI/LFM3-Omni-2B; the tests here cover what LFM3-Omni adds:
images and video, alone and next to speech."""

import pathlib

import pytest

from liquidonnx.lfm3_audio.config import REVISIONS

from ..test_lfm3_audio.conftest import CHECKPOINT_PATTERNS
from ..test_lfm3_audio.remote import load_reference
from ..test_lfm3_vl.conftest import ASSETS


@pytest.fixture(scope="session")
def model_id(request) -> str:
    """The checkpoint under test, --lfm3-omni-model."""
    return request.config.getoption("--lfm3-omni-model")


@pytest.fixture(scope="session")
def checkpoint(model_id) -> pathlib.Path:
    """The checkpoint at its pinned revision (REVISIONS); HF_HUB_OFFLINE=1 reads the cache."""
    from huggingface_hub import snapshot_download

    repo, revision = model_id, REVISIONS.get(model_id)
    return pathlib.Path(
        snapshot_download(repo, revision=revision, allow_patterns=CHECKPOINT_PATTERNS)
    )


@pytest.fixture(scope="session")
def export_dir(exports_dir, model_id) -> pathlib.Path:
    from helpers import get_export_dir

    path = get_export_dir(exports_dir, model_id)
    if not (path / "onnx" / "vision_encoder.onnx").exists():
        pytest.fail(
            f"No export in {path}\nExport with: uv run lfm3-omni-export {model_id} --precision"
        )
    return path


@pytest.fixture(scope="session")
def reference(checkpoint, device):
    """float32 Lfm3OmniForConditionalGeneration on --device."""
    return load_reference(checkpoint).to(device)


@pytest.fixture(scope="session")
def reference_processor(checkpoint):
    """The checkpoint's Lfm3OmniProcessor (images, video, audio, chat template)."""
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(str(checkpoint), trust_remote_code=True)


@pytest.fixture(scope="session")
def onnx_models(export_dir, ep):
    """(precision, tf32=False) -> session.ONNXOmniModel with sessions on --ep and its
    preprocessing, cached one precision at a time (the tests run precision by precision) to bound
    GPU memory."""
    from liquidonnx.lfm3_omni.session import ONNXOmniModel

    models = {}

    def get(precision: str, tf32: bool = False):
        if (precision, tf32) not in models:
            if any(key[0] != precision for key in models):
                models.clear()
            model = ONNXOmniModel(export_dir, precision, tf32=tf32)
            model.load_sessions(ep=ep)
            model.load_preprocessing()
            models[precision, tf32] = model
        return models[precision, tf32]

    return get


def load_images(names: list[str]) -> list:
    from PIL import Image

    return [Image.open(ASSETS / name).convert("RGB") for name in names]


def video_frames():
    """A 4 s clip at 2 fps (8 frames of 320x240, the cardinal panning right) and its metadata:
    4 temporal patches of 16x20 patches, 80 tokens each."""
    import numpy as np

    [cardinal] = load_images(["cardinal.jpg"])
    cardinal = cardinal.resize((480, 360))
    frames = [np.asarray(cardinal.crop((20 * i, 60, 20 * i + 320, 300))) for i in range(8)]
    return np.stack(frames), {"fps": 2.0, "total_num_frames": len(frames)}
