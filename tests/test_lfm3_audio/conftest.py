"""LFM3-Audio fixtures."""

import functools
import hashlib
import pathlib
import shutil
import sys

import pytest

from liquidonnx.compare import checkpoint_revision
from liquidonnx.lfm3_audio.config import REVISIONS

from .remote import load_codec_reference, load_reference
from .tiny import write_omni_checkpoint, write_release_files, write_tiny_checkpoint

CHECKPOINT_PATTERNS = [
    "*.json",
    "*.jinja",
    "*.py",
    "model*.safetensors",
    "audio_detokenizer/*",
    "audio_tokenizer/*",
]


@pytest.fixture(scope="session")
def tiny_checkpoint(tmp_path_factory) -> pathlib.Path:
    return write_tiny_checkpoint(tmp_path_factory.mktemp("lfm3_audio_tiny"))


@pytest.fixture(scope="session")
def tiny_reference(tiny_checkpoint):
    return load_reference(tiny_checkpoint)


@pytest.fixture(scope="session")
def tiny_codec_reference(tiny_checkpoint):
    return load_codec_reference(tiny_checkpoint)


@pytest.fixture(scope="session")
def tiny_exports(tiny_checkpoint, tmp_path_factory):
    """export(cli, release) -> the folder of lfm3-audio-export's or lfm3-omni-export's main() with
    --precision q4 q4f16, built once, on the tiny checkpoint in release's layout (LFM3-Audio-2B,
    or LFM3-Omni-2B: write_omni_checkpoint) with release's generation_config.json and tokenizer
    (write_release_files). The tiny checkpoint itself has neither: its reference's generate would
    read the file."""
    from liquidonnx.lfm3_audio import export as audio_export
    from liquidonnx.lfm3_omni import export as omni_export

    root = tmp_path_factory.mktemp("lfm3_audio_tiny_exports")

    @functools.cache
    def checkpoint(release: str) -> pathlib.Path:
        directory = root / f"tiny-{release}"
        if release == "LFM3-Omni-2B":
            write_omni_checkpoint(tiny_checkpoint, directory)
        else:
            shutil.copytree(tiny_checkpoint, directory)
        return write_release_files(directory, release)

    @functools.cache
    def export(cli: str, release: str) -> pathlib.Path:
        output_dir = root / cli
        argv = [cli, str(checkpoint(release)), "--output-dir", str(output_dir)]
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(sys, "argv", [*argv, "--precision", "q4", "q4f16"])
            (omni_export if cli == "lfm3-omni-export" else audio_export).main()
        return output_dir / "exports" / f"tiny-{release}-ONNX"

    return export


# === Real checkpoint ===


@pytest.fixture(scope="session")
def model_id(request) -> str:
    """The checkpoint under test, --lfm3-audio-model."""
    return request.config.getoption("--lfm3-audio-model")


@pytest.fixture(scope="session")
def checkpoint(model_id) -> pathlib.Path:
    """The checkpoint at its validated revision (config.REVISIONS); HF_HUB_OFFLINE=1 reads the
    cache."""
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
        pytest.fail(
            f"No export in {path}\nExport with: uv run lfm3-audio-export {model_id} --precision"
        )
    return path


@pytest.fixture(scope="session")
def reference(checkpoint, device):
    """float32 reference model on --device."""
    return load_reference(checkpoint).to(device)


@pytest.fixture(scope="session")
def slot_cache(request, model_id, checkpoint):
    """cache(*clip paths) -> the cache argument of test_runtime.reference_slots: the folder of
    --reference-cache and a key of the checkpoint's revision and the clips' bytes; None without
    the option."""
    folder = request.config.getoption("--reference-cache")
    revision = checkpoint_revision(model_id, checkpoint)

    def cache(*clips: pathlib.Path):
        if not folder:
            return None
        digests = tuple(hashlib.sha256(clip.read_bytes()).hexdigest() for clip in clips)
        return pathlib.Path(folder), (revision, *digests)

    return cache


@pytest.fixture(scope="session")
def codec_reference(checkpoint, device):
    """float32 reference codec on --device."""
    return load_codec_reference(checkpoint).to(device)


@pytest.fixture(scope="session")
def reference_processor(checkpoint):
    """The checkpoint's processor (feature extractor, tokenizer, chat template)."""
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(str(checkpoint), trust_remote_code=True)


@pytest.fixture(scope="session")
def onnx_models(export_dir, ep):
    """(precision, tf32=False) -> session.ONNXAudioModel with sessions on --ep and the tokenizer
    (one precision's cached, so a GPU holds one bundle at a time)."""
    from liquidonnx.lfm3_audio.session import ONNXAudioModel

    models = {}

    def get(precision: str, tf32: bool = False):
        if (precision, tf32) not in models:
            if any(key[0] != precision for key in models):
                models.clear()
            model = ONNXAudioModel(export_dir, precision, tf32=tf32)
            model.load_sessions(ep=ep)
            model.load_preprocessing()
            models[precision, tf32] = model
        return models[precision, tf32]

    return get
