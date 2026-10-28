"""LFM3-VL fixtures."""

import pathlib
import shutil

import pytest

from liquidonnx.lfm3_vl.config import REVISIONS

from ..test_lfm3.tiny import write_tokenizer
from .reference.loader import load_reference
from .tiny import write_tiny_checkpoint


@pytest.fixture(scope="session")
def tiny_checkpoint(tmp_path_factory):
    """The tiny checkpoint with the files lfm3-vl-infer reads: LFM3-VL-2B's processor_config.json,
    and a tokenizer with LFM3-1.7B's special ids and chat template (LFM3-VL-2B has the same)."""
    directory = write_tiny_checkpoint(tmp_path_factory.mktemp("lfm3_vl_tiny"))
    write_tokenizer(directory)
    assets = pathlib.Path(__file__).parent / "assets" / "LFM3-VL-2B"
    shutil.copy2(assets / "processor_config.json", directory)
    return directory


@pytest.fixture(scope="session")
def tiny_reference(tiny_checkpoint):
    return load_reference(tiny_checkpoint)


@pytest.fixture(scope="session")
def tiny_export(tiny_checkpoint, tmp_path_factory):
    """Export folder of the tiny checkpoint with every precision."""
    from liquidonnx.lfm3_vl.export import copy_checkpoint_files, derive, export_fp32

    output_dir = tmp_path_factory.mktemp("lfm3_vl_tiny_export")
    onnx_dir = output_dir / "onnx"
    onnx_dir.mkdir()
    export_fp32(tiny_checkpoint, onnx_dir)
    copy_checkpoint_files(tiny_checkpoint, output_dir)
    derive(onnx_dir, ["fp16", "q4", "q8", "q4f16", "q8f16"], block_size=32)
    return output_dir


# === Real checkpoint ===

ASSETS = pathlib.Path(__file__).parent.parent / "test_lfm2_vl" / "assets"


@pytest.fixture(scope="session")
def model_id(request) -> str:
    """The checkpoint under test, --lfm3-vl-model."""
    return request.config.getoption("--lfm3-vl-model")


@pytest.fixture(scope="session")
def checkpoint(model_id):
    """The checkpoint at its validated revision (config.REVISIONS), if pinned."""
    from huggingface_hub import snapshot_download

    repo, revision = model_id, REVISIONS.get(model_id)
    return pathlib.Path(
        snapshot_download(
            repo,
            revision=revision,
            allow_patterns=["*.json", "*.jinja", "*.py", "*.safetensors"],
        )
    )


@pytest.fixture(scope="session")
def checkpoint_files(model_id):
    """The checkpoint's tokenizer, processor and config files (and code), without the weights."""
    from huggingface_hub import snapshot_download

    repo, revision = model_id, REVISIONS.get(model_id)
    return pathlib.Path(
        snapshot_download(repo, revision=revision, allow_patterns=["*.json", "*.jinja", "*.py"])
    )


@pytest.fixture(scope="session")
def export_dir(exports_dir, model_id):
    from helpers import get_export_dir

    path = get_export_dir(exports_dir, model_id)
    if not (path / "onnx" / "decoder.onnx").exists():
        pytest.fail(
            f"No export in {path}\nExport with: uv run lfm3-vl-export {model_id} --precision"
        )
    return path


@pytest.fixture(scope="session")
def reference_processor(checkpoint_files):
    """The checkpoint's own processor (its remote code), as the reference model is."""
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(str(checkpoint_files), trust_remote_code=True)


@pytest.fixture(scope="session")
def reference(checkpoint, device):
    """float32 reference model on --device."""
    return load_reference(checkpoint).to(device)


@pytest.fixture(scope="session")
def onnx_models(export_dir, ep):
    """precision -> ONNXVLModel with sessions on --ep (the last one cached, so a GPU holds one
    bundle at a time; the tests run precision outermost)."""
    from liquidonnx.lfm3_vl.session import ONNXVLModel

    models = {}

    def get(precision: str):
        if precision not in models:
            models.clear()
            model = ONNXVLModel(export_dir, precision, tf32=False)
            model.load_sessions(ep=ep)
            models[precision] = model
        return models[precision]

    return get


def chat(images: list[str], question: str) -> tuple[list[dict], list]:
    from PIL import Image

    content = [{"type": "image"} for _ in images] + [{"type": "text", "text": question}]
    pil = [Image.open(ASSETS / name).convert("RGB") for name in images]
    return [{"role": "user", "content": content}], pil
