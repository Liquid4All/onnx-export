"""LFM3 test fixtures."""

import gc
import logging

import pytest
import torch
import transformers
from huggingface_hub.errors import StrictDataclassError
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig

from liquidonnx.genai_builder import resolve_checkpoint
from liquidonnx.lfm3.session import CHECKPOINT_PATTERNS, REVISIONS, load_tokenizer

logger = logging.getLogger(__name__)


def checkpoint(model_id: str) -> str:
    """The checkpoint at its validated revision (session.REVISIONS), if pinned."""
    repo, revision = model_id, REVISIONS.get(model_id)
    return str(resolve_checkpoint(repo, CHECKPOINT_PATTERNS, revision=revision))


def load_reference(model_id: str):
    """float32 PyTorch model of a checkpoint's remote code, or a skip.

    Each checkpoint's remote code builds only on some transformers releases: LFM3-1.7B's on
    5.8-5.14 and 5.18+, the older format's on 5.0-5.1. 5.15-5.17 rename the config's "conv"
    layers to "linear_attention", which the restore below undoes. The skip names the release so
    the test can be rerun with uv run --with transformers==X.
    """
    path = checkpoint(model_id)
    raw_layer_types = PretrainedConfig.get_config_dict(path)[0].get("layer_types")
    try:
        config = AutoConfig.from_pretrained(path, trust_remote_code=True)
        if raw_layer_types is not None:
            config.layer_types = raw_layer_types
        model = AutoModelForCausalLM.from_pretrained(
            path, config=config, dtype=torch.float32, trust_remote_code=True
        )
    except (ValueError, TypeError, StrictDataclassError) as e:
        pytest.skip(
            f"transformers {transformers.__version__} cannot build {model_id}: "
            f"{type(e).__name__}: {str(e)[:200]}"
        )
    return model.eval()


def pytest_generate_tests(metafunc):
    """pytorch_model runs on --lfm3-model."""
    if "pytorch_model" in metafunc.fixturenames:
        model = metafunc.config.getoption("--lfm3-model")
        metafunc.parametrize("pytorch_model", [model], indirect=True)


@pytest.fixture(scope="module")
def model_tokenizer(request):
    """(model_id, tokenizer) for a model ID given with indirect=True."""
    model_id = request.param
    return model_id, load_tokenizer(checkpoint(model_id))


@pytest.fixture(scope="module")
def pytorch_model(request, device):
    """(model_id, model, tokenizer) for a model ID given with indirect=True, on --device."""
    model_id = request.param
    logger.info(f"Loading PyTorch model from {model_id}...")
    model = load_reference(model_id).to(device)
    yield model_id, model, load_tokenizer(checkpoint(model_id))

    logger.info(f"Cleaning up {model_id} model...")
    del model
    gc.collect()
