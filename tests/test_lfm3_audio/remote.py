"""
The PyTorch references of LFM3-Audio checkpoints: their own remote code (trust_remote_code), fp32,
sdpa, TF32 off. LFM3-Audio-2B and LFM3-Omni-2B build on transformers 5.19+ (their code imports
5.19 APIs: WeightRenaming, create_recurrent_attention_mask, ...).

reference/ vendors the modeling and configuration files of LFM3-Audio-2B at revision 9d49701
(Apache-2.0 headers), with the checkpoint names in their docstrings set to the released ones, for
the tiny checkpoint of tiny.py, which loads them from its own folder like any checkpoint.

MaskGIT orders come from torch.randperm in the reference's sample_audio_frame; inject_orders feeds
it the orders the ONNX runtime takes from session.rng_orders, so greedy runs match frame for frame.
"""

import contextlib
import hashlib
import logging
import pathlib
from collections.abc import Callable, Iterable

import numpy as np
import pytest
import torch
import transformers

from liquidonnx.compare import atomic_write
from liquidonnx.lfm3_audio.config import codec_dir

logger = logging.getLogger(__name__)

SPEECH_SAMPLING = (
    "audio_temperature",
    "audio_top_k",
    "audio_repetition_penalty",
    "audio_repetition_window",
)


def remote_class(checkpoint: str | pathlib.Path, name: str):
    """A class of the checkpoint's remote code, e.g. "modeling_liquid_codec.LiquidCodecModel"."""
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    return get_class_from_dynamic_module(name, str(checkpoint))


def fp32_matmuls():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def load_reference(checkpoint: str | pathlib.Path):
    """float32 Lfm3AudioForConditionalGeneration (or Lfm3OmniForConditionalGeneration) of a
    checkpoint directory with every tensor loaded, or a skip naming the transformers release."""
    from transformers import AutoModelForMultimodalLM

    fp32_matmuls()
    try:
        model, info = AutoModelForMultimodalLM.from_pretrained(
            str(checkpoint),
            dtype=torch.float32,
            trust_remote_code=True,
            attn_implementation="sdpa",
            output_loading_info=True,
        )
    except ImportError as e:
        pytest.skip(f"transformers {transformers.__version__} cannot build {checkpoint}: {e}")
    problems = {k: v for k, v in info.items() if v}
    if problems:
        raise ValueError(f"reference loaded {checkpoint} with {problems}")
    # LFM3-Audio-2B's and LFM3-Omni-2B's generation_config.json hold their cards' speech sampling,
    # which their generate reads when a call omits it; the tests' reference speech stays greedy
    for name in SPEECH_SAMPLING:
        setattr(model.generation_config, name, None)
    return model.eval()


def load_codec_reference(checkpoint: str | pathlib.Path):
    """float32 LiquidCodecModel of the checkpoint's codec folder."""
    fp32_matmuls()
    codec = remote_class(checkpoint, "modeling_liquid_codec.LiquidCodecModel")
    model, info = codec.from_pretrained(
        str(checkpoint),
        subfolder=codec_dir(checkpoint).name,
        dtype=torch.float32,
        attn_implementation="sdpa",
        output_loading_info=True,
    )
    problems = {k: v for k, v in info.items() if v}
    if problems:
        raise ValueError(f"codec reference loaded {checkpoint} with {problems}")
    return model.eval()


def cached_slots(
    folder: pathlib.Path,
    key: tuple,
    device: str,
    compute: Callable[[], tuple[list[int], np.ndarray]],
) -> tuple[list[int], np.ndarray]:
    """compute()'s (generated tokens, frames [F, 12]) on device, kept in folder (--reference-cache)
    as <digest of key>.npz: a file there is read instead, whichever host and device wrote it."""
    path = folder / f"{hashlib.sha256(repr(key).encode()).hexdigest()[:16]}.npz"
    if path.exists():
        with np.load(path) as saved:
            logger.info(f"  reference slots computed on {saved['device']}: {path}")
            return saved["tokens"].tolist(), saved["frames"]
    tokens, frames = compute()
    with atomic_write(path) as f:
        np.savez(f, tokens=np.asarray(tokens, dtype=np.int64), frames=frames, device=device)
    return tokens, frames


def maskgit_orders(seed: int, count: int, num_codebooks: int = 12) -> list[np.ndarray]:
    """count orders of numpy.random.default_rng(seed), as session.rng_orders(seed) draws them."""
    rng = np.random.default_rng(seed)
    return [rng.permutation(num_codebooks) for _ in range(count)]


@contextlib.contextmanager
def inject_orders(orders: Iterable[np.ndarray]):
    """torch.randperm returns the next of orders while the context is open: the reference's
    MaskGIT order of each sampled frame."""
    orders = iter(orders)

    def randperm(n, *args, device=None, **kwargs):
        order = next(orders)
        if len(order) != n:
            raise ValueError(f"injected order of {len(order)} codebooks, randperm({n})")
        return torch.as_tensor(np.asarray(order), dtype=torch.long, device=device)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(torch, "randperm", randperm)
        yield
