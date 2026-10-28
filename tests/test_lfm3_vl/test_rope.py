"""
The LFM3-VL decoder's RoPE tables of a flat config.json (no download): that layout computes
inv_freq as theta ** -x, an ulp off the 1 / theta ** x of the text models and LFM3-VL-2B, which
tests/test_lfm3/test_rope.py checks.

Run with:
    uv run pytest tests/test_lfm3_vl/test_rope.py -v
"""

import dataclasses

import numpy as np
import pytest
import torch

from liquidonnx.lfm3.builder import LFM3Builder
from liquidonnx.lfm3_vl.config import parse_config

from .tiny import TINY_CONFIG

MAX_POSITIONS = 40000
HEAD_DIM = 128


def reference_tables(theta: float, partial: float) -> tuple[np.ndarray, np.ndarray]:
    """cos and sin of the rotated pairs, theta ** -x angles in float32."""
    pairs = int(HEAD_DIM * partial) // 2
    frequencies = theta ** (-(2.0 * torch.arange(pairs, dtype=torch.float32) / HEAD_DIM))
    angles = torch.arange(MAX_POSITIONS).float()[:, None] * frequencies
    return angles.cos().numpy(), angles.sin().numpy()


@pytest.mark.parametrize("theta,partial", [(1e6, 0.5), (1e4, 1.0)], ids=["full", "sliding"])
def test_decoder_rope_tables_match_reference(theta, partial):
    language, _, _ = parse_config(TINY_CONFIG)
    assert language.rope_negative_power
    config = dataclasses.replace(language, head_dim=HEAD_DIM, max_position_embeddings=MAX_POSITIONS)
    cos, sin = LFM3Builder(config)._compute_rope_cache(theta, partial)

    pairs = int(HEAD_DIM * partial) // 2
    ref_cos, ref_sin = reference_tables(theta, partial)
    assert np.abs(cos[:, :pairs] - ref_cos).max() < 2e-7
    assert np.abs(sin[:, :pairs] - ref_sin).max() < 2e-7
