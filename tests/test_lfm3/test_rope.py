"""
The builder's RoPE tables against Lfm3RotaryEmbedding's float32 arithmetic (no download).

Run with:
    uv run pytest tests/test_lfm3/test_rope.py -v
"""

import numpy as np
import pytest
import torch

from liquidonnx.lfm3.builder import LFM3Builder, LFM3Config

# Past the 32K training context, where an exact (float64) angle is ~2e-3 off the float32 one.
MAX_POSITIONS = 40000
HEAD_DIM = 128


def reference_tables(theta: float, partial: float) -> tuple[np.ndarray, np.ndarray]:
    """compute_default_rope_parameters + forward of the LFM3 remote code, half-width."""
    rope_angles = int(partial * HEAD_DIM) // 2
    exponents = torch.arange(0, 2 * rope_angles, 2, dtype=torch.int64).float() / HEAD_DIM
    inv_freq = torch.cat([1.0 / (theta**exponents), torch.zeros(HEAD_DIM // 2 - rope_angles)])
    positions = torch.arange(MAX_POSITIONS).float()
    # The remote code's [D/2, 1] @ [1, P] as an elementwise product, one float32 multiply each: a
    # CPU matmul did not always return those on an x86 host (cos(1) 0.5403335, not 0.5403023)
    freqs = positions[:, None] * inv_freq[None, :]
    return freqs.cos().numpy(), freqs.sin().numpy()


@pytest.mark.parametrize(
    "theta,partial", [(1e6, 0.5), (1e4, 1.0)], ids=["full_attention", "sliding_attention"]
)
def test_rope_tables_match_pytorch(theta: float, partial: float):
    config = LFM3Config(
        hidden_size=2048,
        num_hidden_layers=1,
        num_attention_heads=16,
        num_key_value_heads=4,
        vocab_size=128000,
        layer_types=["full_attention"],
        head_dim=HEAD_DIM,
        intermediate_size=7680,
        max_position_embeddings=MAX_POSITIONS,
    )
    cos, sin = LFM3Builder(config)._compute_rope_cache(theta, partial)
    ref_cos, ref_sin = reference_tables(theta, partial)
    assert np.abs(cos - ref_cos).max() < 2e-7
    assert np.abs(sin - ref_sin).max() < 2e-7
