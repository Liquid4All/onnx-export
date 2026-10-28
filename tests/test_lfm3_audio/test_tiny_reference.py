"""
The tiny checkpoint's PyTorch reference (no download): it builds from the vendored remote code,
reads the trainer's names, and takes injected MaskGIT orders. The graph parity tests build on it.

Run with:
    uv run pytest tests/test_lfm3_audio/test_tiny_reference.py -v
"""

import numpy as np
import torch

from .remote import inject_orders, maskgit_orders


def test_forward_and_frame(tiny_reference):
    config = tiny_reference.config
    # 3 encoder tokens (17..24 mel frames), 2 replayed speech frames
    features = torch.randn(1, 128, 20, generator=torch.Generator().manual_seed(0))
    codes = torch.tensor([[5] * 12, [2016] * 12])
    ids = torch.tensor(
        [[124894] + [config.audio_token_id] * 3 + [config.audio_out_token_id] * 2 + [124899]]
    )
    with torch.no_grad():
        out = tiny_reference.model(input_ids=ids, input_features=features, audio_out_codes=codes)
    hidden = out.last_hidden_state[0, -1]
    assert out.last_hidden_state.shape == (1, ids.shape[1], 64)

    # Greedy frames are a function of the hidden state and the order
    randperm = torch.randperm
    [order] = maskgit_orders(seed=1, count=1)
    with torch.no_grad(), inject_orders([order, order]):
        frame = tiny_reference.sample_audio_frame(hidden)
        again = tiny_reference.sample_audio_frame(hidden)
    assert torch.randperm is randperm
    assert frame.shape == (12,) and 0 <= int(frame.min()) and int(frame.max()) < 2017
    assert torch.equal(frame, again)
    assert np.array_equal(order, np.random.default_rng(1).permutation(12))


def test_codec_decode(tiny_codec_reference):
    codes = torch.randint(0, 2016, (1, 12, 5), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        wave = tiny_codec_reference.decode(codes).audio_values
    assert wave.shape == (1, 5 * 1920) and torch.isfinite(wave).all()
