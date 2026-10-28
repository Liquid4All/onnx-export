"""
codec_decoder.onnx of the tiny checkpoint (no download) against its LiquidCodecModel on the CPU:
the STFT features of a batch of two utterances, and their waveforms through session.stft_to_wave.
42 positions span the tiny codec's window of 8 several times.

Run with:
    uv run pytest tests/test_lfm3_audio/test_tiny_codec.py -v
"""

import numpy as np
import onnxruntime as ort
import torch

from liquidonnx.lfm3_audio.codec import INPUT, OUTPUT, LFM3AudioCodecBuilder
from liquidonnx.lfm3_audio.config import load_config
from liquidonnx.lfm3_audio.session import stft_to_wave


def rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def test_codec_matches_reference(tiny_checkpoint, tiny_codec_reference):
    config = load_config(tiny_checkpoint)
    model = LFM3AudioCodecBuilder(config).build(tiny_checkpoint)
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])

    codes = torch.randint(0, 2016, (2, 12, 7), generator=torch.Generator().manual_seed(0))
    captured = {}
    hook = tiny_codec_reference.decoder.lin.register_forward_hook(
        lambda module, args, output: captured.update(stft=output.numpy())
    )
    with torch.no_grad():
        waves = tiny_codec_reference.decode(codes).audio_values.numpy()
    hook.remove()

    stft = session.run([OUTPUT], {INPUT: codes.numpy()})[0]
    assert stft.shape == (2, 42, 1282)
    assert rel(stft, captured["stft"]) < 1e-5
    for features, wave in zip(stft, waves, strict=True):
        ours = stft_to_wave(features, config.codec_n_fft, config.codec_hop_length)
        assert ours.shape == (7 * config.frame_samples,)
        assert rel(ours, wave) < 1e-5
