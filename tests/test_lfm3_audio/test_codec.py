"""
codec_decoder.onnx (fp32 and fp16, built from the checkpoint's codec folder; test_exported_codec_wave:
the export's files under --exports-dir) against the float32 LiquidCodecModel on --device, sessions
on --ep (CUDA runs the fp32 graph's attention on the CPU).
Waveforms go through session.stft_to_wave; a bfloat16 copy of the reference (its iSTFT float32,
as from_pretrained keeps it) gives the noise floor.

Cases:
    tts      46 frames of greedy speech of the reference of LFM3-Audio-2B at 9d49701, its release
             (assets/codec_tts_codes.npy, the TTS "fox" prompt with numpy default_rng(20261024)
             orders, the end-of-audio frame dropped)
    rand200  200 frames of torch.randint(0, 2016) with a CPU generator seeded 0

LFM3-Audio-2B's and LFM3-Omni-2B's audio_tokenizer/ hold the same weights; each runs against its
own remote code.

Run with:
    uv run pytest tests/test_lfm3_audio/test_codec.py -v
    uv run pytest tests/test_lfm3_audio/test_codec.py -v --device cuda
    uv run pytest tests/test_lfm3_audio/test_codec.py -v --lfm3-audio-model LiquidAI/LFM3-Omni-2B
"""

import copy
import logging
import pathlib

import numpy as np
import pytest
import torch

from liquidonnx.lfm3.session import load_session
from liquidonnx.lfm3_audio.codec import INPUT, OUTPUT, LFM3AudioCodecBuilder
from liquidonnx.lfm3_audio.config import load_config
from liquidonnx.lfm3_audio.session import stft_to_wave
from liquidonnx.quantize import convert_to_fp16, save_model

logger = logging.getLogger(__name__)

ASSETS = pathlib.Path(__file__).parent / "assets"
CASES = ("tts", "rand200")
FILES = {"fp32": "codec_decoder.onnx", "fp16": "codec_decoder_fp16.onnx"}


def rel(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def case_codes(case: str) -> torch.Tensor:
    """[1, 12, F] int64"""
    if case == "tts":
        return torch.from_numpy(np.load(ASSETS / "codec_tts_codes.npy").astype(np.int64))[None]
    return torch.randint(0, 2016, (1, 12, 200), generator=torch.Generator().manual_seed(0))


@pytest.fixture(scope="module")
def codec_dir(checkpoint, tmp_path_factory) -> pathlib.Path:
    """FILES built from the checkpoint as lfm3-audio-export builds them."""
    out = tmp_path_factory.mktemp("codec")
    model = LFM3AudioCodecBuilder(load_config(checkpoint)).build(checkpoint)
    save_model(model, out / FILES["fp32"])
    convert_to_fp16(out / FILES["fp32"], out / FILES["fp16"], keep_io=True)
    return out


@pytest.fixture(scope="module")
def references(codec_reference, device) -> dict:
    """case -> codes, float32 stft_features [6F, 1282] and wave, bfloat16 wave"""
    bf16 = copy.deepcopy(codec_reference).to(torch.bfloat16)
    bf16.decoder.istft.float()
    captured = {}
    hook = codec_reference.decoder.lin.register_forward_hook(
        lambda module, args, output: captured.update(stft=output[0].float().cpu().numpy())
    )
    references = {}
    with torch.no_grad():
        for case in CASES:
            codes = case_codes(case)
            wave = codec_reference.decode(codes.to(device)).audio_values[0].cpu().numpy()
            floor = bf16.decode(codes.to(device)).audio_values[0].float().cpu().numpy()
            references[case] = {
                "codes": codes,
                "stft": captured["stft"],
                "wave": wave,
                "bf16": floor,
            }
    hook.remove()
    return references


def assert_codec_wave(path: pathlib.Path, case, precision, references, checkpoint, ep):
    config = load_config(checkpoint)
    ref = references[case]
    session = load_session(path, ep, tf32=False)
    stft = session.run([OUTPUT], {INPUT: ref["codes"].numpy()})[0][0]
    wave = stft_to_wave(stft, config.codec_n_fft, config.codec_hop_length)

    assert wave.shape == ref["wave"].shape == (ref["codes"].shape[-1] * config.frame_samples,)
    stft_rel, wave_rel, floor = (
        rel(stft, ref["stft"]),
        rel(wave, ref["wave"]),
        rel(ref["bf16"], ref["wave"]),
    )
    logger.info(
        f"{case} {path.name} ({ep}): stft rel {stft_rel:.2e}, wave rel {wave_rel:.2e} "
        f"(SNR {-20 * np.log10(wave_rel):.1f} dB), bf16 floor {floor:.2e}"
    )
    bound = 1e-4 if precision == "fp32" else 2 * floor
    assert wave_rel <= bound, f"wave rel {wave_rel:.2e} > {bound:.2e}"


@pytest.mark.parametrize("precision", FILES)
@pytest.mark.parametrize("case", CASES)
def test_codec_wave(case, precision, references, codec_dir, checkpoint, ep):
    assert_codec_wave(codec_dir / FILES[precision], case, precision, references, checkpoint, ep)


@pytest.mark.parametrize("precision", FILES)
@pytest.mark.parametrize("case", CASES)
def test_exported_codec_wave(case, precision, references, export_dir, checkpoint, ep):
    """The export's codec files, as the bundles load them (--exports-dir)."""
    path = export_dir / "onnx" / FILES[precision]
    assert_codec_wave(path, case, precision, references, checkpoint, ep)
