"""
ASR (Automatic Speech Recognition) tests for LFM2.5-Audio ONNX exports.

Tests transcription quality across all precisions (fp32, fp16, q4, q8).

Run with:
    uv run pytest tests/test_lfm2_audio/test_asr.py -v
    uv run pytest tests/test_lfm2_audio/test_asr.py -v -k "fp16"
    uv run pytest tests/test_lfm2_audio/test_asr.py -v -k "short"
"""

import logging
import pathlib

import pytest

logger = logging.getLogger(__name__)

# Precision configurations
PRECISION_CONFIGS = [
    pytest.param(None, id="fp32"),
    pytest.param("fp16", id="fp16"),
    pytest.param("q4", id="q4"),
    pytest.param("q8", id="q8"),
]

# Expected transcription keywords for validation
ASR_KEYWORDS = {
    "fool_me_once_mono.wav": ["tennessee", "texas", "fool"],
    "woodworks_question.wav": ["woodwork", "slogan", "tagline"],
}


def get_model_dir(exports_dir: pathlib.Path) -> pathlib.Path:
    """Get the ONNX model directory."""
    return exports_dir / "LFM2.5-Audio-1.5B-ONNX"


def load_onnx_model(model_dir: pathlib.Path, precision: str | None):
    """Load ONNX model with specified precision."""
    from liquidonnx.lfm2_audio.infer import LFM2AudioInference, resolve_precision_files

    files = resolve_precision_files(precision)
    return LFM2AudioInference(
        model_dir,
        decoder_file=files["decoder"],
        audio_embedding_file=files["audio_embedding"],
        audio_encoder_file=files["audio_encoder"],
        audio_detokenizer_file=files["audio_detokenizer"],
        vocoder_depthformer_file=files["vocoder_depthformer"],
    )


def check_precision_available(model_dir: pathlib.Path, precision: str | None) -> bool:
    """Check if the precision models are available."""
    onnx_dir = model_dir / "onnx"
    if precision is None:
        return (onnx_dir / "decoder.onnx").exists()
    return (onnx_dir / f"decoder_{precision}.onnx").exists()


@pytest.mark.parametrize("precision", PRECISION_CONFIGS)
def test_asr_short_audio(
    exports_dir: pathlib.Path,
    sample_audio_short: pathlib.Path,
    precision: str | None,
):
    """Test ASR transcription on short audio across precisions."""
    model_dir = get_model_dir(exports_dir)
    if not model_dir.exists():
        pytest.skip(f"Model not found: {model_dir}")
    if not check_precision_available(model_dir, precision):
        pytest.skip(f"Precision {precision or 'fp32'} not available")

    model = load_onnx_model(model_dir, precision)

    logger.info(f"Testing ASR ({precision or 'fp32'}): {sample_audio_short.name}")

    transcription = model.transcribe(str(sample_audio_short))

    logger.info(f"  Transcription: {transcription}")

    # Validate output
    assert transcription is not None
    assert len(transcription) > 0
    assert isinstance(transcription, str)

    # Check for expected keywords
    audio_name = sample_audio_short.name
    if audio_name in ASR_KEYWORDS:
        text_lower = transcription.lower()
        found_keywords = [kw for kw in ASR_KEYWORDS[audio_name] if kw in text_lower]
        assert len(found_keywords) > 0, (
            f"Expected keywords {ASR_KEYWORDS[audio_name]} not found in: {transcription}"
        )
        logger.info(f"  Found keywords: {found_keywords}")

    del model


@pytest.mark.parametrize("precision", PRECISION_CONFIGS)
def test_asr_long_audio(
    exports_dir: pathlib.Path,
    sample_audio_long: pathlib.Path,
    precision: str | None,
):
    """Test ASR transcription on longer audio across precisions."""
    model_dir = get_model_dir(exports_dir)
    if not model_dir.exists():
        pytest.skip(f"Model not found: {model_dir}")
    if not check_precision_available(model_dir, precision):
        pytest.skip(f"Precision {precision or 'fp32'} not available")

    model = load_onnx_model(model_dir, precision)

    logger.info(f"Testing ASR ({precision or 'fp32'}): {sample_audio_long.name}")

    transcription = model.transcribe(str(sample_audio_long))

    logger.info(f"  Transcription: {transcription[:100]}...")

    # Validate output
    assert transcription is not None
    assert len(transcription) > 20  # Long audio should produce substantial text

    # Check for expected keywords
    audio_name = sample_audio_long.name
    if audio_name in ASR_KEYWORDS:
        text_lower = transcription.lower()
        found_keywords = [kw for kw in ASR_KEYWORDS[audio_name] if kw in text_lower]
        assert len(found_keywords) > 0, (
            f"Expected keywords {ASR_KEYWORDS[audio_name]} not found in: {transcription}"
        )
        logger.info(f"  Found keywords: {found_keywords}")

    del model


@pytest.mark.parametrize("precision", PRECISION_CONFIGS)
def test_asr_produces_text(
    exports_dir: pathlib.Path,
    sample_audio_short: pathlib.Path,
    precision: str | None,
):
    """Test that ASR produces non-empty text output."""
    model_dir = get_model_dir(exports_dir)
    if not model_dir.exists():
        pytest.skip(f"Model not found: {model_dir}")
    if not check_precision_available(model_dir, precision):
        pytest.skip(f"Precision {precision or 'fp32'} not available")

    model = load_onnx_model(model_dir, precision)

    transcription = model.transcribe(str(sample_audio_short))

    # Basic sanity checks
    assert transcription is not None
    assert isinstance(transcription, str)
    assert len(transcription.strip()) > 0, "ASR produced empty transcription"
    # Should contain actual words (not just special tokens)
    words = transcription.split()
    assert len(words) >= 3, f"ASR produced too few words: {words}"

    del model
