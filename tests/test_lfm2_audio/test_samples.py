"""
Cross-precision consistency tests for LFM2.5-Audio ONNX exports.

Tests that outputs are consistent across precisions (fp32, fp16, q4, q8).

Run with:
    uv run pytest tests/test_lfm2_audio/test_samples.py -v
"""

import logging
import pathlib

import pytest

logger = logging.getLogger(__name__)

# Expected transcription keywords for ASR validation
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


def test_asr_consistency_across_precisions(
    exports_dir: pathlib.Path,
    sample_audio_short: pathlib.Path,
):
    """Test that ASR produces similar results across precisions."""
    model_dir = get_model_dir(exports_dir)
    if not model_dir.exists():
        pytest.skip(f"Model not found: {model_dir}")

    results = {}
    precisions = [None, "fp16", "q4", "q8"]

    for precision in precisions:
        if not check_precision_available(model_dir, precision):
            logger.info(f"Skipping {precision or 'fp32'} - not available")
            continue

        model = load_onnx_model(model_dir, precision)
        transcription = model.transcribe(str(sample_audio_short))
        results[precision or "fp32"] = transcription
        del model

    if len(results) < 2:
        pytest.skip("Need at least 2 precisions to compare")

    logger.info("ASR results across precisions:")
    for prec, text in results.items():
        logger.info(f"  {prec}: {text}")

    # All results should have the same keywords (allowing for minor variations)
    audio_name = sample_audio_short.name
    if audio_name in ASR_KEYWORDS:
        for prec, text in results.items():
            text_lower = text.lower()
            found = [kw for kw in ASR_KEYWORDS[audio_name] if kw in text_lower]
            assert len(found) > 0, f"{prec} missing expected keywords"


def test_tts_consistency_across_precisions(exports_dir: pathlib.Path):
    """Test that TTS produces audio of similar length across precisions."""
    model_dir = get_model_dir(exports_dir)
    if not model_dir.exists():
        pytest.skip(f"Model not found: {model_dir}")

    prompt = "Hello world"
    results = {}
    precisions = [None, "fp16", "q4", "q8"]

    for precision in precisions:
        if not check_precision_available(model_dir, precision):
            logger.info(f"Skipping {precision or 'fp32'} - not available")
            continue

        model = load_onnx_model(model_dir, precision)
        audio_codes = model.synthesize(
            text=prompt,
            max_new_tokens=100,
            audio_temperature=0,  # Deterministic
            text_temperature=0,
        )
        results[precision or "fp32"] = len(audio_codes)
        del model

    if len(results) < 2:
        pytest.skip("Need at least 2 precisions to compare")

    logger.info("TTS frame counts across precisions:")
    for prec, count in results.items():
        logger.info(f"  {prec}: {count} frames")

    # Frame counts should be within 50% of each other (allowing for quantization effects)
    counts = list(results.values())
    max_count = max(counts)
    min_count = min(counts)
    if max_count > 0:
        ratio = min_count / max_count
        assert ratio > 0.5, f"Frame count variation too high: {min_count} vs {max_count}"


def test_interleaved_consistency_across_precisions(
    exports_dir: pathlib.Path,
    sample_audio_short: pathlib.Path,
):
    """Test that interleaved mode produces output across all precisions."""
    model_dir = get_model_dir(exports_dir)
    if not model_dir.exists():
        pytest.skip(f"Model not found: {model_dir}")

    results = {}
    precisions = [None, "fp16", "q4", "q8"]

    for precision in precisions:
        if not check_precision_available(model_dir, precision):
            logger.info(f"Skipping {precision or 'fp32'} - not available")
            continue

        model = load_onnx_model(model_dir, precision)
        text_output, audio_codes = model.generate_interleaved_from_audio(
            audio_path=str(sample_audio_short),
            max_new_tokens=100,
            audio_temperature=0.8,
            text_temperature=0.7,
        )
        results[precision or "fp32"] = {
            "text_len": len(text_output),
            "audio_frames": len(audio_codes),
        }
        del model

    if len(results) < 2:
        pytest.skip("Need at least 2 precisions to compare")

    logger.info("Interleaved results across precisions:")
    for prec, data in results.items():
        logger.info(f"  {prec}: text={data['text_len']} chars, audio={data['audio_frames']} frames")

    # All precisions should produce some output
    for prec, data in results.items():
        total = data["text_len"] + data["audio_frames"]
        assert total > 0, f"{prec} produced no output"
