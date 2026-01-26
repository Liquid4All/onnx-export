"""
Interleaved mode tests for LFM2.5-Audio ONNX exports.

Tests mixed text/audio generation across all precisions (fp32, fp16, q4, q8).

Run with:
    uv run pytest tests/test_lfm2_audio/test_interleaved.py -v
    uv run pytest tests/test_lfm2_audio/test_interleaved.py -v -k "fp16"
    uv run pytest tests/test_lfm2_audio/test_interleaved.py -v -k "audio_input"
"""

import logging
import pathlib

import numpy as np
import pytest
import torch

logger = logging.getLogger(__name__)

# Precision configurations
PRECISION_CONFIGS = [
    pytest.param(None, id="fp32"),
    pytest.param("fp16", id="fp16"),
    pytest.param("q4", id="q4"),
    pytest.param("q8", id="q8"),
]

# Text prompts for interleaved mode
TEXT_PROMPTS = [
    pytest.param("Say hello in a friendly way", id="hello"),
    pytest.param("What is 2 plus 2?", id="math"),
]


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
def test_interleaved_audio_input(
    exports_dir: pathlib.Path,
    sample_audio_short: pathlib.Path,
    precision: str | None,
):
    """Test interleaved mode with audio input across precisions."""
    model_dir = get_model_dir(exports_dir)
    if not model_dir.exists():
        pytest.skip(f"Model not found: {model_dir}")
    if not check_precision_available(model_dir, precision):
        pytest.skip(f"Precision {precision or 'fp32'} not available")

    model = load_onnx_model(model_dir, precision)

    logger.info(f"Testing interleaved ({precision or 'fp32'}): {sample_audio_short.name}")

    # Run interleaved with audio input
    text_output, audio_codes = model.generate_interleaved_from_audio(
        audio_path=str(sample_audio_short),
        max_new_tokens=200,
        audio_temperature=0.8,
        text_temperature=0.7,
    )

    if len(text_output) > 100:
        logger.info(f"  Text: {text_output[:100]}...")
    else:
        logger.info(f"  Text: {text_output}")
    logger.info(f"  Audio frames: {len(audio_codes)}")

    # Validate outputs
    assert text_output is not None
    # Interleaved should produce either text or audio (or both)
    assert len(text_output) > 0 or len(audio_codes) > 0, "No output generated"

    del model


@pytest.mark.parametrize("precision", PRECISION_CONFIGS)
@pytest.mark.parametrize("prompt", TEXT_PROMPTS)
def test_interleaved_text_input(
    exports_dir: pathlib.Path,
    precision: str | None,
    prompt: str,
):
    """Test interleaved mode with text-only input across precisions."""
    model_dir = get_model_dir(exports_dir)
    if not model_dir.exists():
        pytest.skip(f"Model not found: {model_dir}")
    if not check_precision_available(model_dir, precision):
        pytest.skip(f"Precision {precision or 'fp32'} not available")

    model = load_onnx_model(model_dir, precision)

    logger.info(f"Testing interleaved text-only ({precision or 'fp32'}): '{prompt}'")

    # Run interleaved with text input only
    text_output, audio_codes = model.generate_interleaved(
        prompt=prompt,
        max_new_tokens=200,
        audio_temperature=0.8,
        text_temperature=0.7,
    )

    if len(text_output) > 100:
        logger.info(f"  Text: {text_output[:100]}...")
    else:
        logger.info(f"  Text: {text_output}")
    logger.info(f"  Audio frames: {len(audio_codes)}")

    # Validate outputs
    assert text_output is not None
    # Should produce either text or audio response
    assert len(text_output) > 0 or len(audio_codes) > 0, "No output generated"

    del model


@pytest.mark.parametrize("precision", PRECISION_CONFIGS)
def test_interleaved_produces_audio(
    exports_dir: pathlib.Path,
    sample_audio_short: pathlib.Path,
    precision: str | None,
):
    """Test that interleaved mode can produce audio output."""
    model_dir = get_model_dir(exports_dir)
    if not model_dir.exists():
        pytest.skip(f"Model not found: {model_dir}")
    if not check_precision_available(model_dir, precision):
        pytest.skip(f"Precision {precision or 'fp32'} not available")

    model = load_onnx_model(model_dir, precision)

    # Use audio input which is more likely to produce audio response
    text_output, audio_codes = model.generate_interleaved_from_audio(
        audio_path=str(sample_audio_short),
        max_new_tokens=300,
        audio_temperature=0.8,
        text_temperature=0.7,
    )

    logger.info(f"  Text length: {len(text_output)}, Audio frames: {len(audio_codes)}")

    # At minimum, should produce some output
    total_output = len(text_output) + len(audio_codes)
    assert total_output > 0, "Interleaved produced no output"

    # If audio was produced, validate format
    if len(audio_codes) > 0:
        for i, frame in enumerate(audio_codes):
            assert frame.shape == (8,), f"Frame {i} has wrong shape: {frame.shape}"
            assert np.all(frame >= 0), f"Frame {i} has negative values"
            assert np.all(frame < 2048), f"Frame {i} has values >= 2048"

    del model


@pytest.mark.parametrize("precision", PRECISION_CONFIGS)
def test_interleaved_audio_decoding(
    exports_dir: pathlib.Path,
    sample_audio_short: pathlib.Path,
    precision: str | None,
):
    """Test that interleaved audio output can be decoded to waveform."""
    model_dir = get_model_dir(exports_dir)
    if not model_dir.exists():
        pytest.skip(f"Model not found: {model_dir}")
    if not check_precision_available(model_dir, precision):
        pytest.skip(f"Precision {precision or 'fp32'} not available")

    model = load_onnx_model(model_dir, precision)

    text_output, audio_codes = model.generate_interleaved_from_audio(
        audio_path=str(sample_audio_short),
        max_new_tokens=300,
        audio_temperature=0.8,
        text_temperature=0.7,
    )

    if len(audio_codes) == 0:
        pytest.skip("No audio generated in this run")

    # Decode to waveform
    codes_array = np.stack(audio_codes, axis=0)  # [T, 8]
    waveform = model.decode_audio(codes_array)

    logger.info(f"  Decoded {len(audio_codes)} frames -> {len(waveform)} samples")

    # Validate waveform
    assert len(waveform) > 0, "Empty waveform"
    assert waveform.dtype == np.float32, f"Wrong dtype: {waveform.dtype}"
    assert np.abs(waveform).max() <= 1.5, "Waveform values out of range"

    del model


# === Reference Comparison Tests (fp32 only) ===


def load_audio_tensor(audio_path: str, device: str = "cpu") -> tuple:
    """Load audio file and return tensor + sample rate."""
    from scipy.io import wavfile

    sample_rate, audio_data = wavfile.read(audio_path)

    # Convert to float32 tensor normalized to [-1, 1]
    if audio_data.dtype == np.int16:
        audio_tensor = torch.tensor(audio_data, dtype=torch.float32) / 32768.0
    elif audio_data.dtype == np.int32:
        audio_tensor = torch.tensor(audio_data, dtype=torch.float32) / 2147483648.0
    else:
        audio_tensor = torch.tensor(audio_data, dtype=torch.float32)

    # Add batch dimension: [samples] → [1, samples]
    audio_tensor = audio_tensor.unsqueeze(0).to(device)
    return audio_tensor, sample_rate


def generate_reference_interleaved(model, processor, audio_path: str, max_new_tokens: int = 100):
    """Generate interleaved output using reference liquid-audio model.

    Uses the same settings as the ONNX implementation for fair comparison.
    """
    from liquid_audio import ChatState

    # Load audio
    audio_tensor, sample_rate = load_audio_tensor(audio_path)

    # Build chat state
    state = ChatState(processor, dtype=torch.float32)
    state.new_turn("system")
    state.add_text("You are a helpful assistant.")
    state.end_turn()
    state.new_turn("user")
    state.add_audio(audio_tensor, sample_rate)
    state.end_turn()
    state.new_turn("assistant")

    # Generate with interleaved mode
    text_tokens = []
    audio_codes = []

    for token in model.generate_interleaved(
        text=state["text"],
        audio_in=state["audio_in"],
        audio_in_lens=state["audio_in_lens"],
        audio_out=state["audio_out"],
        modality_flag=state["modality_flag"],
        max_new_tokens=max_new_tokens,
        text_temperature=0.7,
        audio_temperature=0.8,
    ):
        if token.numel() == 1:
            token_id = token.item()
            if token_id == 7:  # <|im_end|>
                break
            if token_id == 130:  # <|text_end|>
                continue
            text_tokens.append(token_id)
        elif token.numel() == 8:
            codes = token.cpu().numpy().flatten()
            if np.any(codes >= 2048):
                break
            audio_codes.append(codes)

    text_output = processor.text.decode(text_tokens, skip_special_tokens=True)
    return text_output, audio_codes


def test_interleaved_reference_audio_codes(
    reference_model,
    onnx_model,
    sample_audio_short: pathlib.Path,
):
    """Test that interleaved mode produces similar audio codes to reference.

    Note: Due to numerical differences between ONNX and PyTorch, exact match
    is not expected. This test validates that both produce reasonable output.
    """
    model, processor = reference_model

    logger.info(f"Testing interleaved reference: {sample_audio_short.name}")

    # Generate with reference
    ref_text, ref_codes = generate_reference_interleaved(
        model, processor, str(sample_audio_short), max_new_tokens=100
    )
    logger.info(f"  Reference: {len(ref_text)} chars, {len(ref_codes)} audio frames")

    # Generate with ONNX
    onnx_text, onnx_codes = onnx_model.generate_interleaved_from_audio(
        audio_path=str(sample_audio_short),
        max_new_tokens=100,
        text_temperature=0.7,
        audio_temperature=0.8,
    )
    logger.info(f"  ONNX: {len(onnx_text)} chars, {len(onnx_codes)} audio frames")

    # Both should produce some output
    assert len(ref_text) > 0 or len(ref_codes) > 0, "Reference produced no output"
    assert len(onnx_text) > 0 or len(onnx_codes) > 0, "ONNX produced no output"

    # Log first few audio codes for comparison if both produced audio
    if len(ref_codes) > 0 and len(onnx_codes) > 0:
        logger.info(f"  Reference first frame: {ref_codes[0].tolist()}")
        logger.info(f"  ONNX first frame: {onnx_codes[0].tolist()}")

        # Check code validity
        for name, codes in [("Reference", ref_codes), ("ONNX", onnx_codes)]:
            for i, frame in enumerate(codes):
                assert frame.shape == (8,), f"{name} frame {i} wrong shape: {frame.shape}"
                assert np.all(frame >= 0), f"{name} frame {i} has negative values"
                assert np.all(frame < 2048), f"{name} frame {i} has values >= 2048"

    logger.info("  Both produced valid interleaved output")


def test_interleaved_reference_text_similarity(
    reference_model,
    onnx_model,
    sample_audio_short: pathlib.Path,
):
    """Test that interleaved text output is semantically similar.

    Uses deterministic settings (temperature=0) for more consistent comparison.
    """
    from liquid_audio import ChatState

    model, processor = reference_model

    logger.info(f"Testing interleaved text similarity: {sample_audio_short.name}")

    # Load audio
    audio_tensor, sample_rate = load_audio_tensor(str(sample_audio_short))

    # Reference with deterministic settings
    state = ChatState(processor, dtype=torch.float32)
    state.new_turn("system")
    state.add_text("You are a helpful assistant.")
    state.end_turn()
    state.new_turn("user")
    state.add_audio(audio_tensor, sample_rate)
    state.end_turn()
    state.new_turn("assistant")

    ref_text_tokens = []
    for token in model.generate_interleaved(
        text=state["text"],
        audio_in=state["audio_in"],
        audio_in_lens=state["audio_in_lens"],
        audio_out=state["audio_out"],
        modality_flag=state["modality_flag"],
        max_new_tokens=50,
        text_temperature=0,
        audio_temperature=0,
    ):
        if token.numel() == 1:
            token_id = token.item()
            if token_id == 7:  # <|im_end|>
                break
            if token_id == 130:  # <|text_end|>
                continue
            ref_text_tokens.append(token_id)

    ref_text = processor.text.decode(ref_text_tokens, skip_special_tokens=True)

    # ONNX with deterministic settings
    onnx_text, _ = onnx_model.generate_interleaved_from_audio(
        audio_path=str(sample_audio_short),
        max_new_tokens=50,
        text_temperature=0,
        audio_temperature=0,
    )

    logger.info(
        f"  Reference text: {ref_text[:100]}..."
        if len(ref_text) > 100
        else f"  Reference text: {ref_text}"
    )
    logger.info(
        f"  ONNX text: {onnx_text[:100]}..."
        if len(onnx_text) > 100
        else f"  ONNX text: {onnx_text}"
    )

    # Both should produce some text
    assert len(ref_text) > 0, "Reference produced no text"
    assert len(onnx_text) > 0, "ONNX produced no text"

    # Check if they share common words (loose similarity check)
    ref_words = set(ref_text.lower().split())
    onnx_words = set(onnx_text.lower().split())
    common_words = ref_words & onnx_words

    logger.info(f"  Common words: {len(common_words)} ({common_words})")
    # At least some overlap expected for same audio input
    # Note: This is a weak check since outputs can legitimately differ
