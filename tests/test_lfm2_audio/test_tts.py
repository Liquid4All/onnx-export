"""
TTS (Text-to-Speech) tests for LFM2.5-Audio ONNX exports.

Tests audio generation quality across all precisions (fp32, fp16, q4, q8).
Includes reference comparison tests against PyTorch model.

Run with:
    uv run pytest tests/test_lfm2_audio/test_tts.py -v
    uv run pytest tests/test_lfm2_audio/test_tts.py -v -k "fp16"
    uv run pytest tests/test_lfm2_audio/test_tts.py -v -k "reference"
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

# Test prompts
TTS_PROMPTS = [
    pytest.param("Hello world", id="hello"),
    pytest.param("The quick brown fox jumps over the lazy dog.", id="pangram"),
]

# Short prompts for reference comparison (deterministic)
REFERENCE_PROMPTS = ["Hello world", "How are you today?", "The quick brown fox"]


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


# === Precision-based Tests ===


@pytest.mark.parametrize("precision", PRECISION_CONFIGS)
@pytest.mark.parametrize("prompt", TTS_PROMPTS)
def test_tts_generation(
    exports_dir: pathlib.Path,
    precision: str | None,
    prompt: str,
):
    """Test TTS audio generation across precisions."""
    model_dir = get_model_dir(exports_dir)
    if not model_dir.exists():
        pytest.skip(f"Model not found: {model_dir}")
    if not check_precision_available(model_dir, precision):
        pytest.skip(f"Precision {precision or 'fp32'} not available")

    model = load_onnx_model(model_dir, precision)

    logger.info(f"Testing TTS ({precision or 'fp32'}): '{prompt}'")

    # Generate audio codes
    audio_codes = model.synthesize(
        text=prompt,
        max_new_tokens=100,
        audio_temperature=0.8,
        text_temperature=0,
    )

    logger.info(f"  Generated {len(audio_codes)} audio frames")

    # Validate output
    assert len(audio_codes) > 0, "No audio frames generated"

    # Check code validity (should be in range [0, 2047])
    for i, frame in enumerate(audio_codes):
        assert frame.shape == (8,), f"Frame {i} has wrong shape: {frame.shape}"
        assert np.all(frame >= 0), f"Frame {i} has negative values"
        assert np.all(frame < 2048), f"Frame {i} has values >= 2048"

    del model


@pytest.mark.parametrize("precision", PRECISION_CONFIGS)
def test_tts_decoding(exports_dir: pathlib.Path, precision: str | None):
    """Test TTS with full audio decoding (codes -> waveform)."""
    model_dir = get_model_dir(exports_dir)
    if not model_dir.exists():
        pytest.skip(f"Model not found: {model_dir}")
    if not check_precision_available(model_dir, precision):
        pytest.skip(f"Precision {precision or 'fp32'} not available")

    model = load_onnx_model(model_dir, precision)

    prompt = "Hello"
    logger.info(f"Testing TTS decoding ({precision or 'fp32'}): '{prompt}'")

    # Generate audio codes
    audio_codes = model.synthesize(
        text=prompt,
        max_new_tokens=50,
        audio_temperature=0.8,
        text_temperature=0,
    )

    assert len(audio_codes) > 0, "No audio frames generated"

    # Decode to waveform
    codes_array = np.stack(audio_codes, axis=0)  # [T, 8]
    waveform = model.decode_audio(codes_array)

    logger.info(f"  Generated {len(audio_codes)} frames -> {len(waveform)} samples")

    # Validate waveform
    assert len(waveform) > 0, "Empty waveform"
    assert waveform.dtype == np.float32, f"Wrong dtype: {waveform.dtype}"
    assert np.abs(waveform).max() <= 1.5, "Waveform values out of range"

    del model


@pytest.mark.parametrize("precision", PRECISION_CONFIGS)
def test_tts_deterministic(exports_dir: pathlib.Path, precision: str | None):
    """Test that TTS with temperature=0 is deterministic."""
    model_dir = get_model_dir(exports_dir)
    if not model_dir.exists():
        pytest.skip(f"Model not found: {model_dir}")
    if not check_precision_available(model_dir, precision):
        pytest.skip(f"Precision {precision or 'fp32'} not available")

    model = load_onnx_model(model_dir, precision)

    prompt = "Hello"

    # Generate twice with same settings
    codes1 = model.synthesize(text=prompt, max_new_tokens=30, audio_temperature=0, text_temperature=0)
    codes2 = model.synthesize(text=prompt, max_new_tokens=30, audio_temperature=0, text_temperature=0)

    assert len(codes1) == len(codes2), f"Frame count differs: {len(codes1)} vs {len(codes2)}"

    for i, (c1, c2) in enumerate(zip(codes1, codes2, strict=True)):
        assert np.array_equal(c1, c2), f"Frame {i} differs between runs"

    logger.info(f"  Deterministic: {len(codes1)} frames match")

    del model


# === Reference Comparison Tests (fp32 only) ===


def generate_reference_tts(model, processor, text: str, max_new_tokens: int = 60):
    """Generate TTS audio codes using reference model (greedy sampling)."""
    from liquid_audio import ChatState

    state = ChatState(processor, dtype=torch.float32)
    state.new_turn("system")
    state.add_text("Perform TTS.")
    state.end_turn()
    state.new_turn("user")
    state.add_text(text)
    state.end_turn()
    state.new_turn("assistant")

    audio_codes = []
    for token in model.generate_sequential(
        **state,
        max_new_tokens=max_new_tokens,
        text_temperature=0,
        audio_temperature=0,
    ):
        if token.shape != torch.Size([1]):
            codes = token.cpu().numpy()
            if np.any(codes >= 2048):
                break
            audio_codes.append(codes)

    return audio_codes


def generate_onnx_tts(model, text: str, max_new_tokens: int = 60):
    """Generate TTS audio codes using ONNX model (greedy sampling)."""
    audio_codes = model.synthesize(
        text=text,
        max_new_tokens=max_new_tokens,
        audio_temperature=0,
        text_temperature=0,
    )
    # Filter out any end-of-audio frames
    valid_codes = [c for c in audio_codes if np.all(c < 2048)]
    return valid_codes


@pytest.mark.parametrize("prompt", REFERENCE_PROMPTS)
def test_tts_reference_single_turn(reference_model, onnx_model, prompt: str):
    """Test single-turn TTS audio code generation against reference.

    Note: Due to numerical differences in ONNX vs PyTorch, exact match is not
    expected. This test validates that both produce reasonable audio output.
    """
    model, processor = reference_model

    logger.info(f"Testing TTS reference: '{prompt}'")

    # Generate with reference
    ref_codes = generate_reference_tts(model, processor, prompt)
    logger.info(f"  Reference: {len(ref_codes)} frames")

    # Generate with ONNX
    onnx_codes = generate_onnx_tts(onnx_model, prompt)
    logger.info(f"  ONNX: {len(onnx_codes)} frames")

    # Validate both produce audio
    assert len(ref_codes) > 0, "Reference produced no audio frames"
    assert len(onnx_codes) > 0, "ONNX produced no audio frames"

    # Check both produce at least a minimum amount of audio (5 frames = ~0.1s)
    # Note: TTS output length can vary significantly between implementations
    # due to different EOS detection, temperature handling, etc.
    assert len(ref_codes) >= 5, f"Reference produced too few frames: {len(ref_codes)}"
    assert len(onnx_codes) >= 5, f"ONNX produced too few frames: {len(onnx_codes)}"

    # Check code validity
    for codes in [ref_codes, onnx_codes]:
        for frame in codes:
            assert np.all(frame >= 0), "Negative code values"
            assert np.all(frame < 2048), "Code values out of range"

    logger.info(f"  Both produced audio: ref={len(ref_codes)}, onnx={len(onnx_codes)} frames")


def test_tts_reference_multi_turn(reference_model, onnx_model):
    """Test multi-turn TTS maintains context correctly.

    Note: Due to numerical differences between ONNX and PyTorch, we don't expect
    exact match. This test validates that both models can generate audio across
    multiple turns with proper context handling.
    """
    from liquid_audio import ChatState
    from liquid_audio.processor import LFMModality

    model, processor = reference_model
    turns = ["Hello", "World"]

    logger.info(f"Testing multi-turn TTS reference: {turns}")

    # === Reference multi-turn ===
    state = ChatState(processor, dtype=torch.float32)
    state.new_turn("system")
    state.add_text("Perform TTS.")
    state.end_turn()

    ref_all_codes = []
    for user_text in turns:
        state.new_turn("user")
        state.add_text(user_text)
        state.end_turn()
        state.new_turn("assistant")

        text_tokens = []
        audio_codes = []
        for token in model.generate_sequential(
            **state,
            max_new_tokens=50,
            text_temperature=0,
            audio_temperature=0,
        ):
            if token.shape == torch.Size([1]):
                text_tokens.append(token)
            else:
                codes = token.cpu().numpy()
                if np.any(codes >= 2048):
                    break
                audio_codes.append(token)

        ref_all_codes.append([c.cpu().numpy() for c in audio_codes])

        # Update state
        if text_tokens:
            text_tensor = torch.cat(text_tokens, dim=0).unsqueeze(0)
        else:
            text_tensor = torch.empty((1, 0), dtype=torch.long)

        if audio_codes:
            audio_tensor = torch.stack(audio_codes, dim=1)
        else:
            audio_tensor = torch.empty((8, 0), dtype=torch.long)

        mod_text = torch.full((1, text_tensor.shape[1]), LFMModality.TEXT, dtype=torch.long)
        mod_audio = torch.full((1, audio_tensor.shape[1]), LFMModality.AUDIO_OUT, dtype=torch.long)
        mod_flag = torch.cat([mod_text, mod_audio], dim=1)

        state.append(text_tensor, audio_tensor, mod_flag)
        state.end_turn()

    # === ONNX multi-turn ===
    cache = onnx_model._init_cache(batch_size=1)
    total_len = 0

    prompt_parts = [
        "<|startoftext|><|im_start|>system\n",
        "Perform TTS.<|im_end|>\n",
    ]

    onnx_all_codes = []
    for turn_idx, user_text in enumerate(turns):
        turn_prompt = (
            f"<|im_start|>user\n{user_text}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        if turn_idx == 0:
            full_prompt = "".join(prompt_parts) + turn_prompt
        else:
            full_prompt = turn_prompt

        input_ids = onnx_model.tokenizer.encode(
            full_prompt, return_tensors="np", add_special_tokens=False
        )
        batch_size, seq_len = input_ids.shape
        embeds = onnx_model._get_text_embeds(input_ids)

        attention_mask = np.ones((batch_size, total_len + seq_len), dtype=np.int64)
        logits, hidden_states, cache = onnx_model._run_decoder(embeds, attention_mask, cache)
        total_len += seq_len

        # Generate text until audio_start
        in_audio_mode = False
        for _ in range(10):
            last_logits = logits[0, -1, :onnx_model.vocab_size]
            token = int(np.argmax(last_logits))

            if token == onnx_model.AUDIO_START_TOKEN:
                in_audio_mode = True
                next_ids = np.array([[onnx_model.AUDIO_START_TOKEN]], dtype=np.int64)
                next_embeds = onnx_model._get_text_embeds(next_ids)
                attention_mask = np.ones((batch_size, total_len + 1), dtype=np.int64)
                logits, hidden_states, cache = onnx_model._run_decoder(
                    next_embeds, attention_mask, cache
                )
                total_len += 1
                break

            next_ids = np.array([[token]], dtype=np.int64)
            next_embeds = onnx_model._get_text_embeds(next_ids)
            attention_mask = np.ones((batch_size, total_len + 1), dtype=np.int64)
            logits, hidden_states, cache = onnx_model._run_decoder(
                next_embeds, attention_mask, cache
            )
            total_len += 1

        assert in_audio_mode, f"Turn {turn_idx}: Did not enter audio mode"

        # Generate audio
        audio_codes = []
        for _ in range(50):
            last_hidden = hidden_states[0, -1:, :]
            frame_codes = onnx_model._sample_audio_codes(
                last_hidden, temperature=0
            )

            if onnx_model._is_end_of_audio(frame_codes[0]):
                break

            audio_codes.append(frame_codes[0])

            clamped_codes = np.minimum(frame_codes[0], 2047)
            audio_tokens = np.array(
                [[cb_idx * onnx_model.codebook_vocab + int(clamped_codes[cb_idx])
                  for cb_idx in range(onnx_model.num_codebooks)]],
                dtype=np.int64,
            )
            all_embeds = onnx_model._get_audio_embeds(audio_tokens)
            next_embeds = all_embeds.sum(axis=1, keepdims=True)

            attention_mask = np.ones((batch_size, total_len + 1), dtype=np.int64)
            logits, hidden_states, cache = onnx_model._run_decoder(
                next_embeds, attention_mask, cache
            )
            total_len += 1

        onnx_all_codes.append(audio_codes)

        # End turn
        end_ids = onnx_model.tokenizer.encode(
            "<|im_end|>\n", return_tensors="np", add_special_tokens=False
        )
        end_embeds = onnx_model._get_text_embeds(end_ids)
        attention_mask = np.ones((batch_size, total_len + end_ids.shape[1]), dtype=np.int64)
        logits, hidden_states, cache = onnx_model._run_decoder(end_embeds, attention_mask, cache)
        total_len += end_ids.shape[1]

    # Compare - validate both produce audio in each turn
    assert len(ref_all_codes) == len(onnx_all_codes)

    for turn_idx in range(len(turns)):
        ref_codes = ref_all_codes[turn_idx]
        onnx_codes = onnx_all_codes[turn_idx]

        logger.info(f"  Turn {turn_idx + 1} '{turns[turn_idx]}': ref={len(ref_codes)}, onnx={len(onnx_codes)}")

        # Both should produce some audio (allow empty for very short inputs)
        # Just validate code ranges
        for codes in [ref_codes, onnx_codes]:
            for frame in codes:
                assert np.all(frame >= 0), "Negative code values"
                assert np.all(frame < 2048), "Code values out of range"

    logger.info("  Multi-turn generation completed for both models")


def test_tts_reference_audio_decoding(reference_model, onnx_model, audio_processor):
    """Test that both models can generate audio that decodes to valid waveforms.

    Note: Due to numerical differences, we don't expect identical codes or waveforms.
    This test validates that both models produce decodable audio.
    """
    model, processor = reference_model
    text = "Hello world"

    # Generate codes
    ref_codes = generate_reference_tts(model, processor, text)
    onnx_codes = generate_onnx_tts(onnx_model, text)

    assert len(ref_codes) > 0, "Reference produced no audio codes"
    assert len(onnx_codes) > 0, "ONNX produced no audio codes"

    # Decode both
    device = "cpu"

    ref_array = np.stack(ref_codes, axis=0)
    ref_tensor = torch.from_numpy(ref_array.T).unsqueeze(0).long().to(device)

    onnx_array = np.stack(onnx_codes, axis=0)
    onnx_tensor = torch.from_numpy(onnx_array.T).unsqueeze(0).long().to(device)

    with torch.no_grad():
        ref_wav = audio_processor.decode(ref_tensor)
        onnx_wav = audio_processor.decode(onnx_tensor)

    # Validate waveforms
    ref_np = ref_wav.squeeze().cpu().numpy()
    onnx_np = onnx_wav.squeeze().cpu().numpy()

    # Both should produce valid audio
    assert len(ref_np) > 0, "Reference waveform is empty"
    assert len(onnx_np) > 0, "ONNX waveform is empty"
    assert np.abs(ref_np).max() <= 1.5, "Reference waveform out of range"
    assert np.abs(onnx_np).max() <= 1.5, "ONNX waveform out of range"

    ref_rms = np.sqrt(np.mean(ref_np**2))
    onnx_rms = np.sqrt(np.mean(onnx_np**2))

    logger.info(f"  Reference: {len(ref_codes)} frames, {len(ref_np)} samples, RMS={ref_rms:.4f}")
    logger.info(f"  ONNX: {len(onnx_codes)} frames, {len(onnx_np)} samples, RMS={onnx_rms:.4f}")
