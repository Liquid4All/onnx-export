"""
Test TTS (Text-to-Speech) functionality comparing ONNX vs reference.

Run with:
    uv run pytest tests/test_lfm2_audio/test_tts.py -v
    uv run pytest tests/test_lfm2_audio/test_tts.py -v -k "single_turn"
"""

import logging
import pathlib

import numpy as np
import pytest
import torch

logger = logging.getLogger(__name__)

# Test prompts for TTS
TTS_PROMPTS = [
    "Hello world",
    "How are you today?",
    "The quick brown fox",
]


def generate_reference_tts(model, processor, text: str, max_frames: int = 50):
    """Generate TTS audio codes using reference model."""
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
        max_new_tokens=max_frames + 10,
        text_temperature=None,
        audio_temperature=None,
    ):
        if token.shape != torch.Size([1]):
            codes = token.cpu().numpy()
            if np.any(codes >= 2048):
                break
            audio_codes.append(codes)
            if len(audio_codes) >= max_frames:
                break

    return audio_codes


def generate_onnx_tts(model, text: str, max_frames: int = 50):
    """Generate TTS audio codes using ONNX model."""
    audio_codes = model.synthesize(
        text=text,
        max_audio_frames=max_frames,
        audio_temperature=0,
        text_temperature=0,
    )
    # Filter out any end-of-audio frames
    valid_codes = [c for c in audio_codes if np.all(c < 2048)]
    return valid_codes


@pytest.mark.parametrize("prompt", TTS_PROMPTS)
def test_tts_single_turn(reference_model, onnx_model, prompt: str):
    """Test single-turn TTS audio code generation matches reference."""
    model, processor = reference_model

    logger.info(f"Testing TTS: '{prompt}'")

    # Generate with reference
    ref_codes = generate_reference_tts(model, processor, prompt)
    logger.info(f"  Reference: {len(ref_codes)} frames")

    # Generate with ONNX
    onnx_codes = generate_onnx_tts(onnx_model, prompt)
    logger.info(f"  ONNX: {len(onnx_codes)} frames")

    # Compare
    assert len(ref_codes) > 0, "Reference produced no audio frames"
    assert len(onnx_codes) > 0, "ONNX produced no audio frames"
    assert len(ref_codes) == len(onnx_codes), (
        f"Frame count mismatch: ref={len(ref_codes)}, onnx={len(onnx_codes)}"
    )

    for i, (ref, onnx) in enumerate(zip(ref_codes, onnx_codes)):
        assert np.array_equal(ref, onnx), (
            f"Frame {i} mismatch:\n  ref:  {ref.tolist()}\n  onnx: {onnx.tolist()}"
        )

    logger.info(f"  All {len(ref_codes)} frames match!")


def test_tts_multi_turn(reference_model, onnx_model):
    """Test multi-turn TTS maintains context correctly."""
    from liquid_audio import ChatState
    from liquid_audio.processor import LFMModality

    model, processor = reference_model
    turns = ["Hello", "World"]

    logger.info(f"Testing multi-turn TTS: {turns}")

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
            text_temperature=None,
            audio_temperature=None,
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
            frame_codes = onnx_model._sample_audio_codes_autoregressive(
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

    # Compare
    assert len(ref_all_codes) == len(onnx_all_codes)

    for turn_idx in range(len(turns)):
        ref_codes = ref_all_codes[turn_idx]
        onnx_codes = onnx_all_codes[turn_idx]

        logger.info(f"  Turn {turn_idx + 1} '{turns[turn_idx]}': ref={len(ref_codes)}, onnx={len(onnx_codes)}")

        assert len(ref_codes) == len(onnx_codes), (
            f"Turn {turn_idx + 1} frame count mismatch"
        )

        for i, (ref, onnx) in enumerate(zip(ref_codes, onnx_codes)):
            assert np.array_equal(ref, onnx), (
                f"Turn {turn_idx + 1} frame {i} mismatch"
            )

    logger.info("  All turns match!")


def test_tts_audio_decoding(reference_model, onnx_model, audio_processor):
    """Test that decoded audio waveforms are identical."""
    model, processor = reference_model
    text = "Hello world"

    # Generate codes
    ref_codes = generate_reference_tts(model, processor, text)
    onnx_codes = generate_onnx_tts(onnx_model, text)

    assert len(ref_codes) == len(onnx_codes), "Code count mismatch"
    assert len(ref_codes) > 0, "No audio codes generated"

    # Decode both
    device = "cpu"

    ref_array = np.stack(ref_codes, axis=0)
    ref_tensor = torch.from_numpy(ref_array.T).unsqueeze(0).long().to(device)

    onnx_array = np.stack(onnx_codes, axis=0)
    onnx_tensor = torch.from_numpy(onnx_array.T).unsqueeze(0).long().to(device)

    with torch.no_grad():
        ref_wav = audio_processor.decode(ref_tensor)
        onnx_wav = audio_processor.decode(onnx_tensor)

    # Compare waveforms
    ref_np = ref_wav.squeeze().cpu().numpy()
    onnx_np = onnx_wav.squeeze().cpu().numpy()

    assert ref_np.shape == onnx_np.shape, "Waveform shape mismatch"
    assert np.allclose(ref_np, onnx_np, rtol=1e-5, atol=1e-5), "Waveform values differ"

    logger.info(f"  Waveforms identical: shape={ref_np.shape}, RMS={np.sqrt(np.mean(ref_np**2)):.4f}")
