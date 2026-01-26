#!/usr/bin/env python3
"""Generate TTS using liquid-audio (PyTorch) to verify it works correctly."""

import argparse
import logging

import numpy as np
import torch
from liquid_audio import ChatState, LFM2AudioModel, LFM2AudioProcessor
from scipy.io import wavfile

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Generate TTS using liquid-audio")
    parser.add_argument(
        "text",
        nargs="?",
        default="Hello! This is a test of the text to speech system.",
        help="Text to synthesize",
    )
    parser.add_argument(
        "--output",
        default="tts_liquidaudio.wav",
        help="Output WAV file",
    )
    parser.add_argument(
        "--system",
        default="Perform TTS. Use the UK female voice.",
        help="System prompt for TTS",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load model and processor
    logger.info("Loading liquid-audio model...")
    model = LFM2AudioModel.from_pretrained(
        "LiquidAI/LFM2.5-Audio-1.5B",
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device=device,
    )
    model.eval()  # Disable dropout for inference

    logger.info("Loading processor...")
    processor = LFM2AudioProcessor.from_pretrained(
        "LiquidAI/LFM2.5-Audio-1.5B",
        device=device,
    )

    # Set random seed for reproducibility (after model loading to ensure consistency)
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Create chat state for TTS
    text = args.text
    logger.info(f"Input text: '{text}'")

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    state = ChatState(processor, dtype=dtype)

    # System instruction for TTS
    state.new_turn("system")
    state.add_text(args.system)
    state.end_turn()
    logger.info(f"System prompt: '{args.system}'")

    # User message with text to speak
    state.new_turn("user")
    state.add_text(text)
    state.end_turn()

    # Start assistant turn (model will generate audio)
    state.new_turn("assistant")

    # Generate audio tokens
    logger.info("Generating audio tokens...")
    audio_frames = []
    max_frames = 300

    # Use generate_sequential for TTS (not interleaved which is for dialogue)
    # generate_sequential: text until <|audio_start|> then continuous audio
    generator = model.generate_sequential(
        text=state["text"],
        audio_in=state["audio_in"],
        audio_in_lens=state["audio_in_lens"],
        audio_out=state["audio_out"],
        modality_flag=state["modality_flag"],
        max_new_tokens=max_frames,
        text_temperature=0.7,
        audio_temperature=0.7,
    )

    for token in generator:
        # token shape: [8] or similar
        token_np = token.cpu().numpy().flatten()

        # Check for end-of-audio token (2048)
        if len(token_np) > 0 and token_np[0] == 2048:
            logger.info("End-of-audio token received")
            break

        # Only append audio tokens (8 codebooks)
        if len(token_np) == 8:
            audio_frames.append(token_np)

            if len(audio_frames) % 50 == 0:
                logger.info(f"Generated {len(audio_frames)} frames...")

    logger.info(f"Total audio frames: {len(audio_frames)}")

    if not audio_frames:
        logger.error("No audio frames generated!")
        return

    # Stack into [T, 8]
    audio_codes = np.stack(audio_frames, axis=0)
    audio_codes = np.clip(audio_codes, 0, 2047)
    logger.info(f"Audio codes shape: {audio_codes.shape}")

    # Save codes
    np.save("tts_codes_liquidaudio_fresh.npy", audio_codes)

    # Decode with processor
    # [T, 8] → [1, 8, T]
    codes_tensor = torch.tensor(audio_codes.T, dtype=torch.int64).unsqueeze(0)
    codes_tensor = codes_tensor.to(device)

    logger.info("Decoding audio...")
    with torch.no_grad():
        waveform = processor.decode(codes_tensor).cpu().numpy()[0]

    logger.info(f"Waveform shape: {waveform.shape}")
    logger.info(f"Waveform stats: min={waveform.min():.4f}, max={waveform.max():.4f}")
    logger.info(f"RMS: {np.sqrt(np.mean(waveform**2)):.4f}")

    # Save audio at 24kHz
    sample_rate = 24000
    duration = len(waveform) / sample_rate
    logger.info(f"Duration: {duration:.2f}s")

    # Normalize and save
    max_val = np.abs(waveform).max()
    if max_val > 0:
        normalized = waveform / max_val * 0.9
    else:
        normalized = waveform

    wavfile.write(args.output, sample_rate, (normalized * 32767).astype(np.int16))
    logger.info(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
