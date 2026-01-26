#!/usr/bin/env python3
"""Generate ASR transcription using liquid-audio (PyTorch)."""

import argparse
import logging

import numpy as np
import torch
from liquid_audio import ChatState, LFM2AudioModel, LFM2AudioProcessor
from scipy.io import wavfile

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate ASR using liquid-audio")
    parser.add_argument(
        "audio",
        help="Path to audio file to transcribe",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)


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

    # Load audio file
    logger.info(f"Loading audio: {args.audio}")
    sample_rate, audio_data = wavfile.read(args.audio)
    logger.info(f"Audio sample rate: {sample_rate}, shape: {audio_data.shape}")

    # Convert to float32 tensor normalized to [-1, 1]
    if audio_data.dtype == np.int16:
        audio_tensor = torch.tensor(audio_data, dtype=torch.float32) / 32768.0
    elif audio_data.dtype == np.int32:
        audio_tensor = torch.tensor(audio_data, dtype=torch.float32) / 2147483648.0
    else:
        audio_tensor = torch.tensor(audio_data, dtype=torch.float32)

    # Add batch dimension: [samples] → [1, samples]
    audio_tensor = audio_tensor.unsqueeze(0).to(device)
    logger.info(
        f"Audio tensor shape: {audio_tensor.shape}, range: [{audio_tensor.min():.3f}, {audio_tensor.max():.3f}]"
    )

    # Set random seed for reproducibility (after model loading to ensure consistency)
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Create chat state for ASR
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    state = ChatState(processor, dtype=dtype)

    # System instruction for ASR
    state.new_turn("system")
    state.add_text("Perform ASR.")
    state.end_turn()

    # User message with audio
    state.new_turn("user")
    state.add_audio(audio_tensor, sample_rate)
    state.end_turn()

    # Start assistant turn (model will generate transcription)
    state.new_turn("assistant")

    # Generate text tokens
    logger.info("Generating transcription...")
    max_tokens = 200

    # ASR uses greedy decoding (temperature=None) for deterministic output
    generator = model.generate_sequential(
        text=state["text"],
        audio_in=state["audio_in"],
        audio_in_lens=state["audio_in_lens"],
        audio_out=state["audio_out"],
        modality_flag=state["modality_flag"],
        max_new_tokens=max_tokens,
        text_temperature=None,  # Greedy decoding for ASR
        audio_temperature=None,
    )

    generated_tokens = []
    for i, token in enumerate(generator):
        token_id = token.item() if token.numel() == 1 else token[0].item()

        # Check for end tokens
        if token_id == 7:  # <|im_end|>
            logger.info(f"End of turn token received at position {i}")
            break

        # Skip audio start token (ASR should only generate text)
        if token_id == 128:  # <|audio_start|>
            logger.info("Skipping audio_start token")
            continue

        generated_tokens.append(token_id)

        if len(generated_tokens) % 20 == 0:
            logger.info(f"Generated {len(generated_tokens)} tokens...")

    # Decode tokens to text
    transcription = processor.text.decode(generated_tokens, skip_special_tokens=True)

    logger.info(f"Generated {len(generated_tokens)} tokens")
    print("\n" + "=" * 60)
    print(f"Audio: {args.audio}")
    print(f"Transcription: {transcription}")
    print("=" * 60)


if __name__ == "__main__":
    main()
