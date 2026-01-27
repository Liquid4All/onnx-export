#!/usr/bin/env python3
"""Generate interleaved text+audio response using liquid-audio (PyTorch).

Supports two decoders:
- mimi: Streaming neural codec (used in official chat.py demo)
- detokenizer: LFM2AudioDetokenizer (used in processor.decode(), matches ONNX)
"""

import argparse
import logging

import numpy as np
import torch
from liquid_audio import ChatState, LFM2AudioModel, LFM2AudioProcessor
from scipy.io import wavfile

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate interleaved response using liquid-audio")
    parser.add_argument(
        "audio",
        help="Path to input audio file",
    )
    parser.add_argument(
        "--output",
        default="interleaved_liquidaudio.wav",
        help="Output WAV file",
    )
    parser.add_argument(
        "--decoder",
        choices=["mimi", "detokenizer"],
        default="detokenizer",
        help="Audio decoder: mimi (official demo) or detokenizer (ONNX-compatible)",
    )
    parser.add_argument(
        "--save-codes",
        type=str,
        metavar="FILE",
        help="Save audio codes to numpy file (.npy) for comparison with other decoders",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Select best available device: CUDA > MPS > CPU
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"Using device: {device} ({gpu_name})")
    elif torch.backends.mps.is_available():
        device = "mps"
        logger.info(f"Using device: {device} (Apple Silicon GPU)")
    else:
        device = "cpu"
        logger.info(f"Using device: {device} (no GPU available)")
    logger.info(f"Decoder: {args.decoder}")

    # Load model and processor
    # Use bfloat16 on CUDA for speed, float32 on MPS/CPU (MPS bfloat16 support is limited)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    logger.info(f"Loading liquid-audio model (dtype={dtype})...")
    model = LFM2AudioModel.from_pretrained(
        "LiquidAI/LFM2.5-Audio-1.5B",
        dtype=dtype,
        device=device,
    )
    model.eval()  # Disable dropout for inference

    logger.info("Loading processor...")
    processor = LFM2AudioProcessor.from_pretrained(
        "LiquidAI/LFM2.5-Audio-1.5B",
        device=device,
    )

    # Pre-initialize mimi to ensure consistent random state regardless of decoder choice
    # (processor.mimi is lazily loaded and consumes random numbers on first access)
    _ = processor.mimi

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

    # Set random seed for reproducibility (after model loading to ensure consistency)
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Create chat state for interleaved dialogue (reuses dtype from model loading)
    state = ChatState(processor, dtype=dtype)

    # System instruction for interleaved dialogue (matching official demo)
    state.new_turn("system")
    state.add_text("Respond with interleaved text and audio.")
    state.end_turn()

    # User message with audio
    state.new_turn("user")
    state.add_audio(audio_tensor, sample_rate)
    state.end_turn()

    # Start assistant turn
    state.new_turn("assistant")

    # Generate interleaved text + audio
    logger.info("Generating interleaved response...")
    max_tokens = 300

    text_tokens = []
    audio_frames = []
    audio_chunks = []  # For mimi streaming

    if args.decoder == "mimi":
        # === Mimi streaming decoder (official demo style) ===
        mimi = processor.mimi.eval()

        with torch.no_grad(), mimi.streaming(1):
            generator = model.generate_interleaved(
                text=state["text"],
                audio_in=state["audio_in"],
                audio_in_lens=state["audio_in_lens"],
                audio_out=state["audio_out"],
                modality_flag=state["modality_flag"],
                max_new_tokens=max_tokens,
                text_temperature=1.0,
                audio_temperature=1.0,
                audio_top_k=4,
            )

            for i, token in enumerate(generator):
                if token.numel() == 1:
                    # Text token
                    token_id = token.item()
                    if token_id == 7:  # <|im_end|>
                        logger.info(f"End of turn at position {i}")
                        break
                    if token_id == 130:  # <|text_end|>
                        logger.info(f"Text end at position {i}")
                        continue
                    text_tokens.append(token_id)
                elif token.numel() == 8:
                    # Audio frame (8 codebooks)
                    if (token == 2048).any():
                        logger.info(f"Skipping frame with 2048 at position {i}")
                        continue

                    frame = token.cpu().numpy().flatten()
                    audio_frames.append(frame)

                    # Decode immediately with mimi (streaming)
                    wav_chunk = mimi.decode(token[None, :, None])[0]
                    audio_chunks.append(wav_chunk.cpu())

                if (len(text_tokens) + len(audio_frames)) % 50 == 0:
                    logger.info(
                        f"Generated {len(text_tokens)} text tokens, {len(audio_frames)} audio frames..."
                    )

    else:
        # === Detokenizer batch decoder (ONNX-compatible) ===
        with torch.no_grad():
            generator = model.generate_interleaved(
                text=state["text"],
                audio_in=state["audio_in"],
                audio_in_lens=state["audio_in_lens"],
                audio_out=state["audio_out"],
                modality_flag=state["modality_flag"],
                max_new_tokens=max_tokens,
                text_temperature=1.0,
                audio_temperature=1.0,
                audio_top_k=4,
            )

            for i, token in enumerate(generator):
                if token.numel() == 1:
                    # Text token
                    token_id = token.item()
                    if token_id == 7:  # <|im_end|>
                        logger.info(f"End of turn at position {i}")
                        break
                    if token_id == 130:  # <|text_end|>
                        logger.info(f"Text end at position {i}")
                        continue
                    text_tokens.append(token_id)
                elif token.numel() == 8:
                    # Audio frame (8 codebooks)
                    if (token == 2048).any():
                        logger.info(f"Skipping frame with 2048 at position {i}")
                        continue

                    frame = token.cpu().numpy().flatten()
                    audio_frames.append(frame)

                if (len(text_tokens) + len(audio_frames)) % 50 == 0:
                    logger.info(
                        f"Generated {len(text_tokens)} text tokens, {len(audio_frames)} audio frames..."
                    )

    # Decode text
    transcription = processor.text.decode(text_tokens, skip_special_tokens=True)
    logger.info(f"Generated {len(text_tokens)} text tokens, {len(audio_frames)} audio frames")

    print("\n" + "=" * 60)
    print(f"Audio input: {args.audio}")
    print(f"Decoder: {args.decoder}")
    print(f"Text response: {transcription}")
    print(f"Audio frames: {len(audio_frames)}")

    # Save audio codes for comparison
    if audio_frames:
        audio_codes = np.stack(audio_frames, axis=0)  # [T, 8]
        audio_codes = np.clip(audio_codes, 0, 2047)
        if args.save_codes:
            np.save(args.save_codes, audio_codes)
            print(f"Codes: {args.save_codes} {audio_codes.shape}")

    # Decode and save audio
    if args.decoder == "mimi" and audio_chunks:
        # Concatenate mimi streaming chunks
        waveform = torch.cat(audio_chunks, dim=-1).squeeze().numpy()
        logger.info(
            f"Mimi waveform shape: {waveform.shape}, duration: {len(waveform) / 24000:.2f}s"
        )

        # Normalize and save
        max_val = np.abs(waveform).max()
        if max_val > 0:
            waveform = waveform / max_val * 0.9

        wavfile.write(args.output, 24000, (waveform * 32767).astype(np.int16))
        logger.info(f"Saved audio: {args.output}")
        print(f"Audio output: {args.output}")

    elif args.decoder == "detokenizer" and audio_frames:
        # Batch decode with LFM2AudioDetokenizer
        detokenizer = processor.audio_detokenizer.to(device).eval()

        codes_tensor = torch.tensor(audio_codes.T, dtype=torch.long, device=device).unsqueeze(0)
        logger.info(f"Audio codes shape: {codes_tensor.shape}")

        with torch.no_grad():
            waveform = detokenizer(codes_tensor)

        waveform = waveform.squeeze().cpu().numpy()
        logger.info(
            f"Detokenizer waveform shape: {waveform.shape}, duration: {len(waveform) / 24000:.2f}s"
        )

        # Normalize and save
        max_val = np.abs(waveform).max()
        if max_val > 0:
            waveform = waveform / max_val * 0.9

        wavfile.write(args.output, 24000, (waveform * 32767).astype(np.int16))
        logger.info(f"Saved audio: {args.output}")
        print(f"Audio output: {args.output}")

    print("=" * 60)


if __name__ == "__main__":
    main()
