#!/usr/bin/env python3
"""
Image and speech understanding, speech recognition, speech synthesis and speech-to-speech chat with
an LFM3-Omni ONNX export on onnxruntime (session.ONNXOmniModel).

The modes are lfm3-audio-infer's (lfm3_audio.infer.conversation); --images go into the user turn
before its audio clips and text. chat (the default) answers in text: <|mixed_start|> and
<|audio_start|> are suppressed, as the reference's explicit generation_mode="text" does (with an
inferred mode the model can switch to speech on its own, as it does on an image question under
some system prompts).
Speech is written to --output as a 24 kHz WAV.

Usage:
    uv run lfm3-omni-infer --model exports/LFM3-Omni-2B-ONNX \\
        --images tests/test_lfm2_vl/assets/cardinal.jpg --prompt "What bird is this?"
    uv run lfm3-omni-infer --model exports/LFM3-Omni-2B-ONNX --mode asr \\
        --audio samples/audio/woodworks_question.wav
    uv run lfm3-omni-infer --model exports/LFM3-Omni-2B-ONNX --mode tts \\
        --prompt "The harbor was quiet that morning." --output tts.wav
"""

import argparse
import logging
import time

import numpy as np
from PIL import Image

from liquidonnx.lfm3.session import log_speed
from liquidonnx.lfm3_audio.infer import MODES, conversation, speech_sampling
from liquidonnx.lfm3_audio.session import load_audio
from liquidonnx.lfm3_omni.session import BUNDLES, ONNXOmniModel

logger = logging.getLogger(__name__)


def main():
    """lfm3-audio-infer's arguments, and --images and --max-image-tokens."""
    parser = argparse.ArgumentParser(description="ONNX inference for LFM3-Omni")
    parser.add_argument("--model", required=True, help="Export folder")
    parser.add_argument(
        "--precision",
        choices=sorted(BUNDLES),
        help="default: q4f16, fp16 or q8f16 on CUDA, q4 or q8 on CPU (the first one exported)",
    )
    parser.add_argument("--mode", choices=MODES, default="chat")
    parser.add_argument("--images", nargs="*", default=[], help="Images of the user turn")
    parser.add_argument(
        "--max-image-tokens", type=int, help="Cap tokens per image (smaller resize)"
    )
    parser.add_argument("--audio", nargs="*", default=[], help="WAV clips of the user turn")
    parser.add_argument("--prompt", help="Text of the user turn (the text to speak for tts)")
    parser.add_argument("--output", help="24 kHz WAV of the generated speech")
    parser.add_argument("--language", default="English", help='"any": "Perform ASR."')
    parser.add_argument("--voice", choices=("female", "male"), default="female")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Slots: tokens and frames")
    parser.add_argument("--audio-temperature", type=float)
    parser.add_argument("--audio-top-k", type=int)
    parser.add_argument("--audio-repetition-penalty", type=float)
    parser.add_argument("--audio-repetition-window", type=int, help="Frames")
    parser.add_argument("--greedy", action="store_true", help="Speech codes by argmax")
    parser.add_argument("--seed", type=int, help="MaskGIT orders and speech sampling")
    parser.add_argument("--suppress-mixed", action="store_true", help="Forbid <|mixed_start|>")
    parser.add_argument("--suppress-audio-out", action="store_true", help="Forbid <|audio_start|>")
    parser.add_argument("--no-stream", action="store_true", help="Print the text at the end")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution (skip CUDA)")
    parser.add_argument("--no-tf32", action="store_true", help="Keep CUDA fp32 math in fp32")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.mode == "tts" and not args.prompt:
        parser.error("--mode tts needs --prompt")
    if args.mode == "asr" and not args.audio:
        parser.error("--mode asr needs --audio")
    if not (args.audio or args.prompt or args.images):
        parser.error("give --images, --audio or --prompt")

    model = ONNXOmniModel(args.model, args.precision, force_cpu=args.cpu, tf32=not args.no_tf32)
    model.load()
    logger.info(f"Precision {model.precision}")
    mel = model.config.mel
    waves = [load_audio(path, mel.sampling_rate) for path in args.audio]
    messages, continue_final = conversation(args, len(waves))
    user = next(m for m in messages if m["role"] == "user")
    user["content"][:0] = [{"type": "image"}] * len(args.images)
    start = time.perf_counter()
    image_features, grids = None, None
    if args.images:
        images = [Image.open(path).convert("RGB") for path in args.images]
        image_features, grids = model.encode_images(images, args.max_image_tokens)
    ids = model.prompt_ids(
        messages,
        [mel.num_tokens(mel.num_frames(len(wave))) for wave in waves],
        add_generation_prompt=not continue_final,
        continue_final_message=continue_final,
        grids=grids,
    )
    audio_features = model.encode_audio(waves) if waves else None
    encoded = time.perf_counter()
    sampling = speech_sampling(args)

    streamer = None
    if not args.no_stream:
        from transformers import TextStreamer

        streamer = TextStreamer(model.tokenizer, skip_special_tokens=True)
    # (slot kind, time it was sampled), as lfm3-audio-infer logs them
    slots = []

    def on_token(token: int):
        slots.append(("token", time.perf_counter()))
        if streamer:
            streamer.put(np.array([token]))

    generation = model.generate_ids(
        ids,
        audio_features,
        max_new_tokens=args.max_tokens,
        audio_sampling=sampling,
        suppress_mixed=args.suppress_mixed or args.mode == "chat",
        suppress_audio_out=args.suppress_audio_out or args.mode == "chat",
        seed=args.seed,
        on_token=on_token,
        on_frame=lambda frame: slots.append(("frame", time.perf_counter())),
        image_features=image_features,
    )
    if streamer:
        streamer.end()
    text = [t for t in generation.tokens if t != model.config.tokens.audio_out]
    if args.no_stream:
        print(model.tokenizer.decode(text, skip_special_tokens=True))
    frames = len(generation.audio_codes)
    logger.info(f"{len(text)} text tokens, {frames} speech frames")
    log_speed(start, encoded, slots)
    if frames and args.output:
        # lfm2_audio.infer imports onnxruntime_genai
        from liquidonnx.lfm2_audio.infer import write_wav

        write_wav(args.output, model.decode_audio(generation.audio_codes))
    elif frames:
        logger.info("No --output: the speech is not written")


if __name__ == "__main__":
    main()
