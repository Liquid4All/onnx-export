#!/usr/bin/env python3
"""
Speech recognition, speech synthesis and speech-to-speech chat with an LFM3-Audio ONNX export on
onnxruntime (session.ONNXAudioModel).

Modes (the checkpoint's chat template, prompts of its README):
    asr   system "Perform ASR in <Language>." ("Perform ASR." for any language), user audio
    tts   system "Perform TTS in <Language> using a <female|male> voice.", user text, assistant
          prefilled with <|audio_start|> (continue_final_message)
    s2s   system "Respond with interleaved text and audio.", user audio or text, assistant
          prefilled with <|mixed_start|>
    chat  no system turn, user audio or text, text answer (suppress_mixed and suppress_audio_out)

Speech is written to --output as a 24 kHz WAV (lfm2_audio.infer.write_wav). Text is greedy; speech
is sampled with the checkpoint's settings, from the export's copy of its generation_config.json,
unless the --audio-* flags say otherwise (speech_sampling).

Usage:
    uv run lfm3-audio-infer --model exports/LFM3-Audio-2B-ONNX --mode asr --audio speech.wav
    uv run lfm3-audio-infer --model exports/LFM3-Audio-2B-ONNX --mode tts \\
        --prompt "The harbor was quiet that morning." --output tts.wav
    uv run lfm3-audio-infer --model exports/LFM3-Audio-2B-ONNX --mode s2s \\
        --audio question.wav --output answer.wav --precision fp16
"""

import argparse
import json
import logging
import pathlib
import time

import numpy as np

from liquidonnx.lfm3.session import log_speed
from liquidonnx.lfm3_audio.session import (
    BUNDLES,
    AudioSampling,
    ONNXAudioModel,
    load_audio,
)

logger = logging.getLogger(__name__)

MODES = ("asr", "tts", "s2s", "chat")
SYSTEM_PROMPTS = {
    "asr": "Perform ASR in {language}.",
    "tts": "Perform TTS in {language} using a {voice} voice.",
    "s2s": "Respond with interleaved text and audio.",
}


def speech_sampling(args: argparse.Namespace) -> AudioSampling:
    """The speech sampling of the CLIs' --model, --audio-* and --greedy arguments, as the
    reference's generate() resolves it (speech_setting): each setting from its flag, else from the
    export's generation_config.json, else the reference's default. --greedy takes temperature 0:
    a top-k without a temperature samples at temperature 1."""
    path = pathlib.Path(args.model) / "generation_config.json"
    config = json.loads(path.read_text()) if path.exists() else {}

    def setting(name: str, default=None):
        value = getattr(args, name)
        if value is None:
            value = config.get(name)
        return default if value is None else value

    return AudioSampling(
        temperature=0.0 if args.greedy else setting("audio_temperature"),
        top_k=setting("audio_top_k"),
        repetition_penalty=setting("audio_repetition_penalty"),
        repetition_window=setting("audio_repetition_window", 64),
    )


def conversation(args, clips: int) -> tuple[list[dict], bool]:
    """(messages, continue_final_message) of the mode."""
    content = [{"type": "audio"}] * clips + (
        [{"type": "text", "text": args.prompt}] if args.prompt else []
    )
    messages = []
    if args.mode in SYSTEM_PROMPTS:
        system = SYSTEM_PROMPTS[args.mode].format(language=args.language, voice=args.voice)
        if args.mode == "asr" and args.language == "any":
            system = "Perform ASR."
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": content})
    prefill = {"tts": "<|audio_start|>", "s2s": "<|mixed_start|>", "chat": "<think></think>"}
    if args.mode in prefill:
        messages.append({"role": "assistant", "content": prefill[args.mode]})
    return messages, args.mode in prefill


def main():
    """Arguments: --model, --precision (session.BUNDLES), --mode (MODES), --audio (16 kHz WAVs),
    --prompt, --output, --language (default English), --voice (female|male), --max-tokens,
    --audio-temperature, --audio-top-k, --audio-repetition-penalty, --audio-repetition-window,
    --greedy, --seed, --suppress-mixed, --suppress-audio-out, --no-stream, --cpu, --no-tf32."""
    parser = argparse.ArgumentParser(description="ONNX inference for LFM3-Audio")
    parser.add_argument("--model", required=True, help="Export folder")
    parser.add_argument(
        "--precision",
        choices=sorted(BUNDLES),
        help="default: q4f16, fp16 or q8f16 on CUDA, q4 or q8 on CPU (the first one exported)",
    )
    parser.add_argument("--mode", choices=MODES, default="chat")
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
    if not args.audio and not args.prompt:
        parser.error("give --audio or --prompt")

    model = ONNXAudioModel(args.model, args.precision, force_cpu=args.cpu, tf32=not args.no_tf32)
    model.load()
    logger.info(f"Precision {model.precision}")
    mel = model.config.mel
    waves = [load_audio(path, mel.sampling_rate) for path in args.audio]
    messages, continue_final = conversation(args, len(waves))
    counts = [mel.num_tokens(mel.num_frames(len(wave))) for wave in waves]
    ids = model.prompt_ids(
        messages,
        counts,
        add_generation_prompt=not continue_final,
        continue_final_message=continue_final,
    )
    sampling = speech_sampling(args)

    streamer = None
    if not args.no_stream:
        from transformers import TextStreamer

        streamer = TextStreamer(model.tokenizer, skip_special_tokens=True)
    # (slot kind, time it was sampled): a slot's time is the decoder step before it and its sampling
    slots = []

    def on_token(token: int):
        slots.append(("token", time.perf_counter()))
        if streamer:
            streamer.put(np.array([token]))

    start = time.perf_counter()
    audio_features = model.encode_audio(waves) if waves else None
    encoded = time.perf_counter()
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

        decode_start = time.perf_counter()
        wave = model.decode_audio(generation.audio_codes)
        logger.info(
            f"Codec: {frames} frames ({len(wave) / 24000:.2f} s) in "
            f"{time.perf_counter() - decode_start:.2f} s"
        )
        write_wav(args.output, wave)
    elif frames:
        logger.info("No --output: the speech is not written")


if __name__ == "__main__":
    main()
