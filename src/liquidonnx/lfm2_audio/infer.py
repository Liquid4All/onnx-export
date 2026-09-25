#!/usr/bin/env python3
"""
LFM2.5-Audio through onnxruntime-genai: text chat, ASR, TTS and interleaved text and speech.

onnxruntime-genai runs the speech encoder, the decoder and the depthformer, and returns the audio
codes (one frame of 8 codes per 80 ms). audio_detokenizer.onnx and an inverse STFT turn them into
a 24 kHz waveform here.

Text is decoded greedily in every mode, as in liquid-audio; the audio codes are sampled with the
model card's settings unless --audio-temperature / --audio-top-k say otherwise.

Usage:
    # Text chat
    uv run lfm2-audio-infer exports/LFM2.5-Audio-1.5B-ONNX --prompt "What is the capital of France?"

    # ASR: audio -> text
    uv run lfm2-audio-infer exports/LFM2.5-Audio-1.5B-ONNX --mode asr --audio input.wav

    # TTS: text -> audio
    uv run lfm2-audio-infer exports/LFM2.5-Audio-1.5B-ONNX --mode tts --prompt "Hello world" \\
        --output output.wav

    # Interleaved: spoken or typed question -> text and speech
    uv run lfm2-audio-infer exports/LFM2.5-Audio-1.5B-ONNX --mode interleaved --audio question.wav \\
        --output answer.wav

    # Interactive multi-turn chat (text or interleaved)
    uv run lfm2-audio-infer exports/LFM2.5-Audio-1.5B-ONNX --mode interleaved --chat --output answer.wav
    # Commands:
    #   /audio <file> [text] - Send audio with optional text
    #   <text>               - Send text message
    #   reset                - Clear conversation
    #   quit                 - Exit
"""

import argparse
import json
import logging
import pathlib
from dataclasses import dataclass

import numpy as np
import onnxruntime_genai as og

from liquidonnx.genai_runtime import TokenPrinter, add_runtime_arguments, generate, load_model
from liquidonnx.lfm2_audio.export import (
    AUDIO_TOKEN_ID,
    MODALITY_SWITCH_TOKEN_IDS,
    PRECISIONS,
    bundle,
    genai_files,
)
from liquidonnx.session import load_onnx_session

logger = logging.getLogger(__name__)

SYSTEM_PROMPTS = {
    "text": None,
    "asr": "Perform ASR.",
    "tts": "Perform TTS. Use the UK female voice.",
    "interleaved": "Respond with interleaved text and audio.",
}
# (audio temperature, audio top-k) from the model card
AUDIO_SAMPLING = {"tts": (0.8, 64), "interleaved": (1.0, 4)}
# One frame is 80 ms; 1024 frames are about 82 seconds of speech.
MAX_TOKENS = {"text": 256, "asr": 256, "tts": 1024, "interleaved": 1024}
# Ids in the answer that hold audio frames or switch to speech, not text.
NON_TEXT_TOKEN_IDS = (AUDIO_TOKEN_ID, *MODALITY_SWITCH_TOKEN_IDS)
AUDIO_MARKER = "<|audio|>"  # onnxruntime-genai's placeholder for one clip in the prompt

NUM_CODEBOOKS = 8
SAMPLE_RATE = 24000
N_FFT = 1280
HOP_LENGTH = 320


@dataclass
class Answer:
    text: str
    codes: np.ndarray  # [frames, NUM_CODEBOOKS] int64


def chat_prompt(system: str | None, turns: list[tuple[str, str]]) -> str:
    """ChatML prompt of (role, content) turns, ending where the assistant answers."""
    prompt = "<|startoftext|>"
    if system:
        prompt += f"<|im_start|>system\n{system}<|im_end|>\n"
    for role, content in turns:
        prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    return prompt + "<|im_start|>assistant\n"


def default_precision(model_dir: pathlib.Path) -> str:
    """The precision genai_config.json loads."""
    config = json.loads((model_dir / "genai_config.json").read_text())
    decoder = config["model"]["decoder"]["filename"]
    return next(p for p in ("fp32", *PRECISIONS) if f"onnx/{bundle(p)['decoder']}" == decoder)


def istft_same_padding(spectrum: np.ndarray, n_fft: int, hop_length: int) -> np.ndarray:
    """liquid-audio's ISTFT with "same" padding: [freq, frames] complex -> frames * hop samples."""
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)  # torch.hann_window
    frames = np.fft.irfft(spectrum, n_fft, axis=0) * window[:, None]
    size = (spectrum.shape[1] - 1) * hop_length + n_fft
    wave = np.zeros(size)
    envelope = np.zeros(size)
    for t in range(spectrum.shape[1]):
        wave[t * hop_length : t * hop_length + n_fft] += frames[:, t]
        envelope[t * hop_length : t * hop_length + n_fft] += window**2
    pad = (n_fft - hop_length) // 2
    return wave[pad:-pad] / envelope[pad:-pad]


class Detokenizer:
    """audio_detokenizer.onnx plus the inverse STFT: audio codes -> waveform."""

    def __init__(self, path: pathlib.Path):
        self.session = load_onnx_session(path)

    def __call__(self, codes: np.ndarray) -> np.ndarray:
        # genai drops end-of-audio frames; codebooks 1-7 can still sample 2048, past the table's 2047.
        features = self.session.run(
            ["stft_features"], {"audio_codes": np.minimum(codes, 2047).T[None].astype(np.int64)}
        )[0][0]  # [frames * 6, n_fft + 2]
        bins = N_FFT // 2 + 1
        spectrum = np.exp(features[:, :bins]) * np.exp(1j * features[:, bins:])
        return istft_same_padding(spectrum.T, N_FFT, HOP_LENGTH).astype(np.float32)


def write_wav(path: str, wave: np.ndarray):
    import scipy.io.wavfile

    peak = np.abs(wave).max()
    if peak > 0:
        wave = wave / peak * 0.9
    scipy.io.wavfile.write(path, SAMPLE_RATE, (wave * 32767).astype(np.int16))
    logger.info(f"Saved {path} ({len(wave) / SAMPLE_RATE:.2f}s)")


class AudioChat:
    def __init__(self, model_dir: pathlib.Path, precision: str | None = None, ep: str = "cpu"):
        self.model = load_model(model_dir, precision and genai_files(precision), ep)
        self.tokenizer = og.Tokenizer(self.model)
        self.processor = self.model.create_multimodal_processor()
        files = bundle(precision or default_precision(model_dir))
        self.detokenizer = Detokenizer(model_dir / "onnx" / files["detokenizer"])

    def answer(
        self,
        mode: str,
        turns: list[tuple[str, str]],
        audios: list[str],
        max_new_tokens: int | None = None,
        system: str | None = None,
        stream: bool = False,
        **search_options,
    ) -> Answer:
        """Answer the conversation in a mode; audios are the clips of the <|audio|> markers.

        system defaults to the mode's prompt ("" for none). search_options take
        onnxruntime-genai's audio_temperature, audio_top_k and random_seed; the audio sampling
        defaults to the mode's.
        """
        temperature, top_k = AUDIO_SAMPLING.get(mode, (1.0, 4))
        options = {"audio_temperature": temperature, "audio_top_k": top_k, **search_options}
        system = SYSTEM_PROMPTS[mode] if system is None else system
        inputs = self.processor(
            chat_prompt(system, turns), audios=og.Audios.open(*audios) if audios else None
        )
        prompt_length = inputs["input_ids"].as_numpy().shape[-1]
        printer = TokenPrinter(self.tokenizer, NON_TEXT_TOKEN_IDS) if stream else None
        generator = generate(
            self.model,
            inputs,
            max_new_tokens or MAX_TOKENS[mode],
            printer,
            audio_interleaved=mode == "interleaved",
            **options,
        )
        if stream:
            print()
        tokens = [
            t for t in generator.get_sequence(0)[prompt_length:] if t not in NON_TEXT_TOKEN_IDS
        ]
        codes = np.asarray(generator.get_output("audio_codes")).reshape(-1, NUM_CODEBOOKS)
        return Answer(self.tokenizer.decode(np.array(tokens, dtype=np.int32)), codes)


def user_content(audio: str | None, text: str | None) -> str:
    return (AUDIO_MARKER if audio else "") + (text or "")


def single_turn(text: str | None, audio: str | None = None) -> tuple[list, list[str]]:
    """(turns, audios) of one user message."""
    return [("user", user_content(audio, text))], [str(audio)] if audio else []


def save_audio(chat: AudioChat, answer: Answer, output: str | None, codes_path: str | None):
    if not len(answer.codes):
        return
    print(f"Audio: {len(answer.codes)} frames")
    if codes_path:
        np.save(codes_path, answer.codes)
        print(f"Codes: {codes_path} {answer.codes.shape}")
    if output:
        write_wav(output, chat.detokenizer(answer.codes))
        print(f"Output: {output}")


def per_turn(path: str, turn: int | str) -> str:
    """path with _turn{turn} before its suffix."""
    path = pathlib.Path(path)
    return str(path.with_name(f"{path.stem}_turn{turn}{path.suffix}"))


def run_chat(chat: AudioChat, args: argparse.Namespace, generate_answer):
    """Multi-turn chat; each turn re-sends the conversation, with earlier answers as text."""
    output = args.output or f"{args.mode}_output.wav"
    print("\n" + "=" * 60)
    print("Interactive chat")
    print("  /audio <file> [text] - Send audio with optional text")
    print("  <text>               - Send text message")
    print("  reset                - Clear conversation")
    print("  quit                 - Exit")
    print(f"Audio output: {per_turn(output, 'N')}")
    print("=" * 60 + "\n")

    turns, audios = [], []
    pending = (args.audio, args.prompt) if args.audio or args.prompt else None
    while True:
        if pending:
            audio, text = pending
            pending = None
            print(f"[Turn {len(turns) // 2}] " + " ".join(filter(None, [audio, text])))
        else:
            try:
                line = input(f"[Turn {len(turns) // 2}] > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break
            if not line:
                continue
            if line.lower() == "quit":
                print("Goodbye!")
                break
            if line.lower() == "reset":
                turns, audios = [], []
                print("Conversation reset.\n")
                continue
            audio, text = None, line
            if line.startswith("/audio "):
                audio, _, text = line[len("/audio ") :].strip().partition(" ")
                if not pathlib.Path(audio).exists():
                    print(f"File not found: {audio}")
                    continue

        turns.append(("user", user_content(audio, text)))
        if audio:
            audios.append(audio)
        print("Assistant: ", end="")
        answer = generate_answer(turns, audios)
        turns.append(("assistant", answer.text))
        turn = len(turns) // 2 - 1
        save_audio(
            chat,
            answer,
            per_turn(output, turn),
            args.save_codes and per_turn(args.save_codes, turn),
        )
        print()


def main():
    parser = argparse.ArgumentParser(
        description="LFM2.5-Audio through onnxruntime-genai (text, ASR, TTS, interleaved)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("model_dir", type=pathlib.Path, help="Export folder")
    parser.add_argument("--mode", choices=list(SYSTEM_PROMPTS), default="text")
    add_runtime_arguments(parser, PRECISIONS)
    parser.add_argument("--prompt", help="Text of the (first) user turn")
    parser.add_argument("--audio", help="Audio clip of the (first) user turn")
    parser.add_argument("--system", help="System prompt (default: the mode's)")
    parser.add_argument("--output", help="WAV file for the speech (default: {mode}_output.wav)")
    parser.add_argument("--save-codes", metavar="FILE", help="Save the audio codes (.npy)")
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="Maximum text tokens plus audio frames "
        f"(default: {', '.join(f'{m} {n}' for m, n in MAX_TOKENS.items())})",
    )
    parser.add_argument("--audio-temperature", type=float, help="0 takes the likeliest code")
    parser.add_argument("--audio-top-k", type=int, help="1 takes the likeliest code")
    parser.add_argument("--seed", type=int, default=42, help="Seed of the audio code sampling")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming output")
    parser.add_argument(
        "--chat", action="store_true", help="Interactive multi-turn chat (text or interleaved)"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.chat and args.mode not in ("text", "interleaved"):
        parser.error("--chat works in text and interleaved modes")
    if not args.chat:
        if args.mode == "asr" and not args.audio:
            parser.error("asr needs --audio")
        if args.mode in ("text", "tts") and not args.prompt:
            parser.error(f"{args.mode} needs --prompt")
        if args.mode == "interleaved" and not (args.audio or args.prompt):
            parser.error("interleaved needs --audio or --prompt")

    sampling = {"random_seed": args.seed}
    if args.audio_temperature is not None:
        sampling["audio_temperature"] = args.audio_temperature
    if args.audio_top_k is not None:
        sampling["audio_top_k"] = args.audio_top_k

    chat = AudioChat(args.model_dir, args.precision, args.ep)

    def generate_answer(turns: list[tuple[str, str]], audios: list[str]) -> Answer:
        return chat.answer(
            args.mode,
            turns,
            audios,
            args.max_tokens,
            args.system,
            stream=not args.no_stream,
            **sampling,
        )

    if args.chat:
        run_chat(chat, args, generate_answer)
        return

    system = SYSTEM_PROMPTS[args.mode] if args.system is None else args.system
    print(f"Mode: {args.mode}" + (f" | System: {system}" if system else ""))
    print("Assistant: ", end="")
    answer = generate_answer(*single_turn(args.prompt, args.audio))
    if args.no_stream:
        print(answer.text)
    save_audio(chat, answer, args.output or f"{args.mode}_output.wav", args.save_codes)


if __name__ == "__main__":
    main()
