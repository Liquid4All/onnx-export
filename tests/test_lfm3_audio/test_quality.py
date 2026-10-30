"""
Generation quality of the export's bundles where test_runtime.py's test_onnx_* leave off, greedy,
against the float32 reference without running PyTorch: its transcripts are REFERENCE_TRANSCRIPTS,
and the fp32 bundle on the same EP, which gives the reference's slots and codes exactly
(test_runtime.py's test_onnx_generation[fp32-*]), transcribes speech. So these run on a CPU EP in
minutes where the CPU reference takes hours.

    chat        lfm3-audio-infer's default mode (assistant prefilled with <think></think>, speech
                markers suppressed) on a text question: the answer has none of the faults (no
                end of turn, "</think>", a sentence said twice, a repetition loop) that the fp32
                bundle's lacks
    non_speech  ASR of 1 s of a chirp in noise: the turn ends without a repetition loop
    asr         ASR of the sample clips: WER against the float32 reference's transcript
    spoken      (slow) the speech of test_runtime.py's TTS and S2S runs, transcribed by the fp32
                bundle: the share of words heard that are not the bundle's own text, in its
                order, bounded from fp32's own

KNOWN_FAILURES' cells are xfail (helpers.expect_known_failure).

Run with:
    uv run --no-sync pytest tests/test_lfm3_audio/test_quality.py -v --device cuda --run-slow
    uv run pytest tests/test_lfm3_audio/test_quality.py -v --run-slow \\
        --lfm3-audio-model LiquidAI/LFM3-Omni-2B -k "not f16"
"""

import logging
import re

import numpy as np
import pytest

from liquidonnx.lfm3_audio.session import ONNXAudioModel, load_audio, rng_orders

from .test_runtime import (
    GENERATION,
    IM_END,
    ONNX_CASES,
    PRECISIONS,
    QUESTION,
    SAMPLES,
    SENTENCES,
    distance,
    expect_known_failure,
    onnx_prompt,
    text_of,
    transcribe,
    wer,
    words,
)

logger = logging.getLogger(__name__)

AUDIO, OMNI = "LiquidAI/LFM3-Audio-2B", "LiquidAI/LFM3-Omni-2B"
CHAT_QUESTION = "What is the capital of France? Answer in one sentence."
CHAT_TOKENS = 64
ASR_TOKENS = 256
CLIPS = {
    "woodworks": SAMPLES / "woodworks_question.wav",
    "fool_me_once": SAMPLES / "fool_me_once_mono.wav",  # 44.1 kHz, resampled with scipy
}
# The float32 reference's greedy transcripts ("Perform ASR in English."; CUDA, TF32 off): the same
# for both checkpoints, and bf16 PyTorch's too
TRANSCRIPTS = {
    "woodworks": "Can you help me come up with a slogan for my woodworking side business?",
    "fool_me_once": "There's an old saying in Tennessee, I know it's in Texas, probably in "
    "Tennessee, that says, fool me once, shame on, shame on you. Fool me, we can't get fooled "
    "again.",
}
REFERENCE_TRANSCRIPTS = {AUDIO: TRANSCRIPTS, OMNI: TRANSCRIPTS}
# Upper bound of a bundle's ASR WER against REFERENCE_TRANSCRIPTS
MAX_ASR_WER = 0.0
SPOKEN_CASES = [case for case, (kind, *_) in ONNX_CASES.items() if kind != "asr"]
# A bundle's stray words (stray(): its speech, transcribed by the fp32 bundle, against its own
# text) may exceed fp32's on the same case by this much
SPOKEN_MARGIN = 0.1
# (model, test, parameters' id, host or "*") -> the evidence of a known failure
# (helpers.expect_known_failure)
KNOWN_FAILURES = {
    (OMNI, "test_asr", "q4-fool_me_once", "cpu-arm64"): (
        'int4 rounding on the CPU EP of arm64: q4 hears "It fool me" for "Fool me" (WER 0.032, '
        "CER 0.019); CUDA and x86-64 give the reference's transcript (README 3.1e)"
    ),
}


def non_speech() -> np.ndarray:
    """1 s at 16 kHz: a 200 Hz to 4 kHz chirp in white noise, seeded."""
    t = np.arange(16000) / 16000
    chirp = 0.3 * np.sin(2 * np.pi * (200 * t + 1900 * t**2))
    noise = 0.05 * np.random.default_rng(0).standard_normal(t.size)
    return (chirp + noise).astype(np.float32)


def looping(tokens: list[int], window: int = 32, period: int = 8) -> bool:
    """The last window tokens repeat a span of at most period tokens."""
    tail = tokens[-window:]
    return len(tail) == window and any(tail[p:] == tail[:-p] for p in range(1, period + 1))


def unquoted(text: str) -> str:
    """text without single quotation marks, which words() would keep on a quoted slogan's words
    ("'Crafted"); apostrophes inside words stay."""
    return re.sub(r"\B'|'\B", " ", text)


def stray(text: str, heard: str) -> float:
    """The share of heard's words that are not text's words in its order: 1 - their longest common
    subsequence over heard's word count. Words of text that go unsaid do not count: greedy speech
    runs to the slot cap or ends a few words short in float32 and bf16 PyTorch too (README 3.1d)."""
    said, spoken = words(unquoted(text)), words(unquoted(heard))
    row = [0] * (len(spoken) + 1)
    for word in said:
        prev = 0
        for j, other in enumerate(spoken, 1):
            prev, row[j] = row[j], prev + 1 if word == other else max(row[j], row[j - 1])
    return 1 - row[-1] / max(len(spoken), 1)


def repeated_sentences(text: str) -> list[str]:
    """The sentences (split at . ! ? and <think> tags) that text says more than once."""
    seen, repeated = set(), []
    for sentence in re.split(r"[.!?]|</?think>", text):
        key = " ".join(words(sentence))
        if key in seen:
            repeated.append(key)
        elif key:
            seen.add(key)
    return repeated


@pytest.fixture(scope="module")
def fp32_model(export_dir, ep):
    """The fp32 bundle on --ep, beside onnx_models' bundle: it transcribes their speech."""
    model = ONNXAudioModel(export_dir, "fp32", tf32=False)
    model.load_sessions(ep=ep)
    model.load_preprocessing()
    return model


@pytest.fixture(scope="module")
def bundle(fp32_model, onnx_models):
    return lambda precision: fp32_model if precision == "fp32" else onnx_models(precision)


def chat_faults(model) -> set[str]:
    """The faults of the model's chat answer to CHAT_QUESTION."""
    ids = onnx_prompt(model, "chat", [], CHAT_QUESTION)
    tokens = model.generate_ids(
        ids, max_new_tokens=CHAT_TOKENS, suppress_mixed=True, suppress_audio_out=True
    ).tokens
    text = text_of(model.tokenizer, tokens)
    faults = {
        "no end of turn": tokens[-1] != IM_END,
        "</think>": "</think>" in text,
        "a sentence said twice": bool(repeated_sentences(text)),
        "a repetition loop": looping(tokens),
    }
    logger.info(f"  {len(tokens)} tokens: {text!r}")
    return {fault for fault, found in faults.items() if found}


# precision outermost: onnx_models holds one bundle at a time
@pytest.mark.parametrize("precision", PRECISIONS)
def test_chat(request, bundle, model_id, ep, precision):
    """The chat answer to CHAT_QUESTION within CHAT_TOKENS has no fault the fp32 bundle's lacks.
    LFM3-Audio-2B's own answer, in float32 and bf16 PyTorch too, says "The capital of France is
    Paris.</think>The capital of France is Paris." (README 3.1d)."""
    expect_known_failure(request, KNOWN_FAILURES, model_id, ep)
    faults = chat_faults(bundle(precision))
    assert not faults - chat_faults(bundle("fp32")), f"{precision}: {faults}"


@pytest.mark.parametrize("precision", PRECISIONS)
def test_non_speech(request, bundle, model_id, ep, precision):
    """ASR of non_speech() ends its turn within 64 tokens without a repetition loop."""
    expect_known_failure(request, KNOWN_FAILURES, model_id, ep)
    model = bundle(precision)
    wave = non_speech()
    ids = onnx_prompt(model, "asr", [wave])
    tokens = model.generate_ids(ids, model.encode_audio([wave]), max_new_tokens=64).tokens
    logger.info(f"  {precision}: {len(tokens)} tokens: {text_of(model.tokenizer, tokens)!r}")
    assert not looping(tokens)
    assert tokens[-1] == IM_END


@pytest.mark.parametrize("clip", CLIPS)
@pytest.mark.parametrize("precision", PRECISIONS)
def test_asr(request, bundle, model_id, ep, precision, clip):
    """Greedy ASR of a clip: the turn ends, and the WER against the float32 reference's
    transcript is at most MAX_ASR_WER (the CER is logged)."""
    expect_known_failure(request, KNOWN_FAILURES, model_id, ep)
    expected = REFERENCE_TRANSCRIPTS[model_id][clip]
    model = bundle(precision)
    wave = load_audio(CLIPS[clip], 16000)
    ids = onnx_prompt(model, "asr", [wave])
    tokens = model.generate_ids(ids, model.encode_audio([wave]), max_new_tokens=ASR_TOKENS).tokens
    transcript = text_of(model.tokenizer, tokens)
    ref_chars, chars = " ".join(words(expected)), " ".join(words(transcript))
    error, char_error = wer(expected, transcript), distance(ref_chars, chars) / len(ref_chars)
    logger.info(f"  {precision} {clip}: WER {error:.4f}, CER {char_error:.4f}: {transcript!r}")
    assert tokens[-1] == IM_END
    assert error <= MAX_ASR_WER


@pytest.fixture(scope="module")
def spoken(bundle, fp32_model):
    """(precision, case) -> (generation, its text, the fp32 bundle's transcript of its speech,
    stray words): the greedy runs of test_runtime.py's ONNX_CASES (injected MaskGIT orders),
    cached."""
    done = {}

    def get(precision: str, case: str):
        if (precision, case) not in done:
            model = bundle(precision)
            kind, clip, seed = ONNX_CASES[case]
            waves = [] if clip is None else [load_audio(QUESTION, 16000)]
            ids = onnx_prompt(model, kind, waves)
            generation = model.generate_ids(
                ids,
                model.encode_audio(waves) if waves else None,
                max_new_tokens=GENERATION[kind],
                orders=rng_orders(seed),
            )
            # a TTS answer is speech only: it says the prompt
            text = SENTENCES[0] if kind == "tts" else text_of(model.tokenizer, generation.tokens)
            wave = model.decode_audio(generation.audio_codes)
            transcript = transcribe(fp32_model, wave, ASR_TOKENS)
            done[precision, case] = generation, text, transcript, stray(text, transcript)
        return done[precision, case]

    return get


@pytest.mark.slow
@pytest.mark.parametrize("case", SPOKEN_CASES)
@pytest.mark.parametrize("precision", PRECISIONS)
def test_spoken(request, spoken, model_id, ep, precision, case):
    """The speech of a bundle's TTS or S2S answer says the answer's text and nothing else:
    transcribed by the fp32 bundle, its stray words are at most fp32's on the same case plus
    SPOKEN_MARGIN (the WER, which counts the words left unsaid, is logged)."""
    expect_known_failure(request, KNOWN_FAILURES, model_id, ep)
    generation, text, transcript, error = spoken(precision, case)
    *_, fp32_error = spoken("fp32", case)
    ended = generation.tokens[-1] == IM_END
    word_error = wer(unquoted(text), unquoted(transcript))
    logger.info(
        f"  {precision} {case}: {len(generation.audio_codes)} frames{'' if ended else ' (cap)'}, "
        f"stray {error:.3f} (fp32 {fp32_error:.3f}), WER {word_error:.3f}"
        f"\n    text  {text!r}\n    heard {transcript!r}"
    )
    assert error <= fp32_error + SPOKEN_MARGIN
