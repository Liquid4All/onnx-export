"""
The runtime of session.py against the PyTorch reference, with the reference's own modules behind
the session interface (TorchDecoder for CachedDecoder, the depth decoder for depth_logits), so the
mode machine, MaskGIT sampling and the prompts are checked without the graphs:

- sampling (sample_token against Lfm3AudioForConditionalGeneration._sample), MaskGIT orders, and
  the iSTFT (stft_to_wave against LiquidCodecModel.decode) on the tiny checkpoint (no download);
- greedy generation with injected MaskGIT orders equal to the reference's generate slot for slot
  (tokens and codes, end-of-audio frames included): forced modes, forced end-of-audio frames and
  prompts with speech frames on the tiny checkpoint; ASR, TTS and S2S on a real one;
- prompt_ids (chat_template.jinja) equal to the checkpoint processor's ids for ASR, TTS, S2S,
  speech-to-text chat and a multi-turn replay of a spoken answer.

The test_onnx_* tests run the export's bundles end to end (greedy, injected MaskGIT orders): ASR,
TTS and S2S against the reference's slots, and TTS round trips through the bundle's own ASR.

Run with:
    uv run pytest tests/test_lfm3_audio/test_runtime.py -v -k tiny
    uv run pytest tests/test_lfm3_audio/test_runtime.py -v --device cuda
    uv run --no-sync pytest tests/test_lfm3_audio/test_runtime.py -v \\
        --device cuda -k "onnx and (fp32 or fp16 or f16)"
    uv run pytest tests/test_lfm3_audio/test_runtime.py -v \\
        --lfm3-audio-model LiquidAI/LFM3-Omni-2B
"""

import logging
import pathlib
import platform
import re

import numpy as np
import pytest
import torch
from helpers import processed_probs

from liquidonnx.lfm3_audio import infer as audio_infer
from liquidonnx.lfm3_audio.session import (
    ONNXAudioModel,
    load_audio,
    rng_orders,
    sample_token,
    stft_to_wave,
)
from liquidonnx.lfm3_omni import infer as omni_infer

from .remote import SPEECH_SAMPLING, cached_slots, inject_orders, maskgit_orders, remote_class
from .test_decoder import cosine, kl_divergence
from .tiny import ASSETS

logger = logging.getLogger(__name__)

SAMPLES = pathlib.Path(__file__).parents[2] / "samples" / "audio"
AUDIO, AUDIO_OUT, EOS_CODE = 125031, 125032, 2016
AUDIO_START, TEXT_END, MIXED_START, IM_END = 125012, 125014, 125015, 124900


# === PyTorch stand-ins for the sessions ===


class TorchDecoder:
    """lfm3.session.CachedDecoder's step and last_outputs over the reference's language model,
    embedding as Lfm3AudioModel.forward does and scoring the last position as _generate does."""

    def __init__(self, reference):
        from transformers import DynamicCache

        self.reference = reference
        self.device = next(reference.parameters()).device
        self.cache = DynamicCache(config=reference.config.text_config)
        self.past = 0

    @torch.no_grad()
    def step(self, ids: list[int], audio_features=None, audio_out_codes=None, **_) -> np.ndarray:
        model, config = self.reference.model, self.reference.config
        input_ids = torch.tensor([ids], device=self.device)
        embeds = model.get_input_embeddings()(input_ids)
        if (input_ids == config.audio_token_id).any():
            features = torch.as_tensor(audio_features, device=self.device)
            embeds[input_ids == config.audio_token_id] = features
        if (input_ids == config.audio_out_token_id).any():
            codes = torch.as_tensor(audio_out_codes, device=self.device)
            embeds[input_ids == config.audio_out_token_id] = model.get_audio_out_embeds(codes)
        n = len(ids)
        hidden = model.language_model(
            inputs_embeds=embeds,
            attention_mask=torch.ones((1, self.past + n), dtype=torch.long, device=self.device),
            position_ids=torch.arange(self.past, self.past + n, device=self.device)[None],
            past_key_values=self.cache,
            use_cache=True,
        ).last_hidden_state[0, -1]
        self.past += n
        # one row, as _generate scores it (a batched lm_head could round differently)
        logits = self.reference.lm_head(hidden).float()
        self.last_outputs = {
            "logits": logits[None, None].cpu(),
            "hidden_states": hidden[None, None].cpu(),
        }
        return self.last_outputs["logits"].numpy()


def torch_model(model_dir: pathlib.Path, reference) -> ONNXAudioModel:
    """ONNXAudioModel of a checkpoint folder, its sessions replaced by the reference's modules."""
    model = ONNXAudioModel(model_dir)
    device = next(reference.parameters()).device

    @torch.no_grad()
    def depth_logits(hidden, codes, masked):
        hidden, codes, masked = (
            torch.from_numpy(np.array(a)).to(device) for a in (hidden, codes, masked)
        )
        return reference.model.depth_decoder(hidden, codes, masked).float().cpu().numpy()

    model.cached_decoder = lambda capacity: TorchDecoder(reference)
    model.depth_logits = depth_logits
    return model


@torch.no_grad()
def audio_features(reference, input_features: torch.Tensor | None) -> np.ndarray | None:
    if input_features is None:
        return None
    mask = torch.ones(input_features.shape[0], input_features.shape[-1], dtype=torch.long)
    device = next(reference.parameters()).device
    return (
        reference.model.get_audio_features(input_features.to(device), mask.to(device)).cpu().numpy()
    )


def reference_slots(reference, ids, input_features, codes, order_seed, cache=None, **kwargs):
    """(generated tokens, frames [F, 12]) of the reference's greedy generate with the MaskGIT
    orders rng_orders(order_seed) draws, end-of-audio frames kept. cache is the slot_cache
    fixture's (folder, key) or None: the slots are then kept in that folder (remote.cached_slots)."""
    device = next(reference.parameters()).device

    def compute():
        inputs = {"input_ids": torch.tensor([ids], device=device)}
        if input_features is not None:
            inputs["input_features"] = input_features.to(device)
        if codes is not None:
            inputs["audio_out_codes"] = torch.as_tensor(codes, device=device)
        with torch.no_grad(), inject_orders(maskgit_orders(order_seed, 4096)):
            out = reference.generate(**inputs, return_audio_eos=True, **kwargs)
        # with the marker generate appends for a requested mode
        tokens = out.sequences[0, len(ids) : int(out.sequence_lengths[0])]
        frames = out.audio_codes[0, :, : int(out.audio_lengths[0])].T
        return tokens.tolist(), frames.cpu().numpy()

    if cache is None:
        return compute()
    folder, key = cache
    prompt_codes = None if codes is None else np.asarray(codes).tolist()
    key = (*key, ids, prompt_codes, order_seed, sorted(kwargs.items()))
    return cached_slots(folder, key, device.type, compute)


def pattern(tokens: list[int]) -> str:
    """The slot pattern, e.g. T6 A12 T6 A94 T1."""
    runs = []
    for token in tokens:
        kind = "A" if token == AUDIO_OUT else "T"
        if runs and runs[-1][0] == kind:
            runs[-1][1] += 1
        else:
            runs.append([kind, 1])
    return " ".join(f"{kind}{count}" for kind, count in runs)


def assert_same_slots(
    model,
    reference,
    ids,
    input_features,
    codes,
    order_seed,
    max_new,
    before=None,
    cache=None,
    **kwargs,
):
    """The runtime's greedy slots equal the reference's (cache: reference_slots'); before() runs
    ahead of each."""
    before = before or (lambda: None)
    before()
    ref_tokens, ref_frames = reference_slots(
        reference, ids, input_features, codes, order_seed, cache, max_new_tokens=max_new, **kwargs
    )
    before()
    generation = model.generate_ids(
        ids,
        audio_features(reference, input_features),
        codes,
        max_new_tokens=max_new,
        orders=rng_orders(order_seed),
        return_audio_eos=True,
        **kwargs,
    )
    # a mode the prompt doesn't end in: the reference returns its marker with the generated slots
    marker = len(ref_tokens) - len(generation.tokens)
    assert marker in (0, 1)
    logger.info(f"  slots {pattern(generation.tokens)}")
    assert generation.tokens == ref_tokens[marker:]
    assert np.array_equal(generation.audio_codes, ref_frames)
    return generation


# === Sampling and the iSTFT (tiny checkpoint) ===


def test_greedy_sample_matches_reference(tiny_checkpoint):
    sample = remote_class(
        tiny_checkpoint, "modeling_lfm3_audio.Lfm3AudioForConditionalGeneration"
    )._sample
    rng = np.random.default_rng(0)
    for _ in range(50):
        logits = rng.standard_normal(2017).astype(np.float32) * 3
        history = rng.integers(0, 2017, 20)
        logits[history[:3]] = 50.0  # a penalized favourite, sometimes negative
        logits[history[3]] = -0.01
        for penalty, suppress in ((None, ()), (1.2, ()), (1.2, (int(history[0]), 2016)), (0.7, ())):
            expected = sample(
                torch.from_numpy(logits),
                None,
                None,
                penalty,
                torch.from_numpy(history),
                list(suppress),
            )
            ours = sample_token(
                logits, repetition_penalty=penalty, history=history, suppress=suppress
            )
            assert ours == int(expected)
    assert sample_token(np.array([0.0, 1.0, 1.0]), temperature=0) == 1


@pytest.mark.parametrize(
    "warpers",
    [
        {"temperature": 0.4},
        {"temperature": 0.7, "top_k": 5},
        {"top_k": 3},
        {"temperature": 1.0, "top_p": 0.6},
        {"temperature": 1.0, "min_p": 0.2},
    ],
)
def test_sampling_support(tiny_checkpoint, warpers):
    """Every draw lies where the reference's warpers keep a finite score."""
    model_class = remote_class(
        tiny_checkpoint, "modeling_lfm3_audio.Lfm3AudioForConditionalGeneration"
    )
    built = model_class._build_warpers(
        warpers.get("temperature"), warpers.get("top_k"), warpers.get("top_p"), warpers.get("min_p")
    )
    logits = np.random.default_rng(1).standard_normal(64).astype(np.float32) * 2
    scores = torch.from_numpy(logits)[None]
    for warper in built:
        scores = warper(None, scores)
    support = set(np.flatnonzero(np.isfinite(scores[0].numpy())).tolist())
    rng = np.random.default_rng(2)
    draws = {sample_token(logits, **warpers, rng=rng) for _ in range(500)}
    assert draws <= support and len(draws) > 1


def test_orders():
    orders = rng_orders(20261024)
    for expected in maskgit_orders(20261024, 5):
        assert np.array_equal(orders(), expected)


@pytest.mark.parametrize("codec", ["tiny_codec_reference", "codec_reference"])
def test_stft_to_wave(request, codec):
    """The numpy iSTFT of the codec's STFT features against LiquidCodecModel.decode (the tiny
    codec, and the real one on --device)."""
    codec = request.getfixturevalue(codec)
    device = next(codec.parameters()).device
    codes = torch.randint(0, 2016, (1, 12, 36), generator=torch.Generator().manual_seed(3))
    captured = {}
    handle = codec.decoder.lin.register_forward_hook(lambda m, args, out: captured.update(out=out))
    with torch.no_grad():
        wave = codec.decode(codes.to(device)).audio_values[0].cpu().numpy()
    handle.remove()
    ours = stft_to_wave(captured["out"][0].cpu().numpy(), 1280, 320)
    assert ours.shape == wave.shape == (36 * 1920,)
    error = np.linalg.norm(ours - wave) / np.linalg.norm(wave)
    logger.info(f"  iSTFT rel {error:.2e}")
    assert error < 1e-5


# === Generation (tiny checkpoint) ===


@pytest.fixture
def force_eos_frame(tiny_reference):
    """force(frame): the depth decoder (the reference's and the stand-in's alike) scores the
    end-of-audio code first in every codebook of that frame (3 calls per frame)."""
    state = {"calls": 0, "frame": None}

    def hook(module, args, logits):
        frame, state["calls"] = state["calls"] // 3, state["calls"] + 1
        if frame == state["frame"]:
            logits = logits.clone()
            logits[..., EOS_CODE] = 1e4
        return logits

    handle = tiny_reference.model.depth_decoder.register_forward_hook(hook)

    def force(frame: int | None):
        state["calls"], state["frame"] = 0, frame

    yield force
    handle.remove()


TINY_CASES = {
    # (generation_mode, end-of-audio frame, slots)
    "text": ("text", None, 16),
    "audio": ("audio", 4, 20),  # A5 (EOS frame included), then text
    "interleaved": ("interleaved", None, 45),
    "interleaved-eos": ("interleaved", 15, 50),  # the speech ends inside the 2nd block
}


@pytest.mark.parametrize("case", TINY_CASES)
def test_tiny_generation(tiny_checkpoint, tiny_reference, force_eos_frame, case):
    mode, eos_frame, max_new = TINY_CASES[case]
    model = torch_model(tiny_checkpoint, tiny_reference)
    features = torch.randn(1, 128, 20, generator=torch.Generator().manual_seed(0))
    ids = [124894, 124899, *[AUDIO] * 3, IM_END, 207, 124899, 1001, 1002]
    generation = assert_same_slots(
        model,
        tiny_reference,
        ids,
        features,
        None,
        7,
        max_new,
        before=lambda: force_eos_frame(eos_frame),
        generation_mode=mode,
    )
    if eos_frame is not None:
        assert (generation.audio_codes[-1] == EOS_CODE).all()
        assert generation.tokens.count(AUDIO_OUT) == eos_frame + 1


@pytest.mark.parametrize("last_frame", ["speech", "end-of-audio"])
def test_tiny_prompt_frames(tiny_checkpoint, tiny_reference, force_eos_frame, last_frame):
    """A prompt ending in a replayed speech frame continues the speech, one ending in an
    end-of-audio frame resumes the text; without return_audio_eos the EOS frame is dropped."""
    model = torch_model(tiny_checkpoint, tiny_reference)
    codes = np.random.default_rng(4).integers(0, EOS_CODE, (3, 12))
    if last_frame == "end-of-audio":
        codes[-1] = EOS_CODE
    ids = [124894, 124899, 1001, AUDIO_START, *[AUDIO_OUT] * 3]
    generation = assert_same_slots(
        model, tiny_reference, ids, None, codes, 11, 12, before=lambda: force_eos_frame(2)
    )
    assert (generation.tokens[0] == AUDIO_OUT) == (last_frame == "speech")

    force_eos_frame(2)
    without = model.generate_ids(
        ids, audio_out_codes=codes, max_new_tokens=12, orders=rng_orders(11)
    )
    eos_frames = (generation.audio_codes == EOS_CODE).all(axis=1)
    assert np.array_equal(without.audio_codes, generation.audio_codes[~eos_frames])
    assert without.tokens.count(AUDIO_OUT) == generation.tokens.count(AUDIO_OUT) - eos_frames.sum()


def test_tiny_cached_slots(tiny_reference, tmp_path, monkeypatch):
    """--reference-cache: the reference runs once per key; the next call reads its slots."""
    features = torch.randn(1, 128, 20, generator=torch.Generator().manual_seed(0))
    ids = [124894, 124899, *[AUDIO] * 3, IM_END, 207, 124899, 1001, 1002]
    cache = (tmp_path, ("tiny", "clip digest"))
    kwargs = {"max_new_tokens": 45, "generation_mode": "interleaved"}
    tokens, frames = reference_slots(tiny_reference, ids, features, None, 7, cache, **kwargs)
    assert AUDIO_OUT in tokens and len(frames)
    [saved] = tmp_path.glob("*.npz")

    def no_generate(**inputs):
        raise AssertionError("the reference ran for a cached key")

    monkeypatch.setattr(tiny_reference, "generate", no_generate)
    cached_tokens, cached_frames = reference_slots(
        tiny_reference, ids, features, None, 7, cache, **kwargs
    )
    assert cached_tokens == tokens and np.array_equal(cached_frames, frames)
    with pytest.raises(AssertionError, match="cached key"):
        reference_slots(tiny_reference, ids, features, None, 8, cache, **kwargs)
    assert list(tmp_path.glob("*.npz")) == [saved]


# === Real checkpoint ===

SYSTEM = {
    "asr": "Perform ASR in English.",
    "tts": "Perform TTS in American English using a female voice.",
    "s2s": "Respond with interleaved text and audio.",
}


def conversation(case: str) -> tuple[list[dict], bool]:
    """(messages, continue_final_message) of a case; the user audio is the question clip."""
    audio = {"role": "user", "content": [{"type": "audio"}]}
    if case == "asr":
        return [{"role": "system", "content": SYSTEM["asr"]}, audio], False
    if case == "tts":
        user = {"role": "user", "content": "The quick brown fox jumps over the lazy dog."}
        return [
            {"role": "system", "content": SYSTEM["tts"]},
            user,
            assistant("<|audio_start|>"),
        ], True
    if case == "s2s":
        return [
            {"role": "system", "content": SYSTEM["s2s"]},
            audio,
            assistant("<|mixed_start|>"),
        ], True
    if case == "chat":
        return [audio, assistant("<think></think>")], True
    raise ValueError(case)


def assistant(content: str) -> dict:
    return {"role": "assistant", "content": content}


QUESTION = SAMPLES / "woodworks_question.wav"


@pytest.fixture(scope="module")
def question() -> np.ndarray:
    return load_audio(QUESTION, 16000)


def processor_inputs(processor, messages, continue_final: bool, clips: list[np.ndarray]):
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=not continue_final,
        continue_final_message=continue_final,
    )
    return processor(text=text, audio=clips or None, sampling_rate=16000, return_tensors="pt")


def replay(processor, messages: list[dict], slots: list[int]) -> list[dict]:
    """Turn 2 after an S2S answer replayed verbatim: <|mixed_start|> and the slots' text (one
    <|audio_out|> per frame) as the assistant message, then a spoken question."""
    content = "<|mixed_start|>" + processor.tokenizer.decode(slots, skip_special_tokens=False)
    turn2 = [{"role": "user", "content": [{"type": "audio"}]}, assistant("<|mixed_start|>")]
    return [*messages[:2], assistant(content), *turn2]


@pytest.mark.parametrize("case", ["asr", "tts", "s2s", "chat", "multiturn"])
def test_prompt_ids(checkpoint, reference_processor, question, case):
    model = ONNXAudioModel(checkpoint)
    model.load_preprocessing()
    if case == "multiturn":
        messages, _ = conversation("s2s")
        words = reference_processor.tokenizer("Sure, here it is.", add_special_tokens=False)[
            "input_ids"
        ]
        slots = [*words[:6], *[AUDIO_OUT] * 12, *words[6:], TEXT_END, *[AUDIO_OUT] * 4]
        messages, continue_final, clips = replay(reference_processor, messages, slots), True, 2
    else:
        messages, continue_final = conversation(case)
        clips = 0 if case == "tts" else 1
    inputs = processor_inputs(reference_processor, messages, continue_final, [question] * clips)
    expected = inputs["input_ids"][0].tolist()
    counts = [model.config.mel.num_tokens(model.config.mel.num_frames(len(question)))] * clips
    ids = model.prompt_ids(
        messages,
        counts,
        add_generation_prompt=not continue_final,
        continue_final_message=continue_final,
    )
    assert ids == expected


# Greedy slots of the reference and the runtime: the ASR transcript, TTS to its end-of-audio
# frame, and S2S text and speech blocks to the end of the turn
GENERATION = {"asr": 64, "tts": 160, "s2s": 640}


def text_of(tokenizer, tokens: list[int]) -> str:
    return tokenizer.decode([t for t in tokens if t != AUDIO_OUT], skip_special_tokens=True)


@pytest.mark.parametrize("case", GENERATION)
def test_generation(checkpoint, reference, reference_processor, question, slot_cache, case):
    messages, continue_final = conversation(case)
    clips = [] if case == "tts" else [question]
    inputs = processor_inputs(reference_processor, messages, continue_final, clips)
    ids = inputs["input_ids"][0].tolist()
    model = torch_model(checkpoint, reference)
    generation = assert_same_slots(
        model,
        reference,
        ids,
        inputs.get("input_features"),
        None,
        20261024,
        GENERATION[case],
        cache=slot_cache(*([] if case == "tts" else [QUESTION])),
    )
    text = text_of(reference_processor.tokenizer, generation.tokens)
    logger.info(f"  {case}: {text!r}, {len(generation.audio_codes)} frames")
    if case == "asr":
        assert generation.tokens[-1] == IM_END and AUDIO_OUT not in generation.tokens
    if case == "tts":  # the speech ends, then the turn
        assert (generation.audio_codes[-1] == EOS_CODE).all() and generation.tokens[-1] == IM_END


# === ONNX bundles (the export under --exports-dir, README 5) ===

PRECISIONS = ["fp32", "fp16", "q8", "q8f16", "q4", "q4f16"]
# Greedy TTS round trips: the bundle's speech of each, transcribed by its own ASR
SENTENCES = (
    "The quick brown fox jumps over the lazy dog.",
    "The harbor was quiet that morning.",
    "Please call Stella and ask her to bring these things with her from the store.",
    "A rainbow is a division of white light into many beautiful colors.",
    "We will meet at the library at half past nine tomorrow.",
)
# case -> (conversation, user clip, MaskGIT order seed)
ONNX_CASES = {
    "asr": ("asr", "question", 20261024),
    "tts": ("tts", None, 20261024),
    "s2s": ("s2s", "question", 20261024),
}
# The interleaved schedule of every S2S answer: 6 text slots and 12 frames until the text ends
# (<|text_end|>), speech to its end-of-audio frame, then <|im_end|>; greedy speech may run to the
# slot cap instead (README 3.1d)
S2S_SCHEDULE = r"(T6 A12 )*T[1-6] A\d+( T1)?"
# The rank below which assert_reference_path holds a bundle's S2S text (test_lfm3_omni's
# test_image_answer holds image answers to the reference's top 5 too)
TOP_K = 5
# (model, precision, case, host(ep) or "*") -> the evidence of a known failure of
# test_onnx_generation, xfail(strict=True)
KNOWN_GENERATION: dict[tuple, str] = {}
# Upper bounds of the TTS round trips' WER (SENTENCES pooled)
MAX_ROUND_TRIP_WER = {p: 0.0 for p in PRECISIONS}


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower().replace("’", "'"))


def distance(reference: list, hypothesis: list) -> int:
    """Levenshtein distance of two sequences (of words or characters)."""
    row = list(range(len(hypothesis) + 1))
    for i, item in enumerate(reference, 1):
        prev, row[0] = row[0], i
        for j, other in enumerate(hypothesis, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (item != other))
    return row[-1]


def edits(reference: str, hypothesis: str) -> tuple[int, int]:
    """(word-level edit distance, the reference's word count)"""
    ref = words(reference)
    return distance(ref, words(hypothesis)), len(ref)


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate: the word-level edit distance over the reference's length."""
    errors, count = edits(reference, hypothesis)
    return errors / max(count, 1)


def transcribe(model, wave: np.ndarray, max_new_tokens: int) -> str:
    """Greedy ASR of the codec's 24 kHz speech by model (an ONNXAudioModel)."""
    import scipy.signal

    speech = scipy.signal.resample_poly(wave, 2, 3).astype(np.float32)  # 24 → 16 kHz
    ids = onnx_prompt(model, "asr", [speech])
    generation = model.generate_ids(
        ids, model.encode_audio([speech]), max_new_tokens=max_new_tokens
    )
    return text_of(model.tokenizer, generation.tokens)


def onnx_prompt(model, case: str, waves: list[np.ndarray], text: str | None = None) -> list[int]:
    messages, continue_final = conversation(case)
    if text is not None:
        messages = [
            {"role": "user", "content": text} if m["role"] == "user" else m for m in messages
        ]
    mel = model.config.mel
    return model.prompt_ids(
        messages,
        [mel.num_tokens(mel.num_frames(len(wave))) for wave in waves],
        add_generation_prompt=not continue_final,
        continue_final_message=continue_final,
    )


@pytest.fixture(scope="module")
def onnx_cases(reference, reference_processor, slot_cache):
    """case -> (user clips, prompt ids, the reference's greedy slots and frames), built once."""
    built = {}

    def get(case: str):
        if case not in built:
            kind, clip, seed = ONNX_CASES[case]
            paths = [] if clip is None else [QUESTION]
            waves = [load_audio(path, 16000) for path in paths]
            messages, continue_final = conversation(kind)
            inputs = processor_inputs(reference_processor, messages, continue_final, waves)
            ids = inputs["input_ids"][0].tolist()
            slots = reference_slots(
                reference,
                ids,
                inputs.get("input_features"),
                None,
                seed,
                slot_cache(*paths),
                max_new_tokens=GENERATION[kind],
            )
            built[case] = (waves, ids, *slots)
        return built[case]

    return get


@torch.no_grad()
def assert_reference_path(reference, processor, waves, ids, generation, top_1: bool):
    """The float32 reference, teacher-forced on a bundle's own S2S slots and speech codes, has each
    of the bundle's text tokens in its top TOP_K, and with top_1 on top where it is confident
    (entropy below 1 nat).

    A bundle's S2S answer can word its text differently from the reference's, as bf16 PyTorch
    does: the speech codes read back differ from the first frames, and the text follows them, so
    the reference reads the bundle's own codes. The top-k rule also holds a flip at a near-tie to
    the reference's path, which the confident rule cannot see."""
    device = next(reference.parameters()).device
    messages, continue_final = conversation("s2s")
    inputs = processor_inputs(processor, messages, continue_final, waves)
    tokens = generation.tokens
    hidden = reference.model(
        input_ids=torch.tensor([ids + tokens], device=device),
        input_features=inputs["input_features"].to(device),
        feature_attention_mask=inputs["feature_attention_mask"].to(device),
        audio_out_codes=torch.as_tensor(
            generation.audio_codes[: tokens.count(AUDIO_OUT)], device=device
        ),
        use_cache=False,
    ).last_hidden_state[0, len(ids) - 1 : -1]
    lp = torch.log_softmax(reference.lm_head(hidden).double(), -1).cpu().numpy()
    confident = -(np.exp(lp) * lp).sum(-1) < 1.0
    text = [i for i, t in enumerate(tokens) if t != AUDIO_OUT]
    rank = {i: int((lp[i] > lp[i, tokens[i]]).sum()) for i in text}
    logger.info(
        f"  reference path: {sum(r == 0 for r in rank.values())}/{len(text)} text slots on top, "
        f"worst rank {max(rank.values(), default=0)}, "
        f"{sum(confident[i] for i in text)} confident"
    )
    off_path = [i for i in text if rank[i] >= TOP_K]
    assert not off_path, f"text slots outside the reference's top {TOP_K}: {off_path}"
    flips = [i for i in text if top_1 and confident[i] and rank[i] > 0]
    assert not flips, f"confident text slots where the bundle left the reference's top: {flips}"


def host(ep: str) -> str:
    """The key of a known failure's host: "cuda", or the CPU EP and its machine, "cpu-x86_64" or
    "cpu-arm64"."""
    return ep if ep == "cuda" else f"cpu-{platform.machine()}"


def expect_known_failure(request, known: dict[tuple, str], key: tuple, ep: str):
    """Marks the running test xfail(strict=True) with known's evidence for key + (host(ep),) or
    key + ("*",) (every host), so that a fix (XPASS) fails it as a regression would."""
    reason = known.get((*key, host(ep))) or known.get((*key, "*"))
    if reason:
        request.applymarker(pytest.mark.xfail(strict=True, reason=reason))


def onnx_generation(model, onnx_cases, case: str):
    """The bundle's greedy generation of an ONNX_CASES case, with the reference's orders."""
    kind, _, seed = ONNX_CASES[case]
    waves, ids, _, _ = onnx_cases(case)
    return model.generate_ids(
        ids,
        model.encode_audio(waves) if waves else None,
        max_new_tokens=GENERATION[kind],
        orders=rng_orders(seed),
        return_audio_eos=True,
    )


def assert_generation(
    generation, model, onnx_cases, reference, reference_processor, model_id, precision, case
):
    """test_onnx_generation's rules, against the reference's slots of the case."""
    kind, _, _ = ONNX_CASES[case]
    waves, ids, ref_tokens, ref_frames = onnx_cases(case)
    ref_text, text = (
        text_of(model.tokenizer, ref_tokens),
        text_of(model.tokenizer, generation.tokens),
    )
    error = wer(ref_text, text) if words(ref_text) else 0.0
    same = np.mean([a == b for a, b in zip(generation.tokens, ref_tokens, strict=False)])
    frames = min(len(ref_frames), len(generation.audio_codes))
    codes = np.mean(generation.audio_codes[:frames] == ref_frames[:frames]) if frames else 1.0
    logger.info(
        f"  {precision} {case}: {pattern(generation.tokens)} (reference {pattern(ref_tokens)}), "
        f"slot agreement {same:.3f}, code agreement {codes:.3f}, WER {error:.3f}: {text!r}"
    )
    if kind == "s2s":
        assert re.fullmatch(S2S_SCHEDULE, pattern(ref_tokens))
        assert re.fullmatch(S2S_SCHEDULE, pattern(generation.tokens))
        assert TEXT_END in generation.tokens and AUDIO_OUT in generation.tokens
        # a turn ends as the reference's does: after an end-of-audio frame
        if generation.tokens[-1] == IM_END:
            assert (generation.audio_codes[-1] == EOS_CODE).all()
    if precision == "fp32":
        assert generation.tokens == ref_tokens
        assert np.array_equal(generation.audio_codes, ref_frames)
    elif kind == "asr":
        assert generation.tokens == ref_tokens
    elif kind == "tts":
        assert (generation.audio_codes[-1] == EOS_CODE).all() and generation.tokens[-1] == IM_END
    else:
        top_1 = not precision.startswith("q4")
        assert_reference_path(reference, reference_processor, waves, ids, generation, top_1)


# precision outermost: onnx_models holds one bundle at a time
@pytest.mark.parametrize("case", ONNX_CASES)
@pytest.mark.parametrize("precision", PRECISIONS)
def test_onnx_generation(
    request, onnx_cases, onnx_models, reference, reference_processor, model_id, ep, precision, case
):
    """Greedy generation with injected orders through the bundle's four graphs: fp32 gives the
    reference's slots exactly; the others the reference's ASR transcript, TTS speech that ends,
    and S2S answers in S2S_SCHEDULE with speech, whose turn ends only after an end-of-audio frame,
    with text on the reference's path (assert_reference_path; top-1 at confident slots but for
    int4). The speech codes fed back diverge from the first frames, and the woodworks S2S answer
    can flip at a near-tie early on (bf16 PyTorch's does too), so its words and whether its speech
    ends within the slot cap follow the precision, as they follow the MaskGIT order seed in
    float32 and bf16 PyTorch (README 3.1d). KNOWN_GENERATION's cells are xfail(strict=True)."""
    expect_known_failure(request, KNOWN_GENERATION, (model_id, precision, case), ep)
    model = onnx_models(precision)
    generation = onnx_generation(model, onnx_cases, case)
    assert_generation(
        generation, model, onnx_cases, reference, reference_processor, model_id, precision, case
    )


# The bundles of the CUDA EP (fp32 and the fp16-activation ones), run as lfm3-audio-infer and
# lfm3-omni-infer run them by default: TF32 on, but for the fp32 depth decoder
TF32_PRECISIONS = ["fp32", "fp16", "q8f16", "q4f16"]
# Upper bounds of KL(TF32 off || TF32 on) at the prompt's last position. The fp16-activation
# bundles move only where TF32 moves their fp16 speech encoder's features (a second TF32-off
# session gives the same bits)
TF32_MAX_KL = {"fp32": 5e-6, "fp16": 5e-4, "q8f16": 5e-4, "q4f16": 5e-4}


@pytest.mark.slow
@pytest.mark.parametrize("case", ONNX_CASES)
@pytest.mark.parametrize("precision", TF32_PRECISIONS)
def test_onnx_generation_tf32(
    request, onnx_cases, onnx_models, reference, reference_processor, model_id, ep, precision, case
):
    """The default runtime on CUDA (ONNXAudioModel(tf32=True)) against the same bundle with TF32
    off: KL(off || on) of the prompt's last logits, the speech encoder's features and the codec's
    waveform of TF32 off's frames, and where the greedy generations part (logged; the KL bounded
    by TF32_MAX_KL); then TF32 on's generation, held to test_onnx_generation's rules (and its
    KNOWN_GENERATION cells), ends its turn where TF32 off's does."""
    if ep != "cuda":
        pytest.skip("TF32 is a CUDA EP option")
    expect_known_failure(request, KNOWN_GENERATION, (model_id, precision, case), ep)
    off, on = onnx_models(precision), onnx_models(precision, tf32=True)
    waves, ids, _, _ = onnx_cases(case)
    features = [model.encode_audio(waves) if waves else None for model in (off, on)]
    logits = [
        model.prefill_outputs(ids, feats)[0][-1]
        for model, feats in zip((off, on), features, strict=True)
    ]
    kl = float(kl_divergence(*logits))
    generation_off, generation = (onnx_generation(model, onnx_cases, case) for model in (off, on))

    message = f"  {precision} {case} TF32 on vs off: prompt KL {kl:.2e}"
    if waves:
        message += f", speech features cosine min {cosine(*features).min():.6f}"
    frames = generation_off.audio_codes
    speech = frames[~(frames == EOS_CODE).all(-1)]
    if len(speech):
        wave_off, wave = (model.decode_audio(speech) for model in (off, on))
        error = np.linalg.norm(wave - wave_off) / np.linalg.norm(wave_off)
        message += f", codec SNR {-20 * np.log10(max(error, 1e-12)):.1f} dB"
    tokens_off, tokens = generation_off.tokens, generation.tokens
    parted = next(
        (i for i, (a, b) in enumerate(zip(tokens_off, tokens, strict=False)) if a != b),
        min(len(tokens_off), len(tokens)),
    )
    count = min(len(frames), len(generation.audio_codes))
    codes = np.mean(generation.audio_codes[:count] == frames[:count]) if count else 1.0
    logger.info(
        f"{message}; slots {pattern(tokens)} (off {pattern(tokens_off)}), same to slot {parted}, "
        f"code agreement {codes:.3f}"
    )
    assert kl <= TF32_MAX_KL[precision]
    assert_generation(
        generation, on, onnx_cases, reference, reference_processor, model_id, precision, case
    )
    if tokens_off[-1] == IM_END:
        assert tokens[-1] == IM_END, "TF32 on runs to the slot cap where TF32 off ends the turn"


@pytest.mark.parametrize("precision", PRECISIONS)
def test_onnx_tts_round_trip(onnx_models, precision):
    """Greedy TTS of SENTENCES through the bundle, the codec and the iSTFT, transcribed back by
    the bundle's own ASR: each ends its speech and its turn, and the pooled WER is bounded."""
    model = onnx_models(precision)
    errors, count = 0, 0
    for sentence in SENTENCES:
        ids = onnx_prompt(model, "tts", [], sentence)
        generation = model.generate_ids(ids, max_new_tokens=GENERATION["tts"], seed=0)
        assert generation.tokens[-1] == IM_END and len(generation.audio_codes)
        wave = model.decode_audio(generation.audio_codes)
        assert wave.shape == (len(generation.audio_codes) * 1920,) and np.isfinite(wave).all()
        transcript = transcribe(model, wave, 64)
        sentence_errors, sentence_words = edits(sentence, transcript)
        errors, count = errors + sentence_errors, count + sentence_words
        logger.info(
            f"  {precision}: {len(generation.audio_codes)} frames, {transcript!r}, "
            f"WER {sentence_errors / sentence_words:.3f}"
        )
    logger.info(f"  {precision}: WER {errors / count:.3f} over {len(SENTENCES)} sentences")
    assert errors / count <= MAX_ROUND_TRIP_WER[precision]


def transformers_draw(logits, temperature, top_k, repetition_penalty, history, uniform) -> int:
    """The id that transformers' processors (repetition penalty over history, temperature,
    top-k) and inverse-CDF sampling at uniform give: what numpy's Generator.choice(p=...) draws
    from the same uniform."""
    histories = [[] if history is None else np.asarray(history).tolist()]
    probs = processed_probs(
        np.asarray(logits)[None], temperature, top_k, None, repetition_penalty, histories
    )[0]
    cdf = np.cumsum(probs)
    return int(np.searchsorted(cdf / cdf[-1], uniform, side="right"))


# The speech sampling of the release checkpoints' generation_config.json (remote.SPEECH_SAMPLING's
# fields, assets/), which lfm3-audio-infer and lfm3-omni-infer take: AudioSampling's temperature,
# top_k, repetition_penalty and repetition_window
SPEECH_SETTINGS = {
    "LFM3-Audio-2B": (1.0, 128, 1.1, 4096),
    "LFM3-Omni-2B": (0.4, None, 1.2, 64),
}


def test_sampler_matches_transformers():
    """sample_token's seeded draws are the ids transformers' processors give the same logits
    at the same uniforms, at the release checkpoints' speech sampling and with other top-k."""
    rng = np.random.default_rng(0)
    settings = [setting[:3] for setting in SPEECH_SETTINGS.values()]
    for temperature, top_k, penalty in (*settings, (0.7, 64, 1.2), (1.0, 4, 1.2)):
        draws, replay = np.random.default_rng(1), np.random.default_rng(1)
        for _ in range(200):
            logits = (rng.standard_normal(2017) * 3).astype(np.float32)
            history = rng.integers(0, 2017, 30)
            got = sample_token(
                logits,
                temperature,
                top_k,
                repetition_penalty=penalty,
                history=history,
                rng=draws,
            )
            expected = transformers_draw(
                logits, temperature, top_k, penalty, history, replay.random()
            )
            assert got == expected


@pytest.mark.parametrize("cli", [audio_infer, omni_infer], ids=["audio-infer", "omni-infer"])
@pytest.mark.parametrize("name", SPEECH_SETTINGS)
def test_speech_sampling(tmp_path, name, cli):
    """lfm3-audio-infer's and lfm3-omni-infer's speech sampling (infer.speech_sampling of their
    build_parser's arguments) as the reference's generate() resolves it: a flag, else the export's
    generation_config.json (the release's), else the reference's default; --greedy takes
    temperature 0. No --audio-* flag and no --language has a default of its own."""
    import dataclasses
    import json
    import shutil

    from liquidonnx.lfm3_audio.session import AudioSampling

    def sampling(folder, *flags):
        args = cli.build_parser().parse_args(["--model", str(folder), *flags])
        return audio_infer.speech_sampling(args)

    export, bare = tmp_path / "export", tmp_path / "bare"
    export.mkdir()
    bare.mkdir()
    shutil.copy2(ASSETS / name / "generation_config.json", export)
    args = cli.build_parser().parse_args(["--model", str(export)])
    assert [getattr(args, key) for key in (*SPEECH_SAMPLING, "language")] == [None] * 5
    expected = AudioSampling(*SPEECH_SETTINGS[name])
    assert audio_infer.speech_sampling(args) == expected
    flags = ["--audio-temperature", "0.7", "--audio-top-k", "5"]
    flags += ["--audio-repetition-penalty", "1.3", "--audio-repetition-window", "16"]
    assert sampling(export, *flags) == AudioSampling(0.7, 5, 1.3, 16)
    assert sampling(export, "--greedy") == dataclasses.replace(expected, temperature=0.0)
    # no file, then a file without the fields
    assert sampling(bare) == AudioSampling(None, None, None, 64)
    (bare / "generation_config.json").write_text(json.dumps({"do_sample": False}))
    assert sampling(bare) == AudioSampling(None, None, None, 64)


SAMPLING_SEED = 7


@pytest.mark.parametrize("precision", PRECISIONS)
def test_onnx_seeded_sampling(onnx_models, monkeypatch, precision):
    """LFM3-Omni-2B's speech sampling (SPEECH_SETTINGS) of a TTS run with --seed, through the
    bundle: a second run gives the same slots and codes, and every speech code is the one
    transformers' processors give the depth logits and history it was drawn from, at the uniform
    the seeded generator gave that draw."""
    import inspect

    from liquidonnx.lfm3_audio import session

    model = onnx_models(precision)
    sampling = session.AudioSampling(*SPEECH_SETTINGS["LFM3-Omni-2B"])
    ids = onnx_prompt(model, "tts", [], SENTENCES[0])
    signature, original, calls = inspect.signature(sample_token), session.sample_token, []

    def recording(*args, **kwargs):
        token = original(*args, **kwargs)
        calls.append((signature.bind(*args, **kwargs).arguments, token))
        return token

    monkeypatch.setattr(session, "sample_token", recording)
    runs = [
        model.generate_ids(
            ids, max_new_tokens=GENERATION["tts"], audio_sampling=sampling, seed=SAMPLING_SEED
        )
        for _ in range(2)
    ]
    assert runs[0].tokens == runs[1].tokens
    assert np.array_equal(runs[0].audio_codes, runs[1].audio_codes)

    first_run = calls[: len(calls) // 2]
    draws = [(a, t) for a, t in first_run if a.get("temperature")]
    replay = np.random.default_rng(SAMPLING_SEED)
    misses = [
        i
        for i, (a, token) in enumerate(draws)
        if token
        != transformers_draw(
            a["logits"],
            a["temperature"],
            a.get("top_k"),
            a.get("repetition_penalty"),
            a.get("history"),
            replay.random(),
        )
    ]
    logger.info(
        f"  {precision}: {pattern(runs[0].tokens)}, {len(draws)} sampled codes, "
        f"{len(misses)} off transformers' draw"
    )
    assert len(draws) >= 12 and not misses
