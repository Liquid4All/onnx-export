"""
The exported decoder against the float32 reference on real prompts, teacher-forced in one uncached
run, scored where generation reads it: the logits at the last prompt position and over the
reference's greedy text answer, and the final-norm hidden states at the positions whose frame the
depth decoder samples (each position followed by <|audio_out|>). The decoder gets the reference's
own speech features, so the speech encoder's error stays out.

    asr        two clips (16 kHz) with the ASR system prompt, past the sliding window
    text       a text-only question
    speech     a TTS turn replayed as speech frames (the reference's greedy frames, MaskGIT orders
               injected, closed by an end-of-audio frame), then a text question about it
    multiturn  an S2S turn (the reference's greedy slots, cut at TURN1 slots and closed by an
               end-of-audio frame and <|im_end|>) replayed verbatim, then a second spoken question
               with the assistant prefilled with <|mixed_start|>: KL at the first logits of turn 2

Run with:
    uv run --no-sync pytest tests/test_lfm3_audio/test_decoder.py -v --device cuda
    uv run pytest tests/test_lfm3_audio/test_decoder.py -v -k "not f16 and not fp16" \
        --lfm3-audio-model LiquidAI/LFM3-Omni-2B
"""

import copy
import logging
import pathlib
from dataclasses import dataclass, field

import numpy as np
import pytest
import torch

from liquidonnx.lfm3.session import load_session
from liquidonnx.lfm3_audio.session import BUNDLES, load_audio
from liquidonnx.session import initialize_cache

from .remote import inject_orders, maskgit_orders

logger = logging.getLogger(__name__)

SAMPLES = pathlib.Path(__file__).parents[2] / "samples" / "audio"
AUDIO_OUT, EOS_CODE, IM_END = 125032, 2016, 124900
ANSWER = 24
FRAMES = 24
TURN1 = 120
# Two clips: 62 + 203 speech tokens, past the sliding window of 256
CLIPS = ("woodworks_question.wav", "fool_me_once_mono.wav")
SPEECH = "The harbor was quiet that morning."
S2S = "Respond with interleaved text and audio."
PROMPTS = {
    "asr": [
        {"role": "system", "content": "Perform ASR in English."},
        {"role": "user", "content": [{"type": "audio"}, {"type": "audio"}]},
    ],
    "multiturn": [
        {"role": "system", "content": S2S},
        {"role": "user", "content": [{"type": "audio"}]},
    ],
    "text": [{"role": "user", "content": "Write a haiku about the sea."}],
    "speech": [
        {"role": "system", "content": "Perform TTS in English using a female voice."},
        {"role": "user", "content": SPEECH},
        {"role": "assistant", "content": [{"type": "text", "text": "<|audio_start|>"}]},
    ],
}
QUESTION = "Respond with text. What did you just say?"
# Mean KL(reference || ONNX) over the scored positions, LFM3-VL's bounds (test_lfm3_vl/test_decoder).
# On the CPU EP the q8 recipe's MatMulNBits accuracy_level 4 quantizes the activations to int8,
# which CUDA doesn't, so q8 measures higher there; with accuracy_level 0 it measures CUDA's numbers
# (--matmulnbits-accuracy-level, README 5).
MAX_KL = {"fp32": 1e-6, "fp16": 1e-4, "q8": 5e-4, "q8f16": 5e-4, "q4": 0.1, "q4f16": 0.1}
# Positions where the reference is confident (entropy below 1 nat) keep its top token, except in q4
CONFIDENT = 1.0
# Lowest cosine of the hidden states the depth decoder reads; test_bf16_floor logs bf16 PyTorch's
MIN_COSINE = {
    "fp32": 0.99999,
    "fp16": 0.999,
    "q8": 0.999,
    "q8f16": 0.999,
    "q4": 0.95,
    "q4f16": 0.95,
}


def log_softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64) - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


def kl_divergence(expected: np.ndarray, got: np.ndarray) -> np.ndarray:
    lp, lq = log_softmax(expected), log_softmax(got)
    return (np.exp(lp) * (lp - lq)).sum(-1)


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a * b).sum(-1) / np.linalg.norm(a, axis=-1) / np.linalg.norm(b, axis=-1)


@dataclass
class Case:
    """A prompt (ids) with its speech inputs: the processor's input_features and mask, and speech
    frames [F, 12] for its <|audio_out|> placeholders. build_case adds the reference's greedy text
    answer, its logits over the scored positions [len(answer), V], its final-norm hidden states
    over ids + answer[:-1] [S, H], and its speech features [Na, H], the decoder's input."""

    ids: list[int]
    input_features: torch.Tensor | None = None
    feature_mask: torch.Tensor | None = None
    frames: np.ndarray | None = None
    answer: list[int] = field(default_factory=list)
    logits: np.ndarray | None = None
    hidden: np.ndarray | None = None
    features: np.ndarray | None = None

    def reference_inputs(self, device) -> dict:
        inputs = {}
        if self.input_features is not None:
            inputs["input_features"] = self.input_features.to(device)
            inputs["feature_attention_mask"] = self.feature_mask.to(device)
        if self.frames is not None:
            inputs["audio_out_codes"] = torch.from_numpy(self.frames).to(device)
        return inputs

    @property
    def sequence(self) -> list[int]:
        return self.ids + self.answer[:-1]

    @property
    def speech_positions(self) -> list[int]:
        """The positions whose hidden state the depth decoder samples the next frame from."""
        sequence = self.sequence
        return [i for i in range(len(sequence) - 1) if sequence[i + 1] == AUDIO_OUT]


def greedy_text(reference, ids: list[int], inputs: dict, device) -> list[int]:
    out = reference.generate(
        torch.tensor([ids], device=device),
        **inputs,
        max_new_tokens=ANSWER,
        generation_mode="text",
        suppress_mixed=True,
        suppress_audio_out=True,
    )
    return out.sequences[0, len(ids) : int(out.sequence_lengths[0])].tolist()


def speech_frames(reference, processor, device) -> np.ndarray:
    """The reference's greedy reply to PROMPTS["speech"] (injected MaskGIT orders): up to FRAMES
    frames, closed by an end-of-audio frame as a replayed cut-off reply is."""
    text = processor.apply_chat_template(
        PROMPTS["speech"], tokenize=False, continue_final_message=True
    )
    ids = processor.tokenizer(text, add_special_tokens=False)["input_ids"]
    with inject_orders(maskgit_orders(seed=20261024, count=FRAMES)):
        out = reference.generate(
            torch.tensor([ids], device=device), max_new_tokens=FRAMES, return_audio_eos=True
        )
    # [12, F] → [F, 12], without the end-of-audio frame
    frames = out.audio_codes[0, :, : int(out.audio_lengths[0])].T.cpu().numpy()
    frames = frames[~(frames == EOS_CODE).all(-1)]
    return np.concatenate([frames, np.full((1, 12), EOS_CODE)]).astype(np.int64)


def replayed_turn(reference, processor, device, clip: np.ndarray) -> tuple[list[dict], np.ndarray]:
    """(messages, frames [F, 12]) of PROMPTS["multiturn"] answered by the reference's greedy S2S
    slots (MaskGIT orders of default_rng(3)), replayed as the template renders them: one
    <|audio_out|> per frame, end-of-audio frames included; cut at TURN1 slots, the reply gets an
    end-of-audio frame (the template closes it with <|im_end|>)."""
    messages = [*PROMPTS["multiturn"], {"role": "assistant", "content": "<|mixed_start|>"}]
    text = processor.apply_chat_template(messages, tokenize=False, continue_final_message=True)
    inputs = processor(text=text, audio=[clip], sampling_rate=16000, return_tensors="pt")
    ids = inputs["input_ids"][0].tolist()
    with inject_orders(maskgit_orders(seed=3, count=TURN1)):
        out = reference.generate(
            torch.tensor([ids], device=device),
            input_features=inputs["input_features"].to(device),
            feature_attention_mask=inputs["feature_attention_mask"].to(device),
            max_new_tokens=TURN1,
            return_audio_eos=True,
        )
    slots = out.sequences[0, len(ids) : int(out.sequence_lengths[0])].tolist()
    frames = out.audio_codes[0, :, : int(out.audio_lengths[0])].T.cpu().numpy()
    if slots[-1] == IM_END:
        slots = slots[:-1]
    else:
        slots.append(AUDIO_OUT)
        frames = np.concatenate([frames, np.full((1, 12), EOS_CODE)])
    reply = "<|mixed_start|>" + processor.tokenizer.decode(slots, skip_special_tokens=False)
    turn2 = [
        {"role": "assistant", "content": reply},
        {"role": "user", "content": [{"type": "audio"}]},
        {"role": "assistant", "content": "<|mixed_start|>"},
    ]
    return [*PROMPTS["multiturn"], *turn2], frames.astype(np.int64)


def prompt_case(reference, processor, device, name: str) -> Case:
    messages = PROMPTS[name]
    if name == "multiturn":
        clips = [load_audio(SAMPLES / clip, 16000) for clip in CLIPS]
        messages, frames = replayed_turn(reference, processor, device, clips[0])
        text = processor.apply_chat_template(messages, tokenize=False, continue_final_message=True)
        inputs = processor(text=text, audio=clips, sampling_rate=16000, return_tensors="pt")
        return Case(
            inputs["input_ids"][0].tolist(),
            inputs["input_features"],
            inputs["feature_attention_mask"],
            frames,
        )
    if name == "asr":
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        clips = [load_audio(SAMPLES / clip, 16000) for clip in CLIPS]
        inputs = processor(text=text, audio=clips, sampling_rate=16000, return_tensors="pt")
        return Case(
            inputs["input_ids"][0].tolist(),
            inputs["input_features"],
            inputs["feature_attention_mask"],
        )
    if name == "speech":
        frames = speech_frames(reference, processor, device)
        reply = [{"type": "text", "text": "<|audio_start|>"}, {"type": "audio_out"}]
        history = [
            *messages[:2],
            {"role": "assistant", "content": reply},
            {"role": "user", "content": QUESTION},
        ]
        text = processor.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        ids = []
        for token in processor.tokenizer(text, add_special_tokens=False)["input_ids"]:
            ids += [token] * (len(frames) if token == AUDIO_OUT else 1)
        return Case(ids, frames=frames)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return Case(processor.tokenizer(text, add_special_tokens=False)["input_ids"])


def reference_outputs(model, case: Case, device) -> tuple[np.ndarray, np.ndarray]:
    """(logits at the scored positions [len(answer), V], hidden [S, H]) of one uncached forward of
    model (the reference, or another precision of it) over case.sequence."""
    dtype = next(model.parameters()).dtype
    inputs = {
        name: value.to(dtype) if value.is_floating_point() else value
        for name, value in case.reference_inputs(device).items()
    }
    with torch.no_grad():
        hidden = model.model(
            input_ids=torch.tensor([case.sequence], device=device),
            **inputs,
            use_cache=False,
        ).last_hidden_state[0]
        logits = model.lm_head(hidden[len(case.ids) - 1 :])
    return logits.float().cpu().numpy(), hidden.float().cpu().numpy()


def build_case(reference, processor, device, name: str) -> Case:
    case = prompt_case(reference, processor, device, name)
    inputs = case.reference_inputs(device)
    with torch.no_grad():
        case.answer = greedy_text(reference, case.ids, inputs, device)
        if case.input_features is not None:
            features = reference.model.get_audio_features(
                inputs["input_features"], inputs["feature_attention_mask"]
            )
            case.features = features.float().cpu().numpy()
    case.logits, case.hidden = reference_outputs(reference, case, device)
    return case


@pytest.fixture(scope="module")
def cases(reference, reference_processor, device):
    """name -> Case of PROMPTS[name] with the reference's outputs (built once)."""
    built = {}

    def get(name: str) -> Case:
        if name not in built:
            built[name] = build_case(reference, reference_processor, device, name)
        return built[name]

    return get


@pytest.fixture(scope="module")
def decoders(export_dir, ep):
    """precision -> decoder session of its bundle on --ep (cached)."""
    sessions = {}

    def get(precision: str):
        if precision not in sessions:
            path = export_dir / "onnx" / BUNDLES[precision]["decoder"]
            sessions[precision] = load_session(path, ep=ep, tf32=False)
        return sessions[precision]

    return get


def run_decoder(session, ids, features, frames, hidden_size, keep: int = 0):
    """(logits [K, V], hidden_states [K, H]) of the last keep positions (all with 0) of one
    uncached run over ids."""
    feed = {
        "input_ids": np.array([ids], dtype=np.int64),
        "attention_mask": np.ones((1, len(ids)), dtype=np.int64),
        "position_ids": np.arange(len(ids), dtype=np.int64)[None, :],
        "audio_features": np.zeros((0, hidden_size), np.float32) if features is None else features,
        "audio_out_codes": np.zeros((0, 12), np.int64) if frames is None else frames,
        "logits_to_keep": np.array([keep], dtype=np.int64),
        **initialize_cache(session),
    }
    names = {i.name for i in session.get_inputs()}
    for name in ("image_features", "video_features"):
        if name in names:
            feed[name] = np.zeros((0, hidden_size), np.float32)
    logits, hidden = session.run(["logits", "hidden_states"], feed)
    return logits[0], hidden[0]


def score(case: Case, logits: np.ndarray, hidden: np.ndarray, label: str):
    """(KL per answer position, top-1 agreement, confident positions, cosine of the speech hidden
    states or None) of logits [len(answer), V] and hidden [S, H] against the reference's, logged."""
    kl = kl_divergence(case.logits, logits)
    lp = log_softmax(case.logits)
    confident = -(np.exp(lp) * lp).sum(-1) < CONFIDENT
    same_top = case.logits.argmax(-1) == logits.argmax(-1)
    message = (
        f"{label}: {len(case.ids)} prompt + {len(case.answer)} answer tokens: KL first "
        f"{kl[0]:.2e} mean {kl.mean():.2e} max {kl.max():.2e}, top-1 {same_top.sum()}/"
        f"{len(case.answer)}, {same_top[confident].sum()}/{confident.sum()} confident"
    )
    similarity = None
    if case.frames is not None:
        positions = case.speech_positions
        similarity = cosine(hidden[positions], case.hidden[positions])
        message += f"; {len(positions)} speech hidden states: cosine min {similarity.min():.6f}"
    logger.info(message)
    return kl, same_top, confident, similarity


@pytest.mark.parametrize("precision", list(MAX_KL))
@pytest.mark.parametrize("name", list(PROMPTS))
def test_prefill(cases, decoders, precision, name):
    case = cases(name)
    # the hidden states of speech frames need every position; the logits only the answer's
    keep = 0 if case.frames is not None else len(case.answer)
    logits, hidden = run_decoder(
        decoders(precision), case.sequence, case.features, case.frames, case.hidden.shape[-1], keep
    )
    kl, same_top, confident, similarity = score(
        case, logits[-len(case.answer) :], hidden, f"{precision} {name}"
    )
    if similarity is not None:
        assert similarity.min() > MIN_COSINE[precision]
    assert kl.mean() < MAX_KL[precision]
    if not precision.startswith("q4"):
        assert same_top[confident].all()


@pytest.fixture(scope="module")
def bf16_reference(reference):
    return copy.deepcopy(reference).to(torch.bfloat16)


@pytest.mark.parametrize("name", list(PROMPTS))
def test_bf16_floor(cases, bf16_reference, device, name):
    """The bf16 reference, scored as test_prefill scores the graphs:
    the context of MAX_KL and MIN_COSINE, logged only."""
    case = cases(name)
    logits, hidden = reference_outputs(bf16_reference, case, device)
    score(case, logits, hidden, f"bf16 PyTorch {name}")
