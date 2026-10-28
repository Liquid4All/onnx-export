"""
Multi-step decode parity: the ONNX graph's cached decode against a teacher-forced PyTorch forward.

The ONNX session decodes greedily with its conv and KV caches (prefill, then one token per step).
PyTorch then runs once over the prompt plus the generated tokens, without a cache; its logits at
each position are what the cached step must reproduce. This checks the conv-cache rotation, the
KV cache and the sliding window across decode steps without PyTorch's own cache code, which
differs between transformers releases (the old checkpoints' remote code had a broken decode
rotation, and transformers 5.14+ grew the conv state the newer remote code reads by one column).

test_sampled_distribution scores the bundles' next-token distributions after transformers'
sampling processors (temperature, top-k, top-p, repetition penalty) at the answer positions.

Run with (--lfm3-model: LiquidAI/LFM3-1.7B unless given):
    uv run pytest tests/test_lfm3/test_decode.py -v -k q4
    uv run --no-sync pytest tests/test_lfm3/test_decode.py -v --device cuda  # README 2
    uv run --no-sync pytest tests/test_lfm3/test_decode.py -v --device cuda --run-long \
        -k bf16_floor  # the floors of the int4 and TV bounds
"""

import copy
import logging
import pathlib

import numpy as np
import pytest
import torch
from helpers import get_model_name, get_onnx_dir, processed_probs, total_variation

from liquidonnx.lfm3.session import CachedDecoder, get_onnx_file, kv_feeds_attention, load_session

logger = logging.getLogger(__name__)

PROMPTS = [
    # Shorter than the conv kernel: the first decode steps read a zero-padded conv state.
    "The capital of France is",
    # > 256 tokens: every decode step of the sliding-window layers drops a key.
    "Liquid Foundation Models are a class of foundation models built by Liquid AI. " * 40,
]

MAX_NEW_TOKENS = 24
FP32_MAX_ABS_DIFF = 5e-4
# Top-5 overlap per step. q4f32 (round-to-nearest int4, no int8 layers) can keep 2 of 5 at a step
# of the long prompt where ranks 2-6 are near ties.
MIN_TOP5_OVERLAP = {"q4f32": 2}

PRECISIONS = [
    pytest.param(None, id="fp32"),
    pytest.param("fp16", id="fp16"),
    pytest.param("q4", id="q4"),
    pytest.param("q4f32", id="q4f32"),
    pytest.param("q4f16", id="q4f16"),
    pytest.param("q8", id="q8"),
    pytest.param("q8f16", id="q8f16"),
]


def greedy_onnx(
    sess, input_ids: list[int], max_new_tokens: int, share_kv: bool
) -> tuple[list[int], np.ndarray]:
    """Greedy tokens and the logits that chose each one, decoding with the graph's caches
    bound in place as lfm3-infer does."""
    decoder = CachedDecoder(sess, len(input_ids) + max_new_tokens if share_kv else None)
    tokens, logits = [], []
    ids = input_ids
    for _ in range(max_new_tokens):
        step_logits = decoder.step(ids, logits_to_keep=np.array([1], dtype=np.int64))
        step_logits = step_logits[0, -1].astype(np.float32)
        tokens.append(int(np.argmax(step_logits)))
        logits.append(step_logits)
        ids = [tokens[-1]]
    return tokens, np.stack(logits)


@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize("prompt", PROMPTS)
def test_decode_parity(
    exports_dir: pathlib.Path, ep: str, pytorch_model, precision: str | None, prompt: str
):
    model_id, model, tokenizer = pytorch_model
    onnx_file = get_onnx_file(get_onnx_dir(exports_dir, model_id), precision)
    if not onnx_file.exists():
        precision_arg = f" --precision {precision}" if precision else ""
        pytest.fail(
            f"ONNX file not found: {onnx_file}\nExport with: uv run lfm3-export {model_id}"
            f"{precision_arg}"
        )

    input_ids = tokenizer.encode(prompt)
    sess = load_session(onnx_file, ep=ep, tf32=False)
    tokens, onnx_logits = greedy_onnx(
        sess, input_ids, MAX_NEW_TOKENS, kv_feeds_attention(onnx_file)
    )

    # === Teacher-forced reference: position len(prompt) - 1 + i predicts token i ===
    sequence = torch.tensor([input_ids + tokens[:-1]], device=model.device)
    with torch.no_grad():
        pt_logits = model(input_ids=sequence).logits[0, len(input_ids) - 1 :].cpu().numpy()

    name = f"{get_model_name(model_id)}/{precision or 'fp32'}"
    logger.info(f"  {name}: {len(input_ids)} prompt tokens, {tokenizer.decode(tokens)!r}")
    pt_tokens = pt_logits.argmax(-1).tolist()
    top5_overlap = [
        len(set(np.argsort(p)[-5:]) & set(np.argsort(o)[-5:]))
        for p, o in zip(pt_logits, onnx_logits, strict=True)
    ]
    logger.info(
        f"  top-1 agreement {np.mean(np.equal(pt_tokens, tokens)):.0%}, top-5 overlap {top5_overlap}"
    )

    if precision is None:
        max_diff = float(np.abs(pt_logits - onnx_logits).max())
        logger.info(f"  fp32 max |logit diff| over {len(tokens)} steps = {max_diff:.2e}")
        assert pt_tokens == tokens, f"{name}: cached decode picked other tokens than PyTorch"
        assert max_diff < FP32_MAX_ABS_DIFF, f"{name}: decode logits differ by {max_diff:.2e}"
    else:
        min_overlap = MIN_TOP5_OVERLAP.get(precision, 3)
        assert min(top5_overlap) >= min_overlap, f"{name}: top-5 overlap per step {top5_overlap}"


def chat_ids(tokenizer, messages: list[dict], enable_thinking: bool = True) -> list[int]:
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
    )
    return tokenizer.encode(text, add_special_tokens=False)


def teacher_forced(model, input_ids: list[int], tokens: list[int]) -> np.ndarray:
    """PyTorch's logits [len(tokens), vocab] at the positions that predict tokens, from one
    uncached forward over input_ids + tokens[:-1]."""
    sequence = torch.tensor([input_ids + tokens[:-1]], device=model.device)
    with torch.no_grad():
        logits = model(input_ids=sequence, logits_to_keep=len(tokens), use_cache=False).logits
    return logits[0].float().cpu().numpy()


def greedy_pytorch(model, input_ids: list[int], steps: int) -> tuple[list[int], np.ndarray]:
    """Greedy tokens past the end of turn, as greedy_onnx, and the logits that chose each one,
    one uncached forward per step (PyTorch's own cache differs between transformers releases)."""
    tokens, logits = [], []
    for _ in range(steps):
        logits.append(teacher_forced(model, input_ids + tokens, [0])[0])
        tokens.append(int(logits[-1].argmax()))
    return tokens, np.stack(logits)


def decode_scores(model, input_ids: list[int], tokens: list[int], logits: np.ndarray):
    """(PyTorch's teacher-forced logits, KL(PyTorch || decode) per step, top-5 overlap per step)
    of a greedy decode's logits [steps, vocab]."""
    expected = teacher_forced(model, input_ids, tokens)
    lp = torch.log_softmax(torch.from_numpy(expected).double(), -1)
    lq = torch.log_softmax(torch.from_numpy(logits).double(), -1)
    kl = (lp.exp() * (lp - lq)).sum(-1).numpy()
    top5 = [
        len(set(np.argpartition(p, -5)[-5:]) & set(np.argpartition(o, -5)[-5:]))
        for p, o in zip(expected, logits, strict=True)
    ]
    return expected, kl, top5


# A second turn in thinking mode (the template's default), after an answer whose reasoning the
# template drops from the history
CHAT = [
    {
        "role": "user",
        "content": "If a train travels 60 km in 45 minutes, what is its speed in km/h?",
    },
    {
        "role": "assistant",
        "content": "<think>45 minutes is 0.75 h; 60 / 0.75 = 80.</think>80 km/h.",
    },
    {"role": "user", "content": "How long would it take that train to travel 200 km?"},
]
LONG_DECODE_STEPS = 600
# Mean KL(PyTorch || ONNX) over the steps: test_decode_parity's max |logit diff| grows with the
# number of logits it takes the max of, even for an exact fp32 graph. q8 and q8f16 take
# test_decoder's bound (the CPU EP's int8 activations can pass it, README 3.1b). int4 takes 4 times
# the floor test_long_decode_bf16_floor measures, as test_lfm3_vl's bounds sit at about 4 times
# bf16 PyTorch: bf16's own greedy 600 steps of CHAT, scored as a bundle's.
INT4_BF16_FLOOR = 2.9e-4
INT4 = ("q4", "q4f16", "q4f32")
LONG_DECODE_MAX_KL = {
    None: 1e-6,
    "fp16": 1e-4,
    "q8": 5e-4,
    "q8f16": 5e-4,
    **dict.fromkeys(INT4, 4 * INT4_BF16_FLOOR),
}
# Every step keeps this many of the reference's top 5 (bf16 PyTorch keeps 4)
LONG_DECODE_MIN_TOP5 = 3


@pytest.mark.parametrize(
    "precision",
    [
        pytest.param(None, id="fp32"),
        *(pytest.param(p, id=p) for p in ("fp16", "q8", "q8f16", "q4", "q4f16", "q4f32")),
    ],
)
def test_long_decode_parity(
    exports_dir: pathlib.Path, ep: str, pytorch_model, precision: str | None
):
    """The caches over hundreds of steps: from the short chat, the sliding-window layers start
    dropping keys during decode (step 256 - prompt length)."""
    model_id, model, tokenizer = pytorch_model
    onnx_file = get_onnx_file(get_onnx_dir(exports_dir, model_id), precision)
    if not onnx_file.exists():
        pytest.fail(f"ONNX file not found: {onnx_file}\nExport with: uv run lfm3-export {model_id}")

    input_ids = chat_ids(tokenizer, CHAT)
    sess = load_session(onnx_file, ep=ep, tf32=False)
    tokens, onnx_logits = greedy_onnx(
        sess, input_ids, LONG_DECODE_STEPS, kv_feeds_attention(onnx_file)
    )
    pt_logits, kl, top5_overlap = decode_scores(model, input_ids, tokens, onnx_logits)
    name = f"{get_model_name(model_id)}/{precision or 'fp32'}"
    logger.info(
        f"  {name}: {len(input_ids)} prompt tokens, KL mean {kl.mean():.2e} max {kl.max():.2e}, "
        f"top-1 agreement {np.mean(pt_logits.argmax(-1) == tokens):.0%}, min top-5 overlap "
        f"{min(top5_overlap)} ({sum(t < LONG_DECODE_MIN_TOP5 for t in top5_overlap)} steps under "
        f"{LONG_DECODE_MIN_TOP5}): {tokenizer.decode(tokens[:60])!r}..."
    )
    assert kl.mean() < LONG_DECODE_MAX_KL[precision]
    if precision is None:
        assert pt_logits.argmax(-1).tolist() == tokens, f"{name}: decode left PyTorch's tokens"
    else:
        assert min(top5_overlap) >= LONG_DECODE_MIN_TOP5, f"{name}: top-5 overlap {top5_overlap}"


@pytest.mark.long
def test_long_decode_bf16_floor(pytorch_model):
    """bf16 PyTorch's own greedy 600 steps of CHAT, scored against fp32 PyTorch as
    test_long_decode_parity scores a bundle: the floor of its int4 bounds (INT4_BF16_FLOOR),
    which meets the int8 bundles' bounds."""
    model_id, model, tokenizer = pytorch_model
    input_ids = chat_ids(tokenizer, CHAT)
    bf16 = copy.deepcopy(model).to(torch.bfloat16)
    tokens, logits = greedy_pytorch(bf16, input_ids, LONG_DECODE_STEPS)
    del bf16
    pt_logits, kl, top5_overlap = decode_scores(model, input_ids, tokens, logits)
    logger.info(
        f"  {get_model_name(model_id)}/bf16: KL mean {kl.mean():.2e} max {kl.max():.2e}, top-1 "
        f"agreement {np.mean(pt_logits.argmax(-1) == tokens):.0%}, min top-5 overlap "
        f"{min(top5_overlap)}: {tokenizer.decode(tokens[:60])!r}..."
    )
    assert kl.mean() < LONG_DECODE_MAX_KL["q8"]
    assert min(top5_overlap) >= LONG_DECODE_MIN_TOP5


# Sampling settings as transformers' generate(do_sample=True) applies them: LFM3-VL-2B's
# generation_config.json (LFM3-1.7B's at 40ebe0a has the same), and a common chat setting.
# The text CLIs decode greedily; lfm3-audio-infer's sampled default is its speech (test_depth.py).
SAMPLINGS = {
    "t0.1-k50-r1.1": {"temperature": 0.1, "top_k": 50, "repetition_penalty": 1.1},
    "t0.7-p0.9": {"temperature": 0.7, "top_p": 0.9},
}
SAMPLED_ANSWER = 64
# Mean total variation between the reference's and the bundle's processed next-token
# probabilities over the answer positions: 4 times bf16 PyTorch's
# (test_sampled_distribution_bf16_floor), as test_long_decode_parity's int4 bound, and for the
# exact fp32 graph 1% of it.
SAMPLED_TV_BF16 = {"t0.1-k50-r1.1": 1.9e-3, "t0.7-p0.9": 4.0e-3}
SAMPLED_TV_FACTOR = {"fp32": 0.01, **dict.fromkeys(("fp16", "q8", "q8f16", *INT4), 4)}


@pytest.fixture(scope="module")
def chat_answer(pytorch_model):
    """(CHAT's prompt ids, fp32 PyTorch's greedy answer of SAMPLED_ANSWER tokens, its logits)."""
    _, model, tokenizer = pytorch_model
    input_ids = chat_ids(tokenizer, CHAT)
    tokens, _ = greedy_pytorch(model, input_ids, SAMPLED_ANSWER)
    return input_ids, tokens, teacher_forced(model, input_ids, tokens)


def sampled_tv(input_ids, answer, expected, logits, sampling: str) -> np.ndarray:
    histories = [input_ids + answer[:i] for i in range(len(answer))]
    settings = SAMPLINGS[sampling]
    p = processed_probs(expected, **settings, histories=histories)
    return total_variation(p, processed_probs(logits, **settings, histories=histories))


@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize("sampling", list(SAMPLINGS))
def test_sampled_distribution(
    exports_dir: pathlib.Path, ep: str, pytorch_model, chat_answer, precision, sampling
):
    """The bundle's next-token distribution after the sampling processors against the
    reference's, at the positions of its greedy answer to CHAT (teacher-forced, one prefill)."""
    model_id, _, _ = pytorch_model
    onnx_file = get_onnx_file(get_onnx_dir(exports_dir, model_id), precision)
    if not onnx_file.exists():
        pytest.fail(f"ONNX file not found: {onnx_file}\nExport with: uv run lfm3-export {model_id}")
    input_ids, answer, expected = chat_answer
    decoder = CachedDecoder(load_session(onnx_file, ep=ep, tf32=False))
    keep = np.array([len(answer)], dtype=np.int64)
    logits = decoder.step(input_ids + answer[:-1], logits_to_keep=keep)[0].astype(np.float32)
    tv = sampled_tv(input_ids, answer, expected, logits, sampling)
    key = precision or "fp32"
    logger.info(
        f"  {get_model_name(model_id)}/{key} {sampling}: TV mean {tv.mean():.2e} max "
        f"{tv.max():.2e} (at answer token {tv.argmax()})"
    )
    assert tv.mean() < SAMPLED_TV_FACTOR[key] * SAMPLED_TV_BF16[sampling]


@pytest.mark.long
@pytest.mark.parametrize("sampling", list(SAMPLINGS))
def test_sampled_distribution_bf16_floor(pytorch_model, chat_answer, sampling):
    """test_sampled_distribution's measure for bf16 PyTorch: the floor of its bounds
    (SAMPLED_TV_BF16), which meets the int8 bundles' bound."""
    model_id, model, _ = pytorch_model
    input_ids, answer, expected = chat_answer
    bf16 = copy.deepcopy(model).to(torch.bfloat16)
    logits = teacher_forced(bf16, input_ids, answer)
    del bf16
    tv = sampled_tv(input_ids, answer, expected, logits, sampling)
    logger.info(
        f"  {get_model_name(model_id)}/bf16 {sampling}: TV mean {tv.mean():.2e} max {tv.max():.2e}"
    )
    assert tv.mean() < SAMPLED_TV_FACTOR["q8"] * SAMPLED_TV_BF16[sampling]
