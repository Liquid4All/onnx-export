"""
LFM3 text bundles on long inputs against the float32 PyTorch reference (opt-in: --run-long):

- test_long_chat: a passphrase, 8K, 32K or 128K tokens of wikitext-2 and a question about both in
  one user turn. Each bundle decodes LONG_STEPS greedy steps with its caches after one prefill and
  must say the passphrase; the steps are scored as test_decode.py's test_long_decode_parity scores
  its 600 (mean KL against one teacher-forced forward over the bundle's tokens, fp32 with
  PyTorch's tokens), with its bounds. int4's KL is logged only: tests/test_lfm3 bounds it in
  test_long_decode_parity alone. 128K runs the fp16-activation bundles on CUDA (the others run
  attention on the CPU, README 3.1b), with the reference on flex_attention.
- test_wikitext_kld: lfm2-compare wikitext's protocol (README 5.1) on the decoder: wikitext-2 in 64
  chunks of 512 tokens, the second half of each scored by KLD against the reference's log-
  probabilities (llama.cpp's 16-bit rows). fp32 must be exact; a precision with a ceiling in
  compare.wikitext.BASELINES for the model and EP is gated at it (KLD + 2 SE), the others are
  logged.

Both read wiki.test.raw (--wikitext; README 5.1 downloads it).

Run with:
    uv run --no-sync pytest tests/test_lfm3/test_long_context.py -v --device cuda --run-long \
        --wikitext wikitext-2-raw/wiki.test.raw
    uv run --no-sync pytest tests/test_lfm3/test_long_context.py -v --device cuda --ep cpu \
        --run-long --wikitext wikitext-2-raw/wiki.test.raw
"""

import logging
import pathlib

import numpy as np
import pytest
import torch
from helpers import get_model_name, get_onnx_dir

from liquidonnx.compare import kld_base
from liquidonnx.compare.wikitext import CHUNKS, ceiling, text_tokens
from liquidonnx.lfm3.session import CachedDecoder, get_onnx_file, kv_feeds_attention, load_session

from .test_decode import INT4, LONG_DECODE_MAX_KL, chat_ids, decode_scores, greedy_onnx

logger = logging.getLogger(__name__)

pytestmark = pytest.mark.long

NEEDLE = "Remember this: the secret passphrase is 'amber harbor 47'.\n\n"
QUESTION = (
    "\n\nWhat is the secret passphrase given at the very beginning of the text? "
    "Then name the topic of the last article in the text."
)
PASSPHRASE = "amber harbor 47"
LONG_STEPS = 64
ALL = (None, "fp16", "q8", "q8f16", *INT4)
F16 = ("fp16", "q8f16", "q4f16")
CHATS = [
    *(pytest.param(n, p, id=f"8k-{p or 'fp32'}") for n, p in [(8192, p) for p in ALL]),
    *(pytest.param(n, p, id=f"32k-{p or 'fp32'}") for n, p in [(32768, p) for p in ALL]),
    *(pytest.param(131072, p, id=f"128k-{p}") for p in F16),
]


@pytest.fixture(scope="module")
def wiki_ids(pytorch_model, wikitext) -> list[int]:
    _, _, tokenizer = pytorch_model
    return tokenizer.encode(wikitext.read_text(), add_special_tokens=False)


def long_chat(tokenizer, wiki_ids: list[int], length: int) -> list[int]:
    """Chat ids of about length tokens: the needle, wikitext, the question."""
    base = len(chat_ids(tokenizer, [{"role": "user", "content": NEEDLE + QUESTION}], False))
    document = tokenizer.decode(wiki_ids[: length - base])
    return chat_ids(tokenizer, [{"role": "user", "content": NEEDLE + document + QUESTION}], False)


@pytest.mark.parametrize("length,precision", CHATS)
def test_long_chat(exports_dir: pathlib.Path, ep, pytorch_model, wiki_ids, length, precision):
    model_id, model, tokenizer = pytorch_model
    if length > 32768 and (ep == "cpu" or model.device.type == "cpu"):
        pytest.skip("128K runs on CUDA only")
    onnx_file = get_onnx_file(get_onnx_dir(exports_dir, model_id), precision)
    if not onnx_file.exists():
        pytest.fail(f"ONNX file not found: {onnx_file}\nExport with: uv run lfm3-export {model_id}")
    input_ids = long_chat(tokenizer, wiki_ids, length)
    sess = load_session(onnx_file, ep=ep, tf32=False)
    tokens, logits = greedy_onnx(sess, input_ids, LONG_STEPS, kv_feeds_attention(onnx_file))
    del sess

    attention = model.config._attn_implementation
    if length > 32768:
        model.config._attn_implementation = "flex_attention"
    try:
        expected, kl, top5 = decode_scores(model, input_ids, tokens, logits)
    finally:
        model.config._attn_implementation = attention
    answer = tokenizer.decode(tokens)
    name = f"{get_model_name(model_id)}/{precision or 'fp32'}"
    logger.info(
        f"  {name}: {len(input_ids)} prompt tokens, KL mean {kl.mean():.2e} max {kl.max():.2e}, "
        f"top-1 agreement {np.mean(expected.argmax(-1) == tokens):.0%}, min top-5 overlap "
        f"{min(top5)}: {answer!r}"
    )
    assert PASSPHRASE in answer
    if precision is None:
        assert expected.argmax(-1).tolist() == tokens, f"{name}: decode left PyTorch's tokens"
    if precision not in INT4:
        assert kl.mean() < LONG_DECODE_MAX_KL[precision]


# === wikitext-2 KLD ===

N_CTX = 512
FIRST = N_CTX // 2
FP32_MAX_KLD = 1e-6


@pytest.fixture(scope="module")
def wikitext_reference(pytorch_model, wikitext):
    """(the chunks' ids [CHUNKS, N_CTX], BOS first, and the reference's 16-bit rows per chunk)."""
    _, model, tokenizer = pytorch_model
    tokens = text_tokens(tokenizer, wikitext, N_CTX, CHUNKS)
    tokens[:, 0] = tokenizer.bos_token_id  # llama-perplexity's BOS at the start of every chunk
    vocab = model.config.vocab_size
    rows = []
    with torch.no_grad():
        for chunk in tokens:
            ids = torch.from_numpy(chunk.astype(np.int64))[None].to(model.device)
            logits = model(input_ids=ids, use_cache=False).logits[0, FIRST : N_CTX - 1, :vocab]
            rows.append(kld_base.encode_rows(logits.float().cpu().numpy(), vocab))
    return tokens, rows


@pytest.mark.parametrize("precision", ["fp32", "fp16", "q8", "q8f16", *INT4])
def test_wikitext_kld(exports_dir: pathlib.Path, ep, pytorch_model, wikitext_reference, precision):
    model_id, model, _ = pytorch_model
    onnx_file = get_onnx_file(get_onnx_dir(exports_dir, model_id), precision)
    tokens, rows = wikitext_reference
    vocab = model.config.vocab_size
    sess = load_session(onnx_file, ep=ep, tf32=False)
    keep = np.array([N_CTX - FIRST], dtype=np.int64)
    stats = kld_base.KldStats()
    for chunk, block in zip(tokens, rows, strict=True):
        logits = CachedDecoder(sess).step(chunk.tolist(), logits_to_keep=keep)[0, :-1, :vocab]
        base = kld_base.decode_rows(block, vocab)
        stats.add(logits.astype(np.float32), base, chunk[FIRST + 1 :])
    row = stats.summary()
    limit = ceiling([get_model_name(model_id)], ep, precision, None)
    logger.info(
        f"  {get_model_name(model_id)}/{precision} on {ep}: KLD {row['kld_mean']:.4g} ± "
        f"{row['kld_mean_err_chunks']:.2g}, same top {row['same_top_pct']:.2f}%, PPL "
        f"{row['ppl']:.4f} (reference {row['ppl_base']:.4f}), ceiling {limit}"
    )
    if precision == "fp32":
        assert row["kld_mean"] < FP32_MAX_KLD
    elif limit is not None:
        assert row["kld_mean"] <= limit
