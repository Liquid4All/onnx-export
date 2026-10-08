"""
llama.cpp KL-divergence base files (`llama-perplexity --kl-divergence-base`) and the statistics
`llama-perplexity --kl-divergence` prints.

Layout (tools/perplexity/perplexity.cpp): b"_logits_", int32 n_ctx, n_vocab and n_chunk, int32
tokens[n_chunk * n_ctx], then per chunk n_ctx - 1 - n_ctx / 2 rows of 2 * ((n_vocab + 1) / 2) + 4
uint16. A row holds two float32 (scale, min_log_prob) and the log-softmax of the logits quantized
to uint16 over [max_logit - 16, max_logit].

Vendored from microsoft/olive-recipes#638 (LiquidAI-LFM2.5-1.2B-Instruct/eval/kld_base.py at
a5d4143f). The arithmetic is unchanged, so scores match its eval_onnx.py to the last bit. The
original's license:

    MIT License

    Copyright (c) 2025 Microsoft

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""

import io
import pathlib
import struct

import numpy as np

MAGIC = b"_logits_"


def row_width(n_vocab: int) -> int:
    return 2 * ((n_vocab + 1) // 2) + 4


def scored_rows(n_ctx: int) -> int:
    """Positions scored per chunk: the second half, whose next token is in the chunk."""
    return n_ctx - 1 - n_ctx // 2


def read_header(path: pathlib.Path) -> tuple[int, int, np.ndarray, int]:
    """n_ctx, n_vocab, the tokens [n_chunk, n_ctx] and the offset of the first row."""
    with path.open("rb") as f:
        header = f.read(20)
        if len(header) < 20 or header[:8] != MAGIC:
            raise ValueError(f"{path} is not a llama.cpp KL-divergence base file")
        n_ctx, n_vocab, n_chunk = struct.unpack("<iii", header[8:])
        if min(n_ctx, n_vocab, n_chunk) < 1:
            raise ValueError(f"{path} has n_ctx {n_ctx}, n_vocab {n_vocab}, n_chunk {n_chunk}")
        tokens = np.frombuffer(f.read(4 * n_ctx * n_chunk), dtype=np.int32)
        offset = f.tell()
    if tokens.size != n_ctx * n_chunk:
        raise ValueError(f"{path} ends inside its token stream")
    return n_ctx, n_vocab, tokens.reshape(n_chunk, n_ctx).copy(), offset


def open_rows(path: pathlib.Path) -> tuple[int, int, np.ndarray, np.memmap]:
    """n_ctx, n_vocab, the tokens and the rows [n_chunk, scored rows, row width], memory-mapped."""
    n_ctx, n_vocab, tokens, offset = read_header(path)
    shape = (len(tokens), scored_rows(n_ctx), row_width(n_vocab))
    size = offset + 2 * int(np.prod(shape))
    if path.stat().st_size != size:
        raise ValueError(f"{path} has {path.stat().st_size} bytes; its header implies {size}")
    rows = np.memmap(path, dtype=np.uint16, mode="r", offset=offset, shape=shape)
    return n_ctx, n_vocab, tokens, rows


def decode_rows(block: np.ndarray, n_vocab: int) -> np.ndarray:
    """uint16 rows -> float32 log-probabilities, exactly as kl_divergence() reads them."""
    params = np.ascontiguousarray(block[:, :4]).view(np.float32)  # [rows, 2]
    scale, min_log_prob = params[:, 0:1], params[:, 1:2]
    return scale * block[:, 4 : 4 + n_vocab].astype(np.float32) + min_log_prob


def encode_rows(logits: np.ndarray, n_vocab: int) -> np.ndarray:
    """float32 logits [rows, n_vocab] -> uint16 rows, mirroring log_softmax() in perplexity.cpp."""
    logits = logits.astype(np.float32, copy=False)
    max_logit = logits.max(axis=1, keepdims=True)
    min_logit = np.maximum(logits.min(axis=1, keepdims=True), max_logit - 16)
    sum_exp = np.exp((logits - max_logit).astype(np.float64)).sum(axis=1, keepdims=True)
    log_sum_exp = np.log(sum_exp).astype(np.float32)
    min_log_prob = (min_logit - max_logit - log_sum_exp).astype(np.float32)
    scale = ((max_logit - min_logit) / np.float32(65535)).astype(np.float32)
    out = np.zeros((logits.shape[0], row_width(n_vocab)), dtype=np.uint16)
    out[:, :4] = np.concatenate([scale, min_log_prob], axis=1).view(np.uint16)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.rint((logits - min_logit) / scale)  # ties to even, like nearest_int()
    q = np.where((logits > min_logit) & (scale > 0), q, 0)
    out[:, 4 : 4 + n_vocab] = np.clip(q, 0, 65535).astype(np.uint16)
    return out


def write_header(f: io.BufferedWriter, n_ctx: int, n_vocab: int, tokens: np.ndarray) -> None:
    f.write(MAGIC)
    f.write(struct.pack("<iii", n_ctx, n_vocab, tokens.shape[0]))
    f.write(np.ascontiguousarray(tokens, dtype=np.int32).tobytes())


def log_softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits.astype(np.float32, copy=False)
    max_logit = logits.max(axis=1, keepdims=True)
    log_sum_exp = np.log(
        np.exp((logits - max_logit).astype(np.float64)).sum(axis=1, keepdims=True)
    ).astype(np.float32)
    return logits - (max_logit + log_sum_exp)


class KldStats:
    """Accumulates the statistics llama-perplexity --kl-divergence prints, one chunk per add()."""

    def __init__(self):
        self.nll, self.nll_base, self.kld, self.same_top, self.p_diff = [], [], [], [], []

    def add(self, logits: np.ndarray, base_log_probs: np.ndarray, next_tokens: np.ndarray):
        log_q = log_softmax(logits)
        rows = np.arange(len(next_tokens))
        nll = -log_q[rows, next_tokens]
        nll_base = -base_log_probs[rows, next_tokens]
        mask = base_log_probs > -16.0
        p_base = np.where(mask, np.exp(base_log_probs), 0.0)
        kld = (p_base * np.where(mask, base_log_probs - log_q, 0.0)).sum(axis=1)
        self.nll.append(nll)
        self.nll_base.append(nll_base)
        self.kld.append(kld)
        self.same_top.append(log_q.argmax(axis=1) == base_log_probs.argmax(axis=1))
        self.p_diff.append(np.exp(-nll) - np.exp(-nll_base))

    def arrays(self) -> dict[str, np.ndarray]:
        """Per-token values [n_chunk, scored rows], for paired comparisons (#638's compare.py)."""
        return {
            "kld": np.stack(self.kld),
            "same_top": np.stack(self.same_top),
            "nll": np.stack(self.nll),
        }

    def summary(self) -> dict:
        nll, nll_base = np.concatenate(self.nll), np.concatenate(self.nll_base)
        kld, same = np.concatenate(self.kld), np.concatenate(self.same_top)
        p_diff = np.concatenate(self.p_diff)
        n = len(kld)
        chunk_kld = np.array([c.mean() for c in self.kld])
        return {
            "tokens": int(n),
            "ppl": float(np.exp(nll.mean())),
            # from the 16-bit reference, which clips log-probabilities below max - 16 nats
            "ppl_base": float(np.exp(nll_base.mean())),
            "ln_ppl_ratio": float((nll - nll_base).mean()),
            "kld_mean": float(kld.mean()),
            # llama-perplexity's error, which treats the tokens as independent
            "kld_mean_err": float(kld.std(ddof=1) / np.sqrt(n)),
            # tokens of one chunk are correlated: the spread of the chunk means is the honest error
            "kld_mean_err_chunks": float(chunk_kld.std(ddof=1) / np.sqrt(len(chunk_kld))),
            "kld_median": float(np.median(kld)),
            "kld_p99": float(np.percentile(kld, 99)),
            "same_top_pct": float(100 * same.mean()),
            "rms_dp_pct": float(100 * np.sqrt((p_diff**2).mean())),
        }
