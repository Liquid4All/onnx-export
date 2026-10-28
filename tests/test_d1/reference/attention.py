# Changed from LiquidAI/d1-2B at 97c57047d5852d77e02dfac211f8768aff77a8b2: timing and hardware notes cut.
"""Which tokens see which on a backbone's full-attention layers, in training and serving alike.

| mode | a state token sees | a question token sees |
|---|---|---|
| `causal` | the tokens before it | the tokens before it |
| `state` | the whole state | the whole state, and the question before it |
| `prompt` | the whole prompt: state, question, options | the whole prompt |
| `set` | the state and every question | the state and its own question |

`state` reads the state both ways and never shows it the question, so one state's keys still serve every question
about it. `prompt` reads everything before the answer both ways, the state with its question in view: each question is
then a prompt of its own, with no prefill shared between the questions of one state. `set` keeps `prompt`'s reading
and packs a state's questions after it, `[state][q1]..[qN]`, each ending at its own answer slot: one pass, the state
read once. On these layers a question never attends to another; it hears of them only through the state, which reads
them all. A lone question reads exactly as under `prompt`.

Only full-attention layers change; recurrent, convolution and sliding-window layers stay causal. Under `set` they
run over the packed sequence, so a question also hears the ones before it through them; training packs the same way
(questions of a state in random order), so the served function is the trained one. A
checkpoint trains and serves with one mode, recorded in its config (`d1_attention`, with `d1_set_size`, the most
questions `set` packs after one state).
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from transformers import AttentionInterface, AttentionMaskInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import sdpa_mask

Tensor = torch.Tensor
SET_SIZE = 4  # the most questions `set` packs after one state when a checkpoint records none (`d1_set_size`)
TRAINED_SET_SIZE = 128  # the largest cap training takes (`--set-size`), so the most a checkpoint records
CHUNK = 4096  # queries a `Sees` mask is built for at once


def _fused(q: torch.Tensor) -> bool:
    """Whether the flash kernels take these queries: a GPU in half precision, ROCm or CUDA from Ampere on."""
    if not q.is_cuda or q.dtype not in (torch.float16, torch.bfloat16):
        return False
    return bool(torch.version.hip) or torch.cuda.get_device_capability(q.device)[0] >= 8


def _windowed(q: Tensor, k: Tensor, v: Tensor, window: int, scale: float) -> Tensor:
    """Causal attention within `window`, as local blocks: each block of `window`
    queries sees its own block and the one before it."""
    batch, heads, length, dim = q.shape
    blocks = -(-length // window)

    def split(t: Tensor) -> Tensor:  # (B, H, T, D) -> (B, blocks, H, window, D)
        t = F.pad(t, (0, 0, 0, blocks * window - length))
        return t.reshape(batch, t.shape[1], blocks, window, dim).transpose(1, 2)

    def with_previous(t: Tensor) -> Tensor:
        return torch.cat((F.pad(t, (0, 0, 0, 0, 0, 0, 1, 0))[:, :-1], t), dim=3)

    qb, kb, vb = split(q), with_previous(split(k)), with_previous(split(v))
    qpos = torch.arange(window, device=q.device)[:, None] + window
    kpos = torch.arange(2 * window, device=q.device)[None, :]
    block = torch.arange(blocks, device=q.device)[:, None, None]
    allowed = (kpos <= qpos) & (kpos > qpos - window) & ((block > 0) | (kpos >= window))  # block 0 has none before it
    mask = allowed[:, None].repeat(batch, 1, 1, 1)
    out = F.scaled_dot_product_attention(qb.flatten(0, 1), kb.flatten(0, 1), vb.flatten(0, 1),
                                         attn_mask=mask, scale=scale, enable_gqa=True)
    out = out.reshape(batch, blocks, heads, window, dim).transpose(1, 2)
    return out.reshape(batch, heads, blocks * window, dim)[:, :, :length]


def causal_attention(q: Tensor, k: Tensor, v: Tensor, window: int | None, scale: float) -> Tensor:
    """Right-padded chains: causal, within `window` if set: on the flash kernels their own window at any length (no
    branch on it, so torch.compile keeps one graph across the window), else blocks."""
    if window is None or not _fused(q) and q.shape[2] <= window:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale, enable_gqa=True)
    if not _fused(q):
        return _windowed(q, k, v, window, scale)
    out = torch.ops.aten._flash_attention_forward(*(t.transpose(1, 2) for t in (q, k, v)), None, None, q.shape[2],
                                                  k.shape[2], 0.0, True, False, scale=scale,
                                                  window_size_left=window - 1, window_size_right=0)[0]
    return out.transpose(1, 2)


def both_ways(mode: str, tokenizer, ids: list[int], suffix: str) -> int | None:
    """How many of a prompt's first tokens read each other both ways: none (`causal`), its state, or all of it
    (`prompt`, and `set` for a lone question)."""
    if mode == "causal":
        return None
    return len(ids) if mode in ("prompt", "set") else state_length(tokenizer, ids, suffix)


def state_length(tokenizer, ids: list[int], suffix: str) -> int:
    """A prompt's tokens before its question: the prompt is the state's text then `suffix`, which opens on a token
    boundary (the state ends on a blank line), so its own tokens are the prompt's last ones."""
    return len(ids) - len(tokenizer.encode(suffix, add_special_tokens=False))


def pack_set(tokenizer, prefix: str, suffixes: list[str],
             header: str = "", slot_offsets: list[list[int]] | None = None,
             first_ids: list[int] | None = None,
             ) -> tuple[list[int], int, list[int], list[list[int]]]:
    """`set`'s sequence for one state, in any prompt layout: `prefix` (the state, ending with the question
    `header`) and the first question tokenized as its lone prompt is, so one question is exactly its `prompt`
    row; each further question its own group, `header` then its suffix. Returns the ids, the state's length, each
    token's question (1..N, -1 on the state) and each question's answer positions. `slot_offsets` names slots
    relative to its suffix's last token (0); without it, each question has that one slot."""
    if not suffixes:
        raise ValueError("a question set cannot be empty")
    offsets = [[0] for _ in suffixes] if slot_offsets is None else slot_offsets
    if len(offsets) != len(suffixes):
        raise ValueError("one slot layout is required per question")

    def positions(k: int, length: int, end: int) -> list[int]:
        own = offsets[k]
        if (not own or own != sorted(set(own)) or own[-1] != 0
                or any(isinstance(o, bool) or not isinstance(o, int) or not -length < o <= 0 for o in own)):
            raise ValueError("slots must be increasing positions inside their question, ending at its last token")
        return [end + o for o in own]

    ids = tokenizer.encode(prefix + suffixes[0], add_special_tokens=False) if first_ids is None else list(first_ids)
    state = state_length(tokenizer, ids, suffixes[0])
    question = [-1] * state + [1] * (len(ids) - state)
    ends = [positions(0, len(ids) - state, len(ids) - 1)]
    for k, suffix in enumerate(suffixes[1:], 2):
        own = tokenizer.encode(header + suffix, add_special_tokens=False)
        ids, question = ids + own, question + [k] * len(own)
        ends.append(positions(k - 1, len(own), len(ids) - 1))
    return ids, state, question, ends


def text_backbone(model):
    """The module that returns hidden states, under any wrapper (DDP, a training wrapper)."""
    m = getattr(model, "module", model)
    if hasattr(m, "get_base_model"):
        m = m.get_base_model()
    return getattr(m, "model", m)


def language_model(model):
    """The language model under any wrapper and a vision-language model."""
    m = text_backbone(model)
    return getattr(m, "language_model", m)


def _runs(*parts: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Segments, segment i the runs `(start[i], length[i])` of every part back to back: their flat indices, their
    cumulative lengths (int32, as the varlen kernels take them) and the longest."""
    lengths = sum(n for _, n in parts)
    cu = F.pad(lengths.cumsum(0), (1, 0))
    seg = torch.repeat_interleave(torch.arange(len(lengths), device=lengths.device), lengths)
    off = torch.arange(len(seg), device=seg.device) - cu[seg]
    idx = torch.zeros_like(off)
    for start, n in parts:
        idx = torch.where((off >= 0) & (off < n[seg]), start[seg] + off, idx)
        off = off - n[seg]
    return idx, cu.int(), int(lengths.max()) if len(lengths) else 0


def _varlen(q: Tensor, k: Tensor, v: Tensor, cu_q: Tensor, cu_k: Tensor, longest: Tensor, causal: bool,
            scale: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Flash attention over packed segments: the output [N, H, D], then what its backward reads (the log-sum-exp
    [H, N], the RNG state and its companion). Causal aligns each segment's last query to its last key. `longest`: the
    longest query and key segment, on the CPU. Compiled, the opaque op `d1::varlen`, so the lengths neither break the
    graph nor recompile it; its backward is flash's own (`_varlen_backward`), as eager's is."""
    max_q, max_k = longest.tolist()
    out, lse, rng, unused, _ = torch.ops.aten._flash_attention_forward(q, k, v, cu_q, cu_k, max_q, max_k, 0.0, causal,
                                                                       False, scale=scale)
    return out, lse, rng, unused


def _varlen_backward(ctx, grad, *_):
    q, k, v, out, lse, rng, unused, cu_q, cu_k = ctx.saved_tensors
    max_q, max_k = ctx.longest
    dq, dk, dv = torch.ops.aten._flash_attention_backward(grad.contiguous(), q, k, v, out, lse, cu_q, cu_k, max_q,
                                                          max_k, 0.0, ctx.causal, rng, unused, scale=ctx.scale)
    return dq, dk, dv, None, None, None, None, None


def _varlen_context(ctx, inputs, output):
    q, k, v, cu_q, cu_k, longest, causal, scale = inputs
    ctx.save_for_backward(q, k, v, *output, cu_q, cu_k)
    ctx.longest, ctx.causal, ctx.scale = longest.tolist(), causal, scale


if not hasattr(torch.ops.d1, "varlen"):  # once a process: two repos' copies of this module share the op
    _op = torch.library.custom_op("d1::varlen", _varlen, mutates_args=())
    _op.register_fake(lambda q, k, v, cu_q, cu_k, longest, causal, scale: (
        torch.empty_like(q), q.new_empty(q.shape[1], q.shape[0], dtype=torch.float32),
        q.new_empty((), dtype=torch.int64), q.new_empty((), dtype=torch.int64)))
    _op.register_autograd(_varlen_backward, setup_context=_varlen_context)


@dataclass
class Sees:
    """The full-attention mask of `state_attention`, never built whole: `attend` takes the attention under it, `rows`
    a block of queries' part of it."""

    state: torch.Tensor  # [B] each row's state length
    question: torch.Tensor | None  # [B, T] `set`: each token's question, 1..N, and -1 elsewhere
    real: torch.Tensor  # [B, T] the tokens that are not padding
    _calls: dict = field(default_factory=dict, init=False, repr=False, compare=False)  # once a device, every layer's

    def __post_init__(self):
        """On a GPU, the plan on the mask's device, made before any layer: a compiled forward reads it with no graph
        break. Off a GPU, `attend` masks and never plans."""
        if self.real.is_cuda:
            self._calls[self.real.device] = self._plan(self.real.device)

    def rows(self, start: int, width: int, device) -> torch.Tensor:
        """Which keys queries `start` to `start + width` see, [B, 1, width, T]."""
        q = torch.arange(start, start + width, device=device)[None, :, None]
        kv = torch.arange(self.real.shape[1], device=device)[None, None, :]
        s = self.state.to(device)[:, None, None]
        if self.question is None:  # causal, and the state both ways
            m = (kv <= q) | ((q < s) & (kv < s))
        else:  # the state sees every token and every token the state; a question token its own question
            g = self.question.to(device)
            m = (q < s) | (kv < s) | (g[:, start:start + width, None] == g[:, None, :])
        return (m & self.real.to(device)[:, None, :])[:, None]

    def attend(self, q: Tensor, k: Tensor, v: Tensor, scale: float, window: int | None = None) -> Tensor:
        """Attention under this mask; q [B, H, T, D], k and v [B, Hkv, T, D]. On the flash kernels, no mask: the
        states over their keys in one varlen call, the questions over their state and themselves in another. Off
        them (CPU, fp32), `CHUNK` masked queries at a time. A sliding layer's `window`: causal within it, as every
        mode keeps it (right-padded rows: a pad never reaches a real token)."""
        if window is not None:
            return causal_attention(q, k, v, window, scale)
        b, h, t, d = q.shape
        if not _fused(q):
            starts = range(0, t, CHUNK) if t > CHUNK else (0,)  # a range over t would compile a graph per length
            return torch.cat([F.scaled_dot_product_attention(q[:, :, s:s + CHUNK], k, v, scale=scale, enable_gqa=True,
                                                             attn_mask=self.rows(s, min(CHUNK, t - s), q.device))
                              for s in starts], dim=2)
        if q.device not in self._calls:  # a model over GPUs: once on each
            self._calls[q.device] = self._plan(q.device)
        if self._calls[q.device] is None:  # every row reads all of itself, unpadded: plain attention
            return F.scaled_dot_product_attention(q, k, v, scale=scale, enable_gqa=True)
        flat = [x.transpose(1, 2).reshape(b * t, x.shape[1], d) for x in (q, k, v)]
        out = flat[0].new_zeros(b * t, h, d)
        varlen = torch.ops.d1.varlen if torch.compiler.is_compiling() else _varlen
        for qi, cu_q, ki, cu_k, longest, causal in self._calls[q.device]:
            attended = varlen(flat[0][qi], flat[1][ki], flat[2][ki], cu_q, cu_k, longest, causal, scale)[0]
            out = out.index_copy(0, qi, attended)
        return out.view(b, t, h, d).transpose(1, 2)

    def _plan(self, device) -> list | None:
        """The varlen calls, as (queries, their offsets, keys, theirs, the longest of each, causal): each state over
        its row (`set`) or itself (`state`, `prompt`); each question over its state then itself, causally under
        `state`, where the question part is one run a row. Under `set` a row of one question is one segment: all of
        it sees all of it. None when every row is one such segment with no padding."""
        real = self.real.to(device)
        b, t = real.shape
        n = real.sum(1)
        s = torch.minimum(self.state.to(device), n)
        base = torch.arange(b, device=device) * t
        tail = (real & (torch.arange(t, device=device) >= s[:, None])).flatten().nonzero().squeeze(1)
        g = (torch.zeros(b, t, dtype=torch.long, device=device) if self.question is None
             else self.question.to(device)).flatten()[tail]
        new = F.pad((g[1:] != g[:-1]) | (tail[1:] != tail[:-1] + 1), (1, 0), value=True)[:len(tail)]
        first = new.nonzero().squeeze(1)
        start, length = tail[first], torch.diff(F.pad(first, (0, 1), value=len(tail)))
        if self.question is not None:
            lone = torch.bincount(start // t, minlength=b) == 1
            s = torch.where(lone, n, s)
            start, length = start[~lone[start // t]], length[~lone[start // t]]
        if not len(start) and bool((s == t).all()):
            return None
        calls = [(_runs((base, s)), _runs((base, n if self.question is not None else s)), False)]
        if len(start):
            row = start // t
            calls.append((_runs((start, length)), _runs((base[row], s[row]), (start, length)), self.question is None))
        return [(qi, cu_q, ki, cu_k, torch.tensor([max_q, max_k]), causal)
                for (qi, cu_q, max_q), (ki, cu_k, max_k), causal in calls]


def d1_attention(module, q, k, v, attention_mask, scaling=None, sliding_window=None, **kwargs):
    """transformers' attention function for `attn_implementation="d1"` (registered on import): a `Sees` mask attends
    through `Sees.attend` (a sliding layer within its window), anything else is SDPA's."""
    if isinstance(attention_mask, Sees):
        return attention_mask.attend(q, k, v, scaling, sliding_window).transpose(1, 2), None
    return sdpa_attention_forward(module, q, k, v, attention_mask, scaling=scaling, sliding_window=sliding_window,
                                  **kwargs)


AttentionInterface.register("d1", d1_attention)
# the masks transformers builds itself under "d1" (no `state_attention`; a tower, an audio encoder) are SDPA's: an
# implementation without one gets None for every mask, which drops a sliding window and the padding
AttentionMaskInterface.register("d1", sdpa_mask)
ATTENTION_KINDS = ("full_attention", "sliding_attention")  # the layer kinds a `Sees` masks; the others take padding


def mask_mapping(layer_types, sees: Sees, pad: Tensor) -> dict:
    """The per-kind masks of a language model under `Sees`, keyed as its `layer_types`: the `Sees` on attention
    layers, the 2D padding (None without any) on convolution and linear-attention layers."""
    padding = None if pad.all() else pad
    return {kind: sees if kind in ATTENTION_KINDS else padding for kind in set(layer_types)}


@contextmanager
def state_attention(model, state_len: torch.Tensor | None, question: torch.Tensor | None = None):
    """Within the block, the model's language model reads each row's first `state_len` tokens both ways; with
    `question` (`set`: each token's question, 1..N, and -1 elsewhere), as `Sees` says. The mask is swapped in at the
    language model, after the multimodal wrapper has placed its media: as the mask of a language model that reads it
    (`reads_sees`), else as transformers' per-kind masks (`mask_mapping`), which `d1_attention` reads."""
    if state_len is None:
        yield
        return

    def hook(module, args, kwargs):
        pad, x = kwargs.get("attention_mask"), kwargs.get("input_ids")
        if isinstance(pad, dict):
            raise ValueError("state attention replaces a 2D padding mask, not a prepared mapping")
        if pad is None:
            x = kwargs["inputs_embeds"] if x is None else x
            pad = torch.ones(x.shape[:2], dtype=torch.long, device=x.device)
        sees = Sees(state_len.to(pad.device), question, pad.bool())
        if getattr(module, "reads_sees", False):
            kwargs["attention_mask"] = sees
        elif module.config._attn_implementation == "d1":
            kinds = getattr(module.config, "layer_types", None) or ("full_attention",)  # a config naming none: all full
            kwargs["attention_mask"] = mask_mapping(kinds, sees, pad)
        else:
            raise ValueError(f"{type(module).__name__} reads `state_attention` with attn_implementation='d1'")
        return args, kwargs

    handle = language_model(model).register_forward_pre_hook(hook, with_kwargs=True)
    try:
        yield
    finally:
        handle.remove()
