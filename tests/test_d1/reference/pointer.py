"""Parameter-free readouts over a row's state positions, with BOS0 for nothing."""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .json_spans import json_text, string_paths  # torch-free, so code without torch reads spans too

SPAN_CAPS = {"text": 512, "number": 32, "date": 32}  # a read's longest span, in tokens


@dataclass(frozen=True)
class StatePositions:
    """Token positions mapped to the caller's rendered state; None means not a text position."""

    text: str
    offsets: tuple[tuple[int, int] | None, ...]
    paths: tuple[tuple[int, int, str], ...] = ()

    @property
    def valid(self) -> list[bool]:
        return [i == 0 or span is not None for i, span in enumerate(self.offsets)]

    @property
    def pointer_mask(self) -> list[bool]:
        """`valid` up to its last valid position: the keys a read's pointer scores."""
        valid = self.valid
        return valid[:max(i for i, v in enumerate(valid) if v) + 1]

    def read(self, first: int, last: int, max_tokens: int = SPAN_CAPS["text"]) -> dict:
        if first == last == 0:
            return {"text": None, "span": None}
        if not 0 < first <= last < len(self.offsets) or last - first >= max_tokens:
            raise ValueError("invalid pointer endpoints")
        a, b = self.offsets[first], self.offsets[last]
        if a is None or b is None:
            raise ValueError("pointer leaves the state text")
        span = [a[0], b[1]]
        # A token can carry what is around a value: the separating space, or a JSON string's quote (` €",`). Trim
        # boundary space and clip to the one string the span starts or ends in; never alter a character inside.
        while span[0] < span[1] and self.text[span[0]].isspace():
            span[0] += 1
        while span[1] > span[0] and self.text[span[1] - 1].isspace():
            span[1] -= 1
        held = [(start, end, p) for start, end, p in self.paths if start < span[1] and span[0] < end]
        if len(held) == 1:
            start, end, path = held[0]
            span = [max(span[0], start), min(span[1], end)]
            # inside one JSON string: the caller's characters, not their escapes
            return {"text": json_text(self.text[slice(*span)]), "span": span, "path": path}
        return {"text": self.text[slice(*span)], "span": span}


def state_positions(
    tokenizer, state, prefix: str, full_text: str, style: str = "json_only", *, encoded=None
) -> StatePositions:
    """Align one unchanged rendered prompt to text tokens; template, picture and question tokens stay invalid."""
    from .prompt import plain_offsets, question_header, state_block, state_text

    if not full_text.startswith(prefix):
        raise ValueError("the tokenized prompt must start with the state prefix")
    if state is None:
        body, escaped, boundaries = "", "", [0]
        start = len(prefix)
    else:
        body = state_block(state, style)
        escaped, boundaries = plain_offsets(tokenizer, body)
        tail = escaped + "\n" + question_header(tokenizer, state)
        if not prefix.endswith(tail):
            raise ValueError("state rendering differs from the prompt prefix")
        start = len(prefix) - len(tail)
    encoded = (
        tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True) if encoded is None else encoded
    )
    text = state_text(state, style) if state is not None else ""  # the blank line after it is the template's
    stop = start + bisect_right(boundaries, len(text)) - 1
    offsets = []
    for a, b in encoded["offset_mapping"]:
        span = None
        lo, hi = max(a, start), min(b, stop)
        if lo < hi and not full_text[a:lo].strip() and not full_text[hi:b].strip():
            left, right = boundaries[lo - start], boundaries[hi - start]
            if left < right <= len(text) and text[left:right].strip():
                span = (left, right)
        offsets.append(span)
    if not offsets or offsets[0] is not None:
        raise ValueError("position 0 must be a template token for the nothing readout")
    paths = (
        string_paths(text)
        if state is not None and not isinstance(state, str) and style.endswith("_only") and style.startswith("json")
        else []
    )
    return StatePositions(text, tuple(offsets), tuple(paths))


def position_logits(query: torch.Tensor, state: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """[B, slots, D] against [B, T, D], sharing state keys between questions; valid [B, T], boolean, includes BOS0."""
    dtype = torch.float64 if query.dtype == state.dtype == torch.float64 else torch.float32
    logits = torch.bmm(query.to(dtype), state.to(dtype).transpose(1, 2)) / math.sqrt(query.shape[-1])
    return logits.masked_fill(~valid[:, None], -torch.inf)


def read_logits(hidden: torch.Tensor, states: torch.Tensor, plan: dict) -> torch.Tensor:
    """The flat read slots of a mixed batch; queries share each retained state's keys."""
    device = hidden.device
    query = hidden[plan["queries"].to(device)]
    logits = position_logits(query, states, plan["valid"].to(device))
    return logits[plan["query_row"].to(device), plan["query_col"].to(device)]


def query_plan(rows: list[int], masks: list[torch.Tensor], slots: list[int], count: int) -> dict:
    """The read slots of a batch of `count` slots, grouped by their row's state: each state's keys shared by its
    queries; `code_slots`, the other slots, which the LM head reads."""
    unique = sorted(set(rows))
    index = {r: i for i, r in enumerate(unique)}
    width = max(map(len, masks))
    valid = torch.zeros(len(unique), width, dtype=torch.bool)
    groups = [[] for _ in unique]
    row_index, column_index = [], []
    for slot, row, keep in zip(slots, rows, masks):
        r = index[row]
        if groups[r] and not torch.equal(valid[r, : len(keep)], keep):
            raise ValueError("read questions must share their state-position mask")
        valid[r, : len(keep)] = keep
        row_index.append(r)
        column_index.append(len(groups[r]))
        groups[r].append(slot)
    queries = torch.zeros(len(unique), max(map(len, groups)), dtype=torch.long)
    for r, group in enumerate(groups):
        queries[r, : len(group)] = torch.tensor(group)
    return dict(
        rows=torch.tensor(unique),
        width=width,
        queries=queries,
        valid=valid,
        query_row=torch.tensor(row_index),
        query_col=torch.tensor(column_index),
        slots=torch.tensor(slots),
        code_slots=torch.tensor(sorted(set(range(count)) - set(slots)), dtype=torch.long),
    )


def spans(logits: torch.Tensor, max_tokens: int = SPAN_CAPS["text"]) -> tuple[torch.Tensor, torch.Tensor]:
    """Each read's span from its start and end scores [R, 2, T]: found when P(found) >= 0.5, then the likeliest span
    within `max_tokens`; (0, 0) for nothing."""
    logp = logits.log_softmax(-1)
    start, end = logp.unbind(1)
    found = -torch.expm1(start[:, 0])
    if logits.shape[2] == 1:
        return torch.zeros(logits.shape[0], 2, dtype=torch.long, device=logits.device), found
    cap = min(max_tokens, logits.shape[2] - 1)
    windows, indices = F.max_pool1d(
        F.pad(end[:, None, 1:], (0, cap - 1), value=-torch.inf), cap, stride=1, return_indices=True
    )
    scores = start[:, 1:] + windows[:, 0]
    index = scores.argmax(-1)
    first = index + 1
    last = indices[:, 0].gather(1, index[:, None])[:, 0] + 1
    at = torch.stack((first, last), -1)
    nothing = found < 0.5
    return torch.where(nothing[:, None], torch.zeros_like(at), at), found
