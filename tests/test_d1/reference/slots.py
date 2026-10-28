"""Answer positions of a right-padded batch, read through whatever wraps the model."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .attention import state_attention, text_backbone


@dataclass(frozen=True)
class SlotLogits:
    """A batch's answer slots: the code slots' logits, in slot order, and the read slots' over their rows' state
    positions, in the pointer plan's order (pointer.py `query_plan`)."""

    codes: torch.Tensor
    positions: torch.Tensor | None = None


def split(slots: torch.Tensor, states: torch.Tensor | None, plan: dict | None, project) -> SlotLogits:
    """The hidden states at the slots [S, D]: the code slots through `project` (the LM head, or its option rows),
    the read slots against their rows' state positions `states` [R, T, D]."""
    if plan is None:
        return SlotLogits(project(slots))
    from .pointer import read_logits

    return SlotLogits(project(slots[plan["code_slots"].to(slots.device)]), read_logits(slots, states, plan))


def output_head(model):
    """The LM head, through whatever wrappers are in the way."""
    head = getattr(model, "module", model).get_output_embeddings()
    if head is None:
        raise RuntimeError("no output embedding on this model")
    return head


def slot_logits(
    model,
    ids: torch.Tensor,
    mask: torch.Tensor,
    last: torch.Tensor,
    state_len: torch.Tensor | None = None,
    question: torch.Tensor | None = None,
    row: torch.Tensor | None = None,
    pointer: dict | None = None,
    **vision,
):
    """Logits at one position per row (or at each `row`, `last` pair: a question may have several slots in a row),
    the backbone read directly and only the slots projected, never `B x T x vocab`. `vision`: the processor's picture
    inputs, for a vision-language model."""
    with state_attention(model, state_len, question):  # None: the model's own causal mask
        out = text_backbone(model)(input_ids=ids, attention_mask=mask, use_cache=False, **vision)
    hidden = getattr(out, "last_hidden_state", None)
    if hidden is None:
        hidden = out[0]
    row = torch.arange(hidden.size(0)) if row is None else row
    slots = hidden[row.to(hidden.device), last.to(hidden.device)]  # a model over GPUs ends elsewhere
    states = hidden[pointer["rows"].to(hidden.device), : pointer["width"]] if pointer is not None else None
    return split(slots, states, pointer, output_head(model))
