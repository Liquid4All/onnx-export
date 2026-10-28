# Changed from LiquidAI/d1-2B at 97c57047d5852d77e02dfac211f8768aff77a8b2: timing and hardware notes cut.
"""One-pass typed decisions on any backbone.

The model sees the state and the question once, and the answer is read off the
logits at the answer slot. Nothing is decoded, so a schema violation is
impossible and `tokens generated per decision` is zero.

- Questions are read at their answer slots, a whole prompt or packed after their state (`SlotReader._padded`):
  every typed answer, and every question on a backbone with no faster path of its own (`SystemOne.run`).
- A request is `(state, questions, media)`: its pictures, audio clips and videos head the state, a spoken question's
  clip sits in its own text (`media`).
"""

from __future__ import annotations

import os
from typing import Any, Sequence

import torch

from .api import SystemOneApi
from .attention import SET_SIZE, TRAINED_SET_SIZE, both_ways, language_model, pack_set
from .box import EDGE_READS
from .media import Media, Reader, media_list, spoken
from .pointer import SPAN_CAPS, query_plan, spans, state_positions
from .prompt import (
    DEFAULT_STATE_STYLE,
    DEFAULT_SYSTEM,
    PROMPTS,
    Box,
    Boxes,
    Choice,
    Position,
    Question,
    Read,
    default_lead,
    prefix_text,
    question_header,
    question_slots,
    readout,
    readouts,
    render,
    suffix_layout,
    suffix_text,
    typed,
)
from .slots import slot_logits, text_backbone

# Every picture is bounded at this many pixels before the processor (`media.cap_pixels`), in training and serving alike. A checkpoint
# records its own (`d1_vision_max_pixels` in its config); D1_VISION_MAX_PIXELS, when set, overrides it for a run or a
# check (reading a checkpoint at another resolution).
PIXELS_SET = os.environ.get("D1_VISION_MAX_PIXELS")
VISION_MAX_PIXELS = int(PIXELS_SET or 1024 * 1024)


def group_size(n: int | None) -> int | None:
    """A group size (`set_size`): a whole number from 1 (each question alone), or None for the engine's own."""
    if n is None:
        return None
    if isinstance(n, bool) or not isinstance(n, int):
        raise ValueError(f"set_size must be a whole number, not {n!r}")
    if n < 1:
        raise ValueError(f"set_size must be at least 1, not {n}")
    return n


def remote_ok(model_id) -> bool:
    """Whether a checkpoint's own code may run: a local folder or a LiquidAI repo.
    Any other Hub id loads with transformers' classes only, and one that needs its own code fails."""
    return os.path.isdir(str(model_id)) or str(model_id).startswith("LiquidAI/")


def auto_processor(path: str, revision: str | None = None):
    """The media processor a checkpoint holds."""
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(path, revision=revision, trust_remote_code=remote_ok(path))


def _under(config: dict | None, compiled):
    """`compiled` run under inductor's `config` (`SystemOne.compile`)."""
    if not config:
        return compiled
    import torch._inductor.config as inductor

    def run(*args, **kwargs):
        with inductor.patch(config):
            return compiled(*args, **kwargs)

    return run


class SlotReader:
    """Prompts read at their answer slots, whatever runs the backbone: a request's sequences (`_sequences`), each
    question's slots, and the answers read off them (`_padded`). An engine sets what `SystemOne.__init__` sets (the
    checkpoint's record through `_serve`, its `processor` or `_load_processor`) and gives `_batches` and
    `_slot_logits`."""

    def _serve(self, recorded, codes: int | None = None, max_pixels: int | None = None, attention: str | None = None,
               set_size: int | None = None, numbers: bool | None = None, box_edges: str | None = None,
               prompt: int | None = None) -> None:
        """The option codes, number readout, picture cap, attention mode, set size, box-edge read and prompt version
        `recorded` (a checkpoint's config) holds (`d1_codes`, `d1_numbers`, `d1_vision_max_pixels`, `d1_attention`,
        `d1_set_size`, `d1_box_edges`: the median without it, `d1_prompt`: 1 without it), unless given. A recorded set
        size is 2 to `TRAINED_SET_SIZE`, the default; one given is any whole number from 1 (each question alone):
        larger groups read faster, in packs training never saw."""
        self.tokenizer.d1_codes = codes if codes is not None else getattr(recorded, "d1_codes", 1)
        self.tokenizer.d1_numbers = bool(numbers if numbers is not None else getattr(recorded, "d1_numbers", False))
        self.tokenizer.d1_box_edges = box_edges or getattr(recorded, "d1_box_edges", None) or "median"
        if self.tokenizer.d1_box_edges not in EDGE_READS:
            raise ValueError(f"d1_box_edges is one of {EDGE_READS}, not {self.tokenizer.d1_box_edges!r}")
        self.tokenizer.d1_prompt = prompt or getattr(recorded, "d1_prompt", 1)
        if self.tokenizer.d1_prompt not in PROMPTS:
            raise ValueError(f"d1_prompt is one of {PROMPTS}, not {self.tokenizer.d1_prompt!r}")
        self.max_pixels = (VISION_MAX_PIXELS if PIXELS_SET else
                           max_pixels or getattr(recorded, "d1_vision_max_pixels", None) or VISION_MAX_PIXELS)
        self.attention = attention or getattr(recorded, "d1_attention", None) or "causal"
        self.set_size = group_size(getattr(recorded, "d1_set_size", SET_SIZE) if set_size is None else set_size)
        if set_size is None and not 2 <= self.set_size <= TRAINED_SET_SIZE:
            raise ValueError(f"d1_set_size must be between 2 and {TRAINED_SET_SIZE}")

    def _readout(self, q: Question, logz, tempered: bool = True) -> list[float]:
        """An option answer through the served temperature; a typed answer's slot raw (`tempered=False`): its own
        table calibrates it (`calibration.json` `tables`, fitted on raw reads)."""
        return readout(self.tokenizer, q, logz, self.calibration if tempered else None)

    def _readouts(self, qs: Sequence[Question], logz, tempered: Sequence[bool]) -> list[list[float]]:
        """`_readout` for each code slot of a batch, `logz` as `_slot_logits` gives them."""
        return [self._readout(q, z, t) for q, z, t in zip(qs, logz, tempered)]

    @property
    def media(self) -> Reader:
        """The checkpoint's media reader, its processor loaded on first use."""
        if getattr(self, "_reader", None) is None or self._reader.processor is not self.processor:
            if self.processor is None:
                self.processor = self._load_processor()
            self._reader = Reader(self.processor, self.max_pixels)
        return self._reader

    def _prefix(self, state: Any, media: Sequence[Media]) -> tuple[str, list[dict]]:
        """`prompt.prefix_text` headed by `media`'s markup, and their processor outputs."""
        markup, parts = self.media.encode(media) if media else ("", [])
        return prefix_text(self.tokenizer, state, self.bos, self.state_style, self.system, markup), parts

    def _suffixes(self, qs: Sequence[Question]) -> tuple[list[str], list[list[int]], list[list[dict]]]:
        """Each question's suffix (a spoken one's clip in place), its slots' offsets and its clip's outputs."""
        layouts = [suffix_layout(self.tokenizer, q, self.lead, self.option_style) for q in qs]
        placed = [self.media.place(text, q) if spoken(q) else (text, []) for (text, _), q in zip(layouts, qs)]
        return [text for text, _ in placed], [ends for _, ends in layouts], [parts for _, parts in placed]

    def _sequences(self, requests: Sequence[tuple[Any, list[Question], Sequence[Media]]], *,
                   position_maps=None, set_size: int | None = None) -> list[tuple]:
        """Every request's sequences, `(request, first question, ids, state length, each token's question or None,
        answer slots, media inputs)`: under `set` up to `set_size` questions after their state (`attention.pack_set`),
        else one whole prompt a question. A request's media head its state, a spoken question's clip sits in its
        suffix, each as its processor expands it, so the text is the sequence. `position_maps` gets each read's state
        positions; `set_size`, this call's group size (the engine's by default)."""
        seqs = []
        for i, (state, qs, media) in enumerate(requests):
            media = media_list(media)
            if any(isinstance(q, (Box, Boxes)) for q in qs) and (n := sum(m.type == "image" for m in media)) != 1:
                raise ValueError(f"a box or boxes question reads exactly one picture in `media`; this request has {n}")
            prefix, state_parts = self._prefix(state, media)
            suffixes, offsets, voices = self._suffixes(qs)
            step = (self.set_size if set_size is None else set_size) if self.attention == "set" else 1
            for k in range(0, len(qs), step):
                text = prefix + suffixes[k]
                encoded = None
                if position_maps is not None and any(isinstance(q, Read) for q in qs[k:k + step]):
                    encoded = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
                ids = self.tokenizer.encode(text, add_special_tokens=False) if encoded is None else encoded["input_ids"]
                if self.attention == "set":
                    ids, state_len, question, ends = pack_set(self.tokenizer, prefix, suffixes[k:k + step],
                                                              question_header(self.tokenizer, state),
                                                              offsets[k:k + step], first_ids=ids)
                else:
                    state_len, question = both_ways(self.attention, self.tokenizer, ids, suffixes[k]), None
                    ends = [[len(ids) - 1 + offset for offset in offsets[k]]]
                parts = state_parts + [part for own in voices[k:k + step] for part in own]
                inputs = self.media.inputs(parts, ids) if parts else {}
                if encoded is not None:
                    position_maps[len(seqs)] = state_positions(self.tokenizer, state, prefix, text, self.state_style,
                                                               encoded=encoded)
                seqs.append((i, k, ids, state_len, question, ends, inputs))
        return seqs

    def _padded(self, requests: Sequence[tuple[Any, list[Question], Sequence]],
                set_size: int | None = None) -> list[tuple[list[list[float]], int]]:
        """`_sequences` in the engine's batches (`_batches`), each read at its answer slots as training reads them
        (`_slot_logits`): option codes through the readout, reads through the pointer."""
        position_maps = {}
        seqs = self._sequences(requests, position_maps=position_maps, set_size=set_size)
        read = [0] * len(requests)
        for i, _, ids, *_ in seqs:
            read[i] += len(ids)

        def asked(key: tuple[int, int]) -> Question:
            return requests[key[0]][1][key[1]]

        probs: dict[tuple[int, int], list[float]] = {}
        for batch in self._batches(seqs):
            slots = []  # (row in the batch, (request, question), the slot's readout, its position)
            for r, b in enumerate(batch):
                i, first = seqs[b][:2]
                for k, ends in enumerate(seqs[b][5]):
                    key = (i, first + k)
                    slots.extend((r, key, q, end) for q, end in zip(question_slots(asked(key), self.tokenizer), ends))
            reads = [j for j, s in enumerate(slots) if isinstance(s[2], Position)]  # a read's start then its end
            pointer = None
            if reads:
                masks = [torch.tensor(position_maps[batch[slots[j][0]]].pointer_mask) for j in reads]
                pointer = query_plan([slots[j][0] for j in reads], masks, reads, len(slots))
            logz, positions = self._slot_logits(seqs, batch, slots, pointer)
            if pointer is not None:
                logits = positions.reshape(-1, 2, positions.shape[-1])
                capped = {}  # the reads of each span cap
                for index, j in enumerate(reads[::2]):
                    capped.setdefault(SPAN_CAPS[asked(slots[j][1]).type], []).append((index, j))
                for cap, own in capped.items():
                    at, found = spans(logits[[index for index, _ in own]], max_tokens=cap)
                    for (_, j), span, p in zip(own, at.tolist(), found.tolist()):
                        r, key = slots[j][:2]
                        probs[key] = {**position_maps[batch[r]].read(*span, max_tokens=cap), "found": p}
            codes = pointer["code_slots"].tolist() if pointer else range(len(slots))
            scored = self._readouts([slots[j][2] for j in codes], logz, [not typed(asked(slots[j][1])) for j in codes])
            for j, p in zip(codes, scored):
                key = slots[j][1]
                q = asked(key)
                if isinstance(q, Choice) and q.multiple:
                    probs.setdefault(key, []).append(p[0])
                elif isinstance(q, (Box, Boxes)):
                    probs.setdefault(key, []).append(p)
                else:
                    probs[key] = p
        return [([probs[(i, k)] for k in range(len(qs))], read[i]) for i, (_, qs, _) in enumerate(requests)]


class SystemOne(SlotReader, SystemOneApi):
    """State in, calibrated distribution out, one forward pass."""

    def __init__(
        self,
        model=None,
        tokenizer=None,
        device: str | None = None,
        calibration=None,
        lead: str | None = None,
        state_style: str = DEFAULT_STATE_STYLE,
        system: str = DEFAULT_SYSTEM,
        option_style: str = "desc",
        codes: int | None = None,
        max_pixels: int | None = None,
        attention: str | None = None,
        set_size: int | None = None,
        numbers: bool | None = None,
        box_edges: str | None = None,
        prompt: int | None = None,
        compile: bool = False,
        token_budget: int = 65536,
        revision: str | None = None,
    ):
        """`model` and `tokenizer`: a backbone already loaded (the Hub's class passes itself, and the `revision` its
        processor loads at); it stays on its device unless `device` says otherwise. `codes` and `max_pixels` default
        to what the checkpoint's config records (`d1_codes`, `d1_vision_max_pixels`), else scheme 1 and the cap;
        `attention` likewise (`d1_attention`, else causal; `attention`)."""
        if model is None or tokenizer is None or isinstance(model, str):
            raise TypeError("SystemOne reads a loaded backbone: pass `model=` and `tokenizer=` (a checkpoint's own "
                            "class serves it: AutoModel.from_pretrained(repo, trust_remote_code=True).system_one)")
        self.model, self.tokenizer = model, tokenizer
        self.model_id, device = model.config._name_or_path, device or next(model.parameters()).device
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if getattr(self.model, "hf_device_map", None):  # spread over GPUs at load: inputs go to the embeddings
            self.device = self.model.get_input_embeddings().weight.device
        else:
            self.model.to(self.device)
        self.model.eval()
        if hasattr(self.model, "fuse_projections"):  # one matmul for the linears on one input
            self.model.fuse_projections()
        bos = getattr(self.tokenizer, "bos_token", None)
        self.bos = bos if isinstance(bos, str) else ""
        self.lead = default_lead(model.config.model_type, tokenizer) if lead is None else lead
        self.calibration, self.state_style, self.system, self.option_style = calibration, state_style, system, option_style
        self._serve(self.model.config, codes, max_pixels, attention, set_size, numbers, box_edges, prompt)
        self.token_budget = token_budget
        self.processor, self.revision = None, revision
        if compile:
            self.compile()

    def compile(self, mode: str | None = None) -> None:
        """torch.compile what the passes run: the language model's forward, shapes dynamic; not its hooks, so
        `state_attention`'s plan is made outside the graph. Also the media encoders the backbone names
        (`compiled_parts`), under the inductor config it names (`compile_options`),
        set while they run: dynamo reads `freezing` as it traces, where `torch.compile`'s own `options` do not reach.
        No CUDA graphs: inductor's graph pool refuses the packed varlen calls."""
        if mode in ("reduce-overhead", "max-autotune") and self.attention != "causal":
            raise ValueError(f"mode={mode!r} records CUDA graphs, which `{self.attention}` attention does not take: "
                             "model.compile() fuses without them")
        body, parts = text_backbone(self.model), getattr(self.model, "compiled_parts", ())
        config = getattr(self.model, "compile_options", None)
        for part in (language_model(self.model), *(getattr(body, name) for name in parts)):
            part.forward = _under(config, torch.compile(part.forward, dynamic=True, mode=mode))

    # ---------------------------------------------------------------- prompt

    def render(self, state: Any, q: Question) -> str:
        return render(self.tokenizer, state, q, self.bos, self.lead, self.state_style, self.system, self.option_style)

    # ------------------------------------------------------------------- api

    @torch.inference_mode()
    def run(self, requests: Sequence[tuple[Any, list[Question], Sequence[Media]]],
            set_size: int | None = None) -> list[tuple[list[list[float]], int]]:
        """Each `(state, questions, media)` request's probabilities and the tokens it read, at its answer slots
        (`_padded`), every request in the same batches. `set_size`: this call's questions a group under `set` (the
        engine's by default)."""
        set_size = group_size(set_size)
        requests = [(state, qs, media_list(media)) for state, qs, media in requests]
        return self._padded(requests, set_size)

    def _batches(self, seqs: list[tuple]) -> list[list[int]]:
        """`_sequences` right-padded into batches of at most `token_budget` padded tokens, whatever their media: a
        request's sequences in one batch (one forward) unless they alone pass the budget, requests of similar length
        together. Every row is whole, its media given with it (`_rows`)."""
        own: dict[int, list[int]] = {}
        for j in sorted(range(len(seqs)), key=lambda j: len(seqs[j][2])):
            own.setdefault(seqs[j][0], []).append(j)
        batches: list[list[int]] = [[]]
        for rows in sorted(own.values(), key=lambda rows: len(seqs[rows[-1]][2])):
            if batches[-1] and len(seqs[rows[-1]][2]) * (len(batches[-1]) + len(rows)) > self.token_budget:
                batches.append([])
            for j in rows:  # shortest first, so a batch's width is its last sequence's length
                if batches[-1] and len(seqs[j][2]) * (len(batches[-1]) + 1) > self.token_budget:
                    batches.append([])
                batches[-1].append(j)
        return list(filter(None, batches))

    def _rows(self, seqs: list[tuple], batch: list[int]) -> tuple[torch.Tensor, list[int], dict]:
        """A batch's right-padded ids, each row's length and its media inputs: every row whole, its own ids and its
        media's inputs, joined in row order (`media.Reader.batch`) for the one forward."""
        pad = self.tokenizer.pad_token_id or 0
        lengths = [len(seqs[b][2]) for b in batch]
        ids = torch.tensor([seqs[b][2] + [pad] * (max(lengths) - n) for b, n in zip(batch, lengths)], device=self.device)
        inputs = [seqs[b][6] for b in batch]
        return ids, lengths, self.media.batch(inputs, ids) if any(inputs) else {}

    def _slot_logits(self, seqs: list[tuple], batch: list[int], slots: list[tuple], pointer: dict | None):
        """A batch's code slots' log-probabilities over the vocabulary, and its read slots' position logits
        (`slots.slot_logits`), in one forward (`_rows`)."""
        ids, lengths, media = self._rows(seqs, batch)
        width = ids.shape[1]
        mask = (torch.arange(width, device=self.device) < torch.tensor(lengths, device=self.device)[:, None]).long()
        state = None if self.attention == "causal" else torch.tensor([seqs[b][3] for b in batch])
        question = None if self.attention != "set" else torch.tensor(
            [seqs[b][4] + [-1] * (width - len(seqs[b][4])) for b in batch])
        rows = slot_logits(self.model, ids, mask, torch.tensor([s[3] for s in slots]), state, question,
                           torch.tensor([s[0] for s in slots]), pointer=pointer, **self._on_device(media))
        codes = rows.codes.float()
        return codes - torch.logsumexp(codes, dim=-1, keepdim=True), rows.positions

    def _readouts(self, qs: Sequence[Question], logz: torch.Tensor, tempered: Sequence[bool]) -> list[list[float]]:
        """Every code slot of a batch read at once (`prompt.readouts`): one transfer, not one a slot."""
        return readouts(self.tokenizer, qs, logz, [self.calibration if t else None for t in tempered])

    def tokens(self, state: Any, questions: Sequence[Question]) -> int:
        """The longest prompt one of `questions` makes over a text state: its state's tokens and its own."""
        prefix, _ = self._prefix(state, ())
        return len(self.tokenizer.encode(prefix, add_special_tokens=False)) + max(
            len(self.tokenizer.encode(suffix_text(self.tokenizer, q, self.lead, self.option_style),
                                      add_special_tokens=False)) for q in questions)

    # ----------------------------------------------------------------- media

    def _load_processor(self):
        return auto_processor(self.model.config._name_or_path, self.revision)

    def _on_device(self, inputs: dict) -> dict:
        """The media inputs on the model's device, but what the backbone reads on the host (`host_inputs`)."""
        host = getattr(self.model, "host_inputs", ())
        return {k: (v.to(self.device) if hasattr(v, "to") and k not in host else v) for k, v in inputs.items()}
