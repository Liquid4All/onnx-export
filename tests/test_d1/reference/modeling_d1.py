"""The System One API on any backbone: each `modeling_d1_<model_type>.py` serves `D1Model(SystemOneModel, <its
class>)` to `AutoModel.from_pretrained(repo, trust_remote_code=True)`.

    model.system_one(state, {name: question}, media=None)     # one forward pass; media: pictures, audio, video
    model.system_one_batch([(state, {name: question}), ...])  # many states in one call

A release's `calibration.json` (its temperature) is applied to every answer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import cached_property
from typing import Any

from .api import SCORE_READOUT
from .runner import SystemOne


class SystemOneModel:
    """The System One API over any backbone the runner reads. `attention`: the attention function a transformers
    backbone loads with (`attention.d1_attention`), when it has one."""

    attention: str | None = None

    @classmethod
    def from_pretrained(cls, *args, revision=None, **kwargs):
        if cls.attention:
            kwargs["attn_implementation"] = cls.attention
        loaded = super().from_pretrained(*args, revision=revision, **kwargs)  # with its loading info when asked
        # transformers 5.19 resolves the revision to a commit for the load and keeps no `_commit_hash`
        (loaded[0] if isinstance(loaded, tuple) else loaded).d1_revision = revision
        return loaded

    def d1_file(self, name: str, required: bool = True) -> str | None:
        """A file of the loaded revision, not main's: its path, or None for a missing one not `required`."""
        from transformers.utils import cached_file

        return cached_file(self.name_or_path, name, revision=self._revision(),
                           _raise_exceptions_for_missing_entries=required)

    def _revision(self) -> str | None:
        return getattr(self.config, "_commit_hash", None) or getattr(self, "d1_revision", None)

    @cached_property
    def engine(self) -> SystemOne:
        """The runner over this model, with the loaded revision's tokenizer and temperature."""
        from transformers import AutoTokenizer

        from .calibration import Calibration

        found = self.d1_file("calibration.json", required=False)
        return SystemOne(model=self.eval(), tokenizer=AutoTokenizer.from_pretrained(self.name_or_path,
                                                                                  revision=self._revision()),
                         calibration=Calibration.load(found) if found else None, revision=self._revision())

    def compile(self, mode: str | None = None) -> None:
        """torch.compile what `system_one` runs (`SystemOne.compile`): fused kernels, with no CUDA graphs under `set` attention."""
        self.engine.compile(mode)

    def system_one(self, state: Any, questions: Mapping[str, Any], media: Sequence | None = None,
                   score_readout: str = SCORE_READOUT, set_size: int | None = None) -> dict:
        """Named questions over a state (text, JSON, or None with media alone) and its media (pictures, audio clips,
        videos: `media`): `{"answers": {name: answer}, "usage": {"input_tokens": n, "output_tokens": 0}}`.
        `score_readout`: a score's `median` level (the default), its expected level (`mean`) or its `argmax` level.
        `set_size`: questions read together after the state, the model's trained size by default; larger is faster
        and costs some accuracy on text."""
        return self.engine.system_one(state, questions, media, score_readout, set_size=set_size)

    def system_one_batch(self, requests: Sequence[tuple], score_readout: str = SCORE_READOUT,
                         set_size: int | None = None) -> list[dict]:
        """`(state, questions)` or `(state, questions, media)` requests, one response each."""
        return self.engine.system_one_batch(requests, score_readout, set_size=set_size)
