"""LFM3's transformers classes, one module a model type: `native_lfm3` (text), `native_lfm3_vl` (pictures),
`native_lfm3_omni` (pictures and audio), each importing the one before it, so a Hub repository serves only its
model's files. transformers' own classes once a release ships them; until then copies of transformers' LFM3 code
beside these (each names its commit), registered with the Auto classes, so a checkpoint's config, tokenizer and
processor load and the composite models build their parts.

Shipping LFM3, transformers makes the fallbacks dead: delete them and the copies (the files headed "Vendored from"),
nothing else."""

from __future__ import annotations

try:
    from transformers import Lfm3Config, Lfm3ForCausalLM, Lfm3Model
except ImportError:
    from transformers import AutoConfig, AutoModel

    from .configuration_lfm3 import Lfm3Config
    from .modeling_lfm3 import Lfm3ForCausalLM, Lfm3Model

    # exist_ok: a second copy of these files in one process (two snapshots) registers again
    AutoConfig.register(Lfm3Config.model_type, Lfm3Config, exist_ok=True)
    AutoModel.register(Lfm3Config, Lfm3Model, exist_ok=True)  # the composite models build their text model by config

__all__ = ["Lfm3Config", "Lfm3ForCausalLM", "Lfm3Model"]
