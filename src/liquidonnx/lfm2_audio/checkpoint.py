"""The LFM2.5-Audio checkpoint folder the export reads, given as a path or a Hugging Face model ID."""

import pathlib

from liquidonnx.genai_builder import resolve_checkpoint

# model.safetensors holds every graph but the detokenizer, which liquid-audio loads from
# audio_detokenizer/. The Hub repos also hold liquid-audio's Mimi codec (tokenizer-*.safetensors)
# and demo media, which the export does not read.
CHECKPOINT_PATTERNS = ["*.json", "*.jinja", "model*.safetensors", "audio_detokenizer/*"]


def checkpoint_dir(model: str) -> pathlib.Path:
    return resolve_checkpoint(model, CHECKPOINT_PATTERNS)
