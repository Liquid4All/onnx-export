"""
LFM3-VL safetensors -> float32 arrays for the decoder and vision builders.

The builders read the flat layout, an earlier one: the language model packs q/k/v into
self_attn.qkv and gate/up into feed_forward.gate_up, and stores the conv kernel as the bare
parameter conv.conv; language_weights unpacks them into the names LFM3Builder reads
(model.layers.N.self_attn.{q,k,v}_proj, feed_forward.w1/w3, conv.conv.weight).

LFM3-VL-2B (transformers' layout) stores the same tensors as model.language_model.*,
model.vision_tower.* and model.multi_modal_projector.*, with q/k/v and w1/w3 unfused and the
convolutions as Conv modules (conv.conv.weight, spatial_conv.weight). load_tensors gives them the
names above, so both layouts build the same graphs.
"""

import logging
import pathlib
import re

import numpy as np

from liquidonnx.lfm3.builder import LFM3Config

logger = logging.getLogger(__name__)

PORT_PREFIXES = {
    "model.language_model.": "language_model.",
    "model.vision_tower.": "vision_encoder.",
    "model.multi_modal_projector.": "vision_projector.",
}
PORT_SUFFIXES = {".conv.conv.weight": ".conv.conv", ".spatial_conv.weight": ".spatial_conv"}
# (fused name, its parts in concatenation order): gate_up.chunk(2) and qkv.split
FUSED = (
    (re.compile(r"(.*\.feed_forward\.)w1\.weight$"), "gate_up.weight", ("w1", "w3")),
    (
        re.compile(r"(language_model\.layers\.\d+\.self_attn\.)q_proj\.weight$"),
        "qkv.weight",
        ("q_proj", "k_proj", "v_proj"),
    ),
)


def flat_name(name: str) -> str:
    """The flat checkpoints' name of a tensor of either format (before fusing)."""
    for port, flat in PORT_PREFIXES.items():
        if name.startswith(port):
            name = flat + name[len(port) :]
            for suffix, replacement in PORT_SUFFIXES.items():
                if name.endswith(suffix):
                    return name[: -len(suffix)] + replacement
    return name


def fuse_port_tensors(tensors: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """gate_up and the language qkv concatenated from their parts, as the flat checkpoints store
    them; a no-op on those."""
    for pattern, fused, parts in FUSED:
        for name in [n for n in tensors if pattern.match(n)]:
            base = pattern.match(name).group(1)
            tensors[base + fused] = np.concatenate(
                [tensors.pop(f"{base}{part}.weight") for part in parts]
            )
    return tensors


def load_tensors(checkpoint: pathlib.Path, prefix: str) -> dict[str, np.ndarray]:
    """float32 copies of the tensors whose flat names start with prefix, under those names."""
    import torch
    from safetensors import safe_open

    files = sorted(pathlib.Path(checkpoint).glob("model*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no model*.safetensors in {checkpoint}")
    tensors = {}
    for path in files:
        with safe_open(str(path), framework="pt") as f:
            for name in f.keys():
                flat = flat_name(name)
                if flat.startswith(prefix):
                    tensors[flat] = f.get_tensor(name).to(torch.float32).numpy()
    if not tensors:
        raise KeyError(f"{checkpoint} has no {prefix}* tensors")
    logger.info(f"Loaded {len(tensors)} {prefix}* tensors")
    return fuse_port_tensors(tensors)


def _expect(name: str, array: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    if array.shape != shape:
        raise ValueError(f"{name} has shape {array.shape}, expected {shape}")
    return array


def language_weights(tensors: dict[str, np.ndarray], config: LFM3Config) -> dict[str, np.ndarray]:
    """language_model.* tensors under the names LFM3Builder reads."""
    H, hd = config.hidden_size, config.head_dim
    q_size, kv_size = config.num_attention_heads * hd, config.num_key_value_heads * hd
    inter = config.intermediate_size
    remaining = dict(tensors)
    out = {}

    def take(name: str) -> np.ndarray:
        return remaining.pop(f"language_model.{name}")

    out["model.embed_tokens.weight"] = take("embed_tokens.weight")
    out["model.embedding_norm.weight"] = take("embedding_norm.weight")
    for i, layer_type in enumerate(config.layer_types):
        src, dst = f"layers.{i}", f"model.layers.{i}"
        for norm in ("operator_norm", "post_operator_norm", "ffn_norm", "post_ffn_norm"):
            out[f"{dst}.{norm}.weight"] = take(f"{src}.{norm}.weight")
        # gate_up.chunk(2): gate first, then up
        gate_up = _expect(
            f"{src}.gate_up", take(f"{src}.feed_forward.gate_up.weight"), (2 * inter, H)
        )
        out[f"{dst}.feed_forward.w1.weight"] = gate_up[:inter]
        out[f"{dst}.feed_forward.w3.weight"] = gate_up[inter:]
        out[f"{dst}.feed_forward.w2.weight"] = take(f"{src}.feed_forward.w2.weight")
        if layer_type == "conv":
            out[f"{dst}.conv.in_proj.weight"] = take(f"{src}.conv.in_proj.weight")
            out[f"{dst}.conv.out_proj.weight"] = take(f"{src}.conv.out_proj.weight")
            out[f"{dst}.conv.conv.weight"] = _expect(
                f"{src}.conv.conv", take(f"{src}.conv.conv"), (H, 1, config.conv_L_cache)
            )
            continue
        # qkv.split((q_size, kv_size, kv_size))
        qkv = _expect(f"{src}.qkv", take(f"{src}.self_attn.qkv.weight"), (q_size + 2 * kv_size, H))
        out[f"{dst}.self_attn.q_proj.weight"] = qkv[:q_size]
        out[f"{dst}.self_attn.k_proj.weight"] = qkv[q_size : q_size + kv_size]
        out[f"{dst}.self_attn.v_proj.weight"] = qkv[q_size + kv_size :]
        for name in ("gate_proj", "o_proj", "q_norm", "k_norm"):
            out[f"{dst}.self_attn.{name}.weight"] = take(f"{src}.self_attn.{name}.weight")
    if remaining:
        raise ValueError(f"unused language tensors: {sorted(remaining)[:8]}")
    return out
