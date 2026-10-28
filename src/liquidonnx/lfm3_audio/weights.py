"""
LFM3-Audio / LFM3-Omni safetensors -> float32 arrays per component, under one internal naming.

Names go through two steps:

1. normalize_name: the checkpoint's name -> the HF-native name of Lfm3AudioForConditionalGeneration
   (and Lfm3OmniForConditionalGeneration). LFM3-Audio-2B stores the trainer's names, which TRAINER_RENAMES maps as the remote code's
   register_checkpoint_conversion_mapping("lfm3_audio") does; LFM3-Omni already stores the
   HF-native ones. depth_embeddings.N.to_logits.weight, the
   trainer's copy of the tied depth heads (equal to .embedding.weight), is dropped as the
   reference drops it (_keys_to_ignore_on_load_unexpected).

       lfm.model.*                    model.language_model.*
       encoder.downsample.*           model.audio_encoder.downsample.*
       encoder.lfm.lfm.*              model.audio_encoder.lfm.*
       mlp.model.{0,1,3}.*            model.audio_projector.{layer_norm,linear_1,linear_2}.*
       codebook_embedding.*           model.codebook_embedding.*
       depth_linear.*, mask_embedding, depthformer.model.*, depth_embeddings.*
                                      model.depth_decoder.{depth_linear.*, mask_embedding,
                                                           model.*, depth_embeddings.*}

2. split_name: the HF-native name -> (component, internal name). Each component's internal names
   are what its builder reads; the transformer stacks use LFM3Builder's (model.layers.N.*,
   model.embedding_norm.weight), unfused q/k/v and w1/w3 as both checkpoints store them:

       language  model.embed_tokens.weight [V, H], model.embedding_norm.weight [H],
                 model.layers.N.{operator_norm, post_operator_norm, ffn_norm, post_ffn_norm}.weight,
                 model.layers.N.feed_forward.{w1, w2, w3}.weight,
                 model.layers.N.conv.{in_proj.weight, conv.weight, out_proj.weight},
                 model.layers.N.self_attn.{q_proj, k_proj, v_proj, o_proj, q_norm, k_norm,
                                           gate_proj}.weight,
                 codebook_embedding.weight [num_codebooks * codebook_size, H]
       encoder   model.layers.N.* and model.embedding_norm.weight of the LFM2 stack (attention:
                 q_layernorm, k_layernorm, out_proj, which LFM3Builder._weight also reads; no
                 gate, no sandwich norms),
                 downsample.{conv_in, conv_mid, lin_mid, conv_end, lin_end, lin_out}.{weight, bias},
                 projector.{layer_norm, linear_1, linear_2}.{weight, bias}
       depth     model.layers.N.*, model.embedding_norm.weight, depth_linear.{weight, bias},
                 mask_embedding [D], depth_embeddings.K.{embedding, embedding_norm}.weight
       vision    (lfm3_omni) liquidonnx.lfm3_vl.weights' flat names, vision_encoder.* and
                 vision_projector.*, with feed_forward.gate_up fused as LFM3VLVisionBuilder reads

   The codec (audio_tokenizer/ in LFM3-Audio-2B and LFM3-Omni-2B, audio_detokenizer/ in an
   earlier layout) keeps its decoder side under codec_name:

       codec     decoder.lfm.* -> model.*, decoder.lin.* -> lin.*, decoder.istft.window ->
                 istft.window [n_fft] (float32 in the file), quantizer.project_out.*,
                 from_quantizer.*
                 The encoder side (frontend.*, encoder.*, to_quantizer.*, quantizer.project_in.*)
                 is not exported.

expected_shapes gives every internal name's shape from an LFM3AudioConfig, and load_weights checks
a component against it: a missing, extra or misshapen tensor raises.
"""

import logging
import pathlib
import re

import numpy as np

from liquidonnx.lfm3.builder import LFM3Config
from liquidonnx.lfm3_audio.config import LFM3AudioConfig, codec_dir
from liquidonnx.lfm3_vl.weights import flat_name, fuse_port_tensors

logger = logging.getLogger(__name__)

# modeling_lfm3_audio.py register_checkpoint_conversion_mapping("lfm3_audio")
TRAINER_RENAMES = tuple(
    (re.compile(source), target)
    for source, target in (
        (r"^lfm\.model\.", "model.language_model."),
        (r"^lfm\.lm_head\.", "lm_head."),
        (r"^encoder\.downsample\.", "model.audio_encoder.downsample."),
        (r"^encoder\.lfm\.lfm\.", "model.audio_encoder.lfm."),
        (r"^mlp\.model\.0\.", "model.audio_projector.layer_norm."),
        (r"^mlp\.model\.1\.", "model.audio_projector.linear_1."),
        (r"^mlp\.model\.3\.", "model.audio_projector.linear_2."),
        (r"^codebook_embedding\.", "model.codebook_embedding."),
        (r"^depth_linear\.", "model.depth_decoder.depth_linear."),
        (r"^mask_embedding$", "model.depth_decoder.mask_embedding"),
        (r"^depthformer\.model\.", "model.depth_decoder.model."),
        (r"^depth_embeddings\.", "model.depth_decoder.depth_embeddings."),
    )
)
IGNORED = re.compile(r"depth_embeddings\.\d+\.to_logits\.weight$")
# HF-native prefix -> (component, internal prefix); the longest match wins
COMPONENT_PREFIXES = {
    "model.language_model.": ("language", "model."),
    "model.codebook_embedding.": ("language", "codebook_embedding."),
    "model.audio_encoder.lfm.": ("encoder", "model."),
    "model.audio_encoder.downsample.": ("encoder", "downsample."),
    "model.audio_projector.": ("encoder", "projector."),
    "model.depth_decoder.model.": ("depth", "model."),
    "model.depth_decoder.": ("depth", ""),
}
VISION_PREFIXES = ("model.vision_tower.", "model.multi_modal_projector.")
COMPONENTS = ("language", "encoder", "depth", "vision")
CODEC_PREFIXES = {
    "decoder.lfm.": "model.",
    "decoder.": "",
    "quantizer.project_out.": "quantizer.project_out.",
    "from_quantizer.": "from_quantizer.",
}
CODEC_ENCODER_PREFIXES = ("frontend.", "encoder.", "to_quantizer.", "quantizer.project_in.")


def normalize_name(name: str) -> str | None:
    """The HF-native name of a checkpoint tensor of either format; None for one the reference
    ignores."""
    if IGNORED.search(name):
        return None
    for pattern, target in TRAINER_RENAMES:
        if pattern.match(name):
            return pattern.sub(target, name, count=1)
    return name


def split_name(name: str) -> tuple[str, str]:
    """(component, internal name) of an HF-native name."""
    if name.startswith(VISION_PREFIXES):
        return "vision", flat_name(name)
    for prefix in sorted(COMPONENT_PREFIXES, key=len, reverse=True):
        if name.startswith(prefix):
            component, internal = COMPONENT_PREFIXES[prefix]
            return component, internal + name[len(prefix) :]
    raise KeyError(f"LFM3-Audio export does not know the tensor {name}")


def codec_name(name: str) -> str | None:
    """The internal name of a codec tensor; None for the codec encoder's."""
    if name.startswith(CODEC_ENCODER_PREFIXES):
        return None
    for prefix, internal in CODEC_PREFIXES.items():
        if name.startswith(prefix):
            return internal + name[len(prefix) :]
    raise KeyError(f"LFM3-Audio export does not know the codec tensor {name}")


def stack_shapes(cfg: LFM3Config, lfm2: bool = False) -> dict[str, tuple[int, ...]]:
    """Tensor shapes of an LFM3 (or LFM2, lfm2=True) stack under LFM3Builder's names."""
    H, F, hd = cfg.hidden_size, cfg.intermediate_size, cfg.head_dim
    q, kv = cfg.num_attention_heads * hd, cfg.num_key_value_heads * hd
    shapes = {"model.embedding_norm.weight": (H,)}
    norms = ["operator_norm", "ffn_norm"]
    if cfg.use_peri_ln:
        norms += ["post_operator_norm", "post_ffn_norm"]
    attn = {"q_proj": (q, H), "k_proj": (kv, H), "v_proj": (kv, H)}
    if lfm2:
        attn |= {"q_layernorm": (hd,), "k_layernorm": (hd,), "out_proj": (H, q)}
    else:
        attn |= {"q_norm": (hd,), "k_norm": (hd,), "o_proj": (H, q)}
    if cfg.use_gated_attention:
        heads = cfg.num_attention_heads
        attn["gate_proj"] = (heads if cfg.gated_attention_style == "headwise" else q, H)
    for i, layer_type in enumerate(cfg.layer_types):
        prefix = f"model.layers.{i}"
        shapes |= {f"{prefix}.{norm}.weight": (H,) for norm in norms}
        shapes |= {
            f"{prefix}.feed_forward.w1.weight": (F, H),
            f"{prefix}.feed_forward.w3.weight": (F, H),
            f"{prefix}.feed_forward.w2.weight": (H, F),
        }
        if layer_type == "conv":
            shapes |= {
                f"{prefix}.conv.in_proj.weight": (3 * H, H),
                f"{prefix}.conv.conv.weight": (H, 1, cfg.conv_L_cache),
                f"{prefix}.conv.out_proj.weight": (H, H),
            }
        else:
            shapes |= {f"{prefix}.self_attn.{k}.weight": v for k, v in attn.items()}
    return shapes


def linear_shapes(name: str, out_features: int, in_features: int) -> dict[str, tuple[int, ...]]:
    return {f"{name}.weight": (out_features, in_features), f"{name}.bias": (out_features,)}


def expected_shapes(config: LFM3AudioConfig) -> dict[str, dict[str, tuple[int, ...]]]:
    """component -> internal name -> shape, for language, encoder, depth and codec."""
    text, enc, depth, codec = config.language, config.encoder, config.depth, config.codec
    K, C = config.num_codebooks, config.encoder_conv_channels
    E, D = enc.hidden_size, depth.hidden_size
    language = stack_shapes(text) | {
        "model.embed_tokens.weight": (text.vocab_size, text.hidden_size),
        "codebook_embedding.weight": (K * config.codebook_size, text.hidden_size),
    }
    encoder = stack_shapes(enc, lfm2=True)
    # conv_in has one input channel, conv_mid and conv_end are depthwise
    for name in ("conv_in", "conv_mid", "conv_end"):
        encoder |= {f"downsample.{name}.weight": (C, 1, 3, 3), f"downsample.{name}.bias": (C,)}
    for name in ("lin_mid", "lin_end"):
        encoder |= {f"downsample.{name}.weight": (C, C, 1, 1), f"downsample.{name}.bias": (C,)}
    flat = config.mel.num_mel_bins * C // 8
    encoder |= {"downsample.lin_out.weight": (E, flat, 1), "downsample.lin_out.bias": (E,)}
    encoder |= {"projector.layer_norm.weight": (E,), "projector.layer_norm.bias": (E,)}
    encoder |= linear_shapes("projector.linear_1", text.hidden_size, E)
    encoder |= linear_shapes("projector.linear_2", text.hidden_size, text.hidden_size)
    depth_shapes = stack_shapes(depth) | linear_shapes("depth_linear", K * D, text.hidden_size)
    depth_shapes["mask_embedding"] = (D,)
    for k in range(K):
        depth_shapes[f"depth_embeddings.{k}.embedding.weight"] = (config.codebook_size, D)
        depth_shapes[f"depth_embeddings.{k}.embedding_norm.weight"] = (D,)
    bins = config.codec_n_fft // 2 + 1
    latent = len(config.fsq_levels) * K
    codec_shapes = stack_shapes(codec) | linear_shapes("lin", 2 * bins, codec.hidden_size)
    codec_shapes["istft.window"] = (config.codec_n_fft,)
    # LiquidCodecQuantizer has no projection when the latent is already quantizer_dim wide
    if config.codec_quantizer_dim != latent:
        codec_shapes |= linear_shapes("quantizer.project_out", config.codec_quantizer_dim, latent)
    codec_shapes |= linear_shapes("from_quantizer", codec.hidden_size, config.codec_quantizer_dim)
    return {"language": language, "encoder": encoder, "depth": depth_shapes, "codec": codec_shapes}


def check_shapes(component: str, shapes: dict[str, tuple[int, ...]], config: LFM3AudioConfig):
    """Raise unless shapes (internal name -> shape) are exactly the component's tensors."""
    expected = expected_shapes(config)[component]
    missing, extra = sorted(set(expected) - set(shapes)), sorted(set(shapes) - set(expected))
    wrong = sorted(n for n in set(expected) & set(shapes) if tuple(shapes[n]) != expected[n])
    if missing or extra or wrong:
        details = [f"{n}: {tuple(shapes[n])} != {expected[n]}" for n in wrong[:8]]
        raise ValueError(
            f"{component} tensors do not match the config: missing {missing[:8]}, "
            f"unexpected {extra[:8]}, shapes {details}"
        )


def read_safetensors(path: pathlib.Path, rename) -> dict[str, np.ndarray]:
    """float32 copies of the tensors of one file whose rename(name) is not None, under it."""
    import torch
    from safetensors import safe_open

    tensors = {}
    with safe_open(str(path), framework="pt") as f:
        for name in f.keys():
            internal = rename(name)
            if internal is not None:
                tensors[internal] = f.get_tensor(name).to(torch.float32).numpy()
    return tensors


def load_weights(
    checkpoint: str | pathlib.Path, component: str, config: LFM3AudioConfig
) -> dict[str, np.ndarray]:
    """float32 tensors of one component (COMPONENTS) of the checkpoint, under its internal names,
    checked against the config (vision: lfm3_omni's builder checks them)."""
    if component not in COMPONENTS:
        raise ValueError(f"unknown component {component!r}, expected one of {COMPONENTS}")
    files = sorted(pathlib.Path(checkpoint).glob("model*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no model*.safetensors in {checkpoint}")

    def rename(name: str) -> str | None:
        name = normalize_name(name)
        if name is None:
            return None
        part, internal = split_name(name)
        return internal if part == component else None

    tensors = {}
    for path in files:
        tensors |= read_safetensors(path, rename)
    logger.info(f"Loaded {len(tensors)} {component} tensors from {checkpoint}")
    if component == "vision":
        if not tensors:
            raise KeyError(f"{checkpoint} has no vision tensors")
        return fuse_port_tensors(tensors)
    check_shapes(component, {k: v.shape for k, v in tensors.items()}, config)
    return tensors


def load_codec_weights(checkpoint: str | pathlib.Path, config: LFM3AudioConfig) -> dict:
    """float32 tensors of the codec decoder under their internal names, checked."""
    tensors = read_safetensors(codec_dir(checkpoint) / "model.safetensors", codec_name)
    check_shapes("codec", {k: v.shape for k, v in tensors.items()}, config)
    return tensors
