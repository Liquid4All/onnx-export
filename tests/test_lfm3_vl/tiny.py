"""
A tiny random LFM3-VL checkpoint with the real token ids, for tests without a download.

write_tiny_checkpoint writes it in LFM3-VL-2B's layout: the nested config.json of port_config,
transformers' tensor names and LFM3-VL-2B's remote code from reference/, which is its reference.
TINY_CONFIG is the same model in the flat layout the exporter's parsers work on
(liquidonnx.lfm3_vl.config), and write_flat_checkpoint writes its tensors under the flat names.
"""

import json
import pathlib
import shutil

from .reference.loader import REFERENCE_DIR

TINY_CONFIG = {
    "architectures": ["Lfm3VlForConditionalGeneration"],
    "model_type": "lfm3_vl",
    "bos_token_id": 124894,
    "eos_token_id": 124900,
    "pad_token_id": 124893,
    "image_token_id": 125030,
    "video_token_id": 125029,
    "vision_start_token_id": 125009,
    "vision_end_token_id": 125010,
    "vocab_size": 128000,
    "hidden_size": 64,
    # Even numbers of 32-wide blocks: onnxruntime's k_quant can't pack the int4 zero points of an
    # odd one
    "intermediate_size": 128,
    "num_hidden_layers": 6,
    "layer_types": ["conv", "conv", "full_attention", "conv", "sliding_attention", "conv"],
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "conv_L_cache": 7,
    "conv_bias": False,
    "sliding_window": 8,
    "full_attention_rope_theta": 1000000.0,
    "sliding_attention_rope_theta": 10000,
    "partial_rotary_factor": 0.5,
    "max_position_embeddings": 4096,
    "rms_norm_eps": 1e-05,
    "use_sandwich_norm": True,
    "tie_word_embeddings": True,
    "use_cache": True,
    # 4 heads of 64 keep the real 2D-RoPE width; 2 kv heads (2:1 like 16:8) exercise the
    # GQA head mapping
    "vision_hidden_size": 256,
    "vision_intermediate_size": 96,
    "vision_num_hidden_layers": 3,
    "vision_layer_types": ["conv", "full_attention", "conv"],
    "vision_num_attention_heads": 4,
    "vision_num_key_value_heads": 2,
    "vision_num_channels": 3,
    "vision_patch_size": 16,
    "vision_temporal_patch_size": 2,
    "vision_patch_dim": 1536,
    "vision_spatial_merge_size": 2,
    "downsample_factor": 2,
    "vision_rope_theta": 1000000.0,
    "vision_max_position_embeddings": 2304,
    "projector_hidden_size": 64,
    "projector_layer_norm_eps": 1e-06,
    "dtype": "bfloat16",
}


def flat_shapes(cfg: dict = TINY_CONFIG) -> dict[str, tuple[int, ...]]:
    """name -> shape of the flat layout's tensors, in the order their random values are drawn
    (the LM head is the token table's)."""
    h, i, d = cfg["hidden_size"], cfg["intermediate_size"], cfg["head_dim"]
    heads, kv_heads = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    shapes = {"language_model.embed_tokens.weight": (cfg["vocab_size"], h)}
    for n, kind in enumerate(cfg["layer_types"]):
        layer = f"language_model.layers.{n}."
        for norm in ("operator_norm", "post_operator_norm", "ffn_norm", "post_ffn_norm"):
            shapes[f"{layer}{norm}.weight"] = (h,)
        if kind == "conv":
            shapes[f"{layer}conv.conv"] = (h, 1, cfg["conv_L_cache"])
            shapes[f"{layer}conv.in_proj.weight"] = (3 * h, h)
            shapes[f"{layer}conv.out_proj.weight"] = (h, h)
        else:
            shapes[f"{layer}self_attn.qkv.weight"] = ((heads + 2 * kv_heads) * d, h)
            shapes[f"{layer}self_attn.gate_proj.weight"] = (heads, h)
            shapes[f"{layer}self_attn.o_proj.weight"] = (h, heads * d)
            shapes[f"{layer}self_attn.q_norm.weight"] = (d,)
            shapes[f"{layer}self_attn.k_norm.weight"] = (d,)
        shapes[f"{layer}feed_forward.gate_up.weight"] = (2 * i, h)
        shapes[f"{layer}feed_forward.w2.weight"] = (h, i)
    shapes["language_model.embedding_norm.weight"] = (h,)

    vh, vi = cfg["vision_hidden_size"], cfg["vision_intermediate_size"]
    vd = vh // cfg["vision_num_attention_heads"]
    kv_width = cfg["vision_num_key_value_heads"] * vd
    shapes["vision_encoder.patchifier.weight"] = (vh, cfg["vision_patch_dim"])
    shapes["vision_encoder.embedding_norm.weight"] = (vh,)
    for n, kind in enumerate(cfg["vision_layer_types"]):
        layer = f"vision_encoder.layers.{n}."
        shapes[f"{layer}operator_norm.weight"] = (vh,)
        shapes[f"{layer}ffn_norm.weight"] = (vh,)
        if kind == "conv":
            shapes[f"{layer}conv.spatial_conv"] = (vh, 1, 3, 3)
            shapes[f"{layer}conv.in_proj.weight"] = (3 * vh, vh)
            shapes[f"{layer}conv.out_proj.weight"] = (vh, vh)
        else:
            shapes[f"{layer}self_attn.q_proj.weight"] = (vh, vh)
            shapes[f"{layer}self_attn.k_proj.weight"] = (kv_width, vh)
            shapes[f"{layer}self_attn.v_proj.weight"] = (kv_width, vh)
            shapes[f"{layer}self_attn.out_proj.weight"] = (vh, vh)
            shapes[f"{layer}self_attn.q_layernorm.weight"] = (vd,)
            shapes[f"{layer}self_attn.k_layernorm.weight"] = (vd,)
        shapes[f"{layer}feed_forward.gate_up.weight"] = (2 * vi, vh)
        shapes[f"{layer}feed_forward.w2.weight"] = (vh, vi)
    projector = cfg["projector_hidden_size"]
    shapes["vision_projector.layer_norm.weight"] = (vh,)
    shapes["vision_projector.layer_norm.bias"] = (vh,)
    shapes["vision_projector.linear_1.weight"] = (projector, vh * cfg["downsample_factor"] ** 2)
    shapes["vision_projector.linear_1.bias"] = (projector,)
    shapes["vision_projector.linear_2.weight"] = (h, projector)
    shapes["vision_projector.linear_2.bias"] = (h,)
    return shapes


def write_flat_checkpoint(directory: pathlib.Path) -> pathlib.Path:
    """config.json (TINY_CONFIG) and bf16 model.safetensors of a seeded random LFM3-VL, under the
    flat layout's names."""
    import torch
    from safetensors.torch import save_file

    generator = torch.Generator().manual_seed(0)
    state = {}
    for name, shape in flat_shapes().items():
        if name.endswith(("norm.weight", "layernorm.weight", "layer_norm.weight")):
            value = 0.5 + torch.rand(shape, generator=generator)
        elif name.endswith(".bias"):
            value = 0.02 * torch.randn(shape, generator=generator)
        elif name.endswith(("conv.conv", "spatial_conv")):
            value = 0.3 * torch.randn(shape, generator=generator)
        elif name.endswith("embed_tokens.weight"):
            value = torch.randn(shape, generator=generator)
        else:
            value = torch.randn(shape, generator=generator) * shape[-1] ** -0.5
        state[name] = value.to(torch.bfloat16).contiguous()
    directory.mkdir(parents=True, exist_ok=True)
    save_file(state, str(directory / "model.safetensors"), metadata={"format": "pt"})
    (directory / "config.json").write_text(json.dumps(TINY_CONFIG, indent=2))
    return directory


def write_tiny_checkpoint(directory: pathlib.Path) -> pathlib.Path:
    """The tiny LFM3-VL in LFM3-VL-2B's layout, with its remote code (write_port_checkpoint of
    write_flat_checkpoint's, which goes into a sibling folder)."""
    flat = write_flat_checkpoint(directory.with_name(f"{directory.name}-flat"))
    return write_port_checkpoint(flat, directory)


def port_config(cfg: dict = TINY_CONFIG) -> dict:
    """A flat config in the nested layout of LFM3-VL-2B's config.json."""
    tokens = {k: cfg[k] for k in ("bos_token_id", "eos_token_id", "pad_token_id")}
    text_keys = (
        "conv_L_cache",
        "conv_bias",
        "head_dim",
        "hidden_size",
        "intermediate_size",
        "layer_types",
        "max_position_embeddings",
        "num_attention_heads",
        "num_hidden_layers",
        "num_key_value_heads",
        "sliding_window",
        "tie_word_embeddings",
        "use_cache",
        "use_sandwich_norm",
        "vocab_size",
    )
    return {
        "architectures": cfg["architectures"],
        "auto_map": {
            "AutoConfig": "configuration_lfm3_vl.Lfm3VlConfig",
            "AutoModel": "modeling_lfm3_vl.Lfm3VlModel",
            "AutoModelForImageTextToText": "modeling_lfm3_vl.Lfm3VlForConditionalGeneration",
        },
        "model_type": "lfm3_vl",
        **{k: cfg[k] for k in cfg if k.endswith("token_id")},
        "downsample_factor": cfg["downsample_factor"],
        "dtype": cfg["dtype"],
        "projector_bias": True,
        "projector_hidden_act": "gelu",
        "projector_hidden_size": cfg["projector_hidden_size"],
        "projector_layer_norm_eps": cfg["projector_layer_norm_eps"],
        "tie_word_embeddings": True,
        "transformers_version": "5.19.0.dev0",
        "text_config": {
            **tokens,
            **{k: cfg[k] for k in text_keys},
            "dtype": cfg["dtype"],
            "initializer_range": 0.02,
            "model_type": "lfm3",
            "norm_eps": cfg["rms_norm_eps"],
            "rope_parameters": {
                "full_attention": {
                    "partial_rotary_factor": cfg["partial_rotary_factor"],
                    "rope_theta": cfg["full_attention_rope_theta"],
                    "rope_type": "default",
                },
                "sliding_attention": {
                    "rope_theta": cfg["sliding_attention_rope_theta"],
                    "rope_type": "default",
                },
            },
        },
        "vision_config": {
            "conv_kernel_size": 3,
            "dtype": cfg["dtype"],
            "hidden_size": cfg["vision_hidden_size"],
            "in_channels": cfg["vision_num_channels"],
            "initializer_range": 0.02,
            "intermediate_size": cfg["vision_intermediate_size"],
            "layer_types": cfg["vision_layer_types"],
            "model_type": "lfm3_vl_vision",
            "norm_eps": cfg["rms_norm_eps"],
            "num_attention_heads": cfg["vision_num_attention_heads"],
            "num_hidden_layers": cfg["vision_num_hidden_layers"],
            "num_key_value_heads": cfg["vision_num_key_value_heads"],
            "patch_size": cfg["vision_patch_size"],
            "rope_parameters": {"rope_theta": cfg["vision_rope_theta"], "rope_type": "axial"},
            "spatial_merge_size": cfg["vision_spatial_merge_size"],
            "temporal_patch_size": cfg["vision_temporal_patch_size"],
        },
    }


def write_port_checkpoint(flat: pathlib.Path, directory: pathlib.Path) -> pathlib.Path:
    """The flat checkpoint's tensors under LFM3-VL-2B's names (model.* prefixes, q/k/v and w1/w3
    unfused, the convolutions' kernels as .weight), with port_config and the remote code of
    reference/."""
    from safetensors.torch import load_file, save_file

    cfg = json.loads((flat / "config.json").read_text())
    q_size = cfg["num_attention_heads"] * cfg["head_dim"]
    kv_size = cfg["num_key_value_heads"] * cfg["head_dim"]
    prefixes = {
        "language_model.": "model.language_model.",
        "vision_encoder.": "model.vision_tower.",
        "vision_projector.": "model.multi_modal_projector.",
    }
    state = {}
    for name, value in load_file(str(flat / "model.safetensors")).items():
        flat_prefix = next(p for p in prefixes if name.startswith(p))
        name = prefixes[flat_prefix] + name[len(flat_prefix) :]
        if name.endswith(("conv.conv", "spatial_conv")):
            state[f"{name}.weight"] = value
        elif name.endswith("gate_up.weight"):
            base = name[: -len("gate_up.weight")]
            state[f"{base}w1.weight"], state[f"{base}w3.weight"] = value.chunk(2)
        elif name.endswith("qkv.weight"):
            base = name[: -len("qkv.weight")]
            parts = value.split((q_size, kv_size, kv_size))
            for part, tensor in zip(("q_proj", "k_proj", "v_proj"), parts, strict=True):
                state[f"{base}{part}.weight"] = tensor
        else:
            state[name] = value
    directory.mkdir(parents=True, exist_ok=True)
    state = {k: v.contiguous() for k, v in state.items()}
    save_file(state, str(directory / "model.safetensors"), metadata={"format": "pt"})
    (directory / "config.json").write_text(json.dumps(port_config(cfg), indent=2))
    for path in REFERENCE_DIR.glob("*.py"):
        if path.name not in ("__init__.py", "loader.py"):
            shutil.copy2(path, directory / path.name)
    return directory
