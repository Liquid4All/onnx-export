"""
A tiny random LFM3 checkpoint in LFM3-1.7B's format, for tests without a download: LFM3-1.7B's
config.json (assets/, at its pinned revision) with small widths, its tensor names, its modeling
code (the copy vendored in tests/test_lfm3_vl/reference/, which differs from LFM3-1.7B's only in
the checkpoint ids its docstrings name) and a tokenizer with its special token ids and chat
template.
"""

import json
import pathlib
import shutil

from ..test_lfm3_vl.reference.loader import REFERENCE_DIR

ASSETS = pathlib.Path(__file__).parent / "assets"
RELEASED = json.loads((ASSETS / "LFM3-1.7B" / "config.json").read_text())
TINY_CONFIG = RELEASED | {
    "head_dim": 16,
    "hidden_size": 64,
    # Even numbers of 32-wide blocks: onnxruntime's k_quant can't pack the int4 zero points of an
    # odd one
    "intermediate_size": 128,
    "layer_types": RELEASED["layer_types"][:6],
    "max_position_embeddings": 4096,
    "num_attention_heads": 4,
    "num_hidden_layers": 6,
    "num_key_value_heads": 2,
    "sliding_window": 8,
}
REMOTE_CODE = ("configuration_lfm3.py", "modeling_lfm3.py", "lfm3_conv.py")
# LFM3-1.7B's special tokens that its tokenizer_config.json and chat template name
SPECIAL_TOKENS = {
    "<|pad|>": 124893,
    "<|startoftext|>": 124894,
    "<|im_start|>": 124899,
    "<|im_end|>": 124900,
    "<|vision_start|>": 125009,
    "<|vision_end|>": 125010,
    "<|video|>": 125029,
    "<|image|>": 125030,
    "<|audio|>": 125031,
    "<|audio_out|>": 125032,
}
# Added tokens that are not special, so decoding keeps them
THINK_TOKENS = {"<think>": 124901, "</think>": 124902}


def tensor_shapes(cfg: dict) -> dict[str, tuple[int, ...]]:
    """name -> shape of LFM3-1.7B's tensors under a config (no lm_head: it is the token table), in
    the order their random values are drawn."""
    h, i, d = cfg["hidden_size"], cfg["intermediate_size"], cfg["head_dim"]
    heads, kv_heads = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    shapes = {
        "model.embed_tokens.weight": (cfg["vocab_size"], h),
        "model.embedding_norm.weight": (h,),
    }
    for n, kind in enumerate(cfg["layer_types"]):
        layer = f"model.layers.{n}."
        for norm in ("operator_norm", "post_operator_norm", "ffn_norm", "post_ffn_norm"):
            shapes[f"{layer}{norm}.weight"] = (h,)
        if kind == "conv":
            shapes[f"{layer}conv.conv.weight"] = (h, 1, cfg["conv_L_cache"])
            shapes[f"{layer}conv.in_proj.weight"] = (3 * h, h)
            shapes[f"{layer}conv.out_proj.weight"] = (h, h)
        else:
            shapes[f"{layer}self_attn.q_proj.weight"] = (heads * d, h)
            shapes[f"{layer}self_attn.k_proj.weight"] = (kv_heads * d, h)
            shapes[f"{layer}self_attn.v_proj.weight"] = (kv_heads * d, h)
            shapes[f"{layer}self_attn.gate_proj.weight"] = (heads, h)
            shapes[f"{layer}self_attn.o_proj.weight"] = (h, heads * d)
            shapes[f"{layer}self_attn.q_norm.weight"] = (d,)
            shapes[f"{layer}self_attn.k_norm.weight"] = (d,)
        shapes[f"{layer}feed_forward.w1.weight"] = (i, h)
        shapes[f"{layer}feed_forward.w2.weight"] = (h, i)
        shapes[f"{layer}feed_forward.w3.weight"] = (i, h)
    return shapes


def write_tiny_checkpoint(directory: pathlib.Path) -> pathlib.Path:
    """config.json (TINY_CONFIG), bf16 model.safetensors of a seeded random LFM3, its modeling code
    and write_tokenizer's files."""
    import torch
    from safetensors.torch import save_file

    generator = torch.Generator().manual_seed(0)
    state = {}
    for name, shape in tensor_shapes(TINY_CONFIG).items():
        if name.endswith("norm.weight"):
            value = 0.5 + torch.rand(shape, generator=generator)
        elif name.endswith("conv.conv.weight"):
            value = 0.3 * torch.randn(shape, generator=generator)
        elif name.endswith("embed_tokens.weight"):
            value = torch.randn(shape, generator=generator)
        else:
            value = torch.randn(shape, generator=generator) * shape[-1] ** -0.5
        state[name] = value.to(torch.bfloat16)
    directory.mkdir(parents=True, exist_ok=True)
    save_file(state, str(directory / "model.safetensors"), metadata={"format": "pt"})
    (directory / "config.json").write_text(json.dumps(TINY_CONFIG, indent=2))
    for name in REMOTE_CODE:
        shutil.copy2(REFERENCE_DIR / name, directory)
    write_tokenizer(directory)
    return directory


def write_tokenizer(
    directory: pathlib.Path,
    assets: pathlib.Path = ASSETS / "LFM3-1.7B",
    special_tokens: dict[str, int] = SPECIAL_TOKENS,
):
    """A byte-level BPE tokenizer with special_tokens (LFM3-1.7B's) and the think tokens at their
    ids, and the tokenizer_config.json and chat template of assets (LFM3-1.7B's). Text encodes byte
    by byte (no merges); every other id of the model's vocabulary decodes to " t<id>", so a random
    model's answer reads as text."""
    from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers, processors

    added = special_tokens | THINK_TOKENS
    vocab = {char: i for i, char in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    ids = set(added.values())
    placeholders = range(len(vocab), TINY_CONFIG["vocab_size"])
    vocab |= {f"Ġt{i}": i for i in placeholders if i not in ids} | added
    tokenizer = Tokenizer(models.BPE(vocab, []))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    tokenizer.add_tokens([AddedToken(token, normalized=False) for token in THINK_TOKENS])
    tokenizer.add_special_tokens(
        [AddedToken(token, normalized=False, special=True) for token in special_tokens]
    )
    tokenizer.save(str(directory / "tokenizer.json"))
    for name in ("tokenizer_config.json", "chat_template.jinja"):
        shutil.copy2(assets / name, directory)
