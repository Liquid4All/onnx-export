"""
A tiny random checkpoint in the format of LFM3-Audio-2B (assets/LFM3-Audio-2B), for tests without
a download: the legacy flat config.json (lfm / encoder / depthformer), the trainer's weight names
(with the duplicate depth to_logits heads), a codec in audio_tokenizer/, and the checkpoint's
remote code from reference/ (LFM3-Audio-2B's at revision 9d49701). Real token ids,
codebooks (12 x 2017) and mel / STFT settings; small widths, and windows of 8 so short inputs
reach past them. write_omni_checkpoint makes it an LFM3-Omni one.
"""

import json
import pathlib
import re
import shutil

ASSETS = pathlib.Path(__file__).parent / "assets"
REFERENCE_DIR = pathlib.Path(__file__).parent / "reference"
AUTO_MAP = {
    "AutoConfig": "configuration_lfm3_audio.Lfm3AudioConfig",
    "AutoModel": "modeling_lfm3_audio.Lfm3AudioModel",
    "AutoModelForMultimodalLM": "modeling_lfm3_audio.Lfm3AudioForConditionalGeneration",
}
TINY_CONFIG = {
    "architectures": ["Lfm3AudioForConditionalGeneration"],
    "model_type": "lfm3_audio",
    "format_version": 1,
    "codebooks": 12,
    "audio_vocab_size": 2017,
    "maskgit_steps": 3,
    "interleaved_n_text": 6,
    "interleaved_n_audio": 12,
    "encoder": {
        "dim_in": 128,
        "dim_conv": 16,
        "dim_out": 64,
        "sliding_window": 8,
        "lfm": {
            "model_type": "lfm2",
            "vocab_size": 2,
            "hidden_size": 64,
            # block_auto_adjust_ff_dim: int(2 / 3 * 192) = 128, a multiple of 32
            "intermediate_size": 192,
            "block_auto_adjust_ff_dim": True,
            "block_ffn_dim_multiplier": 1.0,
            "block_multiple_of": 32,
            "num_hidden_layers": 4,
            "layer_types": ["conv", "full_attention", "conv", "full_attention"],
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "conv_L_cache": 3,
            "conv_bias": False,
            "max_position_embeddings": 4096,
            "norm_eps": 1e-05,
            "rope_parameters": {"rope_theta": 1000000.0, "rope_type": "default"},
            "tie_word_embeddings": True,
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": None,
        },
    },
    "depthformer": {
        "model_type": "lfm3",
        "vocab_size": 1,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 3,
        "layer_types": ["full_attention"] * 3,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "max_position_embeddings": 12,
        "norm_eps": 1e-05,
        "conv_L_cache": 7,
        "conv_bias": False,
        "sliding_window": 256,
        "rope_parameters": {
            "full_attention": {
                "rope_type": "default",
                "rope_theta": 1000000.0,
                "partial_rotary_factor": 1.0,
            }
        },
        "use_sandwich_norm": False,
        "use_cache": False,
        "tie_word_embeddings": True,
        "is_causal": False,
        "pad_token_id": None,
        "bos_token_id": None,
        "eos_token_id": None,
    },
    "lfm": {
        "model_type": "lfm3",
        "vocab_size": 128000,
        "hidden_size": 64,
        # Even numbers of 32-wide blocks: onnxruntime's k_quant can't pack the int4 zero points
        # of an odd one
        "intermediate_size": 128,
        "num_hidden_layers": 6,
        "layer_types": ["conv", "conv", "full_attention", "conv", "sliding_attention", "conv"],
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "conv_L_cache": 7,
        "conv_bias": False,
        "sliding_window": 8,
        "rope_parameters": {
            "full_attention": {
                "partial_rotary_factor": 0.5,
                "rope_theta": 1000000.0,
                "rope_type": "default",
            },
            "sliding_attention": {"rope_theta": 10000, "rope_type": "default"},
        },
        "max_position_embeddings": 4096,
        "norm_eps": 1e-05,
        "use_sandwich_norm": True,
        "tie_word_embeddings": True,
        "use_cache": True,
        "bos_token_id": 124894,
        "eos_token_id": 124900,
        "pad_token_id": 124893,
    },
    "auto_map": AUTO_MAP,
}
TINY_CODEC_CONFIG = {
    "architectures": ["LiquidCodecModel"],
    "model_type": "liquid_codec",
    # The only width LiquidCodecConfig.lfm_config takes from the config
    "encoder_dim": 64,
    "encoder_conv_dim": 16,
    "quantizer_dim": 16,
    "fsq_levels": [8, 7, 6, 6],
    "n_q": 12,
    "decoder_window": 8,
    "stft_n_fft": 1280,
    "stft_hop_length": 320,
    "stft_win_length": 1280,
    "stft_frame_stack": 6,
    "sampling_rate": 24000,
    "sample_rate": 24000,
    "frame_samples": 1920,
}
# HF-native prefix -> the trainer's, the inverse of liquidonnx.lfm3_audio.weights.TRAINER_RENAMES
TRAINER_PREFIXES = {
    "model.language_model.": "lfm.model.",
    "model.audio_encoder.downsample.": "encoder.downsample.",
    "model.audio_encoder.lfm.": "encoder.lfm.lfm.",
    "model.audio_projector.layer_norm.": "mlp.model.0.",
    "model.audio_projector.linear_1.": "mlp.model.1.",
    "model.audio_projector.linear_2.": "mlp.model.3.",
    "model.codebook_embedding.": "codebook_embedding.",
    "model.depth_decoder.depth_linear.": "depth_linear.",
    "model.depth_decoder.mask_embedding": "mask_embedding",
    "model.depth_decoder.model.": "depthformer.model.",
    "model.depth_decoder.depth_embeddings.": "depth_embeddings.",
}


def trainer_name(name: str) -> str:
    prefix = next(p for p in TRAINER_PREFIXES if name.startswith(p))
    return TRAINER_PREFIXES[prefix] + name[len(prefix) :]


def random_state(model, seed: int) -> dict:
    """bf16 random values for the model's tensors: norms near 1, small biases, unit-variance
    tables, fan-in scaled matrices; buffers (the iSTFT window) keep their values."""
    import torch

    generator = torch.Generator().manual_seed(seed)
    parameters = dict(model.named_parameters())
    state = {}
    for name, value in model.state_dict().items():
        if name not in parameters:
            state[name] = value.contiguous()
            continue
        shape = value.shape
        if name.endswith(".bias"):
            value = 0.02 * torch.randn(shape, generator=generator)
        elif name.endswith("norm.weight"):
            value = 0.5 + torch.rand(shape, generator=generator)
        elif name.endswith(("embedding.weight", "embed_tokens.weight", "mask_embedding")):
            value = torch.randn(shape, generator=generator)
        else:
            fan_in = value[0].numel() if value.ndim > 1 else shape[-1]
            value = torch.randn(shape, generator=generator) * fan_in**-0.5
        state[name] = value.to(torch.bfloat16).contiguous()
    return state


def write_tiny_checkpoint(directory: pathlib.Path) -> pathlib.Path:
    """config.json, processor_config.json, the remote code, bf16 model.safetensors (trainer names)
    and audio_tokenizer/ of a seeded random lfm3_audio model."""
    from safetensors.torch import save_file

    from .remote import remote_class

    directory.mkdir(parents=True, exist_ok=True)
    for path in REFERENCE_DIR.glob("*.py"):
        shutil.copy2(path, directory / path.name)
    (directory / "config.json").write_text(json.dumps(TINY_CONFIG, indent=2))
    shutil.copy2(ASSETS / "LFM3-Audio-2B" / "processor_config.json", directory)
    codec_dir = directory / "audio_tokenizer"
    codec_dir.mkdir(exist_ok=True)
    (codec_dir / "config.json").write_text(json.dumps(TINY_CODEC_CONFIG, indent=2))

    config_class = remote_class(directory, "configuration_lfm3_audio.Lfm3AudioConfig")
    model_class = remote_class(directory, "modeling_lfm3_audio.Lfm3AudioForConditionalGeneration")
    model = model_class(config_class.from_pretrained(str(directory)))
    state = {}
    for name, value in random_state(model, seed=0).items():
        if name == "lm_head.weight":  # tied to the token table, which the trainer stores alone
            continue
        state[trainer_name(name)] = value
        if re.search(r"depth_embeddings\.\d+\.embedding\.weight$", name):
            # the trainer's export also stores each tied depth head
            state[trainer_name(name).replace(".embedding.", ".to_logits.")] = value.clone()
    save_file(state, str(directory / "model.safetensors"), metadata={"format": "pt"})

    codec_config = remote_class(directory, "configuration_liquid_codec.LiquidCodecConfig")
    codec_class = remote_class(directory, "modeling_liquid_codec.LiquidCodecModel")
    codec = codec_class(codec_config.from_pretrained(str(codec_dir)))
    codec_state = random_state(codec, seed=1)
    save_file(codec_state, str(codec_dir / "model.safetensors"), metadata={"format": "pt"})
    return directory


def write_omni_checkpoint(audio: pathlib.Path, directory: pathlib.Path) -> pathlib.Path:
    """A tiny lfm3_omni checkpoint in LFM3-Omni-2B's layout: the tiny checkpoint audio
    (write_tiny_checkpoint) under the nested config.json, with the vision tower and projector of
    tests/test_lfm3_vl/tiny.py's checkpoint under the transformers port's names, LFM3-Omni-2B's
    processor_config.json and the codec in audio_tokenizer/."""
    from safetensors.torch import load_file, save_file

    from liquidonnx.lfm3_audio.config import OMNI_VISION_KEYS
    from liquidonnx.lfm3_audio.weights import VISION_PREFIXES

    from ..test_lfm3_vl import tiny as vl

    scratch = directory.with_name(f"{directory.name}-vl")
    port = vl.write_tiny_checkpoint(scratch / "port")
    directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(audio / "model.safetensors", directory)
    tensors = load_file(str(port / "model.safetensors"))
    vision = {name: t for name, t in tensors.items() if name.startswith(VISION_PREFIXES)}
    save_file(vision, str(directory / "model-vision.safetensors"), metadata={"format": "pt"})
    shutil.copytree(audio / "audio_tokenizer", directory / "audio_tokenizer")
    shutil.copy2(ASSETS / "LFM3-Omni-2B" / "processor_config.json", directory)
    encoder = TINY_CONFIG["encoder"]
    port_config = json.loads((port / "config.json").read_text())
    config = {
        "architectures": ["Lfm3OmniForConditionalGeneration"],
        "model_type": "lfm3_omni",
        "text_config": TINY_CONFIG["lfm"],
        "audio_encoder_config": {
            "num_mel_bins": encoder["dim_in"],
            "conv_channels": encoder["dim_conv"],
            "hidden_size": encoder["dim_out"],
            "sliding_window": encoder["sliding_window"],
            "text_config": encoder["lfm"],
        },
        "depth_config": TINY_CONFIG["depthformer"],
        **{key: port_config[key] for key in OMNI_VISION_KEYS},
    }
    (directory / "config.json").write_text(json.dumps(config, indent=2))
    return directory
