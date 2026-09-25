"""Parity of the LFM2.5-Audio ONNX graphs against liquid-audio's PyTorch modules.

Each test instantiates the reference module from liquid-audio (or transformers' Lfm2Model)
with a small random config, exports the same weights through our builders, and compares
outputs on identical inputs. No checkpoint is downloaded, so this runs in CI and pins the
export to the reference implementation rather than to a particular set of trained weights.

    uv run pytest tests/test_lfm2_audio/test_reference_parity.py -v
"""

import dataclasses
import pathlib

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from torch import nn

from liquidonnx.genai_builder import export_decoder
from liquidonnx.lfm2_audio.builder.config import ConformerConfig
from liquidonnx.lfm2_audio.builder.conformer_builder import ConformerEncoderBuilder
from liquidonnx.lfm2_audio.builder.depthformer_builder import DepthformerUnifiedBuilder
from liquidonnx.lfm2_audio.builder.detokenizer_builder import AudioDetokenizerBuilder
from liquidonnx.lfm2_audio.export import DECODER_OPTIONS
from liquidonnx.lfm2_audio.infer import Detokenizer
from liquidonnx.session import decoder_inputs, initialize_cache, update_cache

from .synthetic import write_checkpoint

liquid_audio = pytest.importorskip("liquid_audio")

from liquid_audio.detokenizer import LFM2AudioDetokenizer  # noqa: E402
from liquid_audio.model.conformer.encoder import (  # noqa: E402
    ConformerEncoder,
    ConformerEncoderConfig,
)
from liquid_audio.model.mlp import MLP  # noqa: E402
from liquid_audio.model.transformer import (  # noqa: E402
    MHA,
    RawLMBackbone,
    SharedEmbedding,
    StandardBlock,
)
from transformers import Lfm2Config, Lfm2Model  # noqa: E402

torch.manual_seed(0)
CODEBOOKS = 8
AUDIO_VOCAB = 2049


def to_numpy(module: nn.Module, prefix: str = "") -> dict[str, np.ndarray]:
    return {prefix + k: v.detach().float().cpu().numpy() for k, v in module.state_dict().items()}


def session(model: onnx.ModelProto, tmp_path: pathlib.Path, name: str) -> ort.InferenceSession:
    path = tmp_path / name
    onnx.save(model, str(path))
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def randomize_(module: nn.Module, std: float = 0.3) -> None:
    """Perturb every parameter (including norm weights left at 1 by init) so parity is not
    trivially satisfied by defaults."""
    with torch.no_grad():
        for p in module.parameters():
            p.add_(torch.randn_like(p) * std * p.abs().mean().clamp(min=0.05))


# === Depthformer ===


class ReferenceDepthformer(nn.Module):
    """The depthformer sub-modules of liquid_audio.model.lfm2_audio.LFM2AudioModel."""

    def __init__(self, input_dim: int, dim: int, layers: int, num_heads: int, gqa_dim: int):
        super().__init__()
        blocks = [
            StandardBlock(MHA(dim, num_heads=num_heads, gqa_dim=gqa_dim)) for _ in range(layers)
        ]
        self.depthformer = RawLMBackbone(blocks, has_embedding=False)
        self.depth_linear = nn.Linear(input_dim, dim * CODEBOOKS)
        self.depth_embeddings = nn.ModuleList(
            [
                SharedEmbedding(dim=dim, vocab_size=AUDIO_VOCAB, tie_embedding=True)
                for _ in range(CODEBOOKS)
            ]
        )

    @torch.no_grad()
    def greedy_frame(self, hidden: torch.Tensor) -> tuple[list[int], list[np.ndarray]]:
        """LFM2AudioModel._sample_audio_frame with greedy decoding, also returning logits."""
        dim = self.depth_linear.out_features // CODEBOOKS
        depthformer_in = self.depth_linear(hidden).view(CODEBOOKS, dim)
        token_emb = torch.zeros(dim)
        cache = None
        tokens, logits = [], []
        for i in range(CODEBOOKS):
            x = depthformer_in[i] + token_emb
            out, cache = self.depthformer.forward_cached(x[None, None, :], cache)
            step_logits = self.depth_embeddings[i].get_logits(out.squeeze())
            token = int(step_logits.argmax())
            tokens.append(token)
            logits.append(step_logits.numpy().copy())
            token_emb = self.depth_embeddings[i](torch.tensor(token)).squeeze()
        return tokens, logits


def onnx_greedy_frame(
    sess: ort.InferenceSession,
    hidden: np.ndarray,
    layers: int,
    kv_heads: int,
    head_dim: int,
    dim: int,
) -> tuple[list[int], list[np.ndarray]]:
    """One frame of the depthformer's 8-step loop, greedy, also returning the logits."""
    past_k = np.zeros((layers, 1, kv_heads, 0, head_dim), dtype=np.float32)
    past_v = np.zeros_like(past_k)
    depth_slices = np.zeros((1, CODEBOOKS, dim), dtype=np.float32)
    prev_token = 0
    tokens, logits = [], []
    for i in range(CODEBOOKS):
        step_logits, slices, past_k, past_v = sess.run(
            None,
            {
                "hidden_states": hidden[None].astype(np.float32),
                "depth_slices_in": depth_slices,
                "step_idx": np.array(i, dtype=np.int64),
                "prev_token": np.array([prev_token], dtype=np.int64),
                "past_keys": past_k,
                "past_values": past_v,
                "seqlens_k": np.array([i], dtype=np.int32),
                "total_seq_len": np.array(i + 1, dtype=np.int32),
            },
        )
        if i == 0:
            depth_slices = slices
        prev_token = int(step_logits[0].argmax())
        tokens.append(prev_token)
        logits.append(step_logits[0])
    return tokens, logits


def test_depthformer_matches_reference(tmp_path):
    input_dim, dim, layers, num_heads, gqa_dim = 48, 128, 2, 4, 2
    ref = ReferenceDepthformer(input_dim, dim, layers, num_heads, gqa_dim).eval()
    randomize_(ref)

    builder = DepthformerUnifiedBuilder()
    builder.input_hidden_size = input_dim
    builder.dim = dim
    builder.num_layers = layers
    builder.num_heads = num_heads
    builder.num_kv_heads = gqa_dim
    builder.head_dim = dim // num_heads
    builder.weights = to_numpy(ref)
    builder.load_weights = lambda _path: None
    sess = session(builder.build("reference"), tmp_path, "vocoder_depthformer.onnx")

    for trial in range(3):
        hidden = torch.randn(input_dim, generator=torch.Generator().manual_seed(trial))
        ref_tokens, ref_logits = ref.greedy_frame(hidden)
        onnx_tokens, onnx_logits = onnx_greedy_frame(
            sess, hidden.numpy(), layers, gqa_dim, dim // num_heads, dim
        )
        for step, (a, b) in enumerate(zip(ref_logits, onnx_logits, strict=True)):
            np.testing.assert_allclose(
                b, a, rtol=1e-4, atol=1e-4, err_msg=f"trial {trial} step {step}"
            )
        assert onnx_tokens == ref_tokens


# === Conformer encoder + adapter ===


class ReferenceEncoder(nn.Module):
    def __init__(self, cfg: ConformerEncoderConfig, hidden: int):
        super().__init__()
        self.conformer = ConformerEncoder(**dataclasses.asdict(cfg))
        self.audio_adapter = MLP(self.conformer._feat_out, hidden, [hidden])


ENCODER_CONFIG = ConformerEncoderConfig(
    feat_in=128,
    feat_out=-1,
    n_layers=2,
    d_model=64,
    subsampling="dw_striding",
    subsampling_factor=8,
    subsampling_conv_channels=32,
    causal_downsampling=False,
    reduction=None,
    reduction_position=None,
    reduction_factor=1,
    ff_expansion_factor=4,
    self_attention_model="rel_pos",
    n_heads=4,
    att_context_size=[-1, -1],
    xscaling=False,
    untie_biases=True,
    pos_emb_max_len=5000,
    conv_kernel_size=9,
    conv_norm_type="batch_norm",
    conv_context_size=None,
    dropout=0.0,
    dropout_pre_encoder=0.0,
    dropout_emb=0.0,
    dropout_att=0.0,
)


def test_encoder_matches_reference(tmp_path):
    hidden = 40
    ref = ReferenceEncoder(ENCODER_CONFIG, hidden).eval()
    randomize_(ref)
    with torch.no_grad():  # non-trivial BatchNorm statistics
        for m in ref.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.running_mean.normal_(0, 0.5)
                m.running_var.uniform_(0.5, 2.0)

    cfg = ConformerConfig(
        feat_in=128,
        d_model=ENCODER_CONFIG.d_model,
        n_layers=ENCODER_CONFIG.n_layers,
        n_heads=ENCODER_CONFIG.n_heads,
        ff_expansion_factor=ENCODER_CONFIG.ff_expansion_factor,
        conv_kernel_size=ENCODER_CONFIG.conv_kernel_size,
        subsampling_conv_channels=ENCODER_CONFIG.subsampling_conv_channels,
        pos_emb_max_len=ENCODER_CONFIG.pos_emb_max_len,
    )
    builder = ConformerEncoderBuilder(cfg, adapter_output_dim=hidden)
    builder.weights = to_numpy(ref)
    builder.load_weights = lambda _path: None
    sess = session(builder.build("reference"), tmp_path, "audio_encoder.onnx")

    mel = torch.randn(2, 200, 128)
    lengths = torch.tensor([200, 133])
    with torch.no_grad():
        enc, enc_len = ref.conformer(mel.mT, lengths)  # [B, D, T'], [B]
        ref_out = ref.audio_adapter(enc.mT)  # [B, T', hidden]

    out, out_len = sess.run(None, {"mel_spectrogram": mel.numpy(), "mel_lengths": lengths.numpy()})

    assert out_len.tolist() == enc_len.tolist()
    for b, n in enumerate(out_len.tolist()):
        np.testing.assert_allclose(out[b, :n], ref_out[b, :n].numpy(), rtol=1e-4, atol=1e-4)


# === Audio detokenizer ===


def test_detokenizer_matches_reference(tmp_path):
    layer_types = ["conv", "conv", "sliding_attention", "conv"]
    lfm_cfg = Lfm2Config(
        hidden_size=512,  # fixed by LFM2AudioDetokenizer (FusedEmbedding(512), Linear(512, 1282))
        num_hidden_layers=len(layer_types),
        num_attention_heads=16,
        num_key_value_heads=8,
        intermediate_size=3328,
        block_auto_adjust_ff_dim=True,
        block_ffn_dim_multiplier=1.0,
        block_multiple_of=256,
        # liquid-audio renames sliding_attention → full_attention and applies the window
        # as an explicit mask; the ONNX graph bakes the same window into the layer.
        layer_types=[t.replace("sliding_attention", "full_attention") for t in layer_types],
        conv_L_cache=3,
        norm_eps=1e-5,
        rope_theta=1e6,
        max_position_embeddings=4096,
        vocab_size=64,
        sliding_window=30,
    )
    ref = LFM2AudioDetokenizer(lfm_cfg).eval()
    randomize_(ref)

    detok_config = {
        "hidden_size": 512,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "output_size": 1282,
        "norm_eps": 1e-5,
        "sliding_window": 30,
        "rope_theta": 1e6,
        "layer_types": layer_types,
    }
    sess = session(
        AudioDetokenizerBuilder(detok_config, to_numpy(ref)).build(),
        tmp_path,
        "audio_detokenizer.onnx",
    )

    codes = torch.randint(0, 2048, (1, CODEBOOKS, 40))  # 240 frames > sliding window of 30
    with torch.no_grad():
        ref_hidden = ref.lfm(
            inputs_embeds=torch.nn.functional.interpolate(
                ref.emb(codes).mT, 6 * codes.shape[2], mode="nearest-exact"
            ).mT,
            attention_mask=_sliding_window_mask(6 * codes.shape[2], ref.sliding_window_size),
            use_cache=False,
        ).last_hidden_state
        ref_stft = ref.lin(ref_hidden)[0].numpy()
        ref_wave = ref(codes)[0].numpy()

    stft = sess.run(None, {"audio_codes": codes.numpy()})[0][0]  # [T*6, 1282]

    # The graph is responsible for the STFT features; assert those tightly.
    np.testing.assert_allclose(stft, ref_stft, rtol=1e-3, atol=1e-5 * np.abs(ref_stft).max())

    # exp() on the log-magnitudes and the numpy-vs-torch ISTFT widen this to ~0.2%.
    wave = Detokenizer(tmp_path / "audio_detokenizer.onnx")(codes[0].T.numpy())

    assert wave.shape == ref_wave.shape
    np.testing.assert_allclose(wave, ref_wave, rtol=0, atol=5e-3 * np.abs(ref_wave).max())


def _sliding_window_mask(length: int, window: int) -> torch.Tensor:
    """LFM2AudioDetokenizer.forward's causal sliding-window mask."""
    idx = torch.arange(length)
    d_idx = idx - idx[:, None]
    return torch.logical_and(d_idx <= 0, d_idx > -window)[None, None, ...]


# === LFM2 decoder ===


def test_decoder_matches_reference(tmp_path):
    lfm = {
        "hidden_size": 64,
        "num_hidden_layers": 3,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "vocab_size": 96,
        "layer_types": ["conv", "full_attention", "conv"],
        "intermediate_size": 160,
        "conv_L_cache": 3,
        "max_position_embeddings": 4096,
        "norm_eps": 1e-5,
        "rope_theta": 1e6,
        "block_auto_adjust_ff_dim": False,
    }
    ref = Lfm2Model(Lfm2Config(**lfm)).eval()
    randomize_(ref)

    checkpoint = write_checkpoint(
        tmp_path / "checkpoint", to_numpy(ref, prefix="lfm."), {"lfm": lfm}
    )
    decoder = export_decoder(str(checkpoint), tmp_path / "export", "decoder.onnx", DECODER_OPTIONS)
    sess = ort.InferenceSession(str(decoder), providers=["CPUExecutionProvider"])
    outputs = [o.name for o in sess.get_outputs()]

    prefill = torch.randn(1, 7, lfm["hidden_size"])
    step = torch.randn(1, 1, lfm["hidden_size"])
    with torch.no_grad():
        first = ref(inputs_embeds=prefill, use_cache=True)
        second = ref(inputs_embeds=step, past_key_values=first.past_key_values, use_cache=True)
        ref_logits = [
            (o.last_hidden_state @ ref.embed_tokens.weight.T).numpy() for o in (first, second)
        ]
        ref_hidden = [o.last_hidden_state.numpy() for o in (first, second)]

    cache = initialize_cache(sess)
    got = []
    for embeds, past in ((prefill, 0), (step, 7)):
        result = sess.run(None, decoder_inputs(embeds.numpy(), cache, past))
        got.append(dict(zip(outputs, result, strict=True)))
        update_cache(cache, result, sess.get_outputs())

    for i in range(2):
        np.testing.assert_allclose(got[i]["hidden_states"], ref_hidden[i], rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(got[i]["logits"], ref_logits[i], rtol=1e-4, atol=1e-4)
