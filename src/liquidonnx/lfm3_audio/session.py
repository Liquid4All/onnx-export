"""
LFM3-Audio generation on onnxruntime: log-mel features, the speech encoder, the cached decoder
loop with the text / speech mode machine, MaskGIT frame sampling with the depth decoder, and the
codec with a numpy iSTFT. It mirrors Lfm3AudioForConditionalGeneration._generate (remote code of
LiquidAI/LFM3-Audio-2B) for one prompt (batch 1):

    prefill: decoder.onnx(prompt ids, audio_features, audio_out_codes) → logits, hidden (last)
    mode from the prompt's last token: <|audio_start|> or <|audio_out|> → AUDIO,
      <|mixed_start|> → INTERLEAVED, else TEXT (generation_mode only appends the marker the
      requested mode needs; a prompt ending in a replayed end-of-audio frame resumes TEXT)
    each slot (max_new_tokens counts slots: a text token or a frame):
      AUDIO, or INTERLEAVED while speaking → sample_frame(hidden):
          order = orders() (a permutation of the 12 codebooks), split by maskgit_schedule;
          per pass: depth_decoder(hidden, codes, ~revealed) → sample each picked codebook from
          its logits (AudioSampling: repetition penalty over its last repetition_window non-EOS
          codes, temperature, top-k); any code == 2016 → the frame is all 2016
          (end of audio) and the mode becomes TEXT. The next step feeds <|audio_out|> with the
          frame's codes (EOS frames too).
      else → a text token from logits: 125031 and 125032 always suppressed (plus <|mixed_start|>
          with suppress_mixed, <|audio_start|> with suppress_audio_out), repetition penalty over
          the last 32 tokens, greedy unless sampling. <|text_end|> inside an interleaved block
          ends its text (speech goes on until an EOS frame), elsewhere it ends the turn, as
          <|im_end|> does anywhere; outside a block <|audio_start|> → AUDIO and <|mixed_start|> →
          INTERLEAVED (6 text slots, then 12 frames, alternating until the text is done).
    output: generated slots and the frames without EOS → codec_decoder.onnx → numpy iSTFT

The decoder runs through lfm3.session.CachedDecoder (last_outputs["hidden_states"] after each
step), every session through lfm3.session.load_session (CUDA_OPTIONS on CUDA).
"""

import collections
import json
import logging
import math
import pathlib
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from liquidonnx.lfm3.session import (
    PREFERRED,
    CachedDecoder,
    default_ep,
    kv_feeds_attention,
    load_session,
    load_tokenizer,
)
from liquidonnx.lfm3_audio.config import MelConfig, load_config
from liquidonnx.session import initialize_cache

logger = logging.getLogger(__name__)

# precision -> graph role -> file in onnx/. CPU bundles keep the float side graphs fp32 (the
# CPU EP has few fp16 kernels); CUDA bundles take fp16 graphs, as GroupQueryAttention has no
# float32 CUDA kernel. Every bundle takes the fp32 depth decoder: it has no GroupQueryAttention,
# so it runs on CUDA too, and it is the only depth graph that reproduces the reference's greedy
# codes (fp16 and q8 flip a few percent of the argmaxes at near-ties, test_depth.py).
# Side-graph choices change only with numerics evidence.
BUNDLES = {
    "fp32": {
        "decoder": "decoder.onnx",
        "audio_encoder": "audio_encoder.onnx",
        "depth_decoder": "depth_decoder.onnx",
        "codec_decoder": "codec_decoder.onnx",
    },
    "fp16": {
        "decoder": "decoder_fp16.onnx",
        "audio_encoder": "audio_encoder_fp16.onnx",
        "depth_decoder": "depth_decoder.onnx",
        "codec_decoder": "codec_decoder_fp16.onnx",
    },
    "q8": {
        "decoder": "decoder_q8.onnx",
        "audio_encoder": "audio_encoder.onnx",
        "depth_decoder": "depth_decoder.onnx",
        "codec_decoder": "codec_decoder.onnx",
    },
    "q4": {
        "decoder": "decoder_q4.onnx",
        "audio_encoder": "audio_encoder.onnx",
        "depth_decoder": "depth_decoder.onnx",
        "codec_decoder": "codec_decoder.onnx",
    },
    "q8f16": {
        "decoder": "decoder_q8f16.onnx",
        "audio_encoder": "audio_encoder_fp16.onnx",
        "depth_decoder": "depth_decoder.onnx",
        "codec_decoder": "codec_decoder_fp16.onnx",
    },
    "q4f16": {
        "decoder": "decoder_q4f16.onnx",
        "audio_encoder": "audio_encoder_fp16.onnx",
        "depth_decoder": "depth_decoder.onnx",
        "codec_decoder": "codec_decoder_fp16.onnx",
    },
}

# A source of MaskGIT orders: each call returns one permutation of range(num_codebooks) (int64)
OrderSource = Callable[[], np.ndarray]


class Mode(StrEnum):
    TEXT = "text"
    AUDIO = "audio"
    INTERLEAVED = "interleaved"


@dataclass
class TextSampling:
    """Text token sampling; temperature None (the default) is greedy, as do_sample=False."""

    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    min_p: float | None = None
    repetition_penalty: float | None = None
    repetition_window: int = 32


@dataclass
class AudioSampling:
    """Speech code sampling per codebook: the reference generate()'s audio_* settings. The
    defaults are greedy (argmax), as is temperature 0; a top_k without a temperature samples at
    temperature 1 (sample_token). A checkpoint's generation_config.json holds its own settings,
    which lfm3-audio-infer and lfm3-omni-infer take (lfm3_audio.infer.speech_sampling)."""

    temperature: float | None = None
    top_k: int | None = None
    repetition_penalty: float | None = None
    repetition_window: int = 64


@dataclass
class Generation:
    # Generated slots: text token ids, and tokens.audio_out per returned frame
    tokens: list[int] = field(default_factory=list)
    # [F, num_codebooks] int64: the returned frames (end-of-audio frames only with return_audio_eos)
    audio_codes: np.ndarray = field(default_factory=lambda: np.zeros((0, 12), dtype=np.int64))


def available_precisions(
    model_dir: str | pathlib.Path, ep: str = "cpu", bundles: dict = BUNDLES
) -> list[str]:
    """The precisions of bundles (BUNDLES, or lfm3_omni's) whose files the export folder has, in
    lfm3.session.PREFERRED order for ep."""
    onnx_dir = pathlib.Path(model_dir) / "onnx"
    return [
        precision
        for precision in PREFERRED[ep]
        if precision in bundles
        and all((onnx_dir / name).exists() for name in bundles[precision].values())
    ]


# === Features ===


def load_audio(path: str | pathlib.Path, sampling_rate: int) -> np.ndarray:
    """Mono float32 samples [n] of a WAV file at sampling_rate (channels averaged; resampled
    with scipy.signal.resample_poly when the file's rate differs, which is not the reference's
    soxr: feed 16 kHz files for parity)."""
    import scipy.io.wavfile
    import scipy.signal

    rate, samples = scipy.io.wavfile.read(str(path))
    if samples.dtype.kind == "i":  # PCM
        samples = samples / -np.iinfo(samples.dtype).min
    elif samples.dtype.kind != "f":
        raise ValueError(f"{path}: unsupported WAV sample type {samples.dtype}")
    samples = samples.astype(np.float32)
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    if rate != sampling_rate:
        gcd = math.gcd(rate, sampling_rate)
        samples = scipy.signal.resample_poly(samples, sampling_rate // gcd, rate // gcd)
    return samples.astype(np.float32)


def mel_filters(mel: MelConfig) -> np.ndarray:
    """[n_fft // 2 + 1, num_mel_bins] float32: transformers.audio_utils.mel_filter_bank, slaney
    scale and norm, 0 to sampling_rate / 2 Hz (Lfm3AudioFeatureExtractor's)."""
    from transformers.audio_utils import mel_filter_bank

    return mel_filter_bank(
        num_frequency_bins=mel.n_fft // 2 + 1,
        num_mel_filters=mel.num_mel_bins,
        min_frequency=0.0,
        max_frequency=mel.sampling_rate / 2,
        sampling_rate=mel.sampling_rate,
        norm="slaney",
        mel_scale="slaney",
    ).astype(np.float32)


def log_mel(wave: np.ndarray, mel: MelConfig, filters: np.ndarray | None = None) -> np.ndarray:
    """input_features [num_mel_bins, T] float32 of one mono clip [n] at mel.sampling_rate, T =
    mel.num_frames(n): right-padded, |STFT|² (center=False, periodic Hann), mel_filters, log10
    clamped at log_floor, (x + offset) / divisor. Matches the reference extractor to ~1e-6.
    filters: mel_filters(mel), when the caller has them."""
    if filters is None:
        filters = mel_filters(mel)
    wave = np.asarray(wave, dtype=np.float32)
    frames = mel.num_frames(len(wave))
    padded = np.zeros((frames - 1) * mel.hop_length + mel.n_fft, dtype=np.float64)
    padded[: len(wave)] = wave
    # torch.hann_window (periodic, float32), centered in n_fft as torch.stft pads it
    window = np.zeros(mel.n_fft)
    offset = (mel.n_fft - mel.win_length) // 2
    hann = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(mel.win_length) / mel.win_length)
    window[offset : offset + mel.win_length] = hann.astype(np.float32)
    # [T, n_fft] → [T, n_fft // 2 + 1], in float64: torch's float32 STFT is ~1e-7 off either way
    segments = np.lib.stride_tricks.sliding_window_view(padded, mel.n_fft)[:: mel.hop_length]
    power = np.abs(np.fft.rfft(segments * window, axis=-1)) ** 2
    with np.errstate(divide="ignore"):
        features = np.log10(filters.T.astype(np.float64) @ power.T)
    features = np.maximum(features, mel.log_floor)
    return ((features + mel.normalization_offset) / mel.normalization_divisor).astype(np.float32)


# === Sampling ===


def rng_orders(seed: int | None = None, num_codebooks: int = 12) -> OrderSource:
    """MaskGIT orders numpy.random.default_rng(seed).permutation(num_codebooks), one per call:
    the orders the tests inject into the reference's torch.randperm."""
    rng = np.random.default_rng(seed)
    return lambda: rng.permutation(num_codebooks)


def softmax(scores: np.ndarray) -> np.ndarray:
    scores = scores.astype(np.float64)
    exp = np.exp(scores - scores.max())
    return exp / exp.sum()


def sample_token(
    logits: np.ndarray,
    temperature: float | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    min_p: float | None = None,
    repetition_penalty: float | None = None,
    history: np.ndarray | None = None,
    suppress: tuple[int, ...] = (),
    rng: np.random.Generator | None = None,
) -> int:
    """One id from logits [V] as Lfm3AudioForConditionalGeneration._sample: the repetition
    penalty (transformers' RepetitionPenaltyLogitsProcessor) over history, suppress set to -inf,
    then argmax without temperature and top_k (or temperature 0), else the temperature, top-k,
    top-p and min-p warpers and a draw from rng."""
    scores = np.array(logits, dtype=np.float32)
    if repetition_penalty is not None and repetition_penalty != 1.0 and history is not None:
        index = np.asarray(history, dtype=np.int64).reshape(-1)
        picked = scores[index]
        scores[index] = np.where(
            picked < 0, picked * repetition_penalty, picked / repetition_penalty
        )
    if suppress:
        scores[list(suppress)] = -np.inf
    # _build_warpers: an explicit zero temperature is greedy; top_k alone samples at temperature 1
    if temperature == 0 or (temperature is None and not top_k):
        return int(np.argmax(scores))
    if temperature:
        scores = scores / np.float32(temperature)
    if top_k:
        kth = np.partition(scores, -min(top_k, len(scores)))[-min(top_k, len(scores))]
        scores[scores < kth] = -np.inf
    if top_p is not None and top_p < 1.0:
        order = np.argsort(scores, kind="stable")
        cumulative = np.cumsum(softmax(scores[order]))
        remove = cumulative <= 1 - top_p
        remove[-1] = False
        scores[order[remove]] = -np.inf
    if min_p:
        probs = softmax(scores)
        scores[probs < min_p * probs.max()] = -np.inf
    rng = rng if rng is not None else np.random.default_rng()
    return int(rng.choice(len(scores), p=softmax(scores)))


# === Speech out ===


def stft_to_wave(stft_features: np.ndarray, n_fft: int, hop_length: int) -> np.ndarray:
    """Waveform [rows * hop_length] float32 of codec_decoder.onnx's stft_features [rows,
    n_fft + 2] of one utterance: exp(log-magnitude) · e^(i·angle), then
    lfm2_audio.infer.istft_same_padding (irfft, periodic Hann, overlap-add, window² envelope,
    trim (n_fft - hop) / 2 at each end)."""
    from liquidonnx.lfm2_audio.infer import istft_same_padding

    bins = n_fft // 2 + 1
    spectrum = np.exp(stft_features[:, :bins]) * np.exp(1j * stft_features[:, bins:])
    return istft_same_padding(spectrum.T, n_fft, hop_length).astype(np.float32)


class ONNXAudioModel:
    # precision -> graph role -> file; load_sessions sets one session attribute per role
    bundles = BUNDLES

    def __init__(
        self,
        model_dir: str | pathlib.Path,
        precision: str | None = None,
        force_cpu: bool = False,
        tf32: bool = True,
    ):
        """precision: a BUNDLES key; by default the first exported one of PREFERRED for the
        execution provider the sessions load on. tf32: let the CUDA EP run the fp32 decoder,
        encoder and codec in TF32 (faster, ~1e-3 off; the tests turn it off); the depth decoder
        always runs in fp32. Parses config.load_config(model_dir) into self.config
        (LFM3AudioConfig)."""
        self.model_dir = pathlib.Path(model_dir)
        self.requested_precision = precision
        self.precision = precision
        self.force_cpu = force_cpu
        self.tf32 = tf32
        self.config = load_config(self.model_dir)
        self.hidden = self.config.language.hidden_size
        tokens = self.config.tokens
        # decoder input -> its placeholder token: Lfm3*ForConditionalGeneration's
        # _placeholder_token_ids, to which lfm3_omni adds image and video
        raw = json.loads((self.model_dir / "config.json").read_text())
        self.feature_tokens = {"audio_features": tokens.audio, "audio_out_codes": tokens.audio_out}
        for name in ("image", "video"):
            if f"{name}_token_id" in raw:
                self.feature_tokens[f"{name}_features"] = raw[f"{name}_token_id"]
        self.placeholders = tuple(self.feature_tokens.values())

    def load(self):
        """load_sessions() and load_preprocessing()."""
        self.load_sessions()
        self.load_preprocessing()

    def load_sessions(self, ep: str | None = None):
        """The sessions of the bundle on ep, one per role of self.bundles: self.decoder,
        self.audio_encoder, self.depth_decoder, self.codec_decoder."""
        ep = ep or default_ep(self.force_cpu)
        self.precision = self.requested_precision or next(
            iter(available_precisions(self.model_dir, ep, self.bundles)), None
        )
        if self.precision is None:
            raise FileNotFoundError(f"no exported bundle in {self.model_dir / 'onnx'}")
        files = {
            role: self.model_dir / "onnx" / name
            for role, name in self.bundles[self.precision].items()
        }
        for role, path in files.items():
            # TF32 flips the fp32 depth decoder's near-tie argmaxes, the codes it is fp32 to keep
            tf32 = self.tf32 and role != "depth_decoder"
            setattr(self, role, load_session(path, ep=ep, tf32=tf32))
        self.share_kv = kv_feeds_attention(files["decoder"])

    def load_preprocessing(self):
        """self.tokenizer (lfm3.session.load_tokenizer) and the mel filters."""
        self.tokenizer = load_tokenizer(self.model_dir)
        self.mel_filters = mel_filters(self.config.mel)

    def encode_audio(self, waves: list[np.ndarray]) -> np.ndarray:
        """audio_features [sum of ceil(T_i / 8), H] of 16 kHz mono clips: log_mel and one
        audio_encoder.onnx run per clip, concatenated in order."""
        features = [np.zeros((0, self.hidden), dtype=np.float32)]
        for wave in waves:
            mel = log_mel(wave, self.config.mel, self.mel_filters)
            [out] = self.audio_encoder.run(["audio_features"], {"input_features": mel[None]})
            if len(out) != self.config.mel.num_tokens(mel.shape[1]):
                raise ValueError(f"{mel.shape[1]} mel frames gave {len(out)} audio features")
            features.append(out.astype(np.float32))
        return np.concatenate(features)

    def prompt_ids(
        self,
        messages: list[dict],
        audio_token_counts: list[int],
        add_generation_prompt: bool = True,
        continue_final_message: bool = False,
    ) -> list[int]:
        """The checkpoint's chat template (messages' audio items as {"type": "audio"}), with each
        clip's <|audio|> expanded to its token count, as Lfm3AudioProcessor does. TTS and S2S end
        an assistant message with <|audio_start|> / <|mixed_start|> (continue_final_message)."""
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
        )
        ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        audio = self.config.tokens.audio
        if ids.count(audio) != len(audio_token_counts):
            raise ValueError(f"prompt has {ids.count(audio)} clips, got {len(audio_token_counts)}")
        counts = iter(audio_token_counts)
        return [t for token in ids for t in ([token] * next(counts) if token == audio else [token])]

    def check_inputs(self, ids: list[int], side: dict[str, np.ndarray]):
        """Each input of side (side_inputs) has a row per placeholder, and codes are in range."""
        for name, token in self.feature_tokens.items():
            rows = len(side[name])
            if ids.count(token) != rows:
                raise ValueError(f"{ids.count(token)} placeholders {token}, {rows} {name} rows")
        audio_out_codes = side["audio_out_codes"]
        if audio_out_codes.size and not (
            0 <= audio_out_codes.min() and audio_out_codes.max() < self.config.codebook_size
        ):
            raise ValueError(f"audio_out_codes must be in [0, {self.config.codebook_size})")

    def side_inputs(
        self, audio_features=None, audio_out_codes=None, image_features=None, video_features=None
    ) -> dict[str, np.ndarray]:
        """The decoder's feature and code inputs, zero rows when absent (image and video
        features: lfm3_omni's decoder)."""

        def rows(features):
            none = np.zeros((0, self.hidden), dtype=np.float32)
            return none if features is None else features.astype(np.float32)

        codes = np.zeros((0, self.config.num_codebooks), dtype=np.int64)
        return {
            "image_features": rows(image_features),
            "video_features": rows(video_features),
            "audio_features": rows(audio_features),
            "audio_out_codes": codes
            if audio_out_codes is None
            else audio_out_codes.astype(np.int64),
        }

    def prefill_outputs(
        self,
        ids: list[int],
        audio_features: np.ndarray | None = None,
        audio_out_codes: np.ndarray | None = None,
        logits_to_keep: int = 1,
        image_features: np.ndarray | None = None,
        video_features: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """(logits [K, V], hidden_states [K, H]) of the last K positions of one uncached decoder
        run; checks placeholder counts and code ranges first."""
        side = self.side_inputs(audio_features, audio_out_codes, image_features, video_features)
        self.check_inputs(ids, side)
        names = {i.name for i in self.decoder.get_inputs()}
        feed = {
            "input_ids": np.array([ids], dtype=np.int64),
            "attention_mask": np.ones((1, len(ids)), dtype=np.int64),
            "position_ids": np.arange(len(ids), dtype=np.int64)[None],
            "logits_to_keep": np.array([logits_to_keep], dtype=np.int64),
            **{name: value for name, value in side.items() if name in names},
            **initialize_cache(self.decoder),
        }
        logits, hidden = self.decoder.run(["logits", "hidden_states"], feed)
        return logits[0].astype(np.float32), hidden[0].astype(np.float32)

    def cached_decoder(self, capacity: int) -> CachedDecoder:
        """The decoder steps of one generation (tests plug a PyTorch stand-in in here)."""
        return CachedDecoder(self.decoder, capacity if self.share_kv else None)

    def depth_logits(self, hidden: np.ndarray, codes: np.ndarray, masked: np.ndarray) -> np.ndarray:
        """depth_decoder.onnx: hidden [B, H] float32, codes [B, 12] int64, masked [B, 12] bool →
        logits [B, 12, codebook_size] float32."""
        feed = {
            "hidden_states": hidden.astype(np.float32),
            "codes": codes.astype(np.int64),
            "masked": masked.astype(bool),
        }
        return self.depth_decoder.run(["logits"], feed)[0].astype(np.float32)

    def sample_frame(
        self,
        hidden: np.ndarray,
        sampling: AudioSampling,
        orders: OrderSource,
        history: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Codes [12] int64 of one frame from hidden [H] (sample_audio_frame): one order from
        orders, len(maskgit_schedule) depth passes; all audio_eos_code if any code is."""
        config = self.config
        order = np.asarray(orders(), dtype=np.int64)
        if sorted(order.tolist()) != list(range(config.num_codebooks)):
            raise ValueError(f"MaskGIT order {order} is not a permutation of the codebooks")
        codes = np.zeros(config.num_codebooks, dtype=np.int64)
        revealed = np.zeros(config.num_codebooks, dtype=bool)
        for pick in np.split(order, np.cumsum(config.maskgit_schedule)[:-1]):
            logits = self.depth_logits(hidden[None], codes[None], ~revealed[None])[0]
            for k in pick.tolist():
                codes[k] = sample_token(
                    logits[k],
                    sampling.temperature,
                    sampling.top_k,
                    repetition_penalty=sampling.repetition_penalty,
                    history=None if history is None else history[:, k],
                    rng=rng,
                )
            revealed[pick] = True
        if (codes == config.audio_eos_code).any():
            codes[:] = config.audio_eos_code
        return codes

    def generate_ids(
        self,
        input_ids: list[int],
        audio_features: np.ndarray | None = None,
        audio_out_codes: np.ndarray | None = None,
        max_new_tokens: int = 1024,
        generation_mode: Mode | str | None = None,
        text_sampling: TextSampling | None = None,
        audio_sampling: AudioSampling | None = None,
        suppress_mixed: bool = False,
        suppress_audio_out: bool = False,
        return_audio_eos: bool = False,
        orders: OrderSource | None = None,
        seed: int | None = None,
        on_token: Callable[[int], None] | None = None,
        on_frame: Callable[[np.ndarray], None] | None = None,
        image_features: np.ndarray | None = None,
        video_features: np.ndarray | None = None,
    ) -> Generation:
        """The generation loop of the module docstring after input_ids, whose <|audio|> and
        <|audio_out|> placeholders take audio_features' and audio_out_codes' rows (lfm3_omni's
        <|image|> and <|video|> image_features' and video_features'). orders
        defaults to rng_orders(seed); seed also seeds the sampling rng. on_token gets each text
        token, on_frame each non-EOS frame [12]."""
        config, tokens = self.config, self.config.tokens
        text = text_sampling or TextSampling()
        audio = audio_sampling or AudioSampling()
        side = self.side_inputs(audio_features, audio_out_codes, image_features, video_features)
        ids, prompt_codes = list(input_ids), side["audio_out_codes"]
        self.check_inputs(ids, side)
        if orders is None:
            orders = rng_orders(seed, config.num_codebooks)
        rng = np.random.default_rng(seed)
        suppress = (
            *self.placeholders,
            *([tokens.mixed_start] if suppress_mixed else []),
            *([tokens.audio_start] if suppress_audio_out else []),
        )

        # === 1. Start mode ===
        if generation_mode is None:
            mode = {
                tokens.audio_start: Mode.AUDIO,
                tokens.audio_out: Mode.AUDIO,
                tokens.mixed_start: Mode.INTERLEAVED,
            }.get(ids[-1], Mode.TEXT)
            # a replayed end-of-audio frame resumes the text
            if ids[-1] == tokens.audio_out and (prompt_codes[-1] == config.audio_eos_code).any():
                mode = Mode.TEXT
        else:
            mode = Mode(generation_mode)
            markers = {
                Mode.AUDIO: (tokens.audio_start, tokens.audio_out),
                Mode.INTERLEAVED: (tokens.mixed_start,),
            }
            if mode in markers and ids[-1] not in markers[mode]:
                ids.append(markers[mode][0])

        # === 2. Prefill ===
        decoder = self.cached_decoder(len(ids) + max_new_tokens)
        no_codes = np.zeros((0, config.num_codebooks), dtype=np.int64)
        logits = decoder.step(ids, logits_to_keep=np.array([1], dtype=np.int64), **side)
        hidden = decoder.last_outputs["hidden_states"].numpy()[0, -1]
        side = self.side_inputs()

        # === 3. Slots ===
        out, frames = Generation(), []
        text_history = collections.deque(maxlen=text.repetition_window)
        audio_history = collections.deque(maxlen=audio.repetition_window)
        slots_left, speaking, text_done, finished = config.interleaved_n_text, False, False, False
        for step in range(max_new_tokens):
            interleaving = mode is Mode.INTERLEAVED
            if mode is Mode.AUDIO or (interleaving and speaking):
                history = np.stack(audio_history) if audio_history else None
                frame = self.sample_frame(hidden, audio, orders, history, rng)
                is_eos = frame[0] == config.audio_eos_code
                if is_eos:
                    mode = Mode.TEXT
                else:
                    audio_history.append(frame)
                    if on_frame:
                        on_frame(frame)
                if not is_eos or return_audio_eos:
                    frames.append(frame)
                    out.tokens.append(tokens.audio_out)
                next_ids, next_codes = [tokens.audio_out], frame[None]
            else:
                token = sample_token(
                    logits[0, -1],
                    text.temperature,
                    text.top_k,
                    text.top_p,
                    text.min_p,
                    text.repetition_penalty,
                    np.array(text_history, dtype=np.int64) if text_history else None,
                    suppress,
                    rng,
                )
                text_history.append(token)
                out.tokens.append(token)
                if on_token:
                    on_token(token)
                if token == tokens.text_end:
                    # ends an interleaved block's text (its speech goes on), elsewhere the turn
                    if interleaving:
                        text_done = speaking = True
                    else:
                        finished = True
                elif token == tokens.eos:
                    finished = True
                elif interleaving:
                    pass  # inside a block the mode markers are ordinary text
                elif token == tokens.audio_start:
                    mode = Mode.AUDIO
                elif token == tokens.mixed_start:
                    mode = Mode.INTERLEAVED
                    slots_left, speaking, text_done = config.interleaved_n_text, False, False
                next_ids, next_codes = [token], no_codes
            if interleaving and not text_done:
                slots_left -= 1
                if slots_left == 0:
                    speaking = not speaking
                    slots_left = (
                        config.interleaved_n_audio if speaking else config.interleaved_n_text
                    )
            # the last slot needs no hidden state
            if finished or step == max_new_tokens - 1:
                break
            logits = decoder.step(
                next_ids,
                logits_to_keep=np.array([1], dtype=np.int64),
                **{**side, "audio_out_codes": next_codes},
            )
            hidden = decoder.last_outputs["hidden_states"].numpy()[0, -1]
        if frames:
            out.audio_codes = np.stack(frames).astype(np.int64)
        return out

    def decode_audio(self, codes: np.ndarray) -> np.ndarray:
        """24 kHz waveform [F * 1920] float32 of frames [F, 12] (codes < audio_eos_code):
        codec_decoder.onnx and stft_to_wave."""
        config = self.config
        codes = np.asarray(codes, dtype=np.int64)
        if codes.ndim != 2 or codes.shape[1] != config.num_codebooks:
            raise ValueError(f"expected frames [F, {config.num_codebooks}], got {codes.shape}")
        if codes.size and not (0 <= codes.min() and codes.max() < config.audio_eos_code):
            raise ValueError(
                f"codec codes must be in [0, {config.audio_eos_code}): drop EOS frames"
            )
        if not len(codes):
            return np.zeros(0, dtype=np.float32)
        [features] = self.codec_decoder.run(["stft_features"], {"audio_codes": codes.T[None]})
        return stft_to_wave(features[0], config.codec_n_fft, config.codec_hop_length)
