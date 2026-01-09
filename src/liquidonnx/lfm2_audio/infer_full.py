#!/usr/bin/env python3
"""
Full CPU inference for LFM2.5-Audio ONNX models supporting all 3 modes:
- ASR (Automatic Speech Recognition): Audio → Text
- TTS (Text-to-Speech): Text → Audio
- Interleaved: Mixed text and audio I/O

Usage:
    # Text generation (existing functionality)
    uv run lfm2-audio-infer-full /path/to/model --prompt "Hello world"

    # ASR: Transcribe audio to text
    uv run lfm2-audio-infer-full /path/to/model --mode asr --audio input.wav

    # TTS: Generate audio from text
    uv run lfm2-audio-infer-full /path/to/model --mode tts --prompt "Hello world" --output output.wav

    # Interleaved: Mixed text and audio
    uv run lfm2-audio-infer-full /path/to/model --mode interleaved --prompt "Respond with audio"
"""

import argparse
import logging
import pathlib
import time

import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)


def get_onnx_files(model_dir: pathlib.Path, precision: str) -> dict[str, pathlib.Path]:
    """Get paths to ONNX model files for given precision."""
    onnx_dir = model_dir / "onnx"
    suffix = "" if precision == "fp32" else f"_{precision}"

    files = {
        "embed_tokens": onnx_dir / f"embed_tokens{suffix}.onnx",
        "audio_embedding": onnx_dir / f"audio_embedding{suffix}.onnx",
        "decoder": onnx_dir / f"decoder{suffix}.onnx",
        "audio_encoder": onnx_dir / f"audio_encoder{suffix}.onnx",
        "depthformer": onnx_dir / f"depthformer{suffix}.onnx",
        "audio_lm_head": onnx_dir / f"audio_lm_head{suffix}.onnx",
    }

    # Fall back to fp32 if requested precision not available
    for name, path in files.items():
        if not path.exists():
            fp32_path = onnx_dir / f"{name}.onnx"
            if fp32_path.exists():
                logger.info(f"{path.name} not found, using {fp32_path.name}")
                files[name] = fp32_path

    return files


def load_session(model_path: pathlib.Path) -> ort.InferenceSession:
    """Load ONNX model as inference session."""
    providers = ["CPUExecutionProvider"]
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(model_path), sess_options, providers=providers)


class LFM2AudioInferenceFull:
    """Full ONNX inference for LFM2.5-Audio supporting all modes."""

    # Special tokens
    AUDIO_START_TOKEN = 65528  # <|audio|>
    AUDIO_END_TOKEN = 65529  # <|/audio|>
    AUDIO_CODE_START = 65536  # Audio codes start here

    def __init__(self, model_dir: pathlib.Path, precision: str = "fp32"):
        self.model_dir = model_dir
        self.precision = precision

        # Load tokenizer
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)

        # Load ONNX sessions
        files = get_onnx_files(model_dir, precision)

        logger.info(f"Loading embed_tokens from {files['embed_tokens'].name}...")
        self.embed_session = load_session(files["embed_tokens"])

        logger.info(f"Loading audio_embedding from {files['audio_embedding'].name}...")
        self.audio_embed_session = load_session(files["audio_embedding"])

        logger.info(f"Loading decoder from {files['decoder'].name}...")
        self.decoder_session = load_session(files["decoder"])

        if files["audio_encoder"].exists():
            logger.info(f"Loading audio_encoder from {files['audio_encoder'].name}...")
            self.audio_encoder_session = load_session(files["audio_encoder"])
        else:
            logger.warning("audio_encoder not found, ASR mode unavailable")
            self.audio_encoder_session = None

        if files["depthformer"].exists():
            logger.info(f"Loading depthformer from {files['depthformer'].name}...")
            self.depthformer_session = load_session(files["depthformer"])
        else:
            logger.warning("depthformer not found, TTS mode may be limited")
            self.depthformer_session = None

        if files["audio_lm_head"].exists():
            logger.info(f"Loading audio_lm_head from {files['audio_lm_head'].name}...")
            self.audio_lm_head_session = load_session(files["audio_lm_head"])
        else:
            logger.warning("audio_lm_head not found, TTS mode may be limited")
            self.audio_lm_head_session = None

        self._load_config()

    def _load_config(self):
        """Load model config from config.json."""
        import json

        config_path = self.model_dir / "config.json"
        with open(config_path) as f:
            config = json.load(f)

        lfm_config = config.get("lfm", {})
        self.hidden_size = lfm_config.get("hidden_size", 2048)
        self.num_layers = lfm_config.get("num_hidden_layers", 16)
        self.num_kv_heads = lfm_config.get("num_key_value_heads", 8)
        self.head_dim = self.hidden_size // lfm_config.get("num_attention_heads", 32)
        self.conv_L = lfm_config.get("conv_L_cache", 3)
        self.layer_types = lfm_config.get("layer_types", [])
        self.vocab_size = lfm_config.get("vocab_size", 65536)

        # Audio config
        self.audio_vocab_size = 16392  # 8 codebooks * 2049
        self.num_codebooks = 8
        self.codebook_vocab = 2049

    def _init_cache(self, batch_size: int = 1) -> dict[str, np.ndarray]:
        """Initialize KV cache for generation."""
        cache = {}

        for idx, layer_type in enumerate(self.layer_types):
            if layer_type == "conv":
                cache[f"past_conv.{idx}"] = np.zeros(
                    (batch_size, self.hidden_size, self.conv_L), dtype=np.float32
                )
            else:
                cache[f"past_key_values.{idx}.key"] = np.zeros(
                    (batch_size, self.num_kv_heads, 0, self.head_dim), dtype=np.float32
                )
                cache[f"past_key_values.{idx}.value"] = np.zeros(
                    (batch_size, self.num_kv_heads, 0, self.head_dim), dtype=np.float32
                )

        return cache

    def _update_cache(self, cache: dict, outputs: dict) -> dict:
        """Update cache with decoder outputs."""
        for key in cache:
            if key.startswith("past_conv."):
                idx = int(key.split(".")[1])
                cache[key] = outputs[f"present_conv.{idx}"]
            elif key.startswith("past_key_values."):
                parts = key.split(".")
                idx = int(parts[1])
                kv_type = parts[2]
                cache[key] = outputs[f"present.{idx}.{kv_type}"]
        return cache

    def _sample(self, logits: np.ndarray, temperature: float, top_p: float) -> int:
        """Sample next token using temperature and top-p sampling."""
        if temperature == 0:
            return int(np.argmax(logits))

        logits = logits / temperature
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()

        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumsum = np.cumsum(sorted_probs)

        cutoff_idx = np.searchsorted(cumsum, top_p) + 1
        top_indices = sorted_indices[:cutoff_idx]
        top_probs = probs[top_indices]
        top_probs = top_probs / top_probs.sum()

        return int(np.random.choice(top_indices, p=top_probs))

    def _get_text_embeds(self, input_ids: np.ndarray) -> np.ndarray:
        """Get text embeddings."""
        return self.embed_session.run(["inputs_embeds"], {"input_ids": input_ids})[0]

    def _get_audio_embeds(self, audio_codes: np.ndarray) -> np.ndarray:
        """Get audio code embeddings."""
        return self.audio_embed_session.run(["audio_embeds"], {"audio_codes": audio_codes})[0]

    def _run_decoder(
        self, embeds: np.ndarray, attention_mask: np.ndarray, cache: dict
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Run decoder and return logits, hidden_states, and updated cache."""
        inputs = {
            "inputs_embeds": embeds.astype(np.float32),
            "attention_mask": attention_mask,
            **cache,
        }

        outputs = self.decoder_session.run(None, inputs)
        output_names = [o.name for o in self.decoder_session.get_outputs()]
        output_dict = dict(zip(output_names, outputs, strict=True))

        logits = output_dict["logits"]
        hidden_states = output_dict.get("hidden_states")
        cache = self._update_cache(cache, output_dict)

        return logits, hidden_states, cache

    def _run_depthformer(self, hidden_states: np.ndarray) -> np.ndarray:
        """Run depthformer to predict 8 codebook logits from hidden states."""
        if self.depthformer_session is None:
            raise RuntimeError("depthformer not loaded")

        # hidden_states: [batch, hidden_size]
        outputs = self.depthformer_session.run(
            ["codebook_logits"], {"hidden_states": hidden_states.astype(np.float32)}
        )
        return outputs[0]  # [batch, 8, 2049]

    def _sample_audio_codes(
        self, codebook_logits: np.ndarray, temperature: float = 0.9
    ) -> np.ndarray:
        """Sample audio codes from depthformer logits."""
        # codebook_logits: [batch, 8, 2049]
        batch_size, num_codebooks, vocab_size = codebook_logits.shape
        codes = np.zeros((batch_size, num_codebooks), dtype=np.int64)

        for cb_idx in range(num_codebooks):
            logits = codebook_logits[:, cb_idx, :]  # [batch, vocab_size]
            for b in range(batch_size):
                codes[b, cb_idx] = self._sample(logits[b], temperature, top_p=0.95)

        return codes

    # === Text Generation ===

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """Generate text from prompt."""
        input_ids = self.tokenizer.encode(prompt, return_tensors="np")
        batch_size, seq_len = input_ids.shape

        embeds = self._get_text_embeds(input_ids)
        cache = self._init_cache(batch_size)

        attention_mask = np.ones((batch_size, seq_len), dtype=np.int64)
        logits, _, cache = self._run_decoder(embeds, attention_mask, cache)

        next_logits = logits[0, -1, : self.vocab_size]
        next_token = self._sample(next_logits, temperature, top_p)

        generated_tokens = [next_token]
        total_len = seq_len + 1

        start_time = time.time()

        for _ in range(max_new_tokens - 1):
            if next_token == self.tokenizer.eos_token_id:
                break

            next_ids = np.array([[next_token]], dtype=np.int64)
            next_embeds = self._get_text_embeds(next_ids)
            attention_mask = np.ones((batch_size, total_len), dtype=np.int64)

            logits, _, cache = self._run_decoder(next_embeds, attention_mask, cache)

            next_logits = logits[0, -1, : self.vocab_size]
            next_token = self._sample(next_logits, temperature, top_p)

            generated_tokens.append(next_token)
            total_len += 1

        elapsed = time.time() - start_time
        tokens_per_sec = len(generated_tokens) / elapsed if elapsed > 0 else 0
        logger.info(
            f"Generated {len(generated_tokens)} tokens in {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s)"
        )

        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # === ASR (Audio → Text) ===

    def transcribe(
        self,
        audio_path: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
    ) -> str:
        """Transcribe audio to text."""
        if self.audio_encoder_session is None:
            raise RuntimeError("audio_encoder not loaded, ASR unavailable")

        # Load and preprocess audio
        import torchaudio

        waveform, sample_rate = torchaudio.load(audio_path)

        # Resample to 16kHz if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
            sample_rate = 16000

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Compute mel spectrogram
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=512,
            hop_length=160,
            n_mels=128,
            power=2.0,
        )
        mel_spec = mel_transform(waveform)
        mel_spec = mel_spec.log2().clamp(min=-10)

        # [1, 128, time] → [1, time, 128]
        mel_features = mel_spec.squeeze(0).transpose(0, 1).unsqueeze(0).numpy()
        mel_lengths = np.array([mel_features.shape[1]], dtype=np.int64)

        # Encode audio
        audio_embeds, _ = self.audio_encoder_session.run(
            ["audio_embeddings", "output_lengths"],
            {"mel_features": mel_features.astype(np.float32), "mel_lengths": mel_lengths},
        )

        # Run decoder
        batch_size = 1
        seq_len = audio_embeds.shape[1]
        cache = self._init_cache(batch_size)

        attention_mask = np.ones((batch_size, seq_len), dtype=np.int64)
        logits, _, cache = self._run_decoder(audio_embeds, attention_mask, cache)

        # Generate text tokens
        next_logits = logits[0, -1, : self.vocab_size]
        next_token = self._sample(next_logits, temperature, top_p=0.9)

        generated_tokens = [next_token]
        total_len = seq_len + 1

        for _ in range(max_new_tokens - 1):
            if next_token == self.tokenizer.eos_token_id:
                break

            next_ids = np.array([[next_token]], dtype=np.int64)
            next_embeds = self._get_text_embeds(next_ids)
            attention_mask = np.ones((batch_size, total_len), dtype=np.int64)

            logits, _, cache = self._run_decoder(next_embeds, attention_mask, cache)

            next_logits = logits[0, -1, : self.vocab_size]
            next_token = self._sample(next_logits, temperature, top_p=0.9)

            generated_tokens.append(next_token)
            total_len += 1

        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # === TTS (Text → Audio) ===

    def synthesize(
        self,
        text: str,
        max_audio_frames: int = 100,
        audio_temperature: float = 0.9,
        text_temperature: float = 0.7,
    ) -> list[np.ndarray]:
        """Synthesize audio from text using depthformer.

        Returns list of audio code frames (8 codes each).
        Each frame is [8] array of codebook indices.
        """
        if self.depthformer_session is None:
            raise RuntimeError("depthformer not loaded, TTS unavailable")

        # Encode the text prompt
        input_ids = self.tokenizer.encode(text, return_tensors="np")
        batch_size, seq_len = input_ids.shape

        # Get text embeddings and run decoder
        embeds = self._get_text_embeds(input_ids)
        cache = self._init_cache(batch_size)

        attention_mask = np.ones((batch_size, seq_len), dtype=np.int64)
        logits, hidden_states, cache = self._run_decoder(embeds, attention_mask, cache)
        total_len = seq_len

        audio_codes = []
        start_time = time.time()

        # Generate audio frames using depthformer
        for frame_idx in range(max_audio_frames):
            # Get hidden states for the last position: [1, hidden_size]
            last_hidden = hidden_states[0, -1:, :]  # [1, hidden_size]

            # Run depthformer to get codebook logits
            codebook_logits = self._run_depthformer(last_hidden)  # [1, 8, 2049]

            # Sample audio codes for all 8 codebooks
            frame_codes = self._sample_audio_codes(codebook_logits, audio_temperature)  # [1, 8]
            audio_codes.append(frame_codes[0])  # [8]

            # Create audio embedding for next step
            # Flatten to single audio token index for embedding lookup
            # The audio_embedding expects a token in range [0, 16392)
            audio_token = 0
            for cb_idx in range(self.num_codebooks):
                audio_token += int(frame_codes[0, cb_idx]) * (self.codebook_vocab ** cb_idx)
            audio_token = min(audio_token, self.audio_vocab_size - 1)

            # Get audio embedding and continue generation
            audio_ids = np.array([[audio_token]], dtype=np.int64)
            next_embeds = self._get_audio_embeds(audio_ids)

            attention_mask = np.ones((batch_size, total_len + 1), dtype=np.int64)
            logits, hidden_states, cache = self._run_decoder(next_embeds, attention_mask, cache)
            total_len += 1

            # Check for end condition (simple heuristic: all codes near zero)
            if frame_idx > 10 and np.max(frame_codes) < 10:
                break

        elapsed = time.time() - start_time
        frames_per_sec = len(audio_codes) / elapsed if elapsed > 0 else 0
        logger.info(
            f"Generated {len(audio_codes)} audio frames in {elapsed:.2f}s "
            f"({frames_per_sec:.1f} frames/s)"
        )
        return audio_codes

    # === Interleaved Mode ===

    def generate_interleaved(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        audio_temperature: float = 0.9,
        text_temperature: float = 0.7,
    ) -> tuple[str, list[np.ndarray]]:
        """Generate interleaved text and audio using depthformer for audio."""
        input_ids = self.tokenizer.encode(prompt, return_tensors="np")
        batch_size, seq_len = input_ids.shape

        embeds = self._get_text_embeds(input_ids)
        cache = self._init_cache(batch_size)

        attention_mask = np.ones((batch_size, seq_len), dtype=np.int64)
        logits, hidden_states, cache = self._run_decoder(embeds, attention_mask, cache)

        text_tokens = []
        audio_codes = []
        total_len = seq_len
        in_audio_mode = False

        for _ in range(max_new_tokens):
            last_logits = logits[0, -1, :]

            if in_audio_mode:
                # Use depthformer to generate audio frame
                if self.depthformer_session is not None and hidden_states is not None:
                    last_hidden = hidden_states[0, -1:, :]
                    codebook_logits = self._run_depthformer(last_hidden)
                    frame_codes = self._sample_audio_codes(codebook_logits, audio_temperature)
                    audio_codes.append(frame_codes[0])

                    # Create audio token for embedding
                    audio_token = 0
                    for cb_idx in range(self.num_codebooks):
                        audio_token += int(frame_codes[0, cb_idx]) * (self.codebook_vocab ** cb_idx)
                    audio_token = min(audio_token, self.audio_vocab_size - 1)

                    # Check for end of audio (heuristic)
                    if len(audio_codes) > 5 and np.max(frame_codes) < 10:
                        in_audio_mode = False
                        continue

                    next_embeds = self._get_audio_embeds(
                        np.array([[audio_token]], dtype=np.int64)
                    )
                else:
                    # Fallback: sample from audio vocabulary
                    audio_logits = last_logits[
                        self.AUDIO_CODE_START : self.AUDIO_CODE_START + self.audio_vocab_size
                    ]
                    token = self._sample(audio_logits, audio_temperature, top_p=0.95)

                    if token < 0 or last_logits[self.AUDIO_END_TOKEN] > last_logits[self.AUDIO_CODE_START + token]:
                        in_audio_mode = False
                        token = self.AUDIO_END_TOKEN
                        next_embeds = self._get_text_embeds(np.array([[token]], dtype=np.int64))
                    else:
                        frame_codes = []
                        remaining = token
                        for _ in range(self.num_codebooks):
                            code = remaining % self.codebook_vocab
                            remaining //= self.codebook_vocab
                            frame_codes.append(code)
                        audio_codes.append(np.array(frame_codes))

                        next_embeds = self._get_audio_embeds(np.array([[token]], dtype=np.int64))
            else:
                # Sample from text vocabulary
                text_logits = last_logits[: self.vocab_size]
                token = self._sample(text_logits, text_temperature, top_p=0.9)

                if token == self.tokenizer.eos_token_id:
                    break

                if token == self.AUDIO_START_TOKEN:
                    in_audio_mode = True

                text_tokens.append(token)
                next_embeds = self._get_text_embeds(np.array([[token]], dtype=np.int64))

            attention_mask = np.ones((batch_size, total_len + 1), dtype=np.int64)
            logits, hidden_states, cache = self._run_decoder(next_embeds, attention_mask, cache)
            total_len += 1

        text_output = self.tokenizer.decode(text_tokens, skip_special_tokens=True)
        logger.info(f"Generated {len(text_tokens)} text tokens, {len(audio_codes)} audio frames")

        return text_output, audio_codes


def audio_codes_to_wav(
    audio_codes: list[np.ndarray],
    output_path: str,
    model_dir: pathlib.Path | None = None,
    sample_rate: int = 24000,
):
    """Convert audio codes to WAV file.

    Tries ONNX-based decoding first (if model_dir provided), then falls back to PyTorch.
    """
    if len(audio_codes) < 2:
        logger.warning("Not enough audio codes to generate audio")
        return False

    # Stack codes: [T, 8] → [8, T]
    codes = np.stack(audio_codes, axis=0)  # [T, 8]
    codes = np.clip(codes, 0, 2047)
    codes_transposed = codes.T  # [8, T]

    # Try ONNX-based decoding
    if model_dir is not None:
        onnx_dir = model_dir / "onnx"
        # Check for audio_detokenizer.onnx (builder version)
        detok_path = onnx_dir / "audio_detokenizer.onnx"
        if not detok_path.exists():
            # Fall back to audio_detokenizer_lfm.onnx (torch version)
            detok_path = onnx_dir / "audio_detokenizer_lfm.onnx"
        istft_config_path = onnx_dir / "istft_config.json"

        if detok_path.exists() and istft_config_path.exists():
            try:
                return _decode_audio_onnx(
                    codes_transposed, detok_path, istft_config_path, output_path, sample_rate
                )
            except Exception as e:
                logger.warning(f"ONNX decode failed: {e}, trying PyTorch fallback")

    # Fallback to PyTorch
    return _decode_audio_pytorch(codes, output_path, sample_rate)


def _decode_audio_onnx(
    codes: np.ndarray,
    detok_path: pathlib.Path,
    istft_config_path: pathlib.Path,
    output_path: str,
    sample_rate: int,
) -> bool:
    """Decode audio using ONNX detokenizer + scipy ISTFT."""
    import json

    import scipy.io.wavfile
    import scipy.signal

    # Load ISTFT config
    with open(istft_config_path) as f:
        istft_config = json.load(f)

    n_fft = istft_config.get("n_fft", 1280)
    hop_length = istft_config.get("hop_length", 320)

    # Load ONNX detokenizer
    detok_session = load_session(detok_path)

    # Run detokenizer: [1, 8, T] → [1, T, 1282]
    codes_batch = codes[np.newaxis, :, :].astype(np.int64)  # [1, 8, T]
    stft_features = detok_session.run(["stft_features"], {"audio_codes": codes_batch})[0]

    # stft_features shape: [1, T, 1282] where 1282 = n_fft//2 + 1 = 641 complex values * 2 (real, imag)
    stft_features = stft_features[0]  # [T, 1282]

    # Split into real and imaginary parts
    n_freqs = n_fft // 2 + 1  # 641
    real_part = stft_features[:, :n_freqs]  # [T, 641]
    imag_part = stft_features[:, n_freqs:]  # [T, 641]

    # Reconstruct complex STFT: [T, 641] → [641, T]
    stft_complex = (real_part + 1j * imag_part).T

    # Load ISTFT window if available
    onnx_dir = detok_path.parent
    window_path = onnx_dir / "istft_window.npy"
    if window_path.exists():
        window = np.load(str(window_path))
    else:
        window = scipy.signal.windows.hann(n_fft, sym=False)

    # Run ISTFT
    _, waveform = scipy.signal.istft(
        stft_complex,
        fs=sample_rate,
        window=window,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        input_onesided=True,
    )

    # Normalize and save
    waveform = waveform.astype(np.float32)
    max_val = np.abs(waveform).max()
    if max_val > 0:
        waveform = waveform / max_val

    # Convert to int16 for WAV
    waveform_int16 = (waveform * 32767).astype(np.int16)
    scipy.io.wavfile.write(output_path, sample_rate, waveform_int16)

    duration = len(waveform) / sample_rate
    logger.info(f"Saved audio to {output_path} ({duration:.2f}s) [ONNX decode]")
    return True


def _decode_audio_pytorch(codes: np.ndarray, output_path: str, sample_rate: int) -> bool:
    """Decode audio using PyTorch LFM2AudioProcessor."""
    try:
        import torch
        import torchaudio
        from liquid_audio import LFM2AudioProcessor

        # codes: [T, 8] → [1, 8, T]
        codes_tensor = torch.tensor(codes.T, dtype=torch.int64).unsqueeze(0)
        codes_tensor = torch.clamp(codes_tensor, 0, 2047)

        # Load processor for decoding
        processor = LFM2AudioProcessor.from_pretrained(
            "LiquidAI/LFM2.5-Audio-1.5B", device="cpu"
        )

        with torch.no_grad():
            waveform = processor.decode(codes_tensor)

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        torchaudio.save(output_path, waveform.float().cpu(), sample_rate)
        duration = waveform.shape[-1] / sample_rate
        logger.info(f"Saved audio to {output_path} ({duration:.2f}s) [PyTorch decode]")
        return True
    except Exception as e:
        logger.error(f"Failed to decode audio with PyTorch: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="LFM2.5-Audio ONNX inference (all modes)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "model_dir",
        type=pathlib.Path,
        help="Path to exported ONNX model directory",
    )
    parser.add_argument(
        "--mode",
        choices=["text", "asr", "tts", "interleaved"],
        default="text",
        help="Inference mode (default: text)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The capital of France is",
        help="Input prompt for text/tts/interleaved modes",
    )
    parser.add_argument(
        "--audio",
        type=str,
        help="Input audio file for ASR mode",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output audio file for TTS mode",
    )
    parser.add_argument(
        "--precision",
        choices=["fp32", "fp16", "q4", "q8"],
        default="fp32",
        help="Model precision to use",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum tokens/frames to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--audio-temperature",
        type=float,
        default=0.9,
        help="Audio sampling temperature",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    logger.info(f"Loading model from {args.model_dir}...")
    model = LFM2AudioInferenceFull(args.model_dir, precision=args.precision)

    if args.mode == "text":
        logger.info("Mode: Text Generation")
        logger.info(f"Prompt: {args.prompt}")
        output = model.generate_text(
            args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        print("\n" + "=" * 60)
        print(f"Input:  {args.prompt}")
        print(f"Output: {output}")
        print("=" * 60)

    elif args.mode == "asr":
        if not args.audio:
            parser.error("ASR mode requires --audio argument")
        logger.info("Mode: ASR (Speech Recognition)")
        logger.info(f"Audio: {args.audio}")
        transcription = model.transcribe(
            args.audio,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        print("\n" + "=" * 60)
        print(f"Audio:         {args.audio}")
        print(f"Transcription: {transcription}")
        print("=" * 60)

    elif args.mode == "tts":
        logger.info("Mode: TTS (Text-to-Speech)")
        logger.info(f"Text: {args.prompt}")
        audio_codes = model.synthesize(
            args.prompt,
            max_audio_frames=args.max_tokens,
            audio_temperature=args.audio_temperature,
            text_temperature=args.temperature,
        )
        print("\n" + "=" * 60)
        print(f"Input: {args.prompt}")
        print(f"Generated {len(audio_codes)} audio frames")

        if args.output and audio_codes:
            if audio_codes_to_wav(audio_codes, args.output, model_dir=args.model_dir):
                print(f"Output: {args.output}")
        print("=" * 60)

    elif args.mode == "interleaved":
        logger.info("Mode: Interleaved")
        logger.info(f"Prompt: {args.prompt}")
        text_output, audio_codes = model.generate_interleaved(
            args.prompt,
            max_new_tokens=args.max_tokens,
            audio_temperature=args.audio_temperature,
            text_temperature=args.temperature,
        )
        print("\n" + "=" * 60)
        print(f"Input:  {args.prompt}")
        print(f"Text:   {text_output}")
        print(f"Audio:  {len(audio_codes)} frames")

        if args.output and audio_codes:
            if audio_codes_to_wav(audio_codes, args.output, model_dir=args.model_dir):
                print(f"Output: {args.output}")
        print("=" * 60)


if __name__ == "__main__":
    main()
