#!/usr/bin/env python3
"""
ONNX inference for LFM2.5-Audio supporting all 3 modes:
- ASR (Automatic Speech Recognition): Audio → Text
- TTS (Text-to-Speech): Text → Audio
- Interleaved: Mixed text and audio I/O

Uses ONNX models:
- decoder.onnx: LFM2 backbone (embeddings → logits/hidden_states)
- audio_encoder.onnx: Conformer encoder for ASR
- audio_embedding.onnx: Audio code embeddings for TTS
- audio_detokenizer.onnx: Audio codes → STFT features for waveform synthesis
- vocoder_depthformer.onnx: Autoregressive audio codebook prediction (8× per frame)

All components use ONNX-only inference.

Usage:
    # Text generation
    uv run lfm2-audio-infer /path/to/model --prompt "Hello world"

    # ASR: Transcribe audio to text
    uv run lfm2-audio-infer /path/to/model --mode asr --audio input.wav

    # TTS: Generate audio from text
    uv run lfm2-audio-infer /path/to/model --mode tts --prompt "Hello world" --output output.wav

    # Interleaved with text prompt
    uv run lfm2-audio-infer /path/to/model --mode interleaved --prompt "Respond with audio"

    # Interleaved with audio input (single turn)
    uv run lfm2-audio-infer /path/to/model --mode interleaved --audio input.wav --output output.wav

    # Interactive chat mode (multi-turn with stateful KV cache)
    uv run lfm2-audio-infer /path/to/model --mode interleaved --chat --output output.wav
    # Commands:
    #   /audio <file> [text] - Send audio with optional text
    #   <text>               - Send text message
    #   reset                - Clear conversation
    #   quit                 - Exit

    # Chat with initial audio
    uv run lfm2-audio-infer /path/to/model --mode interleaved --chat \\
        --audio input.wav --prompt "Please respond briefly" --output output.wav
"""

import argparse
import logging
import pathlib
import time

import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)

# === Default System Prompts ===
DEFAULT_SYSTEM_PROMPT_ASR = "Perform ASR."
DEFAULT_SYSTEM_PROMPT_TTS = "Perform TTS. Use the UK female voice."
DEFAULT_SYSTEM_PROMPT_INTERLEAVED = "Respond with interleaved text and audio."

# Max tokens defaults (matching liquid-audio)
# Each audio frame = 80ms (6x upsampling in detokenizer, 320 hop, 24kHz)
# 1024 frames ≈ 82 seconds of audio
DEFAULT_MAX_TOKENS_AUDIO = 1024  # TTS and interleaved modes
DEFAULT_MAX_TOKENS_TEXT = 100  # ASR and text modes


def resolve_precision_files(precision: str | None) -> dict[str, str | None]:
    """Resolve file names from precision shorthand.

    Args:
        precision: One of "fp16", "q4", "q8", or None for default (fp32)

    Returns:
        Dict mapping component name to filename (or None for default)
    """
    if precision is None:
        return {
            "decoder": None,
            "audio_embedding": None,
            "audio_encoder": None,
            "audio_detokenizer": None,
            "vocoder_depthformer": None,
        }

    precision = precision.lower()
    if precision not in ("fp16", "q4", "q8"):
        raise ValueError(f"Invalid precision: {precision}. Use fp16, q4, or q8.")

    return {
        "decoder": f"decoder_{precision}.onnx",
        "audio_embedding": f"audio_embedding_{precision}.onnx",
        "audio_encoder": f"audio_encoder_{precision}.onnx",
        "audio_detokenizer": f"audio_detokenizer_{precision}.onnx",
        "vocoder_depthformer": f"vocoder_depthformer_{precision}.onnx",
    }


def load_session(model_path: pathlib.Path) -> ort.InferenceSession:
    """Load ONNX model as inference session."""
    providers = ["CPUExecutionProvider"]
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(model_path), sess_options, providers=providers)


def load_embed_tokens_weight(onnx_dir: pathlib.Path) -> np.ndarray:
    """Load embed_tokens.weight from exported binary file.

    Why load from binary instead of extracting from decoder.onnx?

    While Python CAN extract weights via `onnx.load()` + graph.initializer,
    JavaScript CANNOT - ONNX Runtime Web only exposes inference APIs.

    By loading from the same binary file, both Python and JS use identical
    artifacts and code paths, ensuring consistent behavior across platforms.

    Files:
        embed_tokens.bin - raw float32 binary [vocab_size * hidden_size]
        embed_tokens.json - metadata {vocab_size, hidden_size, dtype}

    Falls back to extracting from decoder.onnx for backwards compatibility.
    """
    import json

    bin_path = onnx_dir / "embed_tokens.bin"
    meta_path = onnx_dir / "embed_tokens.json"

    if bin_path.exists() and meta_path.exists():
        # Load from binary (same as JavaScript)
        with open(meta_path) as f:
            meta = json.load(f)

        weight = np.fromfile(bin_path, dtype=np.float32)
        weight = weight.reshape(meta["vocab_size"], meta["hidden_size"])
        logger.info(f"Loaded embed_tokens from {bin_path.name}: {weight.shape}")
        return weight

    # Fallback: extract from decoder.onnx (for backwards compatibility)
    logger.warning("embed_tokens.bin not found, extracting from decoder.onnx")
    import onnx

    decoder_path = onnx_dir / "decoder.onnx"
    model = onnx.load(str(decoder_path), load_external_data=True)

    for initializer in model.graph.initializer:
        if initializer.name == "model.embed_tokens.weight":
            return onnx.numpy_helper.to_array(initializer)

    raise ValueError("embed_tokens.weight not found")


def load_audio_embedding_weight(onnx_dir: pathlib.Path) -> np.ndarray | None:
    """Load audio_embedding.weight from exported binary file.

    Returns None if binary file not found (falls back to ONNX model).

    Files:
        audio_embedding.bin - raw float32 binary [vocab_size * hidden_size]
        audio_embedding.json - metadata {vocab_size, hidden_size, num_codebooks, ...}
    """
    import json

    bin_path = onnx_dir / "audio_embedding.bin"
    meta_path = onnx_dir / "audio_embedding.json"

    if not (bin_path.exists() and meta_path.exists()):
        return None

    with open(meta_path) as f:
        meta = json.load(f)

    weight = np.fromfile(bin_path, dtype=np.float32)
    weight = weight.reshape(meta["vocab_size"], meta["hidden_size"])
    logger.info(f"Loaded audio_embedding from {bin_path.name}: {weight.shape}")
    return weight


class LFM2AudioInference:
    """ONNX inference for LFM2.5-Audio supporting all modes."""

    # Special tokens (from tokenizer)
    IM_END_TOKEN = 7  # <|im_end|>
    AUDIO_START_TOKEN = 128  # <|audio_start|>
    TEXT_START_TOKEN = 129  # <|text_start|>
    TEXT_END_TOKEN = 130  # <|text_end|>
    MIXED_START_TOKEN = 131  # <|mixed_start|>
    MIXED_END_TOKEN = 132  # <|mixed_end|>

    # Interleaved mode switching thresholds (matching liquid-audio)
    INTERLEAVED_N_TEXT = 6  # Text tokens before switching to audio
    INTERLEAVED_N_AUDIO = 12  # Audio frames before switching to text

    def __init__(
        self,
        model_dir: pathlib.Path,
        decoder_file: str | None = None,
        audio_embedding_file: str | None = None,
        audio_encoder_file: str | None = None,
        audio_detokenizer_file: str | None = None,
        vocoder_depthformer_file: str | None = None,
    ):
        self.model_dir = model_dir
        self.onnx_dir = model_dir / "onnx"

        # Store file name for vocoder loading
        self._vocoder_depthformer_file = vocoder_depthformer_file

        # Load tokenizer
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)

        # Resolve file paths (use provided or default)
        decoder_path = self.onnx_dir / (decoder_file or "decoder.onnx")
        audio_embedding_path = self.onnx_dir / (audio_embedding_file or "audio_embedding.onnx")
        audio_encoder_path = self.onnx_dir / (audio_encoder_file or "audio_encoder.onnx")
        audio_detokenizer_path = self.onnx_dir / (
            audio_detokenizer_file or "audio_detokenizer.onnx"
        )

        logger.info(f"Loading decoder from {decoder_path.name}...")
        self.decoder_session = load_session(decoder_path)

        # Load embed_tokens.weight for text embedding lookup
        logger.info("Loading embed_tokens.weight...")
        self.embed_tokens_weight = load_embed_tokens_weight(self.onnx_dir)

        # Try loading audio embedding from binary (faster), fallback to ONNX
        self.audio_embedding_weight = load_audio_embedding_weight(self.onnx_dir)
        if self.audio_embedding_weight is not None:
            self.audio_embed_session = None  # Not needed when using binary
        else:
            logger.info(f"Loading audio_embedding from {audio_embedding_path.name}...")
            self.audio_embed_session = load_session(audio_embedding_path)

        if audio_encoder_path.exists():
            logger.info(f"Loading audio_encoder from {audio_encoder_path.name}...")
            self.audio_encoder_session = load_session(audio_encoder_path)
        else:
            logger.warning(f"{audio_encoder_path.name} not found, ASR mode unavailable")
            self.audio_encoder_session = None

        if audio_detokenizer_path.exists():
            logger.info(f"Loading audio_detokenizer from {audio_detokenizer_path.name}...")
            self.audio_detokenizer_session = load_session(audio_detokenizer_path)
        else:
            logger.warning(f"{audio_detokenizer_path.name} not found, TTS output unavailable")
            self.audio_detokenizer_session = None

        # Load ONNX depthformer for autoregressive inference
        self.onnx_depthformer = None
        self._load_onnx_depthformer()

        self._load_config()

        # === Stateful cache for multi-turn conversation ===
        # Cache is preserved across calls until explicitly reset
        self.cache = None
        self.cache_seq_len = 0

    def reset(self):
        """Reset conversation state (KV cache).

        Call this to start a new conversation. Without calling reset(),
        the cache is preserved across generate calls for multi-turn chat.
        """
        self.cache = None
        self.cache_seq_len = 0
        logger.info("Conversation state reset")

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

    def _load_onnx_depthformer(self):
        """Load ONNX vocoder model for autoregressive audio codebook prediction.

        vocoder_depthformer.onnx takes hidden_states [B, 2048] and generates
        audio codebook logits. Called 8× per audio frame (one per codebook).
        """
        depthformer_path = self.onnx_dir / (
            self._vocoder_depthformer_file or "vocoder_depthformer.onnx"
        )

        if not depthformer_path.exists():
            raise FileNotFoundError(f"Vocoder depthformer not found: {depthformer_path}")

        logger.info(f"Loading vocoder_depthformer from {depthformer_path.name}...")

        self.onnx_depthformer = load_session(depthformer_path)
        logger.info("ONNX vocoder ready for TTS")

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

    def _sample(
        self,
        logits: np.ndarray,
        temperature: float,
        top_p: float | None = None,
        top_k: int | None = None,
    ) -> int:
        """Sample next token using temperature and optional top-p/top-k sampling.

        Args:
            logits: Raw logits from model
            temperature: Sampling temperature (0 = greedy)
            top_p: Optional nucleus sampling threshold
            top_k: Optional top-k sampling (matches liquid-audio audio_top_k=4)
        """
        if temperature == 0:
            return int(np.argmax(logits))

        logits = logits / temperature

        # Apply top-k filtering before softmax (matches liquid-audio)
        if top_k is not None and top_k > 0:
            top_k_indices = np.argpartition(logits, -top_k)[-top_k:]
            mask = np.full(logits.shape, -np.inf)
            mask[top_k_indices] = logits[top_k_indices]
            logits = mask

        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()

        if top_p is not None:
            # Nucleus (top-p) sampling
            sorted_indices = np.argsort(probs)[::-1]
            sorted_probs = probs[sorted_indices]
            cumsum = np.cumsum(sorted_probs)

            cutoff_idx = np.searchsorted(cumsum, top_p) + 1
            top_indices = sorted_indices[:cutoff_idx]
            top_probs = probs[top_indices]
            top_probs = top_probs / top_probs.sum()

            return int(np.random.choice(top_indices, p=top_probs))
        else:
            # Pure temperature sampling (matches liquid-audio)
            return int(np.random.choice(len(probs), p=probs))

    def _get_text_embeds(self, input_ids: np.ndarray) -> np.ndarray:
        # [batch, seq_len] → [batch, seq_len, hidden]
        return self.embed_tokens_weight[input_ids].astype(np.float32)

    def _get_audio_embeds(self, audio_codes: np.ndarray) -> np.ndarray:
        """Get audio code embeddings.

        Uses direct numpy indexing if audio_embedding.bin is available,
        otherwise falls back to ONNX model call.

        Args:
            audio_codes: [batch, num_codebooks] token indices

        Returns:
            embeddings: [batch, num_codebooks, hidden_size]
        """
        if self.audio_embedding_weight is not None:
            return self.audio_embedding_weight[audio_codes].astype(np.float32)
        else:
            return self.audio_embed_session.run(["audio_embeds"], {"audio_codes": audio_codes})[0]

    def _run_decoder(
        self, embeds: np.ndarray, attention_mask: np.ndarray, cache: dict
    ) -> tuple[np.ndarray, np.ndarray, dict]:
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

    # End-of-audio token (same across all codebooks)
    END_OF_AUDIO_TOKEN = 2048

    def _sample_audio_codes(
        self,
        hidden_states: np.ndarray,
        temperature: float = 0.9,
        top_k: int | None = None,
    ) -> np.ndarray:
        """Sample audio codes using ONNX depthformer.

        Runs 8 autoregressive steps (one per codebook) to generate a full
        audio frame. Token 2048 is the end-of-audio token.

        Args:
            hidden_states: [batch, hidden_size] or [batch, 1, hidden_size]
            temperature: Sampling temperature
            top_k: Optional top-k sampling (e.g., 4 for liquid-audio interleaved)

        Returns:
            codes: [batch, 8] audio codes for each codebook
        """
        if self.onnx_depthformer is None:
            raise RuntimeError(
                "ONNX depthformer not available for TTS.\n"
                "Ensure vocoder_depthformer.onnx is exported."
            )

        num_codebooks = 8
        num_layers = 6
        num_kv_heads = 8
        head_dim = 32

        # Squeeze to [batch, hidden_size] if needed
        if hidden_states.ndim == 3:
            hidden_states = hidden_states.squeeze(1)
        batch_size = hidden_states.shape[0]

        codes_list = []
        for b in range(batch_size):
            embedding = hidden_states[b : b + 1]  # [1, hidden_size]

            # Initialize KV cache for depthformer (6 layers)
            past_keys = np.zeros((num_layers, 1, 0, num_kv_heads, head_dim), dtype=np.float32)
            past_values = np.zeros((num_layers, 1, 0, num_kv_heads, head_dim), dtype=np.float32)

            out_tokens = []
            prev_token = 0

            for i in range(num_codebooks):
                logits, _, new_keys, new_values = self.onnx_depthformer.run(
                    ["logits", "depth_slices", "new_keys", "new_values"],
                    {
                        "hidden_states": embedding.astype(np.float32),
                        "step_idx": np.array(i, dtype=np.int64),
                        "prev_token": np.array([prev_token], dtype=np.int64),
                        "past_keys": past_keys,
                        "past_values": past_values,
                    },
                )

                past_keys = new_keys
                past_values = new_values

                # Sample from logits including end-of-audio token (2048)
                # Use temperature + optional top_k sampling to match liquid-audio
                all_logits = logits[0]
                if temperature is None or temperature <= 0:
                    token = int(np.argmax(all_logits))
                else:
                    token = self._sample(all_logits, temperature, top_p=None, top_k=top_k)

                out_tokens.append(token)
                # Pass token directly to embedding lookup (table has 2049 entries: 0-2048)
                # Don't clamp - if model predicts 2048, next codebook should see that embedding
                prev_token = token

            codes_list.append(out_tokens)

        return np.array(codes_list, dtype=np.int64)  # [batch, 8]

    def _is_end_of_audio(self, frame_codes: np.ndarray, first_codebook_only: bool = False) -> bool:
        """Check if audio frame indicates end of audio.

        Args:
            frame_codes: [8] array of codebook tokens
            first_codebook_only: If True, only check first codebook (for interleaved mode,
                                matching liquid-audio behavior). If False, check any codebook.

        Returns:
            True if end-of-audio detected
        """
        if first_codebook_only:
            return frame_codes[0] == self.END_OF_AUDIO_TOKEN
        return np.any(frame_codes >= self.END_OF_AUDIO_TOKEN)

    def decode_audio(self, codes: np.ndarray) -> np.ndarray:
        """Decode audio codes to waveform using ONNX detokenizer.

        Args:
            codes: Audio codes with shape [T, 8] where T is number of frames

        Returns:
            Waveform as float32 numpy array in range [-1, 1]
        """
        if self.audio_detokenizer_session is None:
            raise RuntimeError("Audio detokenizer not loaded")

        n_fft = 1280
        hop_length = 320
        win_length = 1280
        n_fft_bins = n_fft // 2 + 1
        window = np.hanning(n_fft).astype(np.float32)

        # Transpose: [T, 8] → [8, T] and add batch dimension → [1, 8, T]
        codes_t = codes.T.astype(np.int64)
        codes_batch = codes_t[np.newaxis, :, :]

        # Run detokenizer: [1, 8, T] → [1, T, 1282]
        stft_features = self.audio_detokenizer_session.run(
            ["stft_features"], {"audio_codes": codes_batch}
        )[0]
        stft_features = stft_features[0]  # [T, 1282]

        # Convert to complex STFT: [log_magnitude | angle] → complex
        log_magnitude = stft_features[:, :n_fft_bins]
        angle = stft_features[:, n_fft_bins:]
        magnitude = np.exp(log_magnitude)
        complex_stft = magnitude * np.exp(1j * angle)

        # ISTFT with 'same' padding
        waveform = _istft_same_padding(complex_stft.T, n_fft, hop_length, win_length, window)

        # Normalize to [-1, 1]
        max_val = np.abs(waveform).max()
        if max_val > 0:
            waveform = waveform / max_val * 0.9

        return waveform.astype(np.float32)

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

    def _compute_mel_features(self, audio_path: str) -> tuple[np.ndarray, np.ndarray]:
        """Compute mel spectrogram features from audio file.

        Uses pure numpy implementation for portability to NPU backends.

        Returns:
            mel_features: [1, time, 128] mel spectrogram
            mel_lengths: [1] length array
        """
        return compute_mel_spectrogram_numpy(audio_path, self.onnx_dir)

    def _format_asr_prompt(self, system_prompt: str = DEFAULT_SYSTEM_PROMPT_ASR) -> str:
        """Format ASR system instruction using ChatML format.

        The audio embeddings will be inserted at the user position.
        """
        return f"<|startoftext|><|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n"

    def _format_asr_suffix(self) -> str:
        """Format the suffix after audio embeddings."""
        return "<|im_end|>\n<|im_start|>assistant\n"

    def transcribe(
        self,
        audio_path: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT_ASR,
    ) -> str:
        """Transcribe audio to text using ChatML format.

        The prompt structure is:
        <|startoftext|><|im_start|>system
        {system_prompt}<|im_end|>
        <|im_start|>user
        [AUDIO EMBEDDINGS]<|im_end|>
        <|im_start|>assistant
        [TRANSCRIPTION OUTPUT]
        """
        if self.audio_encoder_session is None:
            raise RuntimeError("audio_encoder not loaded, ASR unavailable")

        # Compute mel spectrogram using proper preprocessing
        mel_features, mel_lengths = self._compute_mel_features(audio_path)

        # Encode audio
        audio_embeds, _ = self.audio_encoder_session.run(
            ["audio_embeddings", "audio_lengths"],
            {"mel_spectrogram": mel_features.astype(np.float32), "mel_lengths": mel_lengths},
        )

        # Build the prompt: prefix + audio + suffix
        # 1. Encode prefix text (system + user start)
        # Note: add_special_tokens=False since we include <|startoftext|> in the prompt
        prefix_text = self._format_asr_prompt(system_prompt)
        prefix_ids = self.tokenizer.encode(
            prefix_text, return_tensors="np", add_special_tokens=False
        )
        prefix_embeds = self._get_text_embeds(prefix_ids)

        # 2. Encode suffix text (user end + assistant start)
        suffix_text = self._format_asr_suffix()
        suffix_ids = self.tokenizer.encode(
            suffix_text, return_tensors="np", add_special_tokens=False
        )
        suffix_embeds = self._get_text_embeds(suffix_ids)

        # 3. Concatenate: prefix + audio + suffix
        all_embeds = np.concatenate(
            [prefix_embeds, audio_embeds, suffix_embeds],
            axis=1,
        )

        # Run decoder with full context
        batch_size = 1
        seq_len = all_embeds.shape[1]
        cache = self._init_cache(batch_size)

        attention_mask = np.ones((batch_size, seq_len), dtype=np.int64)
        logits, _, cache = self._run_decoder(all_embeds, attention_mask, cache)

        # Generate text tokens
        next_logits = logits[0, -1, : self.vocab_size]
        next_token = self._sample(next_logits, temperature, top_p=None)

        generated_tokens = [next_token]
        total_len = seq_len + 1

        for _ in range(max_new_tokens - 1):
            if next_token == self.tokenizer.eos_token_id:
                break
            if next_token == self.IM_END_TOKEN:
                break

            next_ids = np.array([[next_token]], dtype=np.int64)
            next_embeds = self._get_text_embeds(next_ids)
            attention_mask = np.ones((batch_size, total_len), dtype=np.int64)

            logits, _, cache = self._run_decoder(next_embeds, attention_mask, cache)

            next_logits = logits[0, -1, : self.vocab_size]
            next_token = self._sample(next_logits, temperature, top_p=None)

            generated_tokens.append(next_token)
            total_len += 1

        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # === TTS (Text → Audio) ===

    def _format_tts_prompt(self, text: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT_TTS) -> str:
        """Format text with TTS system instruction using ChatML format."""
        return (
            "<|startoftext|><|im_start|>system\n"
            f"{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{text}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def synthesize(
        self,
        text: str,
        max_new_tokens: int = 100,
        audio_temperature: float = 0.8,
        audio_top_k: int = 64,
        text_temperature: float = 0.7,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT_TTS,
    ) -> list[np.ndarray]:
        """Synthesize audio from text using depthformer.

        The model first generates text tokens until it produces <|audio|>,
        then switches to depthformer-based audio code generation.

        Args:
            text: Text to synthesize.
            max_new_tokens: Maximum number of new tokens (text + audio frames combined),
                matching PyTorch reference behavior.
            audio_temperature: Temperature for audio sampling (0 = greedy).
                Default 0.8 matches liquid-audio's fixed TTS settings.
            audio_top_k: Top-k sampling for audio (64 matches liquid-audio).
            text_temperature: Temperature for text sampling (0 = greedy).
            system_prompt: System prompt for TTS.

        Returns list of audio code frames (8 codes each).
        Each frame is [8] array of codebook indices.
        """
        if self.onnx_depthformer is None:
            raise RuntimeError("ONNX depthformer not loaded, TTS unavailable")

        # Format prompt with TTS system instruction
        # Note: add_special_tokens=False since we include <|startoftext|> in the prompt
        prompt = self._format_tts_prompt(text, system_prompt)
        input_ids = self.tokenizer.encode(prompt, return_tensors="np", add_special_tokens=False)
        batch_size, seq_len = input_ids.shape

        # Get text embeddings and run decoder
        embeds = self._get_text_embeds(input_ids)
        cache = self._init_cache(batch_size)

        attention_mask = np.ones((batch_size, seq_len), dtype=np.int64)
        logits, hidden_states, cache = self._run_decoder(embeds, attention_mask, cache)
        total_len = seq_len

        # Track tokens generated (text + audio) to match PyTorch behavior
        tokens_generated = 0

        # === Phase 1: Generate text until <|audio|> token ===
        in_audio_mode = False
        while tokens_generated < max_new_tokens:
            last_logits = logits[0, -1, : self.vocab_size]
            next_token = self._sample(last_logits, text_temperature, top_p=None)

            if next_token == self.tokenizer.eos_token_id:
                logger.warning("Model produced EOS before audio, TTS may not work")
                break

            tokens_generated += 1

            if next_token == self.AUDIO_START_TOKEN:
                logger.info("Model entered audio mode")
                in_audio_mode = True
                # Feed audio_start token to get hidden states for first audio frame
                next_ids = np.array([[self.AUDIO_START_TOKEN]], dtype=np.int64)
                next_embeds = self._get_text_embeds(next_ids)
                attention_mask = np.ones((batch_size, total_len + 1), dtype=np.int64)
                logits, hidden_states, cache = self._run_decoder(next_embeds, attention_mask, cache)
                total_len += 1
                break

            # Continue text generation
            next_ids = np.array([[next_token]], dtype=np.int64)
            next_embeds = self._get_text_embeds(next_ids)
            attention_mask = np.ones((batch_size, total_len + 1), dtype=np.int64)
            logits, hidden_states, cache = self._run_decoder(next_embeds, attention_mask, cache)
            total_len += 1

        if not in_audio_mode:
            logger.warning("Model did not enter audio mode, forcing audio generation")
            # Force audio start token and feed it to decoder
            next_ids = np.array([[self.AUDIO_START_TOKEN]], dtype=np.int64)
            next_embeds = self._get_text_embeds(next_ids)
            attention_mask = np.ones((batch_size, total_len + 1), dtype=np.int64)
            logits, hidden_states, cache = self._run_decoder(next_embeds, attention_mask, cache)
            total_len += 1
            tokens_generated += 1

        # === Phase 2: Generate audio frames using depthformer ===
        audio_codes = []
        start_time = time.time()

        while tokens_generated < max_new_tokens:
            # Get hidden states for the last position: [1, hidden_size]
            last_hidden = hidden_states[0, -1:, :]  # [1, hidden_size]

            # Sample audio codes (autoregressive sampling, matches liquid-audio)
            frame_codes = self._sample_audio_codes(
                last_hidden, audio_temperature, top_k=audio_top_k
            )  # [1, 8]

            # Check for end-of-audio (first codebook only, matching liquid-audio)
            if self._is_end_of_audio(frame_codes[0], first_codebook_only=True):
                logger.info(f"End of audio detected at frame {len(audio_codes)}")
                break

            audio_codes.append(frame_codes[0])  # [8]
            tokens_generated += 1

            # Feed back audio codes to continue generation
            # Preserve 2048 (end-of-audio) for embedding lookup, clamp others to 0-2047
            clamped_codes = np.where(
                frame_codes[0] == self.END_OF_AUDIO_TOKEN,
                self.END_OF_AUDIO_TOKEN,
                np.minimum(frame_codes[0], 2047),
            )
            audio_tokens = np.array(
                [
                    [
                        cb * self.codebook_vocab + int(clamped_codes[cb])
                        for cb in range(self.num_codebooks)
                    ]
                ],
                dtype=np.int64,
            )
            next_embeds = self._get_audio_embeds(audio_tokens).sum(axis=1, keepdims=True)

            attention_mask = np.ones((batch_size, total_len + 1), dtype=np.int64)
            logits, hidden_states, cache = self._run_decoder(next_embeds, attention_mask, cache)
            total_len += 1

        elapsed = time.time() - start_time
        frames_per_sec = len(audio_codes) / elapsed if elapsed > 0 else 0
        logger.info(
            f"Generated {len(audio_codes)} audio frames in {elapsed:.2f}s "
            f"({frames_per_sec:.1f} frames/s)"
        )

        # Debug: analyze code distribution
        if audio_codes:
            codes_array = np.array(audio_codes)  # [T, 8]
            logger.info(
                f"Audio codes stats: min={codes_array.min()}, max={codes_array.max()}, "
                f"mean={codes_array.mean():.1f}, std={codes_array.std():.1f}"
            )

        return audio_codes

    # === Interleaved Mode ===

    def _format_interleaved_prompt(
        self, text: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT_INTERLEAVED
    ) -> str:
        """Format text with interleaved system instruction using ChatML format."""
        return (
            "<|startoftext|><|im_start|>system\n"
            f"{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{text}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def generate_interleaved(
        self,
        prompt: str,
        max_new_tokens: int = 20,
        audio_temperature: float = 0,
        text_temperature: float = 0,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT_INTERLEAVED,
    ) -> tuple[str, list[np.ndarray]]:
        """Generate interleaved text and audio from text prompt.

        Defaults match liquid-audio library defaults (greedy decoding).
        """
        # Note: add_special_tokens=False since we include <|startoftext|> in the prompt
        formatted_prompt = self._format_interleaved_prompt(prompt, system_prompt)
        input_ids = self.tokenizer.encode(
            formatted_prompt, return_tensors="np", add_special_tokens=False
        )
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
                # Use ONNX depthformer to generate audio frame
                if self.onnx_depthformer is None or hidden_states is None:
                    logger.warning("ONNX depthformer unavailable, exiting audio mode")
                    in_audio_mode = False
                    continue

                last_hidden = hidden_states[0, -1:, :]

                # Autoregressive sampling (matches reference)
                frame_codes = self._sample_audio_codes(last_hidden, audio_temperature)

                # Check for end of audio (only first codebook, matching liquid-audio)
                if self._is_end_of_audio(frame_codes[0], first_codebook_only=True):
                    logger.info(f"End of audio detected at frame {len(audio_codes)}")
                    # Set all codes to 2048 and feed back (matching liquid-audio)
                    frame_codes[0][:] = self.END_OF_AUDIO_TOKEN
                    in_audio_mode = False

                    # Still need to feed back the end-of-audio embedding
                    audio_tokens = np.array(
                        [
                            [
                                cb * self.codebook_vocab + self.END_OF_AUDIO_TOKEN
                                for cb in range(self.num_codebooks)
                            ]
                        ],
                        dtype=np.int64,
                    )
                    next_embeds = self._get_audio_embeds(audio_tokens).sum(axis=1, keepdims=True)
                    attention_mask = np.ones((batch_size, total_len + 1), dtype=np.int64)
                    logits, hidden_states, cache = self._run_decoder(
                        next_embeds, attention_mask, cache
                    )
                    total_len += 1
                    continue

                audio_codes.append(frame_codes[0])

                # Feed all 8 codebook tokens as a summed embedding
                audio_tokens = np.array(
                    [
                        [
                            cb * self.codebook_vocab + int(frame_codes[0][cb])
                            for cb in range(self.num_codebooks)
                        ]
                    ],
                    dtype=np.int64,
                )
                next_embeds = self._get_audio_embeds(audio_tokens).sum(axis=1, keepdims=True)

                attention_mask = np.ones((batch_size, total_len + 1), dtype=np.int64)
                logits, hidden_states, cache = self._run_decoder(next_embeds, attention_mask, cache)
                total_len += 1
                continue  # Skip the decoder update at the end of the loop
            else:
                # Sample from text vocabulary
                text_logits = last_logits[: self.vocab_size]
                token = self._sample(text_logits, text_temperature, top_p=None)

                if token == self.tokenizer.eos_token_id:
                    break

                if token == self.AUDIO_START_TOKEN:
                    logger.info("Model entered audio mode")
                    in_audio_mode = True
                    # Feed audio_start token to get hidden states for first audio frame
                    next_ids = np.array([[self.AUDIO_START_TOKEN]], dtype=np.int64)
                    next_embeds = self._get_text_embeds(next_ids)
                    attention_mask = np.ones((batch_size, total_len + 1), dtype=np.int64)
                    logits, hidden_states, cache = self._run_decoder(
                        next_embeds, attention_mask, cache
                    )
                    total_len += 1
                    text_tokens.append(token)
                    continue

                text_tokens.append(token)
                next_embeds = self._get_text_embeds(np.array([[token]], dtype=np.int64))

            attention_mask = np.ones((batch_size, total_len + 1), dtype=np.int64)
            logits, hidden_states, cache = self._run_decoder(next_embeds, attention_mask, cache)
            total_len += 1

        text_output = self.tokenizer.decode(text_tokens, skip_special_tokens=True)
        logger.info(f"Generated {len(text_tokens)} text tokens, {len(audio_codes)} audio frames")

        return text_output, audio_codes

    def _generate_interleaved_response(
        self,
        logits: np.ndarray,
        hidden_states: np.ndarray,
        max_new_tokens: int,
        text_temperature: float,
        audio_temperature: float,
        audio_top_k: int,
    ) -> tuple[str, list[np.ndarray]]:
        """Shared generation loop for stateful interleaved mode.

        Uses counter-based mode switching matching liquid-audio:
        - INTERLEAVED_N_TEXT tokens of text, then switch to audio
        - INTERLEAVED_N_AUDIO frames of audio, then switch to text
        - TEXT_END_TOKEN forces switch to audio mode

        Args:
            logits: Initial logits from decoder [1, seq_len, vocab_size]
            hidden_states: Initial hidden states [1, seq_len, hidden_size]
            max_new_tokens: Maximum tokens to generate
            text_temperature: Sampling temperature for text
            audio_temperature: Sampling temperature for audio
            audio_top_k: Top-k sampling for audio

        Returns:
            Tuple of (text_response, audio_codes)
        """
        batch_size = 1
        text_tokens = []
        audio_codes = []
        total_len = self.cache_seq_len
        in_audio_mode = False
        modality_left = self.INTERLEAVED_N_TEXT
        text_done = False

        for step in range(max_new_tokens):
            modality_left -= 1

            if in_audio_mode:
                if self.onnx_depthformer is None or hidden_states is None:
                    logger.warning("Depthformer unavailable, exiting audio mode")
                    in_audio_mode = False
                    modality_left = self.INTERLEAVED_N_TEXT
                    continue

                last_hidden = hidden_states[0, -1:, :]
                frame_codes = self._sample_audio_codes(
                    last_hidden, temperature=audio_temperature, top_k=audio_top_k
                )
                frame = frame_codes[0]

                # Switch back to text after N audio frames (if text not done)
                if modality_left <= 0 and not text_done:
                    in_audio_mode = False
                    modality_left = self.INTERLEAVED_N_TEXT

                # Check for end of audio - first codebook == 2048 (matching liquid-audio)
                if frame[0] == self.END_OF_AUDIO_TOKEN:
                    logger.info(f"End of audio token at step {step}")
                    frame[:] = self.END_OF_AUDIO_TOKEN
                    in_audio_mode = False
                else:
                    clamped_frame = np.minimum(frame, 2047)
                    audio_codes.append(clamped_frame.copy())

                    if len(audio_codes) % 20 == 0:
                        logger.info(f"Generated {len(audio_codes)} audio frames...")

                # Get embeddings for next step (always feed back, even for end-of-audio)
                feed_codes = np.where(
                    frame == self.END_OF_AUDIO_TOKEN,
                    self.END_OF_AUDIO_TOKEN,
                    np.minimum(frame, 2047),
                )
                audio_tokens = np.array(
                    [
                        [
                            cb * self.codebook_vocab + int(feed_codes[cb])
                            for cb in range(self.num_codebooks)
                        ]
                    ],
                    dtype=np.int64,
                )
                next_embeds = self._get_audio_embeds(audio_tokens).sum(axis=1, keepdims=True)
            else:
                # Generate text token
                last_logits = logits[0, -1, :]
                text_logits = last_logits[: self.vocab_size]
                token = self._sample(text_logits, text_temperature, top_p=None)

                if token == self.tokenizer.eos_token_id or token == self.IM_END_TOKEN:
                    logger.info(f"End of turn at step {step}")
                    break

                if token == self.TEXT_END_TOKEN:
                    logger.info(f"Text end at step {step}")
                    text_done = True

                # Switch to audio after N text tokens OR text_end
                if modality_left <= 0 or text_done:
                    in_audio_mode = True
                    modality_left = self.INTERLEAVED_N_AUDIO

                text_tokens.append(token)
                next_ids = np.array([[token]], dtype=np.int64)
                next_embeds = self._get_text_embeds(next_ids)

            # Update decoder
            attention_mask = np.ones((batch_size, total_len + 1), dtype=np.int64)
            logits, hidden_states, self.cache = self._run_decoder(
                next_embeds, attention_mask, self.cache
            )
            total_len += 1

        # Feed im_end token to finalize this turn in the cache
        im_end_embeds = self._get_text_embeds(np.array([[self.IM_END_TOKEN]], dtype=np.int64))
        attention_mask = np.ones((batch_size, total_len + 1), dtype=np.int64)
        _, _, self.cache = self._run_decoder(im_end_embeds, attention_mask, self.cache)
        total_len += 1

        # Update sequence length for next turn
        self.cache_seq_len = total_len

        text_output = self.tokenizer.decode(text_tokens, skip_special_tokens=True)
        logger.info(f"Generated {len(text_tokens)} text tokens, {len(audio_codes)} audio frames")
        logger.info(f"Cache seq_len: {self.cache_seq_len}")

        return text_output, audio_codes

    def generate_interleaved_from_audio(
        self,
        audio_path: str,
        max_new_tokens: int = 300,
        text_temperature: float = 1.0,
        audio_temperature: float = 1.0,
        audio_top_k: int = 4,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT_INTERLEAVED,
        text_prompt: str | None = None,
    ) -> tuple[str, list[np.ndarray]]:
        """Generate interleaved text+audio response from audio input.

        Stateful: KV cache is preserved across calls for multi-turn conversation.
        Call reset() to start a new conversation.

        Args:
            audio_path: Path to input audio file
            max_new_tokens: Maximum tokens to generate
            text_temperature: Sampling temperature for text (1.0 matches liquid-audio)
            audio_temperature: Sampling temperature for audio (1.0 matches liquid-audio)
            audio_top_k: Top-k sampling for audio (4 matches liquid-audio)
            system_prompt: System prompt for interleaved mode (only used on first turn).
            text_prompt: Optional text to include in user turn alongside audio.

        Returns:
            Tuple of (text_response, audio_codes)
        """
        batch_size = 1

        # Encode audio
        mel_features, mel_lengths = self._compute_mel_features(audio_path)
        audio_embeds, _ = self.audio_encoder_session.run(
            ["audio_embeddings", "audio_lengths"],
            {"mel_spectrogram": mel_features.astype(np.float32), "mel_lengths": mel_lengths},
        )

        # === Build prompt based on conversation state ===
        if self.cache is None:
            prefix_text = (
                f"<|startoftext|><|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n"
            )
            self.cache = self._init_cache(batch_size)
            self.cache_seq_len = 0
            logger.info("Starting new conversation")
        else:
            prefix_text = "<|im_start|>user\n"
            logger.info(f"Continuing conversation (cache seq_len={self.cache_seq_len})")

        suffix_text = "<|im_end|>\n<|im_start|>assistant\n"

        prefix_ids = self.tokenizer.encode(
            prefix_text, return_tensors="np", add_special_tokens=False
        )
        suffix_ids = self.tokenizer.encode(
            suffix_text, return_tensors="np", add_special_tokens=False
        )
        prefix_embeds = self._get_text_embeds(prefix_ids)
        suffix_embeds = self._get_text_embeds(suffix_ids)

        # Build embeddings: prefix + audio + [text_prompt] + suffix
        if text_prompt:
            text_prompt_ids = self.tokenizer.encode(
                text_prompt, return_tensors="np", add_special_tokens=False
            )
            text_prompt_embeds = self._get_text_embeds(text_prompt_ids)
            logger.info(f"Text prompt: {text_prompt} ({text_prompt_ids.shape[1]} tokens)")
            all_embeds = np.concatenate(
                [prefix_embeds, audio_embeds, text_prompt_embeds, suffix_embeds], axis=1
            )
        else:
            all_embeds = np.concatenate([prefix_embeds, audio_embeds, suffix_embeds], axis=1)

        # Run initial prefill
        new_seq_len = all_embeds.shape[1]
        total_len = self.cache_seq_len + new_seq_len
        attention_mask = np.ones((batch_size, total_len), dtype=np.int64)
        logits, hidden_states, self.cache = self._run_decoder(
            all_embeds, attention_mask, self.cache
        )
        self.cache_seq_len = total_len

        return self._generate_interleaved_response(
            logits, hidden_states, max_new_tokens, text_temperature, audio_temperature, audio_top_k
        )

    def generate_interleaved_from_text(
        self,
        user_text: str,
        max_new_tokens: int = 300,
        text_temperature: float = 1.0,
        audio_temperature: float = 1.0,
        audio_top_k: int = 4,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT_INTERLEAVED,
    ) -> tuple[str, list[np.ndarray]]:
        """Generate interleaved text+audio response from text input.

        Stateful: KV cache is preserved across calls for multi-turn conversation.
        Call reset() to start a new conversation.

        Args:
            user_text: User's text message
            max_new_tokens: Maximum tokens to generate
            text_temperature: Sampling temperature for text
            audio_temperature: Sampling temperature for audio
            audio_top_k: Top-k sampling for audio
            system_prompt: System prompt (only used on first turn).

        Returns:
            Tuple of (text_response, audio_codes)
        """
        batch_size = 1

        # === Build prompt based on conversation state ===
        if self.cache is None:
            prefix_text = f"<|startoftext|><|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"
            self.cache = self._init_cache(batch_size)
            self.cache_seq_len = 0
            logger.info("Starting new conversation")
        else:
            prefix_text = f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"
            logger.info(f"Continuing conversation (cache seq_len={self.cache_seq_len})")

        prefix_ids = self.tokenizer.encode(
            prefix_text, return_tensors="np", add_special_tokens=False
        )
        prefix_embeds = self._get_text_embeds(prefix_ids)

        # Run initial prefill
        new_seq_len = prefix_embeds.shape[1]
        total_len = self.cache_seq_len + new_seq_len
        attention_mask = np.ones((batch_size, total_len), dtype=np.int64)
        logits, hidden_states, self.cache = self._run_decoder(
            prefix_embeds, attention_mask, self.cache
        )
        self.cache_seq_len = total_len

        return self._generate_interleaved_response(
            logits, hidden_states, max_new_tokens, text_temperature, audio_temperature, audio_top_k
        )


# === Numpy Mel Spectrogram ===


def _resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio to target sample rate.

    This is INPUT PREPROCESSING only - converts audio files to 16kHz before
    mel spectrogram computation. Not part of the ONNX model.

    Uses torchaudio for best accuracy (matches PyTorch pipeline exactly).
    Falls back to scipy if torchaudio is unavailable.

    Args:
        audio: Audio waveform as numpy array
        orig_sr: Original sample rate
        target_sr: Target sample rate (16000 for this model)

    Returns:
        Resampled audio as numpy array
    """
    if orig_sr == target_sr:
        return audio

    try:
        import torch
        import torchaudio

        audio_tensor = torch.from_numpy(audio).unsqueeze(0)  # [1, samples]
        resampled = torchaudio.functional.resample(audio_tensor, orig_sr, target_sr)
        return resampled.squeeze(0).numpy()
    except ImportError:
        # Fallback to scipy (slightly different results due to FFT-based algorithm)
        import scipy.signal

        num_samples = int(len(audio) * target_sr / orig_sr)
        return scipy.signal.resample(audio, num_samples)


def compute_mel_spectrogram_numpy(
    audio_path: str,
    onnx_dir: pathlib.Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute mel spectrogram for ONNX audio encoder.

    This implementation matches liquid_audio's AudioToMelSpectrogramPreprocessor
    for compatibility with the ONNX audio encoder.

    Args:
        audio_path: Path to audio file (WAV)
        onnx_dir: Path to ONNX directory containing mel_config.json
                  (mel filterbank and window are generated at runtime via librosa)

    Returns:
        mel_features: [1, time, 128] mel spectrogram
        mel_lengths: [1] length array
    """
    import json

    import librosa
    import scipy.io.wavfile

    # Mel spectrogram config (matching liquid_audio's AudioToMelSpectrogramPreprocessor)
    # Load from config file if available, otherwise use defaults
    config_path = onnx_dir / "mel_config.json"
    if config_path.exists():
        with open(config_path) as f:
            mel_config = json.load(f)
    else:
        mel_config = {
            "sample_rate": 16000,
            "n_fft": 512,
            "win_length": 400,
            "hop_length": 160,
            "n_mels": 128,
            "fmin": 0,
            "fmax": 8000,
            "preemph": 0.97,
            "log_zero_guard": 5.960464477539063e-08,
            "mel_norm": "slaney",
        }

    # Extract config
    target_sr = mel_config["sample_rate"]
    n_fft = mel_config["n_fft"]
    win_length = mel_config["win_length"]
    hop_length = mel_config["hop_length"]
    preemph = mel_config["preemph"]
    log_zero_guard = mel_config["log_zero_guard"]

    # Generate mel filterbank at runtime using librosa (same as NeMo/liquid_audio)
    mel_filterbank = librosa.filters.mel(
        sr=target_sr,
        n_fft=n_fft,
        n_mels=mel_config.get("n_mels", 128),
        fmin=mel_config.get("fmin", 0),
        fmax=mel_config.get("fmax", target_sr // 2),
        norm=mel_config.get("mel_norm", "slaney"),
    ).astype(np.float32)

    # Generate Hann window at runtime
    hann_window = np.hanning(win_length).astype(np.float32)

    # === 1. Load audio ===
    sample_rate, audio = scipy.io.wavfile.read(audio_path)

    # Convert to float32 in [-1, 1]
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    elif audio.dtype == np.float64:
        audio = audio.astype(np.float32)

    # Convert stereo to mono
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    # === 2. Resample to 16kHz (input preprocessing) ===
    audio = _resample_audio(audio, sample_rate, target_sr)

    # === 3. Pre-emphasis filter ===
    # y[t] = x[t] - preemph * x[t-1]
    audio_preemph = np.concatenate([[audio[0]], audio[1:] - preemph * audio[:-1]])

    # === 4. STFT (matching torch.stft with center=True) ===
    # Pad for center=True
    pad_amount = n_fft // 2
    audio_padded = np.pad(audio_preemph, (pad_amount, pad_amount), mode="constant")

    # Frame the signal
    num_frames = 1 + (len(audio_padded) - n_fft) // hop_length
    frames = np.zeros((num_frames, n_fft), dtype=np.float32)

    # Center the window in the frame (matching torch.stft behavior)
    pad_left = (n_fft - win_length) // 2
    padded_window = np.zeros(n_fft, dtype=np.float32)
    padded_window[pad_left : pad_left + win_length] = hann_window

    for i in range(num_frames):
        start = i * hop_length
        frames[i] = audio_padded[start : start + n_fft] * padded_window

    # FFT
    stft_complex = np.fft.rfft(frames, axis=1).T  # [n_fft//2+1, time]

    # === 5. Magnitude and power spectrum ===
    magnitude = np.abs(stft_complex)  # [freq, time]
    power_spec = magnitude**2

    # === 6. Apply mel filterbank ===
    # mel_filterbank: [n_mels, n_fft//2+1], power_spec: [n_fft//2+1, time]
    mel_spec = np.dot(mel_filterbank, power_spec)  # [n_mels, time]

    # === 7. Log with zero guard ===
    mel_spec = np.log(mel_spec + log_zero_guard)

    # === 8. Per-feature normalization ===
    # Compute valid length first (matches FilterbankFeatures.get_seq_len)
    input_samples = len(audio)  # after resampling, before padding
    valid_len = input_samples // hop_length

    # Normalize using only valid frames (matching liquid-audio's normalize_batch)
    # Uses Bessel's correction (N-1) for std
    total_frames = mel_spec.shape[1]
    if valid_len > 1:
        valid_mel = mel_spec[:, :valid_len]
        mel_mean = valid_mel.mean(axis=1, keepdims=True)
        mel_std = (
            np.sqrt(np.sum((valid_mel - mel_mean) ** 2, axis=1, keepdims=True) / (valid_len - 1))
            + 1e-5
        )
        mel_spec = (mel_spec - mel_mean) / mel_std
        # Zero out frames beyond valid length
        if total_frames > valid_len:
            mel_spec[:, valid_len:] = 0.0

    # === 9. Format output ===
    # [n_mels, time] -> [1, time, n_mels]
    mel_features = mel_spec.T[np.newaxis, :, :].astype(np.float32)
    # Return actual number of frames (not valid_len which is used only for normalization)
    # This matches liquid-audio's ChatState behavior which uses the actual tensor length
    actual_frames = mel_features.shape[1]
    mel_lengths = np.array([actual_frames], dtype=np.int64)

    logger.info(f"Computed mel spectrogram: {mel_features.shape}")
    return mel_features, mel_lengths


def audio_codes_to_wav(
    audio_codes: list[np.ndarray],
    output_path: str,
    model_dir: pathlib.Path | None = None,
    sample_rate: int = 24000,
    audio_detokenizer_file: str | None = None,
):
    """Convert audio codes to WAV file using ONNX-only decoding.

    Uses ONNX audio_detokenizer + numpy ISTFT. No PyTorch required.
    """
    if len(audio_codes) < 2:
        logger.warning("Not enough audio codes to generate audio")
        return False

    # Stack codes: [T, 8] → [8, T]
    codes = np.stack(audio_codes, axis=0)  # [T, 8]
    codes = np.clip(codes, 0, 2047)
    codes_transposed = codes.T  # [8, T]

    if model_dir is None:
        raise ValueError("model_dir required for ONNX decoding")

    onnx_dir = model_dir / "onnx"
    detok_path = onnx_dir / (audio_detokenizer_file or "audio_detokenizer.onnx")

    if not detok_path.exists():
        raise FileNotFoundError(f"{detok_path.name} not found in {onnx_dir}")

    return _decode_audio_onnx_numpy(
        codes_transposed, detok_path, onnx_dir, output_path, sample_rate
    )


class StreamingISTFT:
    """Streaming ISTFT implementation matching llama.cpp mtmd_audio_streaming_istft."""

    def __init__(self, n_fft: int, hop_length: int):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_fft_bins = n_fft // 2 + 1

        # Hann window (periodic)
        self.hann_window = np.array(
            [0.5 * (1.0 - np.cos(2.0 * np.pi * i / n_fft)) for i in range(n_fft)],
            dtype=np.float32,
        )

        # Streaming state
        self.overlap_buffer = np.zeros(n_fft, dtype=np.float32)
        self.window_sum_buffer = np.zeros(n_fft, dtype=np.float32)
        self.padding_to_remove = (n_fft - hop_length) // 2

    def reset(self):
        self.overlap_buffer.fill(0)
        self.window_sum_buffer.fill(0)
        self.padding_to_remove = (self.n_fft - self.hop_length) // 2

    def process_frame(self, frame_spectrum: np.ndarray) -> np.ndarray:
        """Process a single STFT frame.

        Args:
            frame_spectrum: [n_fft_bins * 2] interleaved real/imag

        Returns:
            output: up to hop_length samples
        """
        # Build full complex spectrum for IFFT
        ifft_in = np.zeros(self.n_fft, dtype=np.complex64)

        # Copy positive frequencies
        for j in range(self.n_fft_bins):
            ifft_in[j] = frame_spectrum[j * 2] + 1j * frame_spectrum[j * 2 + 1]

        # Mirror negative frequencies (conjugate)
        for j in range(1, self.n_fft_bins - 1):
            mirror_idx = self.n_fft - j
            ifft_in[mirror_idx] = ifft_in[j].conjugate()

        # IFFT
        ifft_out = np.fft.ifft(ifft_in).real.astype(np.float32)

        # Update window sum and overlap buffer
        self.window_sum_buffer += self.hann_window * self.hann_window
        self.overlap_buffer += ifft_out * self.hann_window

        # Extract hop_length samples with normalization
        output = np.zeros(self.hop_length, dtype=np.float32)
        for i in range(self.hop_length):
            if self.window_sum_buffer[i] > 1e-8:
                output[i] = self.overlap_buffer[i] / self.window_sum_buffer[i]
            else:
                output[i] = self.overlap_buffer[i]

        # Shift buffers left by hop_length
        self.overlap_buffer = np.roll(self.overlap_buffer, -self.hop_length)
        self.overlap_buffer[-self.hop_length :] = 0

        self.window_sum_buffer = np.roll(self.window_sum_buffer, -self.hop_length)
        self.window_sum_buffer[-self.hop_length :] = 0

        # Remove padding if needed
        if self.padding_to_remove > 0:
            to_remove = min(self.padding_to_remove, len(output))
            output = output[to_remove:]
            self.padding_to_remove -= to_remove

        return output

    def flush(self) -> np.ndarray:
        """Flush remaining samples at end of stream."""
        output = []
        remaining = self.n_fft - self.hop_length

        while remaining > 0:
            chunk_size = min(remaining, self.hop_length)
            chunk = np.zeros(chunk_size, dtype=np.float32)

            for i in range(chunk_size):
                if self.window_sum_buffer[i] > 1e-8:
                    chunk[i] = self.overlap_buffer[i] / self.window_sum_buffer[i]
                else:
                    chunk[i] = self.overlap_buffer[i]

            output.append(chunk)

            # Shift buffers
            self.overlap_buffer = np.roll(self.overlap_buffer, -chunk_size)
            self.overlap_buffer[-chunk_size:] = 0
            self.window_sum_buffer = np.roll(self.window_sum_buffer, -chunk_size)
            self.window_sum_buffer[-chunk_size:] = 0

            remaining -= chunk_size

        return np.concatenate(output) if output else np.array([], dtype=np.float32)


def _istft_same_padding(
    spec: np.ndarray,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: np.ndarray,
) -> np.ndarray:
    """ISTFT with 'same' padding matching liquid_audio.

    This uses the same algorithm as liquid_audio/detokenizer.py ISTFT class
    which pads to ensure output length matches input length * hop_length.

    Args:
        spec: Complex STFT [freq, time]
        n_fft: FFT size
        hop_length: Hop length between frames
        win_length: Window length
        window: Window function array

    Returns:
        Audio waveform as numpy array
    """
    N, T = spec.shape
    pad = (win_length - hop_length) // 2

    # Inverse FFT
    ifft = np.fft.irfft(spec, n_fft, axis=0, norm="backward")  # [n_fft, T]
    ifft = ifft * window[:, None]

    # Overlap and Add
    output_size = (T - 1) * hop_length + win_length
    audio = np.zeros(output_size)
    for t in range(T):
        start = t * hop_length
        audio[start : start + win_length] += ifft[:, t]

    # Window envelope for normalization
    window_sq = window**2
    window_envelope = np.zeros(output_size)
    for t in range(T):
        start = t * hop_length
        window_envelope[start : start + win_length] += window_sq

    # Normalize and trim padding
    audio_trimmed = audio[pad:-pad] / window_envelope[pad:-pad]
    return audio_trimmed


def _decode_audio_onnx_numpy(
    codes: np.ndarray,
    detok_path: pathlib.Path,
    onnx_dir: pathlib.Path,
    output_path: str,
    sample_rate: int,
) -> bool:
    """Decode audio using ONNX detokenizer + numpy ISTFT.

    Pure numpy implementation - no PyTorch required.
    Uses ISTFT with 'same' padding to match liquid_audio behavior.
    """
    import scipy.io.wavfile

    # ISTFT parameters (fixed for this model)
    n_fft = 1280
    hop_length = 320
    win_length = 1280
    n_fft_bins = n_fft // 2 + 1

    # Generate Hann window at runtime (~18µs, faster than loading from disk)
    window = np.hanning(n_fft).astype(np.float32)

    # Load ONNX detokenizer
    detok_session = load_session(detok_path)

    # Run detokenizer: [1, 8, T] → [1, T*6, 1282] (6x upsampling)
    # Input codes are already [8, T] from audio_codes_to_wav
    codes_batch = codes[np.newaxis, :, :].astype(np.int64)  # [1, 8, T]
    stft_features = detok_session.run(["stft_features"], {"audio_codes": codes_batch})[0]
    stft_features = stft_features[0]  # [T*6, 1282]

    # Convert to complex STFT: [log_magnitude | angle] → complex
    log_magnitude = stft_features[:, :n_fft_bins]
    angle = stft_features[:, n_fft_bins:]
    magnitude = np.exp(log_magnitude)
    complex_stft = magnitude * np.exp(1j * angle)

    # ISTFT with 'same' padding
    waveform = _istft_same_padding(complex_stft.T, n_fft, hop_length, win_length, window)

    # Normalize to prevent clipping
    max_val = np.abs(waveform).max()
    if max_val > 0:
        waveform = waveform / max_val * 0.9

    # Save as WAV
    waveform_int16 = (waveform * 32767).astype(np.int16)
    scipy.io.wavfile.write(output_path, sample_rate, waveform_int16)

    duration = len(waveform) / sample_rate
    logger.info(f"Saved audio to {output_path} ({duration:.2f}s)")
    return True


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
        help="Input audio file for ASR/interleaved modes",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output audio file (default: tts_output.wav or interleaved_output.wav)",
    )
    parser.add_argument(
        "--precision",
        choices=["fp16", "q4", "q8"],
        help="Model precision shorthand (default: fp32)",
    )
    parser.add_argument(
        "--decoder",
        metavar="FILE",
        help="Decoder ONNX file (relative to onnx/ dir)",
    )
    parser.add_argument(
        "--audio-embedding",
        metavar="FILE",
        help="Audio embedding ONNX file (relative to onnx/ dir)",
    )
    parser.add_argument(
        "--audio-encoder",
        metavar="FILE",
        help="Audio encoder ONNX file (relative to onnx/ dir)",
    )
    parser.add_argument(
        "--audio-detokenizer",
        metavar="FILE",
        help="Audio detokenizer ONNX file (relative to onnx/ dir)",
    )
    parser.add_argument(
        "--vocoder-depthformer",
        metavar="FILE",
        help="Vocoder depthformer ONNX file (relative to onnx/ dir)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help=f"Maximum tokens/frames to generate (default: {DEFAULT_MAX_TOKENS_AUDIO} for tts/interleaved, {DEFAULT_MAX_TOKENS_TEXT} for asr/text)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Text sampling temperature (default: 0 for ASR, 0.7 for TTS/text)",
    )
    parser.add_argument(
        "--audio-temperature",
        type=float,
        default=None,
        help="Audio sampling temperature (default: 0.8 for TTS, 1.0 for interleaved)",
    )
    parser.add_argument(
        "--audio-top-k",
        type=int,
        default=None,
        help="Top-k sampling for audio (default: 64 for TTS, 4 for interleaved)",
    )
    parser.add_argument(
        "--system",
        type=str,
        default=None,
        help="System prompt (mode-specific defaults: ASR='Perform ASR.', "
        "TTS='Perform TTS. Use the UK female voice.', "
        "interleaved='Respond with interleaved text and audio.')",
    )
    parser.add_argument(
        "--save-codes",
        type=str,
        metavar="FILE",
        help="Save audio codes to numpy file (.npy) for comparison with other decoders",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible generation (default: 42)",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Interactive chat mode for interleaved mode. "
        "Preserves conversation state across turns. Enter audio file paths, 'reset' to clear, 'quit' to exit.",
    )

    args = parser.parse_args()

    # Apply mode-specific system prompt defaults
    if args.system is None:
        if args.mode == "asr":
            args.system = DEFAULT_SYSTEM_PROMPT_ASR
        elif args.mode == "tts":
            args.system = DEFAULT_SYSTEM_PROMPT_TTS
        elif args.mode == "interleaved":
            args.system = DEFAULT_SYSTEM_PROMPT_INTERLEAVED

    # Apply mode-specific temperature defaults
    if args.mode == "asr":
        # ASR uses greedy decoding by default
        if args.temperature is None:
            args.temperature = 0
    elif args.mode == "interleaved":
        # Interleaved uses 1.0 temperature (matching liquid-audio demo)
        if args.temperature is None:
            args.temperature = 1.0
        if args.audio_temperature is None:
            args.audio_temperature = 1.0
        if args.audio_top_k is None:
            args.audio_top_k = 4  # Matching liquid-audio interleaved
    elif args.mode == "tts":
        # TTS uses fixed sampling settings to match liquid-audio
        if args.temperature is None:
            args.temperature = 0.7
        if args.audio_temperature is None:
            args.audio_temperature = 0.8  # Matching liquid-audio TTS
        if args.audio_top_k is None:
            args.audio_top_k = 64  # Matching liquid-audio TTS
    else:
        # Text mode
        if args.temperature is None:
            args.temperature = 0.7
        if args.audio_temperature is None:
            args.audio_temperature = 0.8

    # Apply mode-specific max_tokens defaults
    if args.max_tokens is None:
        if args.mode in ("interleaved", "tts"):
            args.max_tokens = DEFAULT_MAX_TOKENS_AUDIO
        else:
            args.max_tokens = DEFAULT_MAX_TOKENS_TEXT

    # Apply mode-specific output defaults for audio-generating modes
    if args.mode == "tts":
        if args.output is None:
            args.output = "tts_output.wav"
        if args.save_codes is None:
            args.save_codes = "tts_output_codes.npy"
    elif args.mode == "interleaved":
        if args.output is None:
            args.output = "interleaved_output.wav"
        if args.save_codes is None:
            args.save_codes = "interleaved_output_codes.npy"

    logging.basicConfig(level=logging.INFO)

    # Set random seed for reproducibility
    np.random.seed(args.seed)
    logger.info(f"Random seed: {args.seed}")

    # Resolve component files from --precision
    files = resolve_precision_files(args.precision)

    # Explicit file args override --precision
    if args.decoder:
        files["decoder"] = args.decoder
    if args.audio_embedding:
        files["audio_embedding"] = args.audio_embedding
    if args.audio_encoder:
        files["audio_encoder"] = args.audio_encoder
    if args.audio_detokenizer:
        files["audio_detokenizer"] = args.audio_detokenizer
    if args.vocoder_depthformer:
        files["vocoder_depthformer"] = args.vocoder_depthformer

    logger.info(f"Loading model from {args.model_dir}...")
    model = LFM2AudioInference(
        args.model_dir,
        decoder_file=files["decoder"],
        audio_embedding_file=files["audio_embedding"],
        audio_encoder_file=files["audio_encoder"],
        audio_detokenizer_file=files["audio_detokenizer"],
        vocoder_depthformer_file=files["vocoder_depthformer"],
    )

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
        logger.info(f"System prompt: {args.system}")
        transcription = model.transcribe(
            args.audio,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            system_prompt=args.system,
        )
        print("\n" + "=" * 60)
        print(f"Audio:         {args.audio}")
        print(f"Transcription: {transcription}")
        print("=" * 60)

    elif args.mode == "tts":
        logger.info("Mode: TTS (Text-to-Speech)")
        logger.info(f"Text: {args.prompt}")
        logger.info(f"System prompt: {args.system}")
        logger.info(
            f"Audio sampling: temperature={args.audio_temperature}, top_k={args.audio_top_k}"
        )
        audio_codes = model.synthesize(
            args.prompt,
            max_new_tokens=args.max_tokens,
            audio_temperature=args.audio_temperature,
            audio_top_k=args.audio_top_k,
            text_temperature=args.temperature,
            system_prompt=args.system,
        )
        print("\n" + "=" * 60)
        print(f"Input: {args.prompt}")
        print(f"Generated {len(audio_codes)} audio frames")

        if args.save_codes and audio_codes:
            codes_array = np.stack(audio_codes, axis=0)  # [T, 8]
            np.save(args.save_codes, codes_array)
            print(f"Codes:  {args.save_codes} {codes_array.shape}")

        if args.output and audio_codes:
            if audio_codes_to_wav(
                audio_codes,
                args.output,
                model_dir=args.model_dir,
                audio_detokenizer_file=files["audio_detokenizer"],
            ):
                print(f"Output: {args.output}")
        print("=" * 60)

    elif args.mode == "interleaved":
        logger.info("Mode: Interleaved")
        logger.info(f"System prompt: {args.system}")

        if args.chat:
            # === Interactive chat mode ===
            chat_output = args.output.replace(".wav", "") + "_turn{}.wav"
            chat_codes = args.save_codes.replace(".npy", "") + "_turn{}.npy"

            print("\n" + "=" * 60)
            print("Interactive Chat Mode (stateful)")
            print("Commands:")
            print("  /audio <file> [text] - Send audio with optional text")
            print("  <text>               - Send text message")
            print("  reset                - Clear conversation")
            print("  quit                 - Exit")
            print(f"Audio output: {chat_output.format('N')}")
            print("=" * 60 + "\n")

            turn = 0

            # Process initial --audio if provided
            initial_audio = args.audio
            initial_prompt = args.prompt if args.prompt != "The capital of France is" else None

            while True:
                audio_path = None
                text_prompt = None

                if initial_audio:
                    # Use --audio as first turn
                    audio_path = initial_audio
                    text_prompt = initial_prompt
                    initial_audio = None
                    initial_prompt = None
                    print(
                        f"[Turn {turn}] /audio {audio_path}"
                        + (f" {text_prompt}" if text_prompt else "")
                    )
                else:
                    try:
                        user_input = input(f"[Turn {turn}] > ").strip()
                    except (EOFError, KeyboardInterrupt):
                        print("\nExiting...")
                        break

                    if not user_input:
                        continue

                    if user_input.lower() == "quit":
                        print("Goodbye!")
                        break

                    if user_input.lower() == "reset":
                        model.reset()
                        turn = 0
                        print("Conversation reset.\n")
                        continue

                    # Parse input: "/audio file.wav [text]" or just "text"
                    if user_input.startswith("/audio "):
                        parts = user_input[7:].split(maxsplit=1)
                        audio_path = parts[0]
                        text_prompt = parts[1] if len(parts) > 1 else None
                    else:
                        # Plain text input
                        text_prompt = user_input

                try:
                    if audio_path:
                        # Audio input (with optional text)
                        if not pathlib.Path(audio_path).exists():
                            print(f"File not found: {audio_path}")
                            continue

                        text_output, audio_codes = model.generate_interleaved_from_audio(
                            audio_path,
                            max_new_tokens=args.max_tokens,
                            text_temperature=args.temperature,
                            audio_temperature=args.audio_temperature,
                            system_prompt=args.system,
                            text_prompt=text_prompt,
                        )
                    else:
                        # Text-only input
                        text_output, audio_codes = model.generate_interleaved_from_text(
                            text_prompt,
                            max_new_tokens=args.max_tokens,
                            text_temperature=args.temperature,
                            audio_temperature=args.audio_temperature,
                            system_prompt=args.system,
                        )

                    print(f"\nAssistant: {text_output}")
                    print(f"           ({len(audio_codes)} audio frames)")

                    # Save audio and codes
                    if audio_codes:
                        output_path = chat_output.format(turn)
                        codes_path = chat_codes.format(turn)
                        codes_array = np.stack(audio_codes, axis=0)
                        np.save(codes_path, codes_array)
                        print(f"           Codes: {codes_path}")
                        if audio_codes_to_wav(
                            audio_codes,
                            output_path,
                            model_dir=args.model_dir,
                            audio_detokenizer_file=files["audio_detokenizer"],
                        ):
                            print(f"           Audio: {output_path}")

                    print()
                    turn += 1

                except Exception as e:
                    logger.error(f"Error: {e}")
                    print(f"Error: {e}\n")

        elif args.audio:
            # Single audio input mode (matching liquid-audio demo)
            logger.info(f"Audio: {args.audio}")
            text_output, audio_codes = model.generate_interleaved_from_audio(
                args.audio,
                max_new_tokens=args.max_tokens,
                text_temperature=args.temperature,
                audio_temperature=args.audio_temperature,
                system_prompt=args.system,
            )
            print("\n" + "=" * 60)
            print(f"Audio input: {args.audio}")
            print(f"Text:   {text_output}")
            print(f"Audio:  {len(audio_codes)} frames")

            if args.save_codes and audio_codes:
                codes_array = np.stack(audio_codes, axis=0)  # [T, 8]
                np.save(args.save_codes, codes_array)
                print(f"Codes:  {args.save_codes} {codes_array.shape}")

            if args.output and audio_codes:
                if audio_codes_to_wav(
                    audio_codes,
                    args.output,
                    model_dir=args.model_dir,
                    audio_detokenizer_file=files["audio_detokenizer"],
                ):
                    print(f"Output: {args.output}")
            print("=" * 60)

        else:
            # Text prompt mode
            logger.info(f"Prompt: {args.prompt}")
            text_output, audio_codes = model.generate_interleaved(
                args.prompt,
                max_new_tokens=args.max_tokens,
                audio_temperature=args.audio_temperature,
                text_temperature=args.temperature,
                system_prompt=args.system,
            )
            print("\n" + "=" * 60)
            print(f"Input:  {args.prompt}")
            print(f"Text:   {text_output}")
            print(f"Audio:  {len(audio_codes)} frames")

            if args.save_codes and audio_codes:
                codes_array = np.stack(audio_codes, axis=0)  # [T, 8]
                np.save(args.save_codes, codes_array)
                print(f"Codes:  {args.save_codes} {codes_array.shape}")

            if args.output and audio_codes:
                if audio_codes_to_wav(
                    audio_codes,
                    args.output,
                    model_dir=args.model_dir,
                    audio_detokenizer_file=files["audio_detokenizer"],
                ):
                    print(f"Output: {args.output}")
            print("=" * 60)


if __name__ == "__main__":
    main()
