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
- audio_detokenizer.onnx: Neural vocoder for TTS
- vocoder_projection.onnx: [B, 2048] → [B, 8, 1024] (called 1× per frame)
- vocoder_depthformer.onnx: Transformer+embed+logits (called 8× per frame)

All components including depthformer use ONNX-only inference.

Usage:
    # Text generation
    uv run lfm2-audio-infer /path/to/model --prompt "Hello world"

    # ASR: Transcribe audio to text
    uv run lfm2-audio-infer /path/to/model --mode asr --audio input.wav

    # TTS: Generate audio from text
    uv run lfm2-audio-infer /path/to/model --mode tts --prompt "Hello world" --output output.wav

    # Interleaved with text prompt
    uv run lfm2-audio-infer /path/to/model --mode interleaved --prompt "Respond with audio"

    # Interleaved with audio input (recommended)
    uv run lfm2-audio-infer /path/to/model --mode interleaved --audio input.wav --output output.wav
"""

import argparse
import logging
import pathlib
import time

import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)


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
            "vocoder_projection": None,
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
        "vocoder_projection": f"vocoder_projection_{precision}.onnx",
        "vocoder_depthformer": f"vocoder_depthformer_{precision}.onnx",
    }


def load_session(model_path: pathlib.Path) -> ort.InferenceSession:
    """Load ONNX model as inference session."""
    providers = ["CPUExecutionProvider"]
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(model_path), sess_options, providers=providers)


class LFM2AudioInference:
    """ONNX inference for LFM2.5-Audio supporting all modes."""

    # Special tokens (from tokenizer)
    AUDIO_START_TOKEN = 128  # <|audio_start|>
    TEXT_START_TOKEN = 129  # <|text_start|>
    TEXT_END_TOKEN = 130  # <|text_end|>
    MIXED_START_TOKEN = 131  # <|mixed_start|>
    MIXED_END_TOKEN = 132  # <|mixed_end|>

    def __init__(
        self,
        model_dir: pathlib.Path,
        decoder_file: str | None = None,
        audio_embedding_file: str | None = None,
        audio_encoder_file: str | None = None,
        audio_detokenizer_file: str | None = None,
        vocoder_projection_file: str | None = None,
        vocoder_depthformer_file: str | None = None,
    ):
        self.model_dir = model_dir
        self.onnx_dir = model_dir / "onnx"

        # Store file names for vocoder loading
        self._vocoder_projection_file = vocoder_projection_file
        self._vocoder_depthformer_file = vocoder_depthformer_file

        # Load tokenizer
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)

        # Resolve file paths (use provided or default)
        decoder_path = self.onnx_dir / (decoder_file or "decoder.onnx")
        audio_embedding_path = self.onnx_dir / (audio_embedding_file or "audio_embedding.onnx")
        audio_encoder_path = self.onnx_dir / (audio_encoder_file or "audio_encoder.onnx")
        audio_detokenizer_path = self.onnx_dir / (audio_detokenizer_file or "audio_detokenizer.onnx")

        logger.info(f"Loading decoder from {decoder_path.name}...")
        self.decoder_session = load_session(decoder_path)

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
        self._load_embed_tokens_weight()

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

    def _load_embed_tokens_weight(self):
        """Load embed_tokens weight from model weights for text embedding lookup.

        Tries to load from (in order):
        1. Pre-exported numpy file (onnx/embed_tokens.npy) - no PyTorch needed
        2. Local model.safetensors - requires PyTorch
        3. HuggingFace download - requires PyTorch
        """
        # Option 1: Pre-exported numpy file (no PyTorch dependency)
        numpy_path = self.model_dir / "onnx" / "embed_tokens.npy"
        if numpy_path.exists():
            logger.info(f"Loading embed_tokens from {numpy_path.name}...")
            self.embed_tokens_weight = np.load(numpy_path)
            logger.info(f"embed_tokens weight loaded: {self.embed_tokens_weight.shape}")
            return

        # Option 2/3: Load from safetensors (requires PyTorch for bfloat16)
        logger.info("embed_tokens.npy not found, falling back to safetensors (requires PyTorch)...")

        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        local_weights = self.model_dir / "model.safetensors"
        if local_weights.exists():
            weights_path = str(local_weights)
        else:
            try:
                weights_path = hf_hub_download("LiquidAI/LFM2.5-Audio-1.5B", "model.safetensors")
            except Exception as e:
                logger.warning(f"Could not load model weights: {e}")
                self.embed_tokens_weight = None
                return

        logger.info("Loading embed_tokens weight for text embedding...")
        weights = load_file(weights_path)
        embed_tensor = weights["lfm.embed_tokens.weight"].float()
        self.embed_tokens_weight = embed_tensor.numpy()

        logger.info(f"embed_tokens weight loaded: {self.embed_tokens_weight.shape}")

    def _load_onnx_depthformer(self):
        """Load ONNX vocoder models for autoregressive inference.

        Loads 2-model structure:
        - vocoder_projection.onnx: [B, 2048] → [B, 8, 1024] (called 1× per frame)
        - vocoder_depthformer.onnx: Transformer+embed+logits (called 8× per frame)
        """
        projection_path = self.onnx_dir / (
            self._vocoder_projection_file or "vocoder_projection.onnx"
        )
        depthformer_path = self.onnx_dir / (
            self._vocoder_depthformer_file or "vocoder_depthformer.onnx"
        )

        if not projection_path.exists() or not depthformer_path.exists():
            logger.warning("Vocoder ONNX models not found, TTS will not be available")
            return

        try:
            logger.info(f"Loading vocoder_projection from {projection_path.name}...")
            logger.info(f"Loading vocoder_depthformer from {depthformer_path.name}...")

            self.onnx_depthformer = {}
            self.onnx_depthformer["depth_linear"] = load_session(projection_path)
            self.onnx_depthformer["depthformer_unified"] = load_session(depthformer_path)
            logger.info("ONNX vocoder ready for TTS")

        except Exception as e:
            logger.warning(f"Failed to load ONNX depthformer: {e}")
            self.onnx_depthformer = None

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
        """Get text embeddings via numpy lookup."""
        # input_ids: [batch, seq_len] -> embeds: [batch, seq_len, hidden]
        return self.embed_tokens_weight[input_ids]

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

    # End-of-audio token (same across all codebooks)
    END_OF_AUDIO_TOKEN = 2048

    def _sample_audio_codes(
        self,
        hidden_states: np.ndarray,
        temperature: float = 0.9,
        top_k: int | None = None,
    ) -> np.ndarray:
        """Sample audio codes using ONNX autoregressive depthformer.

        Uses the consolidated depthformer_unified model which combines:
        - Transformer step with KV cache
        - All 8 embedding tables
        - All 8 logits projections

        Token 2048 is the end-of-audio token. When the model predicts this,
        it signals the end of audio generation.

        Args:
            hidden_states: [batch, hidden_size] or [batch, 1, hidden_size]
            temperature: Sampling temperature
            top_k: Optional top-k sampling (e.g., 4 for liquid-audio interleaved)

        Returns:
            codes: [batch, 8] audio codes for each codebook
        """
        df = self.onnx_depthformer

        if df is None or "depthformer_unified" not in df:
            raise RuntimeError(
                "ONNX depthformer not available for TTS.\n"
                "Ensure depthformer_unified.onnx is exported."
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

            # Project to depth dimension: [1, 2048] → [1, 8, 1024]
            depth_hidden = df["depth_linear"].run(
                ["depth_hidden"], {"hidden_states": embedding.astype(np.float32)}
            )[0]  # [1, 8, 1024]

            # Initialize KV cache for depthformer (6 layers)
            past_keys = np.zeros((num_layers, 1, 0, num_kv_heads, head_dim), dtype=np.float32)
            past_values = np.zeros((num_layers, 1, 0, num_kv_heads, head_dim), dtype=np.float32)

            out_tokens = []
            prev_token = 0

            for i in range(num_codebooks):
                logits, _, new_keys, new_values = df["depthformer_unified"].run(
                    ["logits", "token_embed", "new_keys", "new_values"],
                    {
                        "depth_slices": depth_hidden.astype(np.float32),
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

        Uses liquid_audio processor when available for proper preprocessing,
        falls back to torchaudio with approximate parameters otherwise.

        Returns:
            mel_features: [1, time, 128] mel spectrogram
            mel_lengths: [1] length array
        """
        import torch
        import torchaudio

        waveform, sample_rate = torchaudio.load(audio_path)

        # Resample to 16kHz if needed
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
            sample_rate = 16000

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Try to use liquid_audio processor for proper preprocessing
        try:
            from liquid_audio import LFM2AudioProcessor

            processor = LFM2AudioProcessor.from_pretrained(
                "LiquidAI/LFM2.5-Audio-1.5B",
                device="cpu",
            )
            length = torch.tensor([waveform.shape[1]], dtype=torch.long)
            mel, mel_length = processor.audio(waveform, length)

            # mel shape: [1, 128, time] -> [1, time, 128]
            mel_features = mel[0].transpose(0, 1).unsqueeze(0).numpy()
            mel_lengths = np.array([mel_features.shape[1]], dtype=np.int64)
            logger.info("Using liquid_audio processor for mel spectrogram")

        except ImportError as e:
            logger.warning(f"liquid_audio not available ({e}), using torchaudio fallback")
            # Fallback to torchaudio (less accurate)
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

        return mel_features.astype(np.float32), mel_lengths

    def _format_asr_prompt(self) -> str:
        """Format ASR system instruction using ChatML format.

        The audio embeddings will be inserted at the user position.
        """
        return "<|startoftext|><|im_start|>system\nPerform ASR.<|im_end|>\n<|im_start|>user\n"

    def _format_asr_suffix(self) -> str:
        """Format the suffix after audio embeddings."""
        return "<|im_end|>\n<|im_start|>assistant\n"

    def transcribe(
        self,
        audio_path: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
    ) -> str:
        """Transcribe audio to text using ChatML format.

        The prompt structure is:
        <|startoftext|><|im_start|>system
        Perform ASR.<|im_end|>
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
        prefix_text = self._format_asr_prompt()
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
            # Also stop on <|im_end|> token (token 7)
            if next_token == 7:
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

    def _format_tts_prompt(self, text: str) -> str:
        """Format text with TTS system instruction using ChatML format."""
        return (
            "<|startoftext|><|im_start|>system\n"
            "Perform TTS. Use the UK female voice.<|im_end|>\n"
            f"<|im_start|>user\n{text}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def synthesize(
        self,
        text: str,
        max_new_tokens: int = 100,
        audio_temperature: float = 0.7,
        text_temperature: float = 0.7,
    ) -> list[np.ndarray]:
        """Synthesize audio from text using depthformer.

        The model first generates text tokens until it produces <|audio|>,
        then switches to depthformer-based audio code generation.

        Args:
            text: Text to synthesize.
            max_new_tokens: Maximum number of new tokens (text + audio frames combined),
                matching PyTorch reference behavior.
            audio_temperature: Temperature for audio sampling (0 = greedy).
            text_temperature: Temperature for text sampling (0 = greedy).

        Returns list of audio code frames (8 codes each).
        Each frame is [8] array of codebook indices.
        """
        if self.onnx_depthformer is None or "depthformer_unified" not in self.onnx_depthformer:
            raise RuntimeError("ONNX depthformer not loaded, TTS unavailable")

        # Format prompt with TTS system instruction
        # Note: add_special_tokens=False since we include <|startoftext|> in the prompt
        prompt = self._format_tts_prompt(text)
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

            # Sample audio codes (autoregressive sampling, matches reference)
            frame_codes = self._sample_audio_codes(last_hidden, audio_temperature)  # [1, 8]

            # Check for end-of-audio (any codebook outputs 2048)
            if self._is_end_of_audio(frame_codes[0]):
                logger.info(f"End of audio detected at frame {len(audio_codes)}")
                break

            audio_codes.append(frame_codes[0])  # [8]
            tokens_generated += 1

            # Feed back audio codes to continue generation
            # Audio embedding expects tokens in range [0, 16392) where:
            # token = codebook_idx * 2049 + code_value
            # Reference: in_emb = self.audio_embedding(next_token + self.codebook_offsets).sum(0)
            # We get embeddings for all 8 codebooks and SUM them into a single embedding
            # Clamp codes to valid range for embedding lookup (0-2047)
            clamped_codes = np.minimum(frame_codes[0], 2047)
            audio_tokens = np.array(
                [
                    [
                        cb_idx * self.codebook_vocab + int(clamped_codes[cb_idx])
                        for cb_idx in range(self.num_codebooks)
                    ]
                ],
                dtype=np.int64,
            )  # [1, 8]
            all_embeds = self._get_audio_embeds(audio_tokens)  # [1, 8, 2048]
            # Sum embeddings across codebooks (axis=1), keep as [1, 1, 2048]
            next_embeds = all_embeds.sum(axis=1, keepdims=True)  # [1, 1, 2048]

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

    def _format_interleaved_prompt(self, text: str) -> str:
        """Format text with interleaved system instruction using ChatML format."""
        return (
            "<|startoftext|><|im_start|>system\n"
            "Respond with interleaved text and audio.<|im_end|>\n"
            f"<|im_start|>user\n{text}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def generate_interleaved(
        self,
        prompt: str,
        max_new_tokens: int = 20,
        audio_temperature: float = 0,
        text_temperature: float = 0,
    ) -> tuple[str, list[np.ndarray]]:
        """Generate interleaved text and audio from text prompt.

        Defaults match liquid-audio library defaults (greedy decoding).
        """
        # Note: add_special_tokens=False since we include <|startoftext|> in the prompt
        formatted_prompt = self._format_interleaved_prompt(prompt)
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
                if (
                    self.onnx_depthformer is None
                    or "depthformer_unified" not in self.onnx_depthformer
                    or hidden_states is None
                ):
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
                                cb_idx * self.codebook_vocab + self.END_OF_AUDIO_TOKEN
                                for cb_idx in range(self.num_codebooks)
                            ]
                        ],
                        dtype=np.int64,
                    )
                    all_embeds = self._get_audio_embeds(audio_tokens)
                    next_embeds = all_embeds.sum(axis=1, keepdims=True)
                    attention_mask = np.ones((batch_size, total_len + 1), dtype=np.int64)
                    logits, hidden_states, cache = self._run_decoder(
                        next_embeds, attention_mask, cache
                    )
                    total_len += 1
                    continue

                audio_codes.append(frame_codes[0])

                # Feed all 8 codebook tokens as a summed embedding (like PyTorch reference)
                # Token 2048 is valid in the embedding table (2049 entries per codebook)
                audio_tokens = np.array(
                    [
                        [
                            cb_idx * self.codebook_vocab + int(frame_codes[0][cb_idx])
                            for cb_idx in range(self.num_codebooks)
                        ]
                    ],
                    dtype=np.int64,
                )  # [1, 8]
                all_embeds = self._get_audio_embeds(audio_tokens)  # [1, 8, 2048]
                # Sum embeddings across codebooks (axis=1), keep as [1, 1, 2048]
                next_embeds = all_embeds.sum(axis=1, keepdims=True)  # [1, 1, 2048]

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

    def generate_interleaved_from_audio(
        self,
        audio_path: str,
        max_new_tokens: int = 300,
        text_temperature: float = 1.0,
        audio_temperature: float = 1.0,
        audio_top_k: int = 4,
    ) -> tuple[str, list[np.ndarray]]:
        """Generate interleaved text+audio response from audio input.

        Defaults match official liquid-audio demo (not library defaults).
        Uses counter-based mode switching:
        - interleaved_n_text = 6 (text tokens before switching to audio)
        - interleaved_n_audio = 12 (audio frames before switching to text)

        Args:
            audio_path: Path to input audio file
            max_new_tokens: Maximum tokens to generate
            text_temperature: Sampling temperature for text (1.0 matches liquid-audio)
            audio_temperature: Sampling temperature for audio (1.0 matches liquid-audio)
            audio_top_k: Top-k sampling for audio (4 matches liquid-audio)

        Returns:
            Tuple of (text_response, audio_codes)
        """
        # Encode audio
        mel_features, mel_lengths = self._compute_mel_features(audio_path)
        audio_embeds, _ = self.audio_encoder_session.run(
            ["audio_embeddings", "audio_lengths"],
            {"mel_spectrogram": mel_features.astype(np.float32), "mel_lengths": mel_lengths},
        )

        # Build prompt: system + user audio + assistant
        # System prompt matches official liquid-audio demo
        prefix_text = (
            "<|startoftext|><|im_start|>system\n"
            "Respond with interleaved text and audio.<|im_end|>\n"
            "<|im_start|>user\n"
        )
        suffix_text = "<|im_end|>\n<|im_start|>assistant\n"

        prefix_ids = self.tokenizer.encode(
            prefix_text, return_tensors="np", add_special_tokens=False
        )
        suffix_ids = self.tokenizer.encode(
            suffix_text, return_tensors="np", add_special_tokens=False
        )

        prefix_embeds = self._get_text_embeds(prefix_ids)
        suffix_embeds = self._get_text_embeds(suffix_ids)

        # Concatenate: prefix + audio + suffix
        all_embeds = np.concatenate([prefix_embeds, audio_embeds, suffix_embeds], axis=1)

        batch_size = 1
        seq_len = all_embeds.shape[1]
        cache = self._init_cache(batch_size)

        attention_mask = np.ones((batch_size, seq_len), dtype=np.int64)
        logits, hidden_states, cache = self._run_decoder(all_embeds, attention_mask, cache)

        # Generate with counter-based mode switching (matching liquid-audio)
        INTERLEAVED_N_TEXT = 6
        INTERLEAVED_N_AUDIO = 12

        text_tokens = []
        audio_codes = []
        total_len = seq_len
        in_audio_mode = False
        modality_left = INTERLEAVED_N_TEXT
        text_done = False

        for step in range(max_new_tokens):
            modality_left -= 1

            if in_audio_mode:
                if self.onnx_depthformer is None or hidden_states is None:
                    logger.warning("Depthformer unavailable, exiting audio mode")
                    in_audio_mode = False
                    modality_left = INTERLEAVED_N_TEXT
                    continue

                last_hidden = hidden_states[0, -1:, :]
                frame_codes = self._sample_audio_codes(
                    last_hidden, temperature=audio_temperature, top_k=audio_top_k
                )
                frame = frame_codes[0]

                # Switch back to text after N audio frames (if text not done)
                if modality_left <= 0 and not text_done:
                    in_audio_mode = False
                    modality_left = INTERLEAVED_N_TEXT

                # Check for end of audio - ANY codebook with 2048 (matching liquid-audio)
                if (frame == 2048).any():
                    logger.info(f"Skipping frame with 2048 at step {step}")
                    # After text_done, 2048 means END of generation
                    if text_done:
                        logger.info(f"End of audio after text_done at step {step}")
                        break
                    in_audio_mode = False
                    modality_left = INTERLEAVED_N_TEXT
                    continue

                # Clamp and save
                clamped_frame = np.minimum(frame, 2047)
                audio_codes.append(clamped_frame.copy())

                # Get embeddings for next step
                clamped_codes = np.minimum(frame, 2047)
                audio_tokens = np.array(
                    [
                        [
                            cb_idx * self.codebook_vocab + int(clamped_codes[cb_idx])
                            for cb_idx in range(self.num_codebooks)
                        ]
                    ],
                    dtype=np.int64,
                )
                audio_embed = self._get_audio_embeds(audio_tokens)
                next_embeds = audio_embed.sum(axis=1, keepdims=True)

                if len(audio_codes) % 20 == 0:
                    logger.info(f"Generated {len(audio_codes)} audio frames...")
            else:
                # Generate text token
                last_logits = logits[0, -1, :]
                text_logits = last_logits[: self.vocab_size]
                token = self._sample(text_logits, text_temperature, top_p=None)

                if token == self.tokenizer.eos_token_id or token == 7:  # EOS or <|im_end|>
                    logger.info(f"End of turn at step {step}")
                    break

                if token == 130:  # <|text_end|>
                    logger.info(f"Text end at step {step}")
                    text_done = True

                # Switch to audio after N text tokens OR text_end
                if modality_left <= 0 or text_done:
                    in_audio_mode = True
                    modality_left = INTERLEAVED_N_AUDIO

                text_tokens.append(token)
                next_ids = np.array([[token]], dtype=np.int64)
                next_embeds = self._get_text_embeds(next_ids)

            # Update decoder
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
        logger.error("model_dir required for ONNX decoding")
        return False

    onnx_dir = model_dir / "onnx"
    detok_path = onnx_dir / (audio_detokenizer_file or "audio_detokenizer.onnx")

    if not detok_path.exists():
        logger.error(f"{detok_path.name} not found in {onnx_dir}")
        return False

    try:
        return _decode_audio_onnx_numpy(
            codes_transposed, detok_path, onnx_dir, output_path, sample_rate
        )
    except Exception as e:
        logger.error(f"ONNX decode failed: {e}")
        return False


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

    # Load window (or use default hann)
    window_path = onnx_dir / "istft_window.npy"
    if window_path.exists():
        window = np.load(window_path)
    else:
        window = np.hanning(n_fft).astype(np.float32)

    # Load ONNX detokenizer
    detok_session = load_session(detok_path)

    # Run detokenizer: [1, 8, T] → [1, T, 1282]
    codes_batch = codes[np.newaxis, :, :].astype(np.int64)  # [1, 8, T]
    stft_features = detok_session.run(["stft_features"], {"audio_codes": codes_batch})[0]
    stft_features = stft_features[0]  # [T, 1282]

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


def _decode_audio_onnx(
    codes: np.ndarray,
    detok_path: pathlib.Path,
    istft_config_path: pathlib.Path,
    output_path: str,
    sample_rate: int,
) -> bool:
    """Decode audio using ONNX detokenizer + custom ISTFT.

    Legacy function - use _decode_audio_onnx_numpy instead.
    """
    import json

    import scipy.io.wavfile
    import scipy.signal

    # Load ISTFT config
    with open(istft_config_path) as f:
        istft_config = json.load(f)

    n_fft = istft_config.get("n_fft", 1280)
    hop_length = istft_config.get("hop_length", 320)
    win_length = istft_config.get("win_length", 1280)
    n_fft_bins = n_fft // 2 + 1  # 641 for n_fft=1280

    # Load window
    onnx_dir = detok_path.parent
    window_path = onnx_dir / "istft_window.npy"
    if window_path.exists():
        window = np.load(window_path)
    else:
        # Fallback to hann window
        window = scipy.signal.windows.hann(n_fft, sym=False)

    # Load ONNX detokenizer
    detok_session = load_session(detok_path)

    # Run detokenizer: [1, 8, T] → [1, T, 1282]
    codes_batch = codes[np.newaxis, :, :].astype(np.int64)  # [1, 8, T]
    stft_features = detok_session.run(["stft_features"], {"audio_codes": codes_batch})[0]

    # stft_features shape: [1, T, 1282] where 1282 = n_fft_bins * 2
    # Format is [log_magnitude | angle] (NOT real + imag!)
    # Reference: liquid_audio/detokenizer.py lines 133-134
    stft_features = stft_features[0]  # [T, 1282]

    # Convert to complex STFT using polar form: magnitude * exp(i * angle)
    log_magnitude = stft_features[:, :n_fft_bins]  # [T, 641]
    angle = stft_features[:, n_fft_bins:]  # [T, 641]
    magnitude = np.exp(log_magnitude)
    complex_stft = magnitude * np.exp(1j * angle)  # polar to complex

    # Use custom ISTFT with 'same' padding (matches liquid_audio)
    # spec needs to be [freq, time]
    waveform = _istft_same_padding(complex_stft.T, n_fft, hop_length, win_length, window)

    # Normalize and save
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
    """Decode audio using PyTorch LFM2AudioDetokenizer.

    Uses the native liquid_audio detokenizer which has sliding_attention layers.
    This produces correct audio while the ONNX version (with full_attention) does not.
    """
    try:
        import json

        import scipy.io.wavfile
        import torch
        from accelerate import load_checkpoint_in_model
        from liquid_audio import LFM2AudioDetokenizer
        from liquid_audio.utils import get_model_dir
        from transformers import Lfm2Config

        # codes: [T, 8] → [1, 8, T]
        codes_tensor = torch.tensor(codes.T, dtype=torch.int64).unsqueeze(0)
        codes_tensor = torch.clamp(codes_tensor, 0, 2047)

        # Load detokenizer with native config (includes sliding_attention)
        cache_dir = get_model_dir("LiquidAI/LFM2.5-Audio-1.5B")
        config_path = cache_dir / "audio_detokenizer" / "config.json"
        with open(config_path) as f:
            config_dict = json.load(f)

        backbone_config = Lfm2Config(**config_dict)
        detok = LFM2AudioDetokenizer(backbone_config)

        weights_path = cache_dir / "audio_detokenizer" / "model.safetensors"
        load_checkpoint_in_model(detok, str(weights_path))
        detok.eval()

        with torch.no_grad():
            waveform = detok(codes_tensor)

        # Convert to numpy
        waveform_np = waveform[0].cpu().numpy()

        # Normalize
        max_val = np.abs(waveform_np).max()
        if max_val > 0:
            waveform_np = waveform_np / max_val

        # Convert to int16 for WAV
        waveform_int16 = (waveform_np * 32767).astype(np.int16)
        scipy.io.wavfile.write(output_path, sample_rate, waveform_int16)

        duration = len(waveform_np) / sample_rate
        logger.info(f"Saved audio to {output_path} ({duration:.2f}s) [PyTorch decode]")
        return True
    except Exception as e:
        logger.error(f"Failed to decode audio with PyTorch: {e}")
        import traceback

        traceback.print_exc()
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
        help="Input audio file for ASR/interleaved modes",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output audio file for TTS mode",
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
        "--vocoder-projection",
        metavar="FILE",
        help="Vocoder projection ONNX file (relative to onnx/ dir)",
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
        help="Maximum tokens/frames to generate (default: 100 for text/asr/tts, 300 for interleaved)",
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
        help="Audio sampling temperature (default: 0.7 for TTS)",
    )

    args = parser.parse_args()

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
    else:
        # TTS, text use temperature sampling
        if args.temperature is None:
            args.temperature = 0.7
        if args.audio_temperature is None:
            args.audio_temperature = 0.7

    # Apply mode-specific max_tokens defaults
    if args.max_tokens is None:
        if args.mode == "interleaved":
            args.max_tokens = 300
        else:
            args.max_tokens = 100

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

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
    if args.vocoder_projection:
        files["vocoder_projection"] = args.vocoder_projection
    if args.vocoder_depthformer:
        files["vocoder_depthformer"] = args.vocoder_depthformer

    logger.info(f"Loading model from {args.model_dir}...")
    model = LFM2AudioInference(
        args.model_dir,
        decoder_file=files["decoder"],
        audio_embedding_file=files["audio_embedding"],
        audio_encoder_file=files["audio_encoder"],
        audio_detokenizer_file=files["audio_detokenizer"],
        vocoder_projection_file=files["vocoder_projection"],
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
            max_new_tokens=args.max_tokens,
            audio_temperature=args.audio_temperature,
            text_temperature=args.temperature,
        )
        print("\n" + "=" * 60)
        print(f"Input: {args.prompt}")
        print(f"Generated {len(audio_codes)} audio frames")

        if args.output and audio_codes:
            if audio_codes_to_wav(
                audio_codes, args.output, model_dir=args.model_dir, audio_detokenizer_file=files["audio_detokenizer"]
            ):
                print(f"Output: {args.output}")
        print("=" * 60)

    elif args.mode == "interleaved":
        logger.info("Mode: Interleaved")
        if args.audio:
            # Audio input mode (matching liquid-audio demo)
            logger.info(f"Audio: {args.audio}")
            text_output, audio_codes = model.generate_interleaved_from_audio(
                args.audio,
                max_new_tokens=args.max_tokens,
                text_temperature=args.temperature,
                audio_temperature=args.audio_temperature,
            )
            print("\n" + "=" * 60)
            print(f"Audio input: {args.audio}")
        else:
            # Text prompt mode
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
            if audio_codes_to_wav(
                audio_codes, args.output, model_dir=args.model_dir, audio_detokenizer_file=files["audio_detokenizer"]
            ):
                print(f"Output: {args.output}")
        print("=" * 60)


if __name__ == "__main__":
    main()
