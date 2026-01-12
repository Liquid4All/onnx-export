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
        # Prefer audio_detokenizer_lfm (has sliding window attention fix)
        "audio_detokenizer": onnx_dir / f"audio_detokenizer_lfm{suffix}.onnx",
    }

    # Fall back to fp32 if requested precision not available
    for name, path in files.items():
        if not path.exists():
            fp32_path = onnx_dir / f"{name}.onnx"
            if fp32_path.exists():
                logger.info(f"{path.name} not found, using {fp32_path.name}")
                files[name] = fp32_path
            # Special case: audio_detokenizer_lfm -> audio_detokenizer_lfm.onnx
            if name == "audio_detokenizer":
                lfm_path = onnx_dir / "audio_detokenizer_lfm.onnx"
                if lfm_path.exists():
                    logger.info(f"Using {lfm_path.name} (with sliding window attention)")
                    files[name] = lfm_path

    return files


def load_session(model_path: pathlib.Path) -> ort.InferenceSession:
    """Load ONNX model as inference session."""
    providers = ["CPUExecutionProvider"]
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(model_path), sess_options, providers=providers)


class LFM2AudioInferenceFull:
    """Full ONNX inference for LFM2.5-Audio supporting all modes."""

    # Special tokens (from tokenizer)
    AUDIO_START_TOKEN = 128  # <|audio_start|>
    TEXT_START_TOKEN = 129  # <|text_start|>
    TEXT_END_TOKEN = 130  # <|text_end|>
    MIXED_START_TOKEN = 131  # <|mixed_start|>
    MIXED_END_TOKEN = 132  # <|mixed_end|>

    def __init__(
        self,
        model_dir: pathlib.Path,
        precision: str = "fp32",
        use_pytorch_depthformer: bool = True,
    ):
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

        # Load PyTorch depthformer for autoregressive inference (more accurate)
        self.pytorch_depthformer = None
        self.use_pytorch_depthformer = use_pytorch_depthformer
        if use_pytorch_depthformer:
            self._load_pytorch_depthformer()

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

    def _load_pytorch_depthformer(self):
        """Load PyTorch depthformer components for autoregressive inference."""
        try:
            import torch
            from liquid_audio.model.lfm2_audio import LFM2AudioModel

            logger.info("Loading PyTorch model for autoregressive depthformer...")
            model = LFM2AudioModel.from_pretrained(
                "LiquidAI/LFM2.5-Audio-1.5B",
                dtype=torch.float32,
                device="cpu"
            )
            model.eval()

            # Store only the depthformer components (not the full model)
            self.pytorch_depthformer = {
                "depth_linear": model.depth_linear,
                "depthformer": model.depthformer,
                "depth_embeddings": model.depth_embeddings,
                "codebooks": model.codebooks,
                "depthformer_dim": model.depthformer_dim,
            }
            logger.info("PyTorch depthformer loaded successfully")
        except ImportError:
            logger.warning("liquid_audio not available, using ONNX depthformer (parallel, less accurate)")
            self.pytorch_depthformer = None
        except Exception as e:
            logger.warning(f"Failed to load PyTorch depthformer: {e}")
            self.pytorch_depthformer = None

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
        """Run depthformer to predict 8 codebook logits from hidden states.

        This is the parallel (non-autoregressive) version using ONNX.
        For autoregressive inference, use _sample_audio_codes_autoregressive.
        """
        if self.depthformer_session is None:
            raise RuntimeError("depthformer not loaded")

        # hidden_states: [batch, hidden_size]
        outputs = self.depthformer_session.run(
            ["codebook_logits"], {"hidden_states": hidden_states.astype(np.float32)}
        )
        return outputs[0]  # [batch, 8, 2049]

    # End-of-audio token (same across all codebooks)
    END_OF_AUDIO_TOKEN = 2048

    def _sample_audio_codes_autoregressive(
        self, hidden_states: np.ndarray, temperature: float = 0.9
    ) -> np.ndarray:
        """Sample audio codes using autoregressive PyTorch depthformer.

        This is the correct autoregressive implementation that matches the
        reference liquid_audio code. Each codebook prediction depends on the
        sampled token from the previous codebook.

        Token 2048 is the end-of-audio token. When the model predicts this,
        it signals the end of audio generation.
        """
        import torch
        from einops import rearrange

        df = self.pytorch_depthformer
        codebooks = df["codebooks"]
        depthformer_dim = df["depthformer_dim"]

        # Convert to torch tensor and handle different input shapes
        hidden_tensor = torch.from_numpy(hidden_states).float()
        # Squeeze to [batch, hidden_size] if needed
        if hidden_tensor.ndim == 3:
            hidden_tensor = hidden_tensor.squeeze(1)  # [batch, 1, hidden_size] -> [batch, hidden_size]
        batch_size = hidden_tensor.shape[0]

        codes_list = []
        for b in range(batch_size):
            embedding = hidden_tensor[b]  # [hidden_size]

            # Project to depthformer dimensions
            with torch.no_grad():
                depthformer_in = rearrange(
                    df["depth_linear"](embedding),
                    "(C D) -> C D",
                    C=codebooks,
                    D=depthformer_dim
                )

            depthformer_token = torch.zeros_like(depthformer_in[0])
            cache = None
            out_tokens = []

            for i in range(codebooks):
                cur_input = depthformer_in[i] + depthformer_token

                with torch.no_grad():
                    depthformer_out, cache = df["depthformer"].forward_cached(
                        cur_input[None, None, :], cache
                    )
                    logits = df["depth_embeddings"][i].get_logits(
                        depthformer_out.squeeze()
                    )  # [2049]

                # Sample from all logits including end-of-audio token (2048)
                all_logits = logits.numpy()
                if temperature is None or temperature <= 0:
                    token = int(np.argmax(all_logits))
                else:
                    token = self._sample(all_logits, temperature, top_p=0.95)

                out_tokens.append(token)

                # Get embedding for next iteration (use clamped token for embedding lookup)
                embed_token = min(token, 2047)  # Clamp to valid embedding range
                with torch.no_grad():
                    depthformer_token = df["depth_embeddings"][i](
                        torch.tensor(embed_token)
                    ).squeeze()

            codes_list.append(out_tokens)

        return np.array(codes_list, dtype=np.int64)  # [batch, 8]

    def _is_end_of_audio(self, frame_codes: np.ndarray) -> bool:
        """Check if audio frame indicates end of audio.

        End of audio is signaled when any codebook outputs the end token (2048).
        """
        return np.any(frame_codes >= self.END_OF_AUDIO_TOKEN)

    def _sample_audio_codes(
        self, codebook_logits: np.ndarray, temperature: float = 0.9
    ) -> np.ndarray:
        """Sample audio codes from depthformer logits (parallel version).

        The depthformer outputs 2049 logits per codebook:
        - Indices 0-2047: valid audio codes
        - Index 2048: special/padding token (should be ignored for sampling)

        Note: This is the parallel (non-autoregressive) version.
        For more accurate results, use _sample_audio_codes_autoregressive.
        """
        # codebook_logits: [batch, 8, 2049]
        batch_size, num_codebooks, vocab_size = codebook_logits.shape
        codes = np.zeros((batch_size, num_codebooks), dtype=np.int64)

        for cb_idx in range(num_codebooks):
            # Only sample from valid codes (exclude last special token)
            logits = codebook_logits[:, cb_idx, :2048]  # [batch, 2048]
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
        return (
            "<|startoftext|><|im_start|>system\n"
            "Perform ASR.<|im_end|>\n"
            "<|im_start|>user\n"
        )

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
            ["audio_embeddings", "output_lengths"],
            {"mel_features": mel_features.astype(np.float32), "mel_lengths": mel_lengths},
        )

        # Build the prompt: prefix + audio + suffix
        # 1. Encode prefix text (system + user start)
        # Note: add_special_tokens=False since we include <|startoftext|> in the prompt
        prefix_text = self._format_asr_prompt()
        prefix_ids = self.tokenizer.encode(prefix_text, return_tensors="np", add_special_tokens=False)
        prefix_embeds = self._get_text_embeds(prefix_ids)

        # 2. Encode suffix text (user end + assistant start)
        suffix_text = self._format_asr_suffix()
        suffix_ids = self.tokenizer.encode(suffix_text, return_tensors="np", add_special_tokens=False)
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
        next_token = self._sample(next_logits, temperature, top_p=0.9)

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
            next_token = self._sample(next_logits, temperature, top_p=0.9)

            generated_tokens.append(next_token)
            total_len += 1

        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # === TTS (Text → Audio) ===

    def _format_tts_prompt(self, text: str) -> str:
        """Format text with TTS system instruction using ChatML format."""
        return (
            "<|startoftext|><|im_start|>system\n"
            "Perform TTS.<|im_end|>\n"
            f"<|im_start|>user\n{text}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def synthesize(
        self,
        text: str,
        max_audio_frames: int = 100,
        audio_temperature: float = 0.9,
        text_temperature: float = 0.7,
        max_text_tokens: int = 50,
    ) -> list[np.ndarray]:
        """Synthesize audio from text using depthformer.

        The model must first generate text tokens until it produces <|audio|>,
        then we switch to depthformer-based audio code generation.

        Returns list of audio code frames (8 codes each).
        Each frame is [8] array of codebook indices.
        """
        if self.depthformer_session is None:
            raise RuntimeError("depthformer not loaded, TTS unavailable")

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

        # === Phase 1: Generate text until <|audio|> token ===
        in_audio_mode = False
        for _ in range(max_text_tokens):
            last_logits = logits[0, -1, : self.vocab_size]
            next_token = self._sample(last_logits, text_temperature, top_p=0.9)

            if next_token == self.tokenizer.eos_token_id:
                logger.warning("Model produced EOS before audio, TTS may not work")
                break

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

        # === Phase 2: Generate audio frames using depthformer ===
        audio_codes = []
        start_time = time.time()

        # Use autoregressive PyTorch depthformer if available (more accurate)
        use_autoregressive = self.pytorch_depthformer is not None

        for frame_idx in range(max_audio_frames):
            # Get hidden states for the last position: [1, hidden_size]
            last_hidden = hidden_states[0, -1:, :]  # [1, hidden_size]

            # Sample audio codes
            if use_autoregressive:
                # Autoregressive sampling (correct, matches reference)
                frame_codes = self._sample_audio_codes_autoregressive(
                    last_hidden, audio_temperature
                )  # [1, 8]
            else:
                # Parallel sampling via ONNX (faster but less accurate)
                codebook_logits = self._run_depthformer(last_hidden)  # [1, 8, 2049]
                frame_codes = self._sample_audio_codes(codebook_logits, audio_temperature)  # [1, 8]

            # Check for end-of-audio (any codebook outputs 2048)
            if self._is_end_of_audio(frame_codes[0]):
                logger.info(f"End of audio detected at frame {frame_idx}")
                break

            audio_codes.append(frame_codes[0])  # [8]

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
        max_new_tokens: int = 200,
        audio_temperature: float = 0.9,
        text_temperature: float = 0.7,
    ) -> tuple[str, list[np.ndarray]]:
        """Generate interleaved text and audio using depthformer for audio."""
        # Note: add_special_tokens=False since we include <|startoftext|> in the prompt
        formatted_prompt = self._format_interleaved_prompt(prompt)
        input_ids = self.tokenizer.encode(formatted_prompt, return_tensors="np", add_special_tokens=False)
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
                depthformer_available = (
                    self.pytorch_depthformer is not None or
                    self.depthformer_session is not None
                )
                if not depthformer_available or hidden_states is None:
                    logger.warning("Depthformer unavailable, exiting audio mode")
                    in_audio_mode = False
                    continue

                last_hidden = hidden_states[0, -1:, :]

                # Use autoregressive PyTorch depthformer if available
                if self.pytorch_depthformer is not None:
                    frame_codes = self._sample_audio_codes_autoregressive(
                        last_hidden, audio_temperature
                    )
                else:
                    codebook_logits = self._run_depthformer(last_hidden)
                    frame_codes = self._sample_audio_codes(codebook_logits, audio_temperature)

                # Check for end of audio (token 2048 in any codebook)
                if self._is_end_of_audio(frame_codes[0]):
                    logger.info(f"End of audio detected at frame {len(audio_codes)}")
                    in_audio_mode = False
                    continue

                audio_codes.append(frame_codes[0])

                # Feed all 8 codebook tokens as a summed embedding (like PyTorch reference)
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
                continue  # Skip the decoder update at the end of the loop
            else:
                # Sample from text vocabulary
                text_logits = last_logits[: self.vocab_size]
                token = self._sample(text_logits, text_temperature, top_p=0.9)

                if token == self.tokenizer.eos_token_id:
                    break

                if token == self.AUDIO_START_TOKEN:
                    logger.info("Model entered audio mode")
                    in_audio_mode = True
                    # Feed audio_start token to get hidden states for first audio frame
                    next_ids = np.array([[self.AUDIO_START_TOKEN]], dtype=np.int64)
                    next_embeds = self._get_text_embeds(next_ids)
                    attention_mask = np.ones((batch_size, total_len + 1), dtype=np.int64)
                    logits, hidden_states, cache = self._run_decoder(next_embeds, attention_mask, cache)
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


def audio_codes_to_wav(
    audio_codes: list[np.ndarray],
    output_path: str,
    model_dir: pathlib.Path | None = None,
    sample_rate: int = 24000,
    precision: str = "fp32",
    use_onnx: bool = False,
):
    """Convert audio codes to WAV file.

    By default uses PyTorch decoding which produces correct audio.
    Set use_onnx=True to use ONNX (may have quality issues due to
    sliding_attention vs full_attention architecture mismatch).
    """
    if len(audio_codes) < 2:
        logger.warning("Not enough audio codes to generate audio")
        return False

    # Stack codes: [T, 8] → [8, T]
    codes = np.stack(audio_codes, axis=0)  # [T, 8]
    codes = np.clip(codes, 0, 2047)
    codes_transposed = codes.T  # [8, T]

    # Try PyTorch first (preferred - produces correct audio)
    if not use_onnx:
        result = _decode_audio_pytorch(codes, output_path, sample_rate)
        if result:
            return True
        logger.warning("PyTorch decode failed, trying ONNX fallback")

    # Try ONNX-based decoding
    if model_dir is not None:
        onnx_dir = model_dir / "onnx"
        suffix = "" if precision == "fp32" else f"_{precision}"

        # Prefer PyTorch-exported model (audio_detokenizer_lfm.onnx)
        detok_path = onnx_dir / f"audio_detokenizer_lfm{suffix}.onnx"
        if not detok_path.exists():
            detok_path = onnx_dir / "audio_detokenizer_lfm.onnx"
        # Fall back to builder-based model if PyTorch export not available
        if not detok_path.exists():
            detok_path = onnx_dir / f"audio_detokenizer{suffix}.onnx"
        if not detok_path.exists():
            detok_path = onnx_dir / "audio_detokenizer.onnx"
        istft_config_path = onnx_dir / "istft_config.json"

        if detok_path.exists() and istft_config_path.exists():
            try:
                return _decode_audio_onnx(
                    codes_transposed, detok_path, istft_config_path, output_path, sample_rate
                )
            except Exception as e:
                logger.warning(f"ONNX decode failed: {e}")

    logger.error("All audio decoding methods failed")
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


def _decode_audio_onnx(
    codes: np.ndarray,
    detok_path: pathlib.Path,
    istft_config_path: pathlib.Path,
    output_path: str,
    sample_rate: int,
) -> bool:
    """Decode audio using ONNX detokenizer + custom ISTFT.

    Uses custom ISTFT with 'same' padding to match liquid_audio behavior.
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
            if audio_codes_to_wav(audio_codes, args.output, model_dir=args.model_dir, precision=args.precision):
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
            if audio_codes_to_wav(audio_codes, args.output, model_dir=args.model_dir, precision=args.precision):
                print(f"Output: {args.output}")
        print("=" * 60)


if __name__ == "__main__":
    main()
