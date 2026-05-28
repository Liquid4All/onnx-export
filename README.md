<div align="center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/61b8e2ba285851687028d395/2b08LKpev0DNEk6DlnWkY.png" alt="Liquid AI" style="width: 100%; max-width: 100%;">

  <p>
    <a href="https://playground.liquid.ai/"><strong>Try LFM</strong></a> •
    <a href="https://docs.liquid.ai/lfm"><strong>Documentation</strong></a> •
    <a href="https://leap.liquid.ai/"><strong>LEAP</strong></a> •
    <a href="https://www.liquid.ai/blog/"><strong>Blog</strong></a>
  </p>
</div>

# LiquidONNX

ONNX export and inference tools for [LFM2](https://www.liquid.ai/liquid-foundation-models) models.

## 1. Supported Models

| Family | Quant Formats |
|--------|---------------|
| **LFM2.5**, **LFM2** | fp32, fp16, q4, q8 |
| **LFM2.5-VL**, **LFM2-VL** | fp32, fp16, q4, q8 |
| **LFM2.5-8B-A1B**, **LFM2-8B-A1B** | fp32, fp16, q4, q4f16 |
| **LFM2.5-Audio** | fp32, fp16, q4, q8 |


## 2. Installation

```bash
git clone https://github.com/Liquid4All/onnx-export.git
cd onnx-export
uv sync

# For GPU inference support
uv sync --extra gpu

# For development (testing, benchmarking)
uv sync --extra dev
```

## 3. Export

### 3.1 LFM2 Text Models

```bash
# All precisions
uv run lfm2-export LiquidAI/LFM2.5-1.2B-Instruct --precision
```

### 3.2 LFM2-VL Vision-Language Models

```bash
# All precisions
uv run lfm2-vl-export LiquidAI/LFM2.5-VL-1.6B --precision

# Conv2d vision format (alternative to default tiled)
uv run lfm2-vl-export LiquidAI/LFM2.5-VL-1.6B --vision-format conv2d
```

### 3.3 LFM2-MoE Mixture of Experts

```bash
# Current LFM2.5 MoE checkpoint
uv run lfm2-moe-export LiquidAI/LFM2.5-8B-A1B --precision

# Earlier LFM2 MoE checkpoint
uv run lfm2-moe-export LiquidAI/LFM2-8B-A1B --precision
```

## 4. Inference

All inference commands provide interactive multi-turn chat with streaming output. They automatically detect CUDA availability and fall back to CPU if needed.

### 4.1 Text Generation

```bash
# Interactive chat (starts conversation loop)
uv run lfm2-infer --model ./exports/LFM2.5-1.2B-Instruct-ONNX/onnx/model_q4.onnx

# Single prompt (non-interactive)
uv run lfm2-infer --model ./exports/LFM2.5-1.2B-Instruct-ONNX/onnx/model_q4.onnx \
    --prompt "Explain quantum computing"

# Force CPU execution
uv run lfm2-infer --model ./exports/LFM2.5-1.2B-Instruct-ONNX/onnx/model_q4.onnx --cpu
```

### 4.2 Vision-Language

```bash
# Single image analysis
uv run lfm2-vl-infer --model ./exports/LFM2.5-VL-1.6B-ONNX \
    --images photo.jpg \
    --prompt "What do you see in this image?"

# Multi-image comparison (up to 2 images)
uv run lfm2-vl-infer --model ./exports/LFM2.5-VL-1.6B-ONNX \
    --images image1.jpg image2.jpg \
    --prompt "Compare these two images"

# Text-only (no images)
uv run lfm2-vl-infer --model ./exports/LFM2.5-VL-1.6B-ONNX \
    --prompt "Hello, how are you?"
```

> **Note:** VL inference requires the model directory path (not a single .onnx file) since it loads multiple components: `embed_tokens.onnx`, `embed_images.onnx`, and `decoder.onnx`.

### 4.3 MoE

```bash
# Interactive chat
uv run lfm2-moe-infer --model ./exports/LFM2.5-8B-A1B-ONNX/onnx/model_q4.onnx

# Force CPU (when model does not fit VRAM)
uv run lfm2-moe-infer --model ./exports/LFM2.5-8B-A1B-ONNX/onnx/model_q4.onnx --cpu
```

### 4.4 Audio (ASR, TTS, Interleaved)

LFM2.5-Audio is a multimodal audio-language model supporting three modes:
- **ASR** (Automatic Speech Recognition): Transcribe audio to text
- **TTS** (Text-to-Speech): Generate audio from text
- **Interleaved**: Mixed text and audio input/output for conversational audio

The model uses 5 ONNX components:
- `decoder.onnx` - LFM2 language model backbone
- `audio_encoder.onnx` - Conformer encoder for ASR input
- `audio_embedding.onnx` - Audio code embeddings for TTS/interleaved
- `audio_detokenizer.onnx` - Converts audio codes to STFT features
- `vocoder_depthformer.onnx` - Autoregressive audio codebook prediction

```bash
# ASR: Transcribe audio to text
uv run lfm2-audio-infer LFM2.5-Audio-1.5B-ONNX --mode asr \
    --audio input.wav --precision q4

# TTS: Generate speech from text
uv run lfm2-audio-infer LFM2.5-Audio-1.5B-ONNX --mode tts \
    --prompt "Hello, how are you today?" \
    --system "Perform TTS. Use the UK female voice." \
    --output output.wav --precision q4

# Interleaved: Audio input with text+audio response
uv run lfm2-audio-infer LFM2.5-Audio-1.5B-ONNX --mode interleaved \
    --audio question.wav --output response.wav --precision q4

# Interactive chat mode (multi-turn with stateful KV cache)
uv run lfm2-audio-infer LFM2.5-Audio-1.5B-ONNX --mode interleaved --chat \
    --output output.wav --precision q4
# Commands in chat mode:
#   /audio <file> [text] - Send audio with optional text
#   <text>               - Send text message
#   reset                - Clear conversation state
#   quit                 - Exit
```

> **Note:** Audio inference requires the model directory path (not a single .onnx file) since it loads multiple components. Use `--precision` to select quantization level (fp16, q4, q8).

## 5. Testing

Tests verify ONNX exports against PyTorch reference models.

```bash
# Install dev dependencies
uv sync --extra dev

# LFM2 text model tests
uv run pytest tests/test_lfm2/test_decoder.py -v -k "q4"

# LFM2-VL vision-language tests
uv run pytest tests/test_lfm2_vl/test_decoder.py -v -k "450M"
uv run pytest tests/test_lfm2_vl/test_vision_encoder.py -v

# LFM2-MoE tests
uv run pytest tests/test_lfm2_moe/test_decoder.py -v
uv run pytest tests/test_lfm2_moe/test_tokenizer.py -v
```

Benchmarking, compare the CPU
```bash
# Text model benchmark
uv run lfm2-bench --model LiquidAI/LFM2.5-1.2B-Instruct \
    --onnx ./exports/LFM2.5-1.2B-Instruct-ONNX/onnx/model_q4.onnx
```

## 6. Pre-exported Models

### 6.1 LiquidAI

**Text models:**
- [LiquidAI/LFM2.5-1.2B-Base-ONNX](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Base-ONNX)
- [LiquidAI/LFM2.5-1.2B-Instruct-ONNX](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-ONNX)
- [LiquidAI/LFM2.5-1.2B-JP-ONNX](https://huggingface.co/LiquidAI/LFM2.5-1.2B-JP-ONNX)
- [LiquidAI/LFM2-2.6B-Transcript-ONNX](https://huggingface.co/LiquidAI/LFM2-2.6B-Transcript-ONNX)

**Vision-Language:**
- [LiquidAI/LFM2.5-VL-1.6B-ONNX](https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B-ONNX)

**Audio:**
- [LiquidAI/LFM2.5-Audio-1.5B-ONNX](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B-ONNX)

**MoE:**
- `LiquidAI/LFM2.5-8B-A1B` and `LiquidAI/LFM2-8B-A1B` are supported by this exporter.
- A pre-exported LiquidAI ONNX repo for `LFM2.5-8B-A1B` is not listed here yet.

### 6.2 onnx-community

**Text models:**
- [onnx-community/LFM2-350M-ONNX](https://huggingface.co/onnx-community/LFM2-350M-ONNX)
- [onnx-community/LFM2-700M-ONNX](https://huggingface.co/onnx-community/LFM2-700M-ONNX)
- [onnx-community/LFM2-1.2B-ONNX](https://huggingface.co/onnx-community/LFM2-1.2B-ONNX)
- [onnx-community/LFM2-2.6B-ONNX](https://huggingface.co/onnx-community/LFM2-2.6B-ONNX)
- [onnx-community/LFM2-2.6B-Exp-ONNX](https://huggingface.co/onnx-community/LFM2-2.6B-Exp-ONNX)

**Specialized:**
- [onnx-community/LFM2-350M-ENJP-MT-ONNX](https://huggingface.co/onnx-community/LFM2-350M-ENJP-MT-ONNX) — translation
- [onnx-community/LFM2-350M-Extract-ONNX](https://huggingface.co/onnx-community/LFM2-350M-Extract-ONNX)
- [onnx-community/LFM2-350M-Math-ONNX](https://huggingface.co/onnx-community/LFM2-350M-Math-ONNX)
- [onnx-community/LFM2-1.2B-Extract-ONNX](https://huggingface.co/onnx-community/LFM2-1.2B-Extract-ONNX)
- [onnx-community/LFM2-1.2B-RAG-ONNX](https://huggingface.co/onnx-community/LFM2-1.2B-RAG-ONNX)
- [onnx-community/LFM2-1.2B-Tool-ONNX](https://huggingface.co/onnx-community/LFM2-1.2B-Tool-ONNX)

**Vision-Language:**
- [onnx-community/LFM2-VL-450M-ONNX](https://huggingface.co/onnx-community/LFM2-VL-450M-ONNX)
- [onnx-community/LFM2-VL-1.6B-ONNX](https://huggingface.co/onnx-community/LFM2-VL-1.6B-ONNX)
- [onnx-community/LFM2-VL-3B-ONNX](https://huggingface.co/onnx-community/LFM2-VL-3B-ONNX)

**MoE:**
- [onnx-community/LFM2-8B-A1B-ONNX](https://huggingface.co/onnx-community/LFM2-8B-A1B-ONNX)

> **Note:** The onnx-community models are exported using [Transformers.js](https://github.com/huggingface/transformers.js) tooling with a different export pipeline. This project aims to produce compatible graph structures and file naming conventions to ensure interoperability with Transformers.js and other ONNX consumers.

## 7. Acknowledgements

Special thanks to [Joshua Lochner](https://huggingface.co/Xenova) for his work on [Transformers.js](https://github.com/huggingface/transformers.js) and the [onnx-community](https://huggingface.co/onnx-community) models, which inspired and informed this project's ONNX export approach.

## 8. License

See [LICENSE](LICENSE) for details.
