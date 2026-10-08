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
| **LFM2.5**, **LFM2** | fp32, fp16, q4, q4f32, q8 |
| **LFM2.5-VL**, **LFM2-VL** | fp32, fp16, q4, q8 |
| **LFM2.5-8B-A1B**, **LFM2-8B-A1B** | fp32, fp16, q4, q4f16, q8 |
| **LFM2.5-Audio** | fp32, fp16, q4, q8 |

Every export is an [onnxruntime-genai](https://github.com/microsoft/onnxruntime-genai) model folder, and the inference CLIs run on onnxruntime-genai. The onnxruntime-genai model builder builds and quantizes the decoders; this repository builds the vision encoder, the audio graphs and the embedding models that splice image or audio features into the token embeddings, quantizes those, and converts the fp16 and q4f16 decoders. The exports are not loadable by Transformers.js.


## 2. Installation

```bash
git clone https://github.com/Liquid4All/onnx-export.git
cd onnx-export
uv sync

# For development (testing, benchmarking, lfm2-compare audio)
uv sync --extra dev
```

`uv sync` installs onnxruntime-genai 0.17.1 from `uv.lock` (`pyproject.toml` requires 0.17.1 or later), which runs all four families (`lfm2`, `lfm2_moe`, `lfm2_vl`, `lfm2_audio`).

`uv.lock` also pins an onnxruntime nightly from the ORT-Nightly feed (`[tool.uv]` in `pyproject.toml`), because it runs LFM2-MoE on CPU about 44 times faster than onnxruntime 1.30.0 (see [4.3](#43-moe)). `pip install` ignores `uv.lock` and installs the latest onnxruntime release (1.30.0; `pyproject.toml` requires 1.30.0 or later). To switch a pip environment to the nightly:

```bash
pip install --pre --no-deps --upgrade \
    --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/ onnxruntime
```

This repository tracks onnxruntime-genai `main`: the model builder runs at a pinned commit of `main` (`liquidonnx.genai_builder.GENAI_COMMIT`), and CI tests the runtime built from that same commit. To run on that build, build the pinned commit and put its Python package first on the path:

```bash
git clone https://github.com/microsoft/onnxruntime-genai.git && cd onnxruntime-genai
git checkout $(uv run --project ../onnx-export python -c "from liquidonnx.genai_builder import GENAI_COMMIT; print(GENAI_COMMIT)")
uv pip install --python ../onnx-export/.venv/bin/python pip  # the build packages a wheel with pip
# On Apple silicon, append CMAKE_OSX_ARCHITECTURES=arm64 to --cmake_extra_defines.
# Without --ort_home, build.py compiles against the onnxruntime NuGet package cmake/ortlib.cmake pins
# (1.26.0 at GENAI_COMMIT); CI passes --ort_home with the onnxruntime 1.30.0 release archive (the
# uv.lock nightly has none). Either way the build loads the venv's onnxruntime at run time.
# CMAKE_BUILD_PARALLEL_LEVEL caps the build at the core count; build.py --parallel is an unbounded make -j.
CMAKE_BUILD_PARALLEL_LEVEL=$(getconf _NPROCESSORS_ONLN) ../onnx-export/.venv/bin/python build.py \
    --config Release --skip_tests --skip_examples --no_telemetry --cmake_extra_defines ENABLE_TESTS=OFF
export PYTHONPATH=$PWD/build/Linux/Release/wheel  # build/macOS/Release/wheel on macOS
```

## 3. Export

### 3.1 LFM2 Text Models

```bash
# All precisions (fp16, q4, q4f32, q8)
uv run lfm2-export LiquidAI/LFM2.5-1.2B-Instruct --precision

# Specific precisions
uv run lfm2-export LiquidAI/LFM2.5-350M --precision q4 q8
```

Output:

```
exports/LFM2.5-1.2B-Instruct-ONNX/
├── genai_config.json      # onnxruntime-genai config; decoder -> onnx/model_q4.onnx
├── config.json, tokenizer.json, tokenizer_config.json, chat_template.jinja
└── onnx/
    ├── model.onnx         # fp32
    ├── model_fp16.onnx    # fp16 weights, activations and caches; fp32 logits
    ├── model_q4.onnx      # int4 (k_quant); int8 lm_head, tied embedding table and sensitive layers
    ├── model_q4f32.onnx   # int4 MatMuls; fp32 embedding and lm_head
    └── model_q8.onnx      # int8
```

`genai_config.json` points at the first exported precision of q4, q4f16, q8, fp16, q4f32, fp32; override `model.decoder.filename` to load another one.

The first export fetches the onnxruntime-genai model builder at a pinned commit into `~/.cache/liquidonnx` (needs `git`). Set `LIQUIDONNX_GENAI_BUILDER` to a local `src/python/py/models` directory to use another copy.

Exports do not run Python code shipped in a Hugging Face repository (`trust_remote_code`). For a trusted repository that needs custom Transformers code, opt in with `LIQUIDONNX_TRUST_REMOTE_CODE=1`:

```bash
LIQUIDONNX_TRUST_REMOTE_CODE=1 uv run lfm2-vl-export LiquidAI/LFM2.5-VL-1.6B --precision q4
```

### 3.2 LFM2-VL Vision-Language Models

```bash
# All precisions (fp16, q4, q8)
uv run lfm2-vl-export LiquidAI/LFM2.5-VL-1.6B --precision
```

Output:

```
exports/LFM2.5-VL-1.6B-ONNX/
├── genai_config.json            # lfm2_vl pipeline; points at the q4 files
├── genai_processor_config.json  # onnxruntime-genai image preprocessing
├── config.json, processor_config.json, tokenizer.json, tokenizer_config.json, chat_template.jinja
└── onnx/
    ├── decoder.onnx             # fp32 decoder (inputs_embeds in); decoder_{fp16,q4,q8}.onnx
    ├── vision_encoder.onnx      # SigLIP2 + projector; vision_encoder_fp16.onnx
    ├── vision_encoder_q8.onnx   # int8, used with q4 and q8
    ├── embeddings.onnx          # token table + image feature scatter
    ├── embeddings_fp16.onnx     # fp16 table, used with fp16
    └── embeddings_q8.onnx       # int8 table, used with q4 and q8
```

The q4 and q8 decoders have an int8 LM head. Both bundles load one int8 vision encoder (symmetric, block 128, fp32 activations); with int4 weights or int8 activations, some LFM2.5-VL-1.6B image tokens fall below 0.7 cosine.

onnxruntime-genai resizes each image once instead of tiling it, as the upstream processor does with `do_image_splitting=False`. The image preprocessing follows the checkpoint's own processor (bilinear for LFM2.5-VL-450M/1.6B, bicubic for the others).

### 3.3 LFM2-MoE Mixture of Experts

```bash
# Current LFM2.5 MoE checkpoint (fp16, q4, q4f16, q8)
uv run lfm2-moe-export LiquidAI/LFM2.5-8B-A1B --precision

# Earlier LFM2 MoE checkpoint
uv run lfm2-moe-export LiquidAI/LFM2-8B-A1B --precision
```

Same layout as the text models. The experts become `QMoE` int4 (q4, q4f16) or int8 (q8); the routers stay fp32.

### 3.4 LFM2.5-Audio

```bash
# All precisions (fp16, q4, q8)
uv run lfm2-audio-export LiquidAI/LFM2.5-Audio-1.5B --precision
```

Output:

```
exports/LFM2.5-Audio-1.5B-ONNX/
├── genai_config.json            # lfm2_audio pipeline with speech input and output; points at q4
├── config.json, tokenizer.json, tokenizer_config.json
└── onnx/
    ├── decoder.onnx             # fp32 decoder (inputs_embeds in, logits + hidden_states out)
    ├── embeddings.onnx          # token table + audio feature scatter (embeddings_{fp16,q8}.onnx too)
    ├── audio_encoder.onnx       # Conformer speech encoder
    ├── audio_embedding.onnx     # audio codes -> decoder input
    ├── vocoder_depthformer.onnx # decoder hidden state -> frame of 8 audio codes
    ├── audio_detokenizer.onnx   # audio codes -> STFT features, run after generation
    └── ...                      # {graph}_{fp16,q4,q8}.onnx; embed_tokens.bin, mel_config.json for web runtimes
```

The q4 and q8 decoders have an int8 LM head, and their bundles an int8 token table. Their configs keep the depthformer and audio embedding at fp16, because a quantized depthformer changes most audio codes.

## 4. Inference

The inference CLIs run the export folder on onnxruntime-genai, with interactive multi-turn chat and streaming output. They load the precision `genai_config.json` points at; `--precision` picks another one, and `--ep cuda` runs on CUDA (needs `onnxruntime-genai-cuda`).

### 4.1 Text Generation

```bash
# Interactive chat
uv run lfm2-infer --model ./exports/LFM2.5-1.2B-Instruct-ONNX

# A specific precision, starting with a prompt
uv run lfm2-infer --model ./exports/LFM2.5-1.2B-Instruct-ONNX --precision q8 \
    --prompt "Explain quantum computing"

# Load, prefill and decode speed
uv run lfm2-bench --model ./exports/LFM2.5-1.2B-Instruct-ONNX --max-tokens 50
```

> **Note:** Batched inputs to the decoders must be right-padded: GroupQueryAttention takes each row's length from the attention mask sum.

### 4.2 Vision-Language

```bash
# Single image analysis
uv run lfm2-vl-infer --model ./exports/LFM2.5-VL-1.6B-ONNX \
    --images photo.jpg \
    --prompt "What do you see in this image?"

# Multi-image comparison
uv run lfm2-vl-infer --model ./exports/LFM2.5-VL-1.6B-ONNX \
    --images image1.jpg image2.jpg \
    --prompt "Compare these two images"

# Text-only, fp16
uv run lfm2-vl-infer --model ./exports/LFM2.5-VL-1.6B-ONNX --precision fp16 \
    --prompt "Hello, how are you?"
```

In the chat, `images <path> [<path> ...]` attaches images to the next message; they stay in the conversation for later turns.

### 4.3 MoE

```bash
uv run lfm2-moe-infer --model ./exports/LFM2.5-8B-A1B-ONNX
uv run lfm2-moe-infer --model ./exports/LFM2.5-8B-A1B-ONNX --precision q8 --prompt "Hello"
```

On CPU, MoE speed depends on the onnxruntime version. LFM2.5-8B-A1B q4 with onnxruntime-genai 0.17.1 (`lfm2-bench --ep cpu --max-tokens 128`, 201-token prompt, 13-core (26-thread) x86-64 host, median of 3 runs):

| onnxruntime | Load | Prefill | Decode |
|---|---|---|---|
| 1.30.0 (PyPI) | 32 s | 61 tok/s | 1.3 tok/s |
| 1.31.0.dev20261007001 (`uv.lock`) | 3.6 s | 225 tok/s | 59 tok/s |

The inference CLIs log a warning when they load LFM2-MoE on CPU with onnxruntime older than 1.31; [2](#2-installation) shows how to switch a pip environment to the nightly.

### 4.4 Audio (ASR, TTS, Interleaved)

LFM2.5-Audio has four modes, picked with `--mode`:
- **text** (default): chat without a system prompt
- **asr**: transcribe audio to text
- **tts**: generate speech from text
- **interleaved**: answer a spoken or typed question with text and speech

onnxruntime-genai runs the speech encoder, the decoder and the depthformer and returns the audio codes; `audio_detokenizer.onnx` and an inverse STFT turn them into a 24 kHz WAV. Text is decoded greedily, as in liquid-audio, and the audio codes are sampled with the model card's settings (`--audio-temperature`, `--audio-top-k` and `--seed` change them).

```bash
# ASR: Transcribe audio to text
uv run lfm2-audio-infer ./exports/LFM2.5-Audio-1.5B-ONNX --mode asr --audio input.wav

# TTS: Generate speech from text
uv run lfm2-audio-infer ./exports/LFM2.5-Audio-1.5B-ONNX --mode tts \
    --prompt "Hello, how are you today?" \
    --system "Perform TTS. Use the UK female voice." \
    --output output.wav

# Interleaved: Audio input with text+audio response, at q8
uv run lfm2-audio-infer ./exports/LFM2.5-Audio-1.5B-ONNX --mode interleaved \
    --audio question.wav --output response.wav --precision q8

# Interactive multi-turn chat (each turn re-sends the conversation, earlier answers as text)
uv run lfm2-audio-infer ./exports/LFM2.5-Audio-1.5B-ONNX --mode interleaved --chat \
    --output output.wav
# Commands in chat mode:
#   /audio <file> [text] - Send audio with optional text
#   <text>               - Send text message
#   reset                - Clear conversation state
#   quit                 - Exit
```

## 5. Testing

Tests verify ONNX exports against the PyTorch reference models.

```bash
# Install dev dependencies
uv sync --extra dev

# Export pipelines on tiny random checkpoints (no model download)
uv run pytest tests/test_genai_export.py -v
uv run pytest tests/test_genai_export_multimodal.py -v
uv run pytest tests/test_lfm2_audio/test_modes_synthetic.py tests/test_lfm2_audio/test_reference_parity.py -v

# LFM2 text model tests
uv run pytest tests/test_lfm2/test_decoder.py -v -k "q4"

# LFM2-VL vision-language tests
uv run pytest tests/test_lfm2_vl/test_decoder.py -v -k "450M"
uv run pytest tests/test_lfm2_vl/test_vision_encoder.py -v

# LFM2-MoE tests
uv run pytest tests/test_lfm2_moe/test_decoder.py -v
uv run pytest tests/test_lfm2_moe/test_tokenizer.py -v

# LFM2.5-Audio tests
uv run pytest tests/test_lfm2_audio/test_asr.py -v -k "q4"
```

### 5.1 Comparing an export with its reference model

`lfm2-compare` scores every precision of an export folder against the PyTorch model (liquid-audio for audio): teacher-forced KL of the decoder over the reference's greedy answers, greedy answers on plain onnxruntime and on onnxruntime-genai, and for VL the vision encoder and embedding model. It writes a JSON file and a markdown table.

```bash
uv run lfm2-compare text --model LiquidAI/LFM2.5-1.2B-Instruct --export ./exports/LFM2.5-1.2B-Instruct-ONNX
uv run lfm2-compare vl --model LiquidAI/LFM2.5-VL-1.6B --export ./exports/LFM2.5-VL-1.6B-ONNX
uv run lfm2-compare audio --model LiquidAI/LFM2.5-Audio-1.5B --export ./exports/LFM2.5-Audio-1.5B-ONNX
```

`--device cuda` runs the reference, the ONNX sessions and onnxruntime-genai on the GPU with TF32 off, so fp32 stays fp32 (needs `onnxruntime-gpu` and `onnxruntime-genai-cuda`). References are cached per device.

`lfm2-compare wikitext` is the quality gate for quantized precisions. It scores them on wikitext-2 with the protocol of [olive-recipes #638](https://github.com/microsoft/olive-recipes/pull/638): 64 chunks of 512 tokens, the second half of each scored, KLD ± SE over the chunks and same-top %. The reference is an fp32 one in llama.cpp's KL-divergence base format, given with `--reference` or computed from `--model`. The command exits with 1 when a precision is above `--max-kld` or, for the LFM2.5 models it knows, above today's KLD + 2 SE. `--help` explains the token streams.

```bash
uv run lfm2-compare wikitext --export ./exports/LFM2.5-350M-ONNX --reference ref.kld
uv run lfm2-compare wikitext --export ./exports/LFM2.5-350M-ONNX --model LiquidAI/LFM2.5-350M \
    --text wikitext-2-raw/wiki.test.raw --reference-device cuda
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

> **Note:** The onnx-community models are exported using [Transformers.js](https://github.com/huggingface/transformers.js) tooling with a different export pipeline. This project's exports follow onnxruntime-genai instead, so their graphs and file names differ.

## 7. Acknowledgements

Special thanks to [Joshua Lochner](https://huggingface.co/Xenova) for his work on [Transformers.js](https://github.com/huggingface/transformers.js) and the [onnx-community](https://huggingface.co/onnx-community) models, which inspired and informed this project's ONNX export approach.

## 8. License

See [LICENSE](LICENSE) for details.
