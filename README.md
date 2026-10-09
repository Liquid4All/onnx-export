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
| **LFM2.5**, **LFM2** | fp32, fp16, q4, q4f16, q4f32, q8 |
| **LFM2.5-VL**, **LFM2-VL** | fp32, fp16, q4, q8 |
| **LFM2.5-8B-A1B**, **LFM2-8B-A1B** | fp32, fp16, q4, q4f16, q8 |
| **LFM2.5-Audio** | fp32, fp16, q4, q8 |

Every export is an [onnxruntime-genai](https://github.com/microsoft/onnxruntime-genai) model folder, and the inference CLIs run on onnxruntime-genai. The onnxruntime-genai model builder builds and quantizes the decoders; this repository builds the vision encoder, the audio graphs and the embedding models that splice image or audio features into the token embeddings, quantizes those, and converts the fp16 and q4f16 decoders. The exports are not loadable by Transformers.js.

## 2. Installation

```bash
git clone https://github.com/Liquid4All/onnx-export.git
cd onnx-export
uv sync
```

`uv sync` installs onnxruntime-genai 0.17.1 from `uv.lock` (`pyproject.toml` requires 0.17.1 or later), which runs all four families (`lfm2`, `lfm2_moe`, `lfm2_vl`, `lfm2_audio`).

`uv sync` also installs the `dev` dependency group: pytest, ruff, and liquid-audio for `lfm2-compare audio`. The exports and the inference CLIs run without it, so `uv sync --no-dev` is enough to use them; run them with `uv run --no-dev` (or set `UV_NO_DEV=1`) there, because a plain `uv run` syncs the environment first and installs the group again. pip 25.1 or later installs the group with `pip install --group dev`.

`uv.lock` also pins an onnxruntime nightly from the ORT-Nightly feed (`[tool.uv]` in `pyproject.toml`), because it runs LFM2-MoE on CPU about 44 times faster than onnxruntime 1.30.0 (see [4.3](#43-moe)). `pip install` ignores `uv.lock` and installs the latest onnxruntime release (1.30.0; `pyproject.toml` requires 1.30.0 or later). To switch a pip environment to the nightly:

```bash
pip install --pre --no-deps --upgrade \
    --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/ onnxruntime
```

liquidonnx sets `ORT_DISABLE_TELEMETRY=1` unless it is already set. The onnxruntime 1.32 nightlies upload telemetry from a thread that can crash a process that has loaded torch when it exits (SIGSEGV), which fails an export that has already written its files; `ORT_DISABLE_TELEMETRY=0` keeps telemetry on.

`--ep cuda` (Linux) needs the CUDA builds, `onnxruntime-gpu` and `onnxruntime-genai-cuda`, in place of `onnxruntime` and `onnxruntime-genai`. They are separate distributions, so `uv sync` and a plain `uv run` would put the CPU packages back: install the rest of the lock without them and run with `uv run --no-sync`. The PyPI CUDA wheels are built for CUDA 13 and need NVIDIA driver 580 or later. `uv venv --clear` replaces the checkout's `.venv`, CPU packages included; to keep both, run these commands in a second clone (a symlink to the first clone's `exports` shares the exports).

```bash
uv venv --clear
uv export --frozen --no-hashes --no-emit-project \
    | grep -vE '^onnxruntime(-genai)?==' > /tmp/requirements-cuda.txt
uv pip install --no-deps -r /tmp/requirements-cuda.txt
uv pip install --no-deps -e .
uv pip install "onnxruntime-gpu[cuda,cudnn]==1.31.0" onnxruntime-genai-cuda==0.17.1
```

In this environment the CPU execution provider comes from onnxruntime-gpu 1.31.0, which runs LFM2-MoE as fast as the nightly (see [4.3](#43-moe)).

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

# Specific precisions: q4 for the CPU, q4f16 for CUDA
uv run lfm2-export LiquidAI/LFM2.5-350M --precision q4 q4f16

# Add q4f16 to the first export without rebuilding fp32 or q4
uv run lfm2-export LiquidAI/LFM2.5-1.2B-Instruct --precision q4f16 --skip-export
```

Output:

```
exports/LFM2.5-1.2B-Instruct-ONNX/
├── genai_config.json      # onnxruntime-genai config; decoder -> onnx/model_q4.onnx
├── config.json, generation_config.json, tokenizer.json, tokenizer_config.json, chat_template.jinja
└── onnx/
    ├── model.onnx         # fp32
    ├── model_fp16.onnx    # fp16 weights, activations and caches; fp32 logits
    ├── model_q4.onnx      # int4 (k_quant); int8 lm_head, tied embedding table and sensitive layers
    ├── model_q4f16.onnx   # q4 with fp16 activations and caches
    ├── model_q4f32.onnx   # int4 MatMuls; fp32 embedding and lm_head
    ├── model_q8.onnx      # int8, lm_head and tied embedding table included
    └── *.onnx_data        # each graph's weights; above 2 GB split into .onnx_data, .onnx_data_1, ...
```

`genai_config.json` points at the first exported precision of q4, q4f16, q8, fp16, q4f32, fp32; override `model.decoder.filename` to load another one.

`--skip-export` adds precisions to an existing export without rebuilding fp32: fp16 converts `model.onnx`, and q4f16 converts the folder's `model_q4.onnx` if it is of the same checkpoint and `--block-size` (otherwise it builds a q4 first).

Every export also writes the fp32 decoder (`model.onnx`, or `decoder.onnx` for VL and Audio, with its `.onnx_data` files), but builds the quantized decoders from the checkpoint. After a quantized-only export the fp32 decoder can be deleted, unless you will add precisions with `--skip-export`: it takes 1.45 GB of the 1.77 GB of an LFM2.5-350M q4 export, 4.7 GB for LFM2.5-1.2B, LFM2.5-VL-1.6B and LFM2.5-Audio-1.5B, 10.8 GB for LFM2.5-2.6B and 33.9 GB for LFM2.5-8B-A1B.

The first export fetches the onnxruntime-genai model builder at a pinned commit into `~/.cache/liquidonnx` (needs `git`). Set `LIQUIDONNX_GENAI_BUILDER` to a local `src/python/py/models` directory to use another copy.

Exports do not run Python code shipped in a Hugging Face repository (`trust_remote_code`). For a trusted repository that needs custom Transformers code, opt in with `LIQUIDONNX_TRUST_REMOTE_CODE=1`:

```bash
LIQUIDONNX_TRUST_REMOTE_CODE=1 uv run lfm2-vl-export LiquidAI/LFM2.5-VL-1.6B --precision q4
```

### 3.2 LFM2-VL Vision-Language Models

```bash
# All precisions (fp16, q4, q8)
uv run lfm2-vl-export LiquidAI/LFM2.5-VL-1.6B --precision

# q4 only
uv run lfm2-vl-export LiquidAI/LFM2.5-VL-450M --precision q4
```

Output:

```
exports/LFM2.5-VL-1.6B-ONNX/
├── genai_config.json            # lfm2_vl pipeline; points at the q4 files
├── genai_processor_config.json  # onnxruntime-genai image preprocessing
├── config.json, generation_config.json, processor_config.json
├── tokenizer.json, tokenizer_config.json, chat_template.jinja
└── onnx/
    ├── decoder.onnx             # fp32 decoder (inputs_embeds in); decoder_{fp16,q4,q8}.onnx
    ├── vision_encoder.onnx      # SigLIP2 + projector; vision_encoder_fp16.onnx
    ├── vision_encoder_q8.onnx   # int8, used with q4 and q8
    ├── embeddings.onnx          # token table + image feature scatter
    ├── embeddings_fp16.onnx     # fp16 table, used with fp16
    ├── embeddings_q8.onnx       # int8 table, used with q4 and q8
    └── *.onnx_data              # weights, as for the text models
```

The q4 and q8 decoders have an int8 LM head. Both bundles load one int8 vision encoder (symmetric, block 128, fp32 activations); with int4 weights or int8 activations, some LFM2.5-VL-1.6B image tokens fall below 0.7 cosine.

`lfm2-vl-infer` and `lfm2-compare` prepare the images with the checkpoint's own processor (`processor_config.json`) and hand the tensors to onnxruntime-genai (`liquidonnx.lfm2_vl.infer.CheckpointProcessor`). Every LFM2-VL and LFM2.5-VL checkpoint sets `do_image_splitting`, so an image of more than twice `max_image_tokens` worth of pixels (524,288, about 724x724) becomes a grid of 512x512 tiles (`min_tiles` to `max_tiles`, 2 to 10) plus a thumbnail resized once, and the prompt carries the processor's tile tokens (`<|img_row_1_col_1|>` ... `<|img_thumbnail|>`). The vision encoder takes each tile and the thumbnail as a separate image; a tile is 32x32 patches, the `max_num_patches` (1024) of `genai_config.json`, and 256 image tokens, so a large image costs up to 2,816 image tokens instead of 256. `--no-image-splitting` resizes each image once instead, as the processor does with `do_image_splitting=False`.

onnxruntime-genai's own image processor (`genai_processor_config.json`, behind `create_multimodal_processor()`) never tiles: it resizes each image once (bilinear for LFM2.5-VL-450M/1.6B, bicubic for the others), and it rejects a prompt whose tallest image is not also its widest, such as a wide photo with a tall one ("image_sizes reports 288x832, larger than the 704x352 pixel_values batch"). Other applications can build the inputs as `CheckpointProcessor` does: the Hugging Face processor's tensors under the names in `model.vision.inputs` of `genai_config.json`, plus `num_image_tokens`, through `generator.set_inputs`.

### 3.3 LFM2-MoE Mixture of Experts

```bash
# Current LFM2.5 MoE checkpoint (fp16, q4, q4f16, q8)
uv run lfm2-moe-export LiquidAI/LFM2.5-8B-A1B --precision

# Earlier LFM2 MoE checkpoint
uv run lfm2-moe-export LiquidAI/LFM2-8B-A1B --precision
```

Same layout as the text models. The experts become `QMoE` int4 (q4, q4f16) or int8 (q8); the routers stay fp32. The LFM2.5-8B-A1B `--precision` export peaks at about 50 GiB of RAM.

### 3.4 LFM2.5-Audio

```bash
# All precisions (fp16, q4, q8)
uv run lfm2-audio-export LiquidAI/LFM2.5-Audio-1.5B --precision

# q4 only
uv run lfm2-audio-export LiquidAI/LFM2.5-Audio-1.5B --precision q4
```

The model can also be a local checkpoint folder (`hf download LiquidAI/LFM2.5-Audio-1.5B --local-dir LFM2.5-Audio-1.5B`). From the Hub, the export downloads only the files it reads, without liquid-audio's Mimi codec or demo media.

Output:

```
exports/LFM2.5-Audio-1.5B-ONNX/
├── genai_config.json            # lfm2_audio pipeline with speech input and output; points at q4
├── config.json, tokenizer.json, tokenizer_config.json, chat_template.jinja
└── onnx/
    ├── decoder.onnx             # fp32 decoder (inputs_embeds in, logits + hidden_states out)
    ├── embeddings.onnx          # token table + audio feature scatter (embeddings_{fp16,q8}.onnx too)
    ├── audio_encoder.onnx       # Conformer speech encoder
    ├── audio_embedding.onnx     # audio codes -> decoder input
    ├── vocoder_depthformer.onnx # decoder hidden state -> frame of 8 audio codes
    ├── audio_detokenizer.onnx   # audio codes -> STFT features, run after generation
    └── ...                      # {graph}_{fp16,q4,q8}.onnx, *.onnx_data; embed_tokens.bin, mel_config.json for web runtimes
```

The q4 and q8 decoders have an int8 LM head, and their bundles an int8 token table. Their depthformer and audio embedding are int4 (q4) or int8 (q8) as in the [olive-recipes](https://github.com/microsoft/olive-recipes) packages, except the depthformer's per-codebook tables, which stay fp32: quantized, they make each audio frame several times slower on CPU.

### 3.5 Harmless log messages

- `matmul_nbits_quantizer [ERROR] - Gather only supports 4 bits quantization.` (q8 and Audio exports): an 8-bit builder pass leaves a Gather as it is; the export is complete.
- `onnx_ir.serde [WARNING] ... cannot be found in any scope. The model is invalid but we will still create a new input` (three lines in text and MoE q4 exports): logged while the builder quantizes; the saved `model_q4.onnx` is complete.
- `neural_compressor [WARNING] - Model size > 2GB. Please use model path instead of onnx model object to quantize` (q4 builds of text, MoE and VL decoders over 2 GB in fp32, from LFM2-700M up; a q4f16 export without an existing q4 runs one too): the builder passes the model to k_quant in memory rather than as a path; the saved decoder is complete.
- numpy `RuntimeWarning` lines from `onnxruntime/quantization/neural_compressor/weight_only.py`, such as `overflow encountered in divide` or `invalid value encountered in divide` (q4 builds of 2.6B and 8B-A1B): k_quant's arithmetic over- or underflows on blocks of 32 tiny weights. In 8B-A1B (largest values around 1e-36) its search's candidate scales come out NaN and are dropped, so those blocks keep their min-max scale. In 2.6B, 746,432 blocks hold only subnormal values (at most 3.2e-39); their min-max scale comes out 0, so they dequantize to zero. Both q4 exports pass the wikitext gate of [5.1](#51-comparing-an-export-with-its-reference-model).
- On macOS, `objc[...]: Class MATStreamingSessionDelegate is implemented in both ...` (two lines in every process that loads onnxruntime and onnxruntime-genai, with the onnxruntime nightly): no effect on the results.

## 4. Inference

The inference CLIs run the export folder on onnxruntime-genai, with interactive multi-turn chat and streaming output. They load the precision `genai_config.json` points at; `--precision` picks another one, and `--ep cuda` runs on CUDA (with the packages in [2](#2-installation)). The CUDA EP runs fp32 matmuls and convolutions in TF32 unless `--no-tf32` is given, so checks against an fp32 reference need the flag (`lfm2-compare --device cuda` always turns TF32 off).

Text is decoded greedily. An answer ends at the end-of-turn token or after `--max-tokens` tokens: 4096 by default, room for the `<think>` block that LFM2.5-2.6B and LFM2.5-8B-A1B write before they answer (`lfm2-audio-infer` has a default per mode, see [4.4](#44-audio-asr-tts-interleaved)).

On CUDA, use fp16 or q4f16 for text and MoE (bare `--precision` leaves q4f16 out for the text models, so name it as in [3.1](#31-lfm2-text-models)), and fp16 for VL and Audio. The other precisions keep fp32 activations, so their GroupQueryAttention nodes, and the experts of MoE q4, run on the CPU: on an H100 with a 1024-token prompt, LFM2.5-350M decodes at 975 tok/s with q4f16 and 124 tok/s with q4. The MoE decoders mark their QMoE experts `weights_prepacked=0`, without which the CUDA EP misreads them; older MoE exports print garbage on CUDA and need re-exporting.

### 4.1 Text Generation

```bash
# Interactive chat
uv run lfm2-infer --model ./exports/LFM2.5-350M-ONNX

# Starting with a prompt
uv run lfm2-infer --model ./exports/LFM2.5-350M-ONNX --prompt "Explain quantum computing"

# On CUDA, in the environment of 2
uv run --no-sync lfm2-infer --model ./exports/LFM2.5-350M-ONNX --precision q4f16 --ep cuda

# Load, prefill and decode speed
uv run lfm2-bench --model ./exports/LFM2.5-350M-ONNX --max-tokens 50
```

> **Note:** Batched inputs to the decoders must be right-padded: GroupQueryAttention takes each row's length from the attention mask sum.

### 4.2 Vision-Language

```bash
# One image
uv run lfm2-vl-infer --model ./exports/LFM2.5-VL-450M-ONNX \
    --images tests/test_lfm2_vl/assets/cardinal.jpg \
    --prompt "What do you see in this image?"

# Two images
uv run lfm2-vl-infer --model ./exports/LFM2.5-VL-450M-ONNX \
    --images tests/test_lfm2_vl/assets/cardinal.jpg tests/test_lfm2_vl/assets/bluejay.jpg \
    --prompt "Compare these two images"

# Each image resized once instead of tiled: fewer image tokens, less detail
uv run lfm2-vl-infer --model ./exports/LFM2.5-VL-450M-ONNX \
    --images tests/test_lfm2_vl/assets/wide.jpg tests/test_lfm2_vl/assets/tall.jpg \
    --no-image-splitting --prompt "Which bird is in each image?"

# Text only
uv run lfm2-vl-infer --model ./exports/LFM2.5-VL-450M-ONNX --prompt "Hello, how are you?"
```

In the chat, `images <path> [<path> ...]` attaches images to the next message; they stay in the conversation for later turns. Large images are split into tiles plus a thumbnail, as the checkpoint's processor does ([3.2](#32-lfm2-vl-vision-language-models)).

### 4.3 MoE

```bash
uv run lfm2-moe-infer --model ./exports/LFM2.5-8B-A1B-ONNX
uv run lfm2-moe-infer --model ./exports/LFM2.5-8B-A1B-ONNX --prompt "Hello"
```

On CPU, MoE speed depends on the onnxruntime version. LFM2.5-8B-A1B q4 with onnxruntime-genai 0.17.1 (`lfm2-bench --ep cpu --max-tokens 128`, 201-token prompt, 13-core (26-thread) x86-64 host, median of 3 runs):

| onnxruntime | Load | Prefill | Decode |
|---|---|---|---|
| 1.30.0 (PyPI) | 115 s | 70 tok/s | 1.3 tok/s |
| 1.31.0.dev20261007001 (`uv.lock`) | 3.9 s | 248 tok/s | 59 tok/s |

The inference CLIs log a warning when they load LFM2-MoE on CPU with onnxruntime older than 1.31; [2](#2-installation) shows how to switch a pip environment to the nightly.

### 4.4 Audio (ASR, TTS, Interleaved)

LFM2.5-Audio has four modes, picked with `--mode`:
- **text** (default): chat without a system prompt
- **asr**: transcribe audio to text
- **tts**: generate speech from text
- **interleaved**: answer a spoken or typed question with text and speech

onnxruntime-genai runs the speech encoder, the decoder and the depthformer and returns the audio codes; `audio_detokenizer.onnx` and an inverse STFT turn them into a 24 kHz WAV. Text is decoded greedily, as in liquid-audio, and the audio codes are sampled with the model card's settings (`--audio-temperature`, `--audio-top-k` and `--seed` change them). The asr, tts and interleaved modes have their own system prompt (`--system` replaces it), and every mode has its own `--max-tokens` default, which counts text tokens and 80 ms audio frames: 256 for text and asr, 1024 (about 82 s of speech) for tts and interleaved.

```bash
# Text chat
uv run lfm2-audio-infer ./exports/LFM2.5-Audio-1.5B-ONNX --prompt "What is the capital of France?"

# ASR: transcribe speech
uv run lfm2-audio-infer ./exports/LFM2.5-Audio-1.5B-ONNX --mode asr \
    --audio samples/audio/fool_me_once_mono.wav

# TTS: speak a text
uv run lfm2-audio-infer ./exports/LFM2.5-Audio-1.5B-ONNX --mode tts \
    --prompt "Hello, how are you today?" --output output.wav

# Interleaved: answer a spoken question with text and speech
uv run lfm2-audio-infer ./exports/LFM2.5-Audio-1.5B-ONNX --mode interleaved \
    --audio samples/audio/woodworks_question.wav --output response.wav

# Interactive multi-turn chat (each turn re-sends the conversation, earlier answers as text);
# turn N's speech goes to output_turnN.wav
uv run lfm2-audio-infer ./exports/LFM2.5-Audio-1.5B-ONNX --mode interleaved --chat \
    --output output.wav
# Commands in chat mode:
#   /audio <file> [text] - Send audio with optional text
#   <text>               - Send text message
#   reset                - Clear conversation state
#   quit                 - Exit
```

### 4.5 Python API

`TextChat` (text and MoE, `liquidonnx.lfm2.infer`) and `VLChat` (`liquidonnx.lfm2_vl.infer`) are the models of `lfm2-infer` and `lfm2-vl-infer`. Their `answer()` keeps no conversation, so a server can load one and answer each request's OpenAI-style messages:

```python
import pathlib

from liquidonnx.lfm2.infer import TextChat
from liquidonnx.lfm2_vl.infer import VLChat

chat = TextChat(pathlib.Path("exports/LFM2.5-350M-ONNX"), precision="q4")
weather = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
}
chat.answer(
    [{"role": "user", "content": "What is the weather in Paris right now?"}],
    max_new_tokens=256,
    tools=[weather],
    keep_special_tokens=True,
)
# '<|tool_call_start|>[get_weather(city="Paris")]<|tool_call_end|>'

vl = VLChat(pathlib.Path("exports/LFM2.5-VL-450M-ONNX"))
image = {"type": "image", "path": "tests/test_lfm2_vl/assets/cardinal.jpg"}
vl.answer([{"role": "user", "content": [image, {"type": "text", "text": "What bird is this?"}]}])
```

`answer()` takes the messages of an OpenAI chat request and leaves them as they were: it parses tool-call arguments sent as JSON strings, which every LFM2 chat template needs as a mapping, and gives content of `null` or a list of text parts the shape the templates take. `VLChat` takes OpenAI's `{"type": "image_url", "image_url": {"url": ...}}` parts (http(s) or data URLs) and transformers' `{"type": "image"}` items with a `path`, `url` or PIL `image`. `tools` go to the chat template. As onnxruntime-genai decodes, the answer leaves out the tokens `tokenizer.json` marks special unless `keep_special_tokens=True`, which a tool-call parser needs for LFM2-350M, LFM2-700M, LFM2-1.2B and the VL models: their `<|tool_call_start|>` and `<|tool_call_end|>` are special. The LFM2.5 text and MoE models keep those, `<think>` and `</think>` either way. The end-of-turn token is never part of the answer, and `stream=True` prints the answer it returns. Both classes run on the CPU unless `ep="cuda"`. `liquidonnx.lfm2` and `liquidonnx.lfm2_vl` do not re-export them, so that the export CLIs do not load onnxruntime-genai.

## 5. Testing

The tests need the `dev` dependency group, which `uv sync` installs by default ([2](#2-installation)). The export pipelines run on tiny random checkpoints, without downloads:

```bash
uv run pytest tests/test_genai_export.py tests/test_genai_export_multimodal.py tests/test_export_cli.py -v
uv run pytest tests/test_lfm2_audio/test_modes_synthetic.py tests/test_lfm2_audio/test_reference_parity.py \
    tests/test_lfm2_audio/test_graph_structure.py -v
uv run pytest tests/test_compare.py tests/test_compare_wikitext.py -v
```

The other tests run an export in `./exports` (`--exports-dir` sets another base folder), most of them against the checkpoint's PyTorch model, which they download. Each file covers several checkpoints and precisions; `-k` picks one, whose export must exist:

```bash
uv run lfm2-export LiquidAI/LFM2-350M --precision q4
uv run pytest tests/test_lfm2/test_decoder.py -v -k "350M and q4"

uv run lfm2-vl-export LiquidAI/LFM2-VL-450M --precision q4
uv run pytest tests/test_lfm2_vl/test_decoder.py tests/test_lfm2_vl/test_vision_encoder.py -v -k "450M and q4"

uv run pytest tests/test_lfm2_moe/test_decoder.py -v -k "LFM2.5-8B-A1B and q4 and not q4f16"
uv run pytest tests/test_lfm2_audio/test_asr.py -v -k "q4"
```

### 5.1 Comparing an export with its reference model

`lfm2-compare` scores every precision of an export folder against the PyTorch model (liquid-audio for audio): teacher-forced KL of the decoder over the reference's greedy answers, greedy answers on plain onnxruntime and on onnxruntime-genai, and for VL the vision encoder and embedding model. It writes a JSON file and a markdown table.

```bash
uv run lfm2-compare text --model LiquidAI/LFM2.5-1.2B-Instruct --export ./exports/LFM2.5-1.2B-Instruct-ONNX
uv run lfm2-compare vl --model LiquidAI/LFM2.5-VL-1.6B --export ./exports/LFM2.5-VL-1.6B-ONNX
uv run lfm2-compare audio --model LiquidAI/LFM2.5-Audio-1.5B --export ./exports/LFM2.5-Audio-1.5B-ONNX
```

`--device cuda` runs the reference, the ONNX sessions and onnxruntime-genai on the GPU with TF32 off, so fp32 stays fp32 (needs the CUDA packages in [2](#2-installation)). References are cached per device. `lfm2-compare vl --no-image-splitting` resizes each image once in the reference and in onnxruntime-genai instead of tiling it.

`lfm2-compare wikitext` is the quality gate for quantized precisions. It scores them on wikitext-2 with the protocol of [olive-recipes #638](https://github.com/microsoft/olive-recipes/pull/638): 64 chunks of 512 tokens, the second half of each scored, KLD ± SE over the chunks and same-top %. The reference is an fp32 one in llama.cpp's KL-divergence base format, given with `--reference` or computed from `--model`. The command exits with 1 when a precision is above `--max-kld` or, for the LFM2.5 models it knows, above today's KLD + 2 SE on `--device`. The ceilings are per execution provider, because the CPU and CUDA kernels score differently: the CPU ones cover q4 and q4f32, and the CUDA ones (TF32 off) q8, fp16 and q4f16, all measured on one H100 host. Each device scores those precisions by default; a precision without a ceiling on its device is reported as "no ceiling", not as a pass. The ONNX sessions run 13 intra-op threads on any machine, the count the ceilings were measured at, because onnxruntime's CPU MoE kernel gives different scores at different thread counts; `--threads` overrides it. `--help` explains the token streams.

```bash
# Against a reference file, on the CPU and on CUDA (in the environment of 2)
uv run lfm2-compare wikitext --export ./exports/LFM2.5-350M-ONNX --reference ref.kld
uv run --no-sync lfm2-compare wikitext --export ./exports/LFM2.5-350M-ONNX --reference ref.kld \
    --device cuda

# Against a reference computed on the GPU from wiki.test.raw, which these two lines download
# as llama.cpp's scripts/get-wikitext-2.sh does
curl -LO https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip
unzip wikitext-2-raw-v1.zip
uv run lfm2-compare wikitext --export ./exports/LFM2.5-350M-ONNX --model LiquidAI/LFM2.5-350M \
    --text wikitext-2-raw/wiki.test.raw --reference-device cuda
```

## 6. Pre-exported Models

The LFM2 and LFM2.5 ONNX repositories on Hugging Face, under [LiquidAI](https://huggingface.co/LiquidAI) and [onnx-community](https://huggingface.co/onnx-community), come from other export pipelines, such as Transformers.js tooling and versions of this repository from before it built onnxruntime-genai models. None of them has a `genai_config.json`, so the inference CLIs and onnxruntime-genai cannot load them; export the checkpoint as in [3](#3-export) instead.

## 7. Acknowledgements

Special thanks to [Joshua Lochner](https://huggingface.co/Xenova) for his work on [Transformers.js](https://github.com/huggingface/transformers.js) and the [onnx-community](https://huggingface.co/onnx-community) models, which inspired and informed this project's ONNX export approach.

## 8. License

See [LICENSE](LICENSE) for details.
