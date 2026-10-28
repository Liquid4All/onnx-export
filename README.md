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

ONNX export and inference tools for [LFM2 and LFM3](https://www.liquid.ai/liquid-foundation-models) models.

## 1. Supported Models

| Family | Quant Formats |
|--------|---------------|
| **LFM2.5**, **LFM2** | fp32, fp16, q4, q4f16, q4f32, q8 |
| **LFM2.5-VL**, **LFM2-VL** | fp32, fp16, q4, q8 |
| **LFM2.5-24B-A2B**, **LFM2.5-8B-A1B**, **LFM2-8B-A1B** | fp32, fp16, q4, q4f16, q8 |
| **LFM2.5-Audio** | fp32, fp16, q4, q8 |
| **LFM3** | fp32, fp16, q4, q4f16, q4f32, q8, q8f16 |
| **LFM3-VL** | fp32, fp16, q4, q4f16, q8, q8f16 |
| **LFM3-Audio**, **LFM3-Omni** | fp32, fp16, q4, q4f16, q8, q8f16 |
| **d1-2B**, **d1-Omni-2B** | fp32, fp16, q4, q4f16, q8, q8f16 |

The LFM2-based d1 models (d1-3B, d1-3B-w8a8, d1-omni-600M) are not supported. [onnx-community/d1-3B-ONNX](https://huggingface.co/onnx-community/d1-3B-ONNX) and [onnx-community/d1-omni-600M-ONNX](https://huggingface.co/onnx-community/d1-omni-600M-ONNX) are Transformers.js exports of d1-3B and d1-omni-600M from another pipeline.

The LFM2 and LFM2.5 exports are [onnxruntime-genai](https://github.com/microsoft/onnxruntime-genai) model folders, and their inference CLIs run on onnxruntime-genai. The onnxruntime-genai model builder builds and quantizes their decoders; this repository builds the vision encoder, the audio graphs and the embedding models that splice image or audio features into the token embeddings, quantizes those, and converts the fp16 and q4f16 decoders. onnxruntime-genai has no LFM3 model type, so this repository builds every graph of the LFM3 families itself and writes plain ONNX folders, which their CLIs run on onnxruntime ([3.1b](#31b-lfm3-text-models)). The exports are not loadable by Transformers.js.

## 2. Installation

```bash
git clone https://github.com/Liquid4All/onnx-export.git
cd onnx-export
uv sync
```

`uv sync` installs onnxruntime-genai 0.17.1 from `uv.lock` (`pyproject.toml` requires 0.17.1 or later), which runs all four families (`lfm2`, `lfm2_moe`, `lfm2_vl`, `lfm2_audio`). The LFM3 families need only onnxruntime.

`uv sync` also installs the `dev` dependency group: pytest, ruff, and liquid-audio for `lfm2-compare audio`. The exports and the inference CLIs run without it, so `uv sync --no-dev` is enough to use them; run them with `uv run --no-dev` (or set `UV_NO_DEV=1`) there, because a plain `uv run` syncs the environment first and installs the group again. pip 25.1 or later installs the group with `pip install --group dev`.

`uv.lock` also pins an onnxruntime nightly from the ORT-Nightly feed (`[tool.uv]` in `pyproject.toml`), because its CPU QMoE kernels decode LFM2-MoE much faster than onnxruntime 1.30.0 (see [4.3](#43-moe)). `pip install` ignores `uv.lock` and installs the latest onnxruntime release (1.30.0; `pyproject.toml` requires 1.30.0 or later). To switch a pip environment to the nightly:

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

Every export CLI also writes `export_info.json` next to `config.json`, one record per run (a `--skip-export` run adds its own): the liquidonnx commit and whether tracked files had uncommitted changes (`null` for an installed package), the command, the checkpoint with the Hugging Face commit it resolved to and its pin if any, the onnxruntime (or onnxruntime-gpu), onnx, torch, transformers and onnxruntime-genai versions, the host's platform and CPU architecture, and the precisions the run built. The command and a local checkpoint's path are recorded as given. The graphs are the same with or without it.

The first export fetches the onnxruntime-genai model builder at a pinned commit into `~/.cache/liquidonnx` (needs `git`). Set `LIQUIDONNX_GENAI_BUILDER` to a local `src/python/py/models` directory to use another copy.

Exports do not run Python code shipped in a Hugging Face repository (`trust_remote_code`). For a trusted repository that needs custom Transformers code, opt in with `LIQUIDONNX_TRUST_REMOTE_CODE=1`:

```bash
LIQUIDONNX_TRUST_REMOTE_CODE=1 uv run lfm2-vl-export LiquidAI/LFM2.5-VL-1.6B --precision q4
```

### 3.1b LFM3 Text Models

```bash
# All precisions (fp16, q4, q4f16, q4f32, q8, q8f16; fp32 is always written)
uv run lfm3-export LiquidAI/LFM3-1.7B --precision

# q4f16 only: q4 with fp16 activations and KV cache, for CUDA
uv run lfm3-export LiquidAI/LFM3-1.7B-Base --precision q4f16

# Add q8 to the first export without rebuilding fp32
uv run lfm3-export LiquidAI/LFM3-1.7B --precision q8 --skip-export
```

Output:

```
exports/LFM3-1.7B-ONNX/
├── config.json, configuration_lfm3.py, generation_config.json, tokenizer files, chat_template.jinja
└── onnx/
    ├── model.onnx         # fp32
    ├── model_fp16.onnx    # fp16 weights, activations and caches
    ├── model_q4.onnx      # int4 (k_quant); int8 lm_head, which is also the token table, and sensitive layers
    ├── model_q4f16.onnx   # q4 with fp16 activations and caches
    ├── model_q4f32.onnx   # int4 MatMuls (round to nearest); fp32 embedding and lm_head
    ├── model_q8.onnx      # int8, lm_head and token table included
    ├── model_q8f16.onnx   # q8 with fp16 activations and caches
    └── *.onnx_data
```

onnxruntime-genai has no LFM3 model type, so `lfm3-export` builds the graph itself (`liquidonnx.lfm3.builder`) and writes a plain ONNX folder without `genai_config.json`; `lfm3-infer` and `lfm3-bench` run it on onnxruntime. It reads `config.json` and the safetensors directly, without running the checkpoint's Python code, in LFM3-1.7B's format (`rope_parameters` per layer type, `use_sandwich_norm`, `sliding_attention` layers) or an older one (`swa_rope_parameters`, `use_peri_ln`, `sliding_window_attention`). `lfm3.session.REVISIONS` pins the revisions the exporter and tests were validated on, `40ebe0a` for `LiquidAI/LFM3-1.7B` and `545fc8f` for `LiquidAI/LFM3-1.7B-Base`; other models follow `main`, and a local checkpoint folder is exported as it is. The exported `generation_config.json` leaves out the checkpoint's sampling defaults: the CLIs decode greedily.

FP8, NVFP4, GGUF and MLX repositories are not export inputs of any LFM3 family: export the bf16 checkpoint. Each LFM3 exporter reads `config.json` before it downloads weights, and stops on a quantized checkpoint (one with a `quantization_config`) or on another family's checkpoint, naming the exporter that takes it.

The quantized decoders (of every LFM3 family) are derived from the fp32 graph, so `--skip-export` adds any of them to an export folder without reading the checkpoint's weights:

- q4 is the q4 recipe of the LFM2 decoders ([3.1](#31-lfm2-text-models)): k_quant int4 MatMuls, with the LM head and the layers most sensitive to int4 in int8. The int8 layers follow the onnxruntime-genai builder's `mixed_layers` rule: in the 24 layers of these checkpoints, the q, k and v projections of 5 attention layers and the MLP down projection of 12 layers.
- q8 has every MatMul in int8.
- In q4 and q8 the int8 LM head's weights are also the token embedding table, stored once.
- q4f16 and q8f16 are q4 and q8 with fp16 activations and caches.
- q4f32 (text only) has int4 MatMuls, rounded to nearest and symmetric unless `--q4-asymmetric`, next to the fp32 embedding table and LM head.
- The quantized MatMuls run at `accuracy_level` 4: the CPU EP quantizes their activations to int8 too, which costs some accuracy against `accuracy_level` 0 but decodes many times faster there (the CUDA EP ignores it). `--matmulnbits-accuracy-level` ([5](#5-testing)) shows what it costs.

A run rebuilds the files of the precisions it is asked for, also in a folder that already has them, and writes no file of the other precisions; it does still split any data file in the folder that is over `--split-data`, and give every file the umask's mode. q4f16 and q8f16 are converted from a q4 or q8 decoder built in the same run, a scratch file unless that precision is requested too. A q4 build runs onnxruntime's k_quant, which is numpy on the CPU, and logs the messages of [3.5](#35-harmless-log-messages).

Exports are byte for byte reproducible on one CPU architecture, but the int8 weights of the LFM3 families' q8 and q8f16 decoders can differ between x86-64 and arm64 in about 0.01% of their codes, by one. onnxruntime's int8 block quantizer (`quantize_matmul_8bits`, in C++) rounds `w × (1 / scale) + zero point`, which its arm64 build computes with a fused multiply-add and its x86-64 build with two roundings, so a value within a rounding error of a .5 tie can land on either side; the onnxruntime version does not matter. The int4 weights of LFM3's q4f32 decoder, rounded by onnxruntime's int4 block quantizer (`quantize_matmul_4bits`), differ between the two as well. Every other file (fp32, fp16, and the k_quant q4 decoders, which numpy builds) is the same on both architectures, and `export_info.json` names the architecture an export was built on.

The decoders take `input_ids`, `attention_mask`, `position_ids`, `logits_to_keep` (logits of the last N positions; 0 for all) and the `past_conv.N` / `past_key_values.N.*` caches; fp16, q4f16 and q8f16 keep their caches in fp16. The LFM3 CLIs bind the caches in place between steps, sharing each KV buffer between past and present. On CUDA they pass `sdpa_kernel=3` and `cudnn_conv_algo_search=HEURISTIC`: otherwise GroupQueryAttention builds a cuDNN graph per decoded token, and cuDNN re-tunes the short convolutions for every prompt length. Use the fp16-activation files (fp16, q4f16, q8f16) on CUDA: CUDA has no float32 GroupQueryAttention kernel, so q4, q4f32, q8 and fp32 run attention on the CPU. The CLIs pick q4f16, fp16 or q8f16 on CUDA and q4 or q8 on the CPU unless `--precision` says otherwise; the fp16-activation files run on the CPU EP too (`--cpu`), where they score as q4 and q8 do.

```bash
uv run lfm3-infer --model exports/LFM3-1.7B-ONNX --prompt "What is the capital of France?"
uv run lfm3-infer --model exports/LFM3-1.7B-ONNX --precision q8 --cpu --no-think \
    --prompt "What is the capital of France? Answer in one sentence."
uv run lfm3-bench --model LiquidAI/LFM3-1.7B --onnx exports/LFM3-1.7B-ONNX --max-tokens 128
```

The chat template opens `<think>` unless `enable_thinking=False` (`--no-think`); `<think>` and `</think>` are not special tokens, so the answer keeps them. LFM3-1.7B-Base follows the same template, thinking included, so `lfm3-infer` runs it unchanged; given plain text without the template (`lfm3-bench`'s prompt), it continues as a base model.

`tests/test_lfm3` runs the checkpoint's own remote code as its float32 PyTorch reference (TF32 off on CUDA), which builds only on some transformers releases: LFM3-1.7B's on 5.8-5.14 and 5.18+ (5.15-5.17 rename its `conv` layers, which the tests undo), the older format's on 5.0-5.1. A test whose reference can't be built is skipped with the release it needs. The tests without a download run `lfm3-export` and `lfm3-infer` on a tiny random checkpoint in LFM3-1.7B's format, against LFM3-1.7B's modeling code (the copy in `tests/test_lfm3_vl/reference/`). `--lfm3-model` (default `LiquidAI/LFM3-1.7B`) sets the checkpoint of the real-model tests, whose export they take from `--exports-dir`:

```bash
uv run pytest tests/test_lfm3/test_config.py tests/test_lfm3/test_rope.py tests/test_lfm3/test_tiny_export.py \
    tests/test_check_checkpoint.py -v  # no download
uv run pytest tests/test_lfm3/test_decoder.py tests/test_lfm3/test_decode.py -v -k q4
uv run --no-sync pytest tests/test_lfm3 -v --device cuda  # in the CUDA environment of 2
uv run --no-sync pytest tests/test_lfm3 -v --device cuda --lfm3-model LiquidAI/LFM3-1.7B-Base
```

The tests score every precision against the reference:

- `test_decoder.py`: one prefill of four prompts, one past the 256-token window: fp32 within 5e-4 of the reference's logits, and 3 of its top 5 at the last position (fp32 5, fp16 4, q4f32 2).
- `test_decode.py::test_decode_parity`: 24 cached greedy steps of a short and a long prompt against one teacher-forced forward over the same tokens: fp32 picks the reference's tokens within 5e-4, the others keep 3 of its top 5 at every step (q4f32 2).
- `test_decode.py::test_long_decode_parity`: 600 steps of a second chat turn in thinking mode, past the point where the sliding window drops keys, by mean KL: below 1e-6 for fp32 (with the reference's tokens), 1e-4 for fp16, 5e-4 for q8 and q8f16 and 1.16e-3 for int4, 4 times bf16 PyTorch's own 600 greedy steps scored the same way (`test_long_decode_bf16_floor`), with 3 of the top 5 at every step.
- `test_decode.py::test_sampled_distribution`: the mean total variation of the next-token probabilities after transformers' sampling processors (LFM3-1.7B's `generation_config.json`, and temperature 0.7 with top-p 0.9) at the positions of the reference's answer, 4 times bf16 PyTorch's (1% of it for fp32), for runtimes that sample.
- `test_long_context.py` (opt-in, [5](#5-testing)): a passphrase recalled after 8K, 32K and 128K tokens of wikitext-2 (64 cached steps, scored as `test_long_decode_parity` scores its steps), and the decoder's wikitext-2 KLD with `lfm2-compare wikitext`'s protocol ([5.1](#51-comparing-an-export-with-its-reference-model)).

Known limits: int4 rounding moves near ties. On LFM3-1.7B, `test_decode_parity` with q4 and q4f16 keeps only 1 or 2 of the reference's top 5 at one or two steps (bound 3), where the reference puts most of its mass on a top token the int4 graphs keep and little on each of ranks 2 to 6; on arm64 q4f32 keeps 1 of 5 at one step (bound 2). The same int4 rounding puts q4 and q4f16 several times over the bound of `test_sampled_distribution` on LFM3-1.7B, and q4 over that of `test_long_decode_parity`: those int4 bounds are 4 times bf16 PyTorch's own error, and int4's answer KL is far above bf16's. On arm64 the CPU EP's float32 arithmetic also leaves the fp32 graph further from PyTorch than on x86-64 or CUDA: LFM3-1.7B's fp32 logits exceed the 5e-4 bound of `test_decoder` and `test_decode_parity` there, while its answers' KL stays below 1e-9 and its 600-step decode keeps the reference's tokens.

### 3.1c LFM3-VL Vision-Language Models

```bash
# All precisions (fp16, q4, q8, q4f16, q8f16; fp32 is always written)
uv run lfm3-vl-export LiquidAI/LFM3-VL-2B --precision

uv run lfm3-vl-infer --model exports/LFM3-VL-2B-ONNX \
    --images tests/test_lfm2_vl/assets/cardinal.jpg --prompt "What bird is this?"
uv run lfm3-vl-infer --model exports/LFM3-VL-2B-ONNX --precision q4 --cpu --no-think \
    --images tests/test_lfm2_vl/assets/cardinal.jpg tests/test_lfm2_vl/assets/bluejay.jpg \
    --prompt "What birds are in these two images?"
```

Like LFM3, LFM3-VL runs on onnxruntime with its own generation loop (`liquidonnx.lfm3_vl.session.ONNXVLModel`). The folder holds two graphs per precision. `vision_encoder*.onnx` maps one patch grid (an image, or one temporal patch of a video) to its projected features. `decoder*.onnx` is the LFM3 decoder, which scatters the `image_features` and `video_features` inputs over the `<|image|>` / `<|video|>` placeholders of `input_ids`; `logits_to_keep` limits the logits it returns. The bundles are:

| `--precision` | decoder | vision encoder |
|---|---|---|
| fp32 | `decoder.onnx` | `vision_encoder.onnx` |
| fp16 | `decoder_fp16.onnx` | `vision_encoder_fp16.onnx` |
| q8 | `decoder_q8.onnx` | `vision_encoder_fp16.onnx` |
| q4 | `decoder_q4.onnx` | `vision_encoder_q8.onnx` |
| q8f16 | `decoder_q8f16.onnx` | `vision_encoder_fp16.onnx` |
| q4f16 | `decoder_q4f16.onnx` | `vision_encoder_fp16.onnx` |

The decoders are quantized as the LFM3 text ones ([3.1b](#31b-lfm3-text-models)). The image features are precision-sensitive: by answer-token KL from the fp32 reference, the int8 vision encoder costs more than the q8 decoders' own error, so q8 and q8f16 keep fp16 vision, as q4f16 does; next to q4's int4 decoder it doesn't show, so q4 takes the smaller int8 encoder.

`lfm3-vl-export` reads LFM3-VL-2B's `config.json` (`Lfm3VlForConditionalGeneration`, with `text_config` and `vision_config`) and its tensors (`model.language_model.*`, `model.vision_tower.*`, `model.multi_modal_projector.*`), and also an earlier, flat layout of both; `lfm3_vl.config.REVISIONS` pins `LiquidAI/LFM3-VL-2B` at `f83d2b0`. Images go through transformers' `Qwen2VLImageProcessor`, configured from the `image_processor` entry of `processor_config.json` (or a `preprocessor_config.json`); the LFM3-VL processor is that class with LFM3 defaults, and `tests/test_lfm3_vl/test_processor.py` checks that this gives the token ids and pixels of the checkpoint's own processor. Video goes through the graphs, but `lfm3-vl-infer` takes images only; as in `lfm3-infer`, `--no-think` turns thinking off.

The video markers come doubled, from the checkpoint's own files, and the graphs see the ids transformers gives. The chat template writes `<|vision_start|><|video|><|vision_end|>`, and the processor replaces only `<|video|>`, with a timestamp and a `<|vision_start|>`/`<|vision_end|>` pair per temporal patch, so the template's pair stays around them: `<|vision_start|><… seconds><|vision_start|><|video|>…<|vision_end|><… seconds><|vision_start|>…<|vision_end|><|vision_end|>`. `test_video.py::test_video_markers` checks this layout. LFM3-Omni's processor builds the same layout ([3.1e](#31e-lfm3-omni)). The decoder's positions are a plain count, so the markers are ordinary tokens to the graphs, and the tests give the processor's `input_ids` to both the reference and the graphs.

The reference of `tests/test_lfm3_vl` is the checkpoint's own remote code; the tests without a download run a tiny random checkpoint against LFM3-VL-2B's remote code, vendored in `tests/test_lfm3_vl/reference/`. `--lfm3-vl-model` (default `LiquidAI/LFM3-VL-2B`) sets the checkpoint of the real-model tests:

```bash
uv run pytest tests/test_lfm3_vl/test_tiny_export.py tests/test_lfm3_vl/test_config.py tests/test_lfm3_vl/test_rope.py -v  # no download
uv run pytest tests/test_lfm3_vl/test_processor.py -v
uv run --no-sync pytest tests/test_lfm3_vl -v --device cuda          # in the CUDA environment of 2
uv run --no-sync pytest tests/test_lfm3_vl -v -k "(q4 and not q4f16) or (vision and q8)" --device cuda --ep cpu  # int4 bundle on the CPU
```

`test_decoder.py` scores each bundle's prefill (vision encoder and decoder) where generation reads logits, the last prompt position and the reference's greedy answer, teacher-forced: mean KL below 1e-6 (fp32), 1e-4 (fp16), 5e-4 (q8, q8f16) and 0.1 (q4, q4f16), about 4 times bf16 PyTorch's, and the reference's top token at every confident position but in int4. Its prompts are an image, two images, a text prompt (a haiku) and an image with thinking on. `test_vision_encoder.py` holds the vision encoders' token cosines to floors (fp16 mean 0.9998, int8 1st percentile 0.94), `test_decode.py` each bundle's cached decode to the reference's top 5, `test_video.py` a 4-patch video, and the opt-in `test_long.py` a 10 s video and four images in one turn (about 12K and 3.5K tokens).

Known limits of LFM3-VL-2B: on the haiku, q8 and q8f16 are over their 5e-4 bound and q4 and q4f16 over 0.1. The checkpoint is very confident on open-ended text, so the same rounding moves more probability; bf16 PyTorch is itself above the q8 bound there, and over ten other text prompts q8 stays under bf16. On the 4-patch video the fp16 vision encoder's mean token cosine is just under its 0.9998 floor and the fp16 bundle's answer KL just over 1e-4: a few low-norm tokens are ill-conditioned, as in fp16 PyTorch. On the CPU EP q8 and q8f16 score higher than on CUDA (int8 activations, [3.1b](#31b-lfm3-text-models)).

### 3.1d LFM3-Audio

```bash
# All precisions (fp16, q4, q8, q4f16, q8f16; fp32 is always written)
uv run lfm3-audio-export LiquidAI/LFM3-Audio-2B --precision

uv run lfm3-audio-infer --model exports/LFM3-Audio-2B-ONNX --mode asr --audio samples/audio/woodworks_question.wav
uv run lfm3-audio-infer --model exports/LFM3-Audio-2B-ONNX --mode tts \
    --prompt "The quick brown fox jumps over the lazy dog." --output tts.wav
uv run lfm3-audio-infer --model exports/LFM3-Audio-2B-ONNX --mode s2s \
    --audio samples/audio/woodworks_question.wav --output answer.wav
```

LFM3-Audio-2B is an LFM3 backbone with three more parts: a speech encoder (three stride-2 convolutions, then a 17-layer LFM2 stack, then an MLP projector), a 3-layer bidirectional depth decoder that predicts each 12-codebook speech frame with MaskGIT in 3 passes, and the Liquid codec decoder (FSQ codes, then an 8-layer LFM3 stack, then an iSTFT to 24 kHz, 80 ms per frame). Like LFM3, it runs on onnxruntime with its own generation loop (`lfm3_audio.session.ONNXAudioModel`, which follows `Lfm3AudioForConditionalGeneration._generate`), so the folder has no `genai_config.json`; `lfm3_audio.config.REVISIONS` pins `LiquidAI/LFM3-Audio-2B` at `9d49701`.

```
exports/LFM3-Audio-2B-ONNX/
├── config.json, generation_config.json, tokenizer.json, tokenizer_config.json, special_tokens_map.json,
│   chat_template.jinja, preprocessor_config.json, processor_config.json, audio_tokenizer/config.json
└── onnx/
    ├── decoder.onnx          # LFM3 decoder: audio_features scattered over <|audio|>, audio_out_codes [F, 12]
    │                         # embedded over <|audio_out|>; logits and final-norm hidden_states
    ├── audio_encoder.onnx    # input_features [1, 128, T] (numpy log-mel) -> audio_features [ceil(T/8), 2048]
    ├── depth_decoder.onnx    # hidden_states [B, 2048], codes [B, 12], masked [B, 12] -> logits [B, 12, 2017]
    ├── codec_decoder.onnx    # audio_codes [B, 12, F] -> stft_features [B, 6F, 1282]; numpy iSTFT -> 24 kHz
    └── decoder_{fp16,q8,q8f16,q4,q4f16}.onnx, audio_encoder_fp16.onnx, codec_decoder_fp16.onnx, *.onnx_data*
```

| `--precision` | decoder | audio encoder | depth decoder | codec |
|---|---|---|---|---|
| fp32 | `decoder.onnx` | `audio_encoder.onnx` | `depth_decoder.onnx` | `codec_decoder.onnx` |
| fp16 | `decoder_fp16.onnx` | `audio_encoder_fp16.onnx` | `depth_decoder.onnx` | `codec_decoder_fp16.onnx` |
| q8 | `decoder_q8.onnx` | `audio_encoder.onnx` | `depth_decoder.onnx` | `codec_decoder.onnx` |
| q4 | `decoder_q4.onnx` | `audio_encoder.onnx` | `depth_decoder.onnx` | `codec_decoder.onnx` |
| q8f16 | `decoder_q8f16.onnx` | `audio_encoder_fp16.onnx` | `depth_decoder.onnx` | `codec_decoder_fp16.onnx` |
| q4f16 | `decoder_q4f16.onnx` | `audio_encoder_fp16.onnx` | `depth_decoder.onnx` | `codec_decoder_fp16.onnx` |

The decoders are derived from `decoder.onnx` as the LFM3 ones are ([3.1b](#31b-lfm3-text-models)), so `--skip-export` works; the speech-frame codebook table stays float. The CPU bundles (q8, q4) keep the encoder and codec in fp32, and the CUDA ones (fp16, q8f16, q4f16) take their fp16 graphs, since CUDA has no float32 GroupQueryAttention kernel. Every bundle runs the fp32 depth decoder: it has no GroupQueryAttention, so it runs on CUDA too, and it is the only depth graph that keeps the reference's greedy codes (fp16 and q8 flip a few percent of the argmaxes, all at near-ties). The export logs k_quant's messages of [3.5](#35-harmless-log-messages).

`lfm3-audio-infer` has four modes, with the checkpoint's prompts: `asr` (system "Perform ASR in English."), `tts` (system "Perform TTS in American English using a female voice.", the assistant turn opened with `<|audio_start|>`), `s2s` (system "Respond with interleaved text and audio.", the assistant turn opened with `<|mixed_start|>`, then 6 text tokens and 12 speech frames in turn until the text ends) and `chat` (a text answer). `--language` and `--voice` change the prompts. For English, the model cards' TTS prompts take a locale: American English or British English. Text is greedy. Speech frames are sampled with the checkpoint's settings, which `lfm3-audio-infer` takes from the export's `generation_config.json` (the checkpoint's, copied verbatim) as the reference's `generate` takes them from the checkpoint's: each of `audio_temperature`, `audio_top_k`, `audio_repetition_penalty` and `audio_repetition_window` comes from its flag (`--audio-temperature`, `--audio-top-k`, `--audio-repetition-penalty`, `--audio-repetition-window`), else from the file, else from the reference's defaults (none, which is greedy, and a 64-frame window). LFM3-Audio-2B's file samples at temperature 1.0, top-k 128 and repetition penalty 1.1 over 4096 frames, LFM3-Omni-2B's at temperature 0.4 and repetition penalty 1.2 over 64 frames. `--greedy` takes the likeliest codes (temperature 0: a top-k without a temperature samples at temperature 1), and `--seed` makes the orders and draws reproducible. On CUDA the fp32 decoder, speech encoder and codec run in TF32 unless `--no-tf32` is given, as in the other inference CLIs ([4](#4-inference)); the depth decoder always runs in fp32. `--suppress-mixed` and `--suppress-audio-out` forbid `<|mixed_start|>` and `<|audio_start|>` in the answer, as `chat` does. Input WAVs are read at 16 kHz; other rates are resampled with scipy, not the reference's soxr, which changes the top mel bands. Not covered: streaming (live ASR, streaming codec decode), the codec encoder (speech prompts given as codes) and batches.

The reference of `tests/test_lfm3_audio` is the checkpoint's own remote code (float32, sdpa, TF32 off); the tests without a download run a tiny random checkpoint against LFM3-Audio-2B's remote code, vendored in `tests/test_lfm3_audio/reference/`. `--lfm3-audio-model` (default `LiquidAI/LFM3-Audio-2B`) sets the checkpoint of the real-model tests; `LiquidAI/LFM3-Omni-2B` runs them on LFM3-Omni's audio parts ([3.1e](#31e-lfm3-omni)):

```bash
uv run pytest tests/test_lfm3_audio/test_config.py tests/test_lfm3_audio/test_weights.py \
    tests/test_lfm3_audio/test_tiny_reference.py tests/test_lfm3_audio/test_tiny_decoder.py \
    tests/test_lfm3_audio/test_tiny_codec.py tests/test_lfm3_audio/test_gqa_window.py \
    tests/test_lfm3_audio/test_infer.py -v  # no download
uv run pytest tests/test_lfm3_audio/test_runtime.py tests/test_lfm3_audio/test_audio_encoder.py \
    tests/test_lfm3_audio/test_depth.py -v -k "(tiny or sample or sampling or orders) and not onnx"  # no download
uv run --no-sync pytest tests/test_lfm3_audio -v --device cuda \
    -k "not q8 and not q4 or f16"   # in the CUDA environment of 2: fp32 and the CUDA bundles
uv run --no-sync pytest tests/test_lfm3_audio/test_runtime.py -v \
    --device cuda -m slow -k tf32   # opt-in: the CUDA bundles with TF32 on, the CLIs' default
uv run pytest tests/test_lfm3_audio -v -k "not q8f16 and not q4f16"    # CPU EP
```

What they hold the bundles to:

- `test_decoder.py`: the decoder's mean KL over the reference's greedy answer (an ASR prompt of two clips past the 256-token window, a text question, a replayed TTS turn and a replayed S2S turn, each followed by a question): LFM3-VL's bounds ([3.1c](#31c-lfm3-vl-vision-language-models)), the reference's top token at every confident position but in int4, and the final-norm hidden states the depth decoder reads at cosine 0.99999 (fp32), 0.999 (fp16, q8, q8f16) and 0.95 (int4).
- `test_audio_encoder.py`, `test_depth.py` and `test_codec.py`: the speech encoder's tokens, the depth decoder's codebook logits over every call of a greedy TTS and S2S run, and the codec's waveform, each against the reference with bf16 PyTorch as the floor; `test_exported_*` score the export's own files the same way.
- `test_runtime.py`: the runtime's mode machine, MaskGIT sampling and prompts against the reference's `generate`, and each bundle end to end, greedy with the reference's MaskGIT orders injected: fp32 gives the reference's slots and codes exactly in ASR, TTS and S2S; the others the reference's ASR transcript, TTS speech that ends, S2S answers in the interleaved schedule with their text on the reference's path, and TTS round trips that their own ASR transcribes back exactly. Greedy speech often runs to the slot cap instead of ending, in float32 PyTorch too, so the tests do not require a bundle's speech to end where the reference's does. `test_onnx_seeded_sampling` checks that seeded speech sampling draws the codes transformers' processors give, and `test_speech_sampling` how the CLIs resolve the speech settings.

Known limits of LFM3-Audio-2B: on the replayed TTS turn q8 and q8f16 are over the 5e-4 KL bound. This is the int8 rounding of the checkpoint's weights: the float32 reference with its weights rounded as the q8 recipe rounds them scores the same, and every position keeps the reference's top token. On the CPU EP the q8 cells are higher (int8 activations), which can also take q4's hidden-state cosine on the replayed S2S turn under its 0.95 floor. In `chat` mode the checkpoint answers a text question with "… Paris.</think>The capital …" in every precision, as float32 and bf16 PyTorch do.

### 3.1e LFM3-Omni

```bash
# All precisions (fp16, q4, q8, q4f16, q8f16; fp32 is always written)
uv run lfm3-omni-export LiquidAI/LFM3-Omni-2B --precision

# An image question (text answer), ASR, TTS, and an interleaved spoken answer about an image
uv run lfm3-omni-infer --model exports/LFM3-Omni-2B-ONNX \
    --images tests/test_lfm2_vl/assets/cardinal.jpg --prompt "What bird is this?"
uv run lfm3-omni-infer --model exports/LFM3-Omni-2B-ONNX --mode asr \
    --audio samples/audio/woodworks_question.wav
uv run lfm3-omni-infer --model exports/LFM3-Omni-2B-ONNX --mode tts \
    --prompt "The harbor was quiet that morning." --output tts.wav
uv run lfm3-omni-infer --model exports/LFM3-Omni-2B-ONNX --mode s2s \
    --images tests/test_lfm2_vl/assets/bluejay.jpg --prompt "What bird is this?" --output answer.wav
```

`LiquidAI/LFM3-Omni-2B` (`lfm3_omni`) is the LFM3 decoder with the LFM3-VL vision tower and the LFM3-Audio speech stack ([3.1d](#31d-lfm3-audio)). Its code, weights (transformers' names) and `config.json` (nested `text_config`, `vision_config`, `audio_encoder_config`, `depth_config`) follow transformers 5.19+. `lfm3-omni-export` writes LFM3-Audio's graphs (`liquidonnx.lfm3_audio`, whose decoder also scatters `image_features` and `video_features` over `<|image|>` and `<|video|>`) and LFM3-VL's vision encoder (`liquidonnx.lfm3_vl.vision`), and runs on onnxruntime with its own generation loop (`liquidonnx.lfm3_omni.session.ONNXOmniModel`); `lfm3_audio.config.REVISIONS` pins it at `ce2458c`. The checkpoint's files are copied, and `preprocessor_config.json` is written from `processor_config.json`'s `image_processor`. The bundles are LFM3-Audio's, with LFM3-VL's vision encoder for each precision (the q4 bundle's int8, the others' fp32 or fp16 as in [3.1c](#31c-lfm3-vl-vision-language-models)).

The modes and the speech sampling of `lfm3-omni-infer` are `lfm3-audio-infer`'s (the flags, else the export's `generation_config.json`, as the reference's `generate` reads the checkpoint's; [3.1d](#31d-lfm3-audio)), and `--images` adds images to the user turn. The chat template wraps each image and clip as `<|vision_start|><|image|><|vision_end|>` and `<|vision_start|><|audio|><|vision_end|>`, and ends the generation prompt with `<think></think>`. The model can switch to speech on its own, as it does on an image question under some system prompts, so the `chat` mode (the default) suppresses `<|mixed_start|>` and `<|audio_start|>`, as the reference's explicit `generation_mode="text"` does. Video goes through the graphs (the tests cover it, with the processor's timestamped frame groups), but `lfm3-omni-infer` takes images, audio and text only.

```bash
uv run pytest tests/test_lfm3_omni/test_config.py -v                                  # no download
uv run --no-sync pytest tests/test_lfm3_omni -v --device cuda                         # in the CUDA environment of 2
uv run --no-sync pytest tests/test_lfm3_omni -v --device cuda -m slow -k tf32         # opt-in: TF32 on, the CLI's default
uv run --no-sync pytest tests/test_lfm3_audio -v --device cuda --lfm3-audio-model LiquidAI/LFM3-Omni-2B
uv run pytest tests/test_lfm3_omni -v                                                 # CPU EP, reference on the CPU
```

`tests/test_lfm3_omni` covers the image, video and image-plus-speech parts with LFM3-VL's bounds and floors (the fp16 vision encoder's 1st-percentile floor is 0.997: Omni's tower ends in large activations, where fp16 math loses a few tokens), and `tests/test_lfm3_audio` the speech parts. End to end, fp32 gives the reference's image answer and its interleaved spoken answer slot for slot and code for code.

Known limits of LFM3-Omni-2B: on the two-bird prompt fp16, q8 and q8f16 are over their bounds, from the fp16 vision encoder (fp16 PyTorch's tower measures the same). On the replayed S2S turn of `tests/test_lfm3_audio`, q8 and q8f16 are over 5e-4 (the int8 rounding of the weights), and q4 and q4f16 are over 0.1 there and on its text prompt: the int4 rounding of the weights, which the float32 reference loaded with the q4 graph's weights reproduces. On the CPU EPs q8 and q8f16 score higher (int8 activations) and the fp16 vision encoder's 1st percentile on `wide.jpg` falls under its floor; on CUDA the fp16 speech encoder's on the woodworks clip falls just under 0.9999.

### 3.1f d1

```bash
# All precisions (fp16, q4, q8, q4f16, q8f16; fp32 is always written)
uv run d1-export LiquidAI/d1-2B --precision
uv run d1-export LiquidAI/d1-Omni-2B --precision

# Named questions over a state, a picture or a clip; prints the API's JSON answer
uv run d1-infer --model exports/d1-2B-ONNX --request samples/d1/ticket.json
curl -LO http://images.cocodataset.org/val2017/000000039769.jpg  # the model cards' picture
echo "dea9e7ef97386345f7cff32f9055da4982da5471c48d575146c796ab4563b04e  000000039769.jpg" \
    | sha256sum -c -
uv run d1-infer --model exports/d1-2B-ONNX --request samples/d1/picture.json \
    --media 000000039769.jpg --precision q4 --repeat 10
uv run d1-infer --model exports/d1-Omni-2B-ONNX --request samples/d1/clip.json \
    --media samples/audio/woodworks_question.wav
```

`LiquidAI/d1-2B` and `LiquidAI/d1-Omni-2B` are decision models: LFM3-VL-2B and LFM3-Omni-2B post-trained to answer typed, named questions (yes/no, choice, score, number, text, date, box, boxes) in one forward pass, with no generation. Their `config.json` is the base model's (`lfm3_vl`, `lfm3_omni`) with `architectures: ["D1Model"]` and `d1_*` keys, and their remote code (`runner.py`, `prompt.py`, `attention.py`, `pointer.py`, `box.py`, ...) holds everything around the forward: prompts, packing, answer slots, readouts, calibration and typed answers. `d1.config.REVISIONS` pins them at `97c5704` and `4cccbfd`.

A request packs a state and up to `d1_set_size` (16) questions into one row: `<|startoftext|><|im_start|>user\n{media}{state}\n\n\nQUESTION:\n{q1}<|im_end|>\n<|im_start|>assistant\n<think></think>{slots}`, then `QUESTION:\n{q2}...` for each further question. Answers are read at answer slots, from the logits (option codes, digits, yes/no, numbers 0..999 for box edges) or, for text, number and date reads, from a pointer: the start and end slots' final hidden states dotted with those of the state's positions. Under `d1_attention: "set"` the 6 full-attention layers do not see causally: a state token sees every real token of the row, a question token the state and its own question, never another question. The 5 sliding-window layers (window 256) and the 13 short convolutions stay causal over the packed row. So the plain LFM3-VL decoder cannot serve d1, and the state's keys depend on the questions, so no prefix of a call can be cached.

`d1-export` builds, with the LFM3-VL and LFM3-Omni builders:

- `vision_encoder*.onnx`: LFM3-VL's ([3.1c](#31c-lfm3-vl-vision-language-models)).
- `audio_encoder*.onnx` (d1-Omni-2B): LFM3-Audio's ([3.1d](#31d-lfm3-audio)). The depth decoder, codebook table and speech codec are not exported: d1 never generates.
- `decoder*.onnx` (`liquidonnx.d1.decoder`): the LFM3 decoder run once over a packed row, without cache inputs or outputs. It takes `input_ids`, `attention_mask`, `position_ids`, the `image_features`, `video_features` (and `audio_features`) of the row's placeholders, `question_ids` (`[B, S]`: -1 on the state, k on question k) and `slot_positions` (`[N]`, flat positions of the answer slots). It returns `logits` `[N, 128000]` at the slots and the final-norm `hidden_states` `[B, S, 2048]` that the pointer reads. The full-attention layers run MultiHeadAttention with the KV heads repeated and an `attention_bias` of the set mask, built once in the graph from `question_ids` and `attention_mask`: GroupQueryAttention adds a bias on top of its causal mask, so it cannot express it. The sliding layers keep GroupQueryAttention with `local_window_size` 256.

The bundles are LFM3-Omni's, without the depth decoder and codec. The runtime (`liquidonnx.d1.session.load`) runs the checkpoint's own host code, which the export folder carries (`*.py`, `*.json`, `*.jinja` and `LICENSE`, verbatim): `OrtSystemOne` gives its `runner.SlotReader` the forwards on onnxruntime, so `system_one` and `system_one_batch` are the checkpoint's. Loading a folder executes its Python code, as `trust_remote_code` does; a non-Python runtime would have to port that code.

The q4 and q4f16 decoders of d1 keep only the MLP `gate_proj` and `up_proj` in int4 (k_quant, blocks of 32): 48 of the 154 MatMuls, with 44% of the weights. Every other MatMul is k_quant int8, the LM head and token table included (`lfm3.quantize.gate_up_int4_int8_nodes`). d1's outputs are the probabilities themselves, and under LFM3's recipe the int4 rounding error of these checkpoints builds up over the layers with no single group of MatMuls to keep in int8: d1-Omni-2B's answer to a spoken question flips. The float32 reference loaded with that bundle's weights gives the same error, so the rounding of the weights causes it, not the graph. `d1-export` gives this recipe to every d1 checkpoint; every other exporter keeps LFM3's (`lfm3.quantize.q4_int8_nodes`), LFM3-Omni's included.

```bash
uv run pytest tests/test_d1/test_config.py tests/test_d1/test_tiny_export.py \
    tests/test_d1/test_tiny_session.py -v                                             # no download, in CI
uv run --no-sync pytest tests/test_d1 -v --d1-model LiquidAI/d1-2B --device cuda      # in the CUDA environment of 2
uv run --no-sync pytest tests/test_d1 -v --d1-model LiquidAI/d1-Omni-2B --device cuda
uv run pytest tests/test_d1 -v --d1-model LiquidAI/d1-Omni-2B                         # CPU EP, reference on the CPU
```

The requests are the model cards' (a ticket, an invoice, two parcel tickets in one batch, the cats picture and, for d1-Omni-2B, a clip of beeps) and a spoken question (`samples/d1/clip.json`). The picture is downloaded on first use, checked against its sha256 and kept in `~/.cache/liquidonnx`. `test_decoder.py` replays every forward of the ONNX engine through the reference engine's `_slot_logits` and holds the slots' KL to LFM3-VL's bounds; `test_answers.py` compares `system_one_batch`'s answers with the reference's: discrete parts must match, probabilities must be within 1e-3 (fp32), 1e-2 (fp16), 3e-2 (q8, q8f16) or 0.1 (q4, q4f16), and box edges within 2e-3, 5e-3, 1e-2 or 3e-2; two near-ties of d1-Omni-2B may fall on either side. The tiny test checks the decoder against LFM3-VL-2B's vendored remote code with the set mask on its full-attention layers, and runs `d1-export` end to end on a tiny d1 and a tiny d1-Omni checkpoint. `test_tiny_session.py` runs `d1.session` on the tiny d1 export with d1-2B's host code, vendored in `tests/test_d1/reference/`, and a generated tokenizer: a picture request and an invoice's reads on the CPU EP, whose answers it checks for their shapes and ranges (the weights are random).

Known limits: on the CPU EP q8 and q8f16 can exceed their KL bound (int8 activations, [3.1b](#31b-lfm3-text-models)). d1-Omni-2B's picture reads box edges as 1000-way distributions, which are sensitive in this checkpoint: there fp16 on CUDA, q8 and q8f16 are over their bounds but under bf16 PyTorch's KL, and the answers hold. On its spoken question q8 and q8f16 are over their bounds: the int8 rounding of the weights, which the float32 reference loaded with the q8 bundle's weights reproduces. The runtime reads one packed row per forward (the graph's GroupQueryAttention assumes no padding), the set mask is a dense `[B, 1, S, S]` bias, and video goes through the vision encoder but no test covers it. `d1.config` takes `D1Model` with `d1_attention: "set"` only.

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

# LFM2.5-24B-A2B: 40 layers, 64 experts, 4 active per token
uv run lfm2-moe-export LiquidAI/LFM2.5-24B-A2B --precision
```

Same layout as the text models. The experts become `QMoE` int4 (q4, q4f16) or int8 (q8); the routers stay fp32. The LFM2.5-8B-A1B `--precision` export needs much more RAM than the dense models' exports.

The LFM2.5-24B-A2B `--precision` export needs more RAM still, and writes 187 GiB: fp32 89.4 GiB, fp16 44.7, q8 25.2, q4 14.2 and q4f16 12.8. Much of its time goes to the fp16 conversion of the fp32 decoder, which runs on one core. Its fp32 PyTorch model (96 GB) does not fit on an 80 GB GPU, so its `lfm2-compare` references need a GPU with more memory (the ROCm build of PyTorch takes `--device cuda`) or the CPU. Scores on wikitext-2 ([5.1](#51-comparing-an-export-with-its-reference-model)), against an fp32 reference:

| precision | device | KLD ± SE | same top | LFM2.5-8B-A1B |
|---|---|---|---|---|
| q4 | CPU | 0.2129 ± 0.0086 | 79.4% | 0.2103 |
| q4f16 | CUDA | 0.2146 ± 0.0086 | 79.3% | 0.2022 |
| q8 | CUDA | 0.0322 ± 0.0021 | 92.1% | 0.0182 |
| fp16 | CUDA | 0.0146 to 0.0156 ± 0.0020 | 95.1 to 95.4% | 0.0057 |

The fp32 decoder scores 0 (−4e-7, 100% same top on 8 chunks). q4 scores as LFM2.5-8B-A1B's does, q8 and fp16 2 to 3 times as high. That is the checkpoint, not the export: rounding the PyTorch weights to int8 as q8 does moves the next token of "The sky is" by KL 0.57, against 0.016 for LFM2.5-8B-A1B.

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
- `neural_compressor [WARNING] - Model size > 2GB. Please use model path instead of onnx model object to quantize` (q4 builds of decoders over 2 GB in fp32: LFM2 text, MoE and VL from LFM2-700M up, and the LFM3 families; a q4f16 export runs one too, except an LFM2 one that reuses an existing q4): k_quant gets the model in memory rather than as a path, from the genai builder (LFM2) or `liquidonnx.quantize.quantize_matmuls` (LFM3); the saved decoder is complete.
- numpy `RuntimeWarning` lines from `onnxruntime/quantization/neural_compressor/weight_only.py`, such as `overflow encountered in divide` or `invalid value encountered in divide` (q4 builds of 2.6B, 8B-A1B and the LFM3 families): k_quant's arithmetic over- or underflows on blocks of 32 tiny weights. In 8B-A1B (largest values around 1e-36) its search's candidate scales come out NaN and are dropped, so those blocks keep their min-max scale. In 2.6B, 746,432 blocks hold only subnormal values (at most 3.2e-39); their min-max scale comes out 0, so they dequantize to zero. Both q4 exports pass the wikitext gate of [5.1](#51-comparing-an-export-with-its-reference-model). The LFM3 decoders log the `invalid value` pair for a few thousand blocks of near-zero weights, which keep their min-max scale as in 8B-A1B.
- `[E:onnxruntime:Default, cudnn_fe_call.cc:39 CudaErrString<cudnn_frontend::error_object>] [cudnn_frontend] Error: No execution plans support the graph.` (LFM3-Audio and LFM3-Omni on CUDA, a few lines per clip that the fp16 speech encoder of the fp16, q8f16 and q4f16 bundles encodes): cuDNN has no execution plan for one of its graphs, and onnxruntime runs it another way; the encoder's output is unaffected.
- On macOS, `objc[...]: Class MATStreamingSessionDelegate is implemented in both ...` (two lines in every process that loads onnxruntime and onnxruntime-genai, with the onnxruntime nightly): no effect on the results.

## 4. Inference

The LFM3 families' CLIs run on onnxruntime and come with their exports in [3.1b](#31b-lfm3-text-models) and the sections after it; [4.6](#46-lfm3-families-on-webgpu) covers running their graphs on WebGPU. The rest of this section is about the LFM2 families.

The inference CLIs run the export folder on onnxruntime-genai, with interactive multi-turn chat and streaming output. They load the precision `genai_config.json` points at; `--precision` picks another one, and `--ep cuda` runs on CUDA (with the packages in [2](#2-installation)). The CUDA EP runs fp32 matmuls and convolutions in TF32 unless `--no-tf32` is given, so checks against an fp32 reference need the flag (`lfm2-compare --device cuda` always turns TF32 off).

Text is decoded greedily. An answer ends at the end-of-turn token or after `--max-tokens` tokens: 4096 by default, room for the `<think>` block that LFM2.5-2.6B, LFM2.5-8B-A1B and LFM2.5-24B-A2B write before they answer (`lfm2-audio-infer` has a default per mode, see [4.4](#44-audio-asr-tts-interleaved)).

On CUDA, use fp16 or q4f16 for text and MoE (bare `--precision` leaves q4f16 out for the text models, so name it as in [3.1](#31-lfm2-text-models)), and fp16 for VL and Audio. The other precisions keep fp32 activations, so their GroupQueryAttention nodes, and the experts of MoE q4, run on the CPU. The MoE decoders mark their QMoE experts `weights_prepacked=0`, without which the CUDA EP misreads them; older MoE exports print garbage on CUDA and need re-exporting.

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

On CPU, MoE speed depends on the onnxruntime version; `lfm2-bench --ep cpu` measures it on your machine. The inference CLIs log a warning when they load LFM2-MoE on CPU with onnxruntime older than 1.31; [2](#2-installation) shows how to switch a pip environment to the nightly.

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

### 4.6 LFM3 families on WebGPU

This repository has no browser runtime; these are the graphs and settings a WebGPU one (onnxruntime-web) should use for the LFM3 families (LFM3, LFM3-VL, LFM3-Audio, LFM3-Omni, d1), as `tests/web` checks them against onnxruntime's CPU EP on the same files and inputs, with onnxruntime-web 1.30.0 on Apple silicon (Metal).

- Load the native WebGPU EP, `import * as ort from 'onnxruntime-web/webgpu'` (`ort.webgpu.min.js`). The default `onnxruntime-web` import's WebGPU EP (JSEP) can't create a session for any q4 or q4f16 decoder or the q8 vision encoder (its MatMulNBits takes 2 and 4 bits only, and every decoder has an int8 LM head), and it ignores GroupQueryAttention's `local_window_size` without an error.
- Pass `executionProviders: [{ name: 'webgpu', enableMatmulFp32Accumulation: true }]`. 1.30.0 ignores the option; later builds sum q4f16's and q8f16's `MatMulNBits` in float32 with it (below).
- Run the q4f16 decoder with the q4 bundle's side graphs. The q4 decoder, with the same side graphs, suits shorter contexts (**Long prefill** below):

| Decoder | Side graphs | On WebGPU |
|---|---|---|
| `decoder_q4f16.onnx` (text: `model_q4f16.onnx`) | The q4 bundle's: `vision_encoder_q8.onnx`, `audio_encoder.onnx`, `depth_decoder.onnx`, `codec_decoder.onnx` | As far from the fp32 decoder as the CPU EP's q4f16 |
| `decoder_q4.onnx` (text: `model_q4.onnx`) | The same | The decoders match the CPU EP with `MatMulNBits` at `accuracy_level` 0: WebGPU computes float activations on Apple GPUs, where the CPU EP at the exported level 4 quantizes them to int8 |

Every graph of the tiny exports, in every precision, loads, runs, and leaves only int64, int32 and bool plumbing on the CPU EP: the int64 graph inputs' casts, shapes and positions, the placeholder rows the features go to, and d1's set mask below (WebGPU has no int64 kernels unless `enableInt64`, which 1.30.0 can't set from JS, and no NonZero kernel). Sliding layers slide: GroupQueryAttention's `local_window_size` gives WebGPU the CPU EP's outputs, on a prefill past the window and on cached decodes past it.

d1's set mask stays on the CPU EP. The decoder compares the int64 `question_ids` and joins the comparisons with `Or`, which WebGPU has no kernel for, so the CPU EP builds the `[B, S, S]` bool mask and uploads it; no node there reads a float tensor. Building it on the GPU instead, from int32 ids with `Equal`, `Not` and `And`, is no faster, and 1.30.0 gets it wrong: when S is a multiple of 4 and the state ends inside a group of 4 tokens, the other questions also see the first one to three tokens of the first question. The tiny d1 rows cover that case.

Limits of onnxruntime-web 1.30.0:

- **fp16 sums.** WebGPU's MatMul, Gemm and MatMulNBits keep the partial sums of fp16 inputs in fp16, where the CPU EP stays within an fp16 ulp. The fp16 side graphs and the fp16 decoder move much further from fp32 than on the CPU EP, hence the side graphs of the q4 bundle above. Wrapping the fp16 graphs' MatMuls in float32 casts would fix WebGPU but move the CPU EP's own outputs, so the fp16 files stay as they are. `enableMatmulFp32Accumulation` in later builds brings MatMulNBits within an fp16 ulp; nothing changes MatMul's.
- **Long prefill.** Past the 256-token window, the sliding layers' GroupQueryAttention materializes `[B, 16, S, T]` probabilities for S new tokens over T positions (float32 in q4, float16 in q4f16), and at 4 GiB a SafeInt check fails (`Integer overflow`): a single run fails at 8192 tokens in q4 and 11586 in q4f16, 4096-token chunks stop at 16384 positions in q4 and 32768 in q4f16, and 1024-token chunks at 65536 and 131072. Single runs of 5400 and 7200 tokens in q4, and of 8192 and 10800 in q4f16, return non-finite logits in all six released models. Prefill the q4f16 decoders in chunks of at most 1024 tokens, which reach the 128,000-token context of LFM3-VL-2B, LFM3-Audio-2B and LFM3-Omni-2B, and LFM3-1.7B's in chunks of at most 256, which reach its 262,144 (with 512, the chunk that ends there fails). Use the q4 decoders only for shorter contexts: 1024-token chunks stop them at 65536 positions, and 4096-token chunks at 16384. d1 runs one forward per packed row: keep each row at most 4096 tokens with `decoder_q4` and under 8192 with `decoder_q4f16`.
- **Other adapters.** Chromium's SwiftShader adapter (the Linux CI's) has no `shader-f16`, and its 4-wide subgroups break 1.30.0's subgroup FlashAttention, which every adapter but Apple's and NVIDIA's takes for an attention over 32 or more positions without a window or within one, such as a decoder's full layers or its sliding layers up to 256 tokens; other GPUs with that path were not checked. Adapters other than Apple's run `MatMulNBits` at `accuracy_level` 4 on int8 activations (DP4A), as the CPU EP does.
- **Graph capture** needs every node on the GPU and shared KV buffers, which the int64 plumbing and the caches' past/present inputs rule out.

`tests/web` runs every LFM3 graph's tiny export on the WebGPU EP ([tests/web/README.md](tests/web/README.md)):

```bash
uv run python tests/web/export_lfm3.py   # writes tests/web/.lfm3
cd tests/web && npm ci && npx playwright test lfm3
```

The fp16 and q4f16 matmul probes are its expected failures, for the fp16 sums above.

## 5. Testing

The tests need the `dev` dependency group, which `uv sync` installs by default ([2](#2-installation)). The export pipelines run on tiny random checkpoints, without downloads:

```bash
uv run pytest tests/test_genai_export.py tests/test_genai_export_multimodal.py tests/test_export_cli.py -v
uv run pytest tests/test_lfm2_audio/test_modes_synthetic.py tests/test_lfm2_audio/test_reference_parity.py \
    tests/test_lfm2_audio/test_graph_structure.py -v
uv run pytest tests/test_compare.py tests/test_compare_wikitext.py -v
```

Each LFM3 family's tests without a download are listed with its export ([3.1b](#31b-lfm3-text-models) and the sections after it), and CI runs them all (`.github/workflows/check_python.yml`).

The other tests run an export in `./exports` (`--exports-dir` sets another base folder), most of them against the checkpoint's PyTorch model, which they download. Each file covers several checkpoints and precisions; `-k` picks one, whose export must exist:

```bash
uv run lfm2-export LiquidAI/LFM2-350M --precision q4
uv run pytest tests/test_lfm2/test_decoder.py -v -k "350M and q4"

uv run lfm2-vl-export LiquidAI/LFM2-VL-450M --precision q4
uv run pytest tests/test_lfm2_vl/test_decoder.py tests/test_lfm2_vl/test_vision_encoder.py -v -k "450M and q4"

uv run pytest tests/test_lfm2_moe/test_decoder.py -v -k "LFM2.5-8B-A1B and q4 and not q4f16"
uv run pytest tests/test_lfm2_audio/test_asr.py -v -k "q4"

uv run lfm3-export LiquidAI/LFM3-1.7B --precision q4
uv run pytest tests/test_lfm3/test_decoder.py -v -k "q4 and not q4f"
```

The LFM3 families' tests take their checkpoint from an option per family (`--lfm3-model` and the others of [3.1b](#31b-lfm3-text-models) and the sections after it), the device of the reference and the sessions from `--device` (`--ep` when the sessions' execution provider differs), and hold their PyTorch reference and one precision's sessions at a time.

`--reference-cache DIR` keeps the PyTorch reference's greedy slots of `tests/test_lfm3_audio/test_runtime.py` (`test_generation`, `test_onnx_generation`) in `DIR`, one `.npz` per checkpoint revision, prompt ids, clip bytes, MaskGIT seed and slot count, and reads them back instead of running the reference. The key leaves out the device, so a folder filled on CUDA spares a CPU host its reference, which is slow there for an S2S answer; the log names the device that computed each file.

`--matmulnbits-accuracy-level N` is a diagnostic of the q8 recipe: every graph the tests load gets its MatMulNBits nodes at `accuracy_level` N in an in-memory copy that reads the export's weights where they are, so the export is not touched. With 0 the CPU EP keeps the activations float, where the recipe's 4 quantizes them to int8; this shows that cause of the CPU EP's q8 cells, for example:

```bash
uv run pytest tests/test_lfm3/test_decoder.py -v -k "q8 and not q8f16" --matmulnbits-accuracy-level 0
```

Tests marked `long` (long contexts, the wikitext-2 KLD of the LFM3 decoders, bf16 PyTorch floors) are skipped unless `--run-long` is given, and tests marked `slow` (long end-to-end runs, such as the speech bundles with TF32 on) unless `--run-slow` is given; `-m long` or `-m slow` selects them too. The ones that read wikitext-2 take the `wiki.test.raw` of [5.1](#51-comparing-an-export-with-its-reference-model) with `--wikitext`:

```bash
uv run --no-sync pytest tests/test_lfm3/test_long_context.py -v --device cuda --run-long \
    --wikitext wikitext-2-raw/wiki.test.raw
```

### 5.1 Comparing an export with its reference model

`lfm2-compare` scores every precision of an export folder against the PyTorch model (liquid-audio for audio): teacher-forced KL of the decoder over the reference's greedy answers, greedy answers on plain onnxruntime and on onnxruntime-genai, and for VL the vision encoder and embedding model. It writes a JSON file and a markdown table.

```bash
uv run lfm2-compare text --model LiquidAI/LFM2.5-1.2B-Instruct --export ./exports/LFM2.5-1.2B-Instruct-ONNX
uv run lfm2-compare vl --model LiquidAI/LFM2.5-VL-1.6B --export ./exports/LFM2.5-VL-1.6B-ONNX
uv run lfm2-compare audio --model LiquidAI/LFM2.5-Audio-1.5B --export ./exports/LFM2.5-Audio-1.5B-ONNX
```

`--device cuda` runs the reference, the ONNX sessions and onnxruntime-genai on the GPU with TF32 off, so fp32 stays fp32 (needs the CUDA packages in [2](#2-installation)). References are cached per device. `lfm2-compare vl --no-image-splitting` resizes each image once in the reference and in onnxruntime-genai instead of tiling it.

`lfm2-compare wikitext` is the quality gate for quantized precisions. It scores them on wikitext-2 with the protocol of [olive-recipes #638](https://github.com/microsoft/olive-recipes/pull/638): 64 chunks of 512 tokens, the second half of each scored, KLD ± SE over the chunks and same-top %. The reference is an fp32 one in llama.cpp's KL-divergence base format, given with `--reference` or computed from `--model`. The command exits with 1 when a precision is above `--max-kld` or, for the LFM2.5 models it knows, above today's KLD + 2 SE on `--device`. The ceilings are per execution provider, because the CPU and CUDA kernels score differently: the CPU ones cover q4 and q4f32, and the CUDA ones (TF32 off) q8, fp16 and q4f16. Each device scores those precisions by default; a precision without a ceiling on its device is reported as "no ceiling", not as a pass. The ONNX sessions run 13 intra-op threads on any machine, the count the ceilings were measured at, because onnxruntime's CPU MoE kernel gives different scores at different thread counts; `--threads` overrides it. `--help` explains the token streams.

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

The exception is [LiquidAI/LFM2.5-24B-A2B-ONNX](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-ONNX): `lfm2-moe-export --precision` from this repository made it ([3.3](#33-lfm2-moe-mixture-of-experts)), so the inference CLIs load its folder as downloaded.

The LFM3 families' ONNX repositories come from this repository's exporters too, and their CLIs load each folder as downloaded:

- [LiquidAI/LFM3-1.7B-ONNX](https://huggingface.co/LiquidAI/LFM3-1.7B-ONNX): `lfm3-export` ([3.1b](#31b-lfm3-text-models))
- [LiquidAI/LFM3-VL-2B-ONNX](https://huggingface.co/LiquidAI/LFM3-VL-2B-ONNX): `lfm3-vl-export` ([3.1c](#31c-lfm3-vl-vision-language-models))
- [LiquidAI/LFM3-Audio-2B-ONNX](https://huggingface.co/LiquidAI/LFM3-Audio-2B-ONNX): `lfm3-audio-export` ([3.1d](#31d-lfm3-audio))
- [LiquidAI/LFM3-Omni-2B-ONNX](https://huggingface.co/LiquidAI/LFM3-Omni-2B-ONNX): `lfm3-omni-export` ([3.1e](#31e-lfm3-omni))
- [LiquidAI/d1-2B-ONNX](https://huggingface.co/LiquidAI/d1-2B-ONNX) and [LiquidAI/d1-Omni-2B-ONNX](https://huggingface.co/LiquidAI/d1-Omni-2B-ONNX): `d1-export` ([3.1f](#31f-d1))

## 7. Acknowledgements

Special thanks to [Joshua Lochner](https://huggingface.co/Xenova) for his work on [Transformers.js](https://github.com/huggingface/transformers.js) and the [onnx-community](https://huggingface.co/onnx-community) models, which inspired and informed this project's ONNX export approach.

## 8. License

See [LICENSE](LICENSE) for details.
