# WebGPU tests

Runs the LFM2.5-Audio ONNX graphs (`webgpu.spec.js`) and the LFM3 families' graphs
(`lfm3.spec.js`) on the WebGPU execution provider in headless Chromium (via Playwright +
onnxruntime-web) and compares every output against a CPU reference on identical inputs. The
exports come from tiny random checkpoints, so no download is needed.

```bash
uv run python tests/web/export_synthetic.py   # writes tests/web/.synthetic (fp32 + q4)
uv run python tests/web/export_lfm3.py        # writes tests/web/.lfm3
cd tests/web
npm ci
npx playwright install --with-deps chromium   # first time only
npm test                                      # both specs; npx playwright test lfm3 for one
```

Each case asserts:

- the graph loads and runs on `webgpu`
- no compute op runs on the CPU EP: ORT's node placement log (which must be there) may put only
  int64 / bool shape and index plumbing there (`CPU_FALLBACK_ALLOWED` in `checks.js`, each addition
  justified), and, for the LFM3 graphs, also their own plumbing ops (`LFM3_PLUMBING`) but no node
  that reads a float tensor whatever its op (`export_lfm3.py` types each graph's nodes by ONNX type
  inference: `float_nodes`, the manifest's `nodes`)
- every output matches the reference within its precision's tolerance

On Linux CI Chromium uses SwiftShader (software Vulkan); see `playwright.config.js`. CI
(`.github/workflows/test_web.yml`) writes both exports and runs both specs with
`LIQUIDONNX_REQUIRE_LFM3=1`, with which `lfm3.spec.js` fails without its export instead of skipping.
Set `LIQUIDONNX_SYNTHETIC_DIR` / `LIQUIDONNX_LFM3_DIR` to point the server at another export.

SwiftShader has no `shader-f16` and 4-wide subgroups, so `lfm3.spec.js` skips there, with the
reason, the fp16-activation cases and those with an attention without a window over 32 or more
query positions: onnxruntime-web 1.30.0's subgroup FlashAttention, which every adapter but Apple's
and NVIDIA's takes from 32 positions without a window or within one, is 0.2 to 0.6 off on it
(single GroupQueryAttention and MultiHeadAttention nodes, S = 32 to 96, no window or a window of
256; S ≤ 31, and a window of 8 that the positions pass, are exact). No tiny case runs 32 positions
within a window. The `prefill28` and `grid4x6` cases ([LFM3 cases](#lfm3-cases)) give it a prefill
of every decoder and the vision encoders.

On every adapter but Apple's, `MatMulNBits` at accuracy_level 4 (the decoders') quantizes the
activations to int8 (DP4A), as the CPU EP does, so the int4 matmul probes skip there too, and the
tiny q4, q4f32 and q8 decoders, whose references keep float activations, are held there to a tenth
of the reference's scale (`DP4A_DECODER`); SwiftShader puts them up to 0.06 of it apart. On Metal
they stay within 4e-5 of their bound. Chromium's SwiftShader adapter also runs on macOS
(`--use-webgpu-adapter=swiftshader --enable-unsafe-swiftshader`, which `LIQUIDONNX_SWIFTSHADER=1`
adds): there 51 LFM3 cases and the 11 LFM2.5-Audio tests pass and 67 LFM3 cases skip
(onnxruntime-web 1.30.0).

## LFM3 cases

`export_lfm3.py` exports every family in every precision it ships, from the tiny checkpoints of
`tests/test_lfm3_vl/tiny.py` and `tests/test_lfm3_audio/tiny.py` (LFM3-Omni's format and d1's keys
put on top): `lfm3` (text, with q4f32), `lfm3_vl`, `lfm3_audio`, `lfm3_omni`, `d1`, `d1_omni`.
Each graph has the op types and quantization attributes of the real export's file of the same name.
The decoders run a prefill (several image / video / audio spans between text, longer than the
sliding window of 8; d1's set attention over three questions after a 21-token state, 36 tokens in
all, where a set mask built from bool `And`s on the GPU comes out wrong) and a cached decode step
past three windows; the encoders, depth decoder and codec one representative input. Two probes
isolate what the tiny graphs can't show: `gqa_window` (GroupQueryAttention's local_window_size) and
`matmul_k{2048,7680}` (fp16 accumulation at the real decoders' widths). For SwiftShader, the
prefills over 31 positions come again in 28 (`prefill28`; d1's after a 13-token state, the same
mask case) and the vision encoders' 8x12 patch grid in 4x6 (`grid4x6`), in the precisions without
fp16 activations.

References: the wasm EP for the fp32 graphs, q4f32 and the q8 vision encoder; native onnxruntime's
CPU EP, written by the export, for the fp16-activation graphs, the q4 and q8 decoders (the wasm EP
has no `GatherBlockQuantized` kernel, so it can't load them) and the probes. The native reference
runs `MatMulNBits` at accuracy_level 0: WebGPU computes float activations on Apple GPUs, while the
CPU EP at the exported level 4 quantizes them to int8, which moves the tiny decoders' logits by up
to 4.7 of 47.

Expected failures (`EXPECTED_FAILURES` in `lfm3.spec.js`) carry their reason; the suite fails when
one passes, so a fix removes its entry. On onnxruntime-web 1.30.0 on Apple silicon (Metal), with
both specs (118 LFM3 cases, 11 LFM2.5-Audio tests), the expected failures are the fp16 and q4f16
matmul probes, upstream limitations (README, [4.6](../../README.md#46-lfm3-families-on-webgpu)):
WebGPU keeps fp16 partial sums in fp16. `harness.js` sets `enableMatmulFp32Accumulation`, which
1.30.0 ignores; with onnxruntime-web 1.31.0-dev.20260918 it brings the q4f16 probes (MatMulNBits)
within an fp16 ulp, so they pass once the pinned release has it, and leaves fp16 MatMul as it is.

Headless Chromium 1243 in an ssh session on macOS dies when a page closes, so `checks.js` gives
each worker one page that every test reuses.
