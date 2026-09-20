# WebGPU tests

Runs the LFM2.5-Audio ONNX graphs on the WebGPU execution provider in headless Chromium
(via Playwright + onnxruntime-web) and compares every output against the wasm (CPU) EP on
identical inputs. Uses the synthetic export from `tests/test_lfm2_audio/synthetic.py`, so
no checkpoint download is needed.

```bash
uv run python tests/web/export_synthetic.py   # writes tests/web/.synthetic (fp32 + q4)
cd tests/web
npm ci
npx playwright install --with-deps chromium   # first time only
npm test
```

Each case asserts:

- the graph loads and runs on `webgpu` (fp32 and q4 / `MatMulNBits`)
- no compute op falls back to the CPU EP — only int64 shape/index plumbing may
  (`CPU_FALLBACK_ALLOWED` in `webgpu.spec.js`)
- outputs match wasm within `TOLERANCE` (observed max|diff| is ~1e-8 on Metal)

On Linux CI Chromium uses SwiftShader (software Vulkan); see `playwright.config.js`.
Set `LIQUIDONNX_SYNTHETIC_DIR` to point the server at a different export.
