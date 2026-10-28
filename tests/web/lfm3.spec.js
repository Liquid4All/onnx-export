// Runs the tiny LFM3-family graphs of export_lfm3.py on the WebGPU EP in headless Chromium and
// compares each output with the case's reference: native onnxruntime's CPU outputs where the
// export wrote them (fp16-activation graphs, q4 / q8 decoders, probes), else the wasm EP on the
// same feeds.
//
//   uv run python tests/web/export_lfm3.py   # once, writes tests/web/.lfm3
//   cd tests/web && npx playwright test lfm3

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { expect } from '@playwright/test';
import { checkWebGpuCase, test } from './checks.js';
import { startServer } from './server.js';

const dir = process.env.LIQUIDONNX_LFM3_DIR ?? path.join(path.dirname(fileURLToPath(import.meta.url)), '.lfm3');
const manifestFile = path.join(dir, 'manifest.json');
const { cases, nodes } = fs.existsSync(manifestFile)
  ? JSON.parse(fs.readFileSync(manifestFile, 'utf8'))
  : { cases: [], nodes: {} };
// graph → the nodes that read float tensors (export_lfm3.py's float_nodes), for checks.js
const floatNodes = Object.fromEntries(Object.entries(nodes).map(([model, names]) => [model, new Set(names)]));
const FP16_ACTIVATIONS = new Set(['fp16', 'q4f16', 'q8f16']);

// max|diff| <= atol + rtol * max|reference|, per graph precision. On Metal the fp32 graphs stay
// within 4% of their bound, the q4 and q8 ones (webgpu.spec.js's MatMulNBits bound) at 1e-4 on
// logits up to 47. WebGPU runs the fp16-activation graphs in fp16 where the CPU EP computes many
// ops in float32: up to 3.3% of the reference's scale (the codec), 1.9% on the decoders' logits.
// A wrong kernel gives O(1) differences.
const Q = { atol: 1e-3, rtol: 5e-2 };
const F16 = { atol: 1e-2, rtol: 5e-2 };
const TOLERANCE = { fp32: { atol: 1e-5, rtol: 1e-4 }, q4: Q, q4f32: Q, q8: Q, fp16: F16, q4f16: F16, q8f16: F16 };
// Adapters but Apple's run the q4, q4f32 and q8 decoders' MatMulNBits (accuracy_level 4) on int8
// activations (DP4A), where their references keep float ones: up to 0.06 of the scale apart on
// SwiftShader (d1's q8 hidden states on the Linux CI)
const DP4A_DECODER = { atol: 1e-3, rtol: 1e-1 };
// The matmul probes (outputs up to ~4.3) within a few fp16 ulps of the CPU EP, which accumulates
// in float32: accumulating in fp16 instead grows the error with the reduction length.
const ACCUMULATION = { atol: 1e-2, rtol: 0 };

// Cases failing on onnxruntime-web 1.30.0 (Apple silicon, Metal), with the reason: upstream
// limitations no graph change can fix without moving the CPU EP's own outputs (README 4.6).
// The suite fails when a listed case passes: remove it once it is fixed.
const FP16_MATMUL = 'WebGPU keeps the partial sums of fp16 MatMul in fp16: max|diff| from the CPU '
  + 'EP grows with K (0.044 at 2048, 0.075 at 7680, on outputs up to 3.8), where the CPU EP stays '
  + 'within an fp16 ulp of float64. No onnxruntime-web option changes it (1.31.0-dev.20260918 '
  + 'neither); float32 activations (fp32, q4) are within 1e-5';
const FP16_MATMULNBITS = 'WebGPU 1.30.0 keeps the partial sums of fp16 MatMulNBits in fp16 (0.043 '
  + 'at K=2048, 0.074 at 7680). onnxruntime-web 1.31.0-dev.20260918 with enableMatmulFp32Accumulation, '
  + 'which harness.js sets, is within one fp16 ulp (0.002): passes once the pinned release has it';
const EXPECTED_FAILURES = {
  'probe matmul_k2048 fp16 run': FP16_MATMUL,
  'probe matmul_k2048 q4f16 run': FP16_MATMULNBITS,
  'probe matmul_k7680 fp16 run': FP16_MATMUL,
  'probe matmul_k7680 q4f16 run': FP16_MATMULNBITS,
};

// What the adapter can't run, with the reason (the Linux CI's SwiftShader, and Chromium's on macOS
// with --use-webgpu-adapter=swiftshader: no shader-f16, 4-wide subgroups)
const NO_F16 = 'no shader-f16 on this adapter: the WebGPU EP runs fp16 graphs with it';
const DP4A = "on adapters but Apple's the WebGPU EP runs the decoders' MatMulNBits (accuracy_level 4) "
  + "on int8 activations (DP4A), so the probe's float-accumulation bound doesn't apply";
const SUBGROUP_FLASH = 'onnxruntime-web 1.30.0 takes its subgroup FlashAttention from 32 query '
  + 'positions, without a window or within it (Apple and NVIDIA adapters its shared-memory one), '
  + "which is wrong on SwiftShader's 4-wide subgroups: 0.2-0.6 off from S=32 (GroupQueryAttention, "
  + 'MultiHeadAttention)';

let ctx;
let adapter;
test.beforeAll(async ({ harnessPage: page }) => {
  ctx = await startServer(dir);
  await page.goto(`${ctx.baseUrl}/harness/harness.html`);
  // A cold GPU process can return no adapter at first (webgpu.spec.js); the skips below need it
  for (let i = 0; i < 5 && !adapter; i++) {
    adapter = await page.evaluate(() => window.webgpuAdapterInfo());
    if (!adapter) await page.waitForTimeout(500);
  }
  expect(adapter, 'no WebGPU adapter').toBeTruthy();
});
test.afterAll(async () => {
  ctx?.server.close();
});

if (!cases.length) {
  const missing = `no ${manifestFile}: run uv run python tests/web/export_lfm3.py`;
  // CI sets LIQUIDONNX_REQUIRE_LFM3=1: there a missing export fails instead of skipping
  if (process.env.LIQUIDONNX_REQUIRE_LFM3) test(missing, () => { throw new Error(missing); });
  else test.skip(missing, () => {});
}

for (const c of cases) {
  test(c.id, async ({ harnessPage: page }) => {
    test.fail(c.id in EXPECTED_FAILURES, EXPECTED_FAILURES[c.id]);
    test.skip(FP16_ACTIVATIONS.has(c.precision) && !adapter.shaderF16, NO_F16);
    test.skip(c.full_attention_queries >= 32 && adapter.architecture === 'swiftshader', SUBGROUP_FLASH);
    test.skip(c.graph.startsWith('matmul') && c.precision.startsWith('q4') && adapter.vendor !== 'apple', DP4A);
    const dp4aDecoder = ['model', 'decoder'].includes(c.graph)
      && ['q4', 'q4f32', 'q8'].includes(c.precision) && adapter.vendor !== 'apple';
    const tolerance = c.graph.startsWith('matmul') ? ACCUMULATION
      : dp4aDecoder ? DP4A_DECODER : TOLERANCE[c.precision];
    const url = (file) => `${ctx.baseUrl}/model/${file}`;
    await checkWebGpuCase(page, ctx.baseUrl, {
      modelUrl: url(c.model),
      externalData: c.external_data,
      feeds: { url: url(c.feeds.file), tensors: c.feeds.tensors },
      expected: c.expected && { url: url(c.expected.file), tensors: c.expected.tensors },
    }, tolerance, floatNodes[c.model]);
  });
}
