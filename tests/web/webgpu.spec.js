// Runs every synthetic LFM2.5-Audio graph on the WebGPU EP in headless Chromium and
// compares each output against the wasm (CPU) EP on identical inputs.
//
//   uv run python tests/web/export_synthetic.py   # once, writes tests/web/.synthetic
//   cd tests/web && npm test

import { test, expect } from '@playwright/test';
import { startServer } from './server.js';

const SEQ = 8; // decoder prefill length
const MEL_FRAMES = 200;
const DETOK_FRAMES = 12;

// Observed max|diff| is ~1e-8 on Metal for both precisions; the bounds leave room for
// SwiftShader and for MatMulNBits kernels that dequantize in a different order.
const TOLERANCE = {
  fp32: { atol: 1e-5, rtol: 1e-4 },
  // MatMulNBits dequantizes in a different order per backend; SwiftShader (Linux CI) drifts
  // ~1e-2 relative where Metal stays at 1e-8. A wrong kernel gives O(1) diffs, not this.
  q4: { atol: 1e-3, rtol: 5e-2 },
};

const EP_SESSION_MARKER = '=== ep session ===';

// int64 shape/index plumbing has no WebGPU kernel unless enableInt64 is set, and ORT keeps
// it on the CPU anyway. Anything outside this list is compute silently leaving the GPU.
const CPU_FALLBACK_ALLOWED = new Set([
  'Add', 'Sub', 'Mul', 'Cast', 'Clip', 'Concat', 'Equal', 'Gather', 'Greater', 'LessOrEqual',
  'Range', 'ReduceSum', 'Reshape', 'Shape', 'Slice', 'Transpose', 'Unsqueeze',
]);

function depthformerFeeds(m, past) {
  const layers = 6;
  return [
    { name: 'hidden_states', type: 'float32', dims: [1, m.hidden], fill: 'randn' },
    { name: 'depth_slices_in', type: 'float32', dims: [1, 8, 1024], fill: past ? 'randn' : 'zeros' },
    { name: 'step_idx', type: 'int64', dims: [], fill: 'const', value: past },
    { name: 'prev_token', type: 'int64', dims: [1], fill: 'const', value: past ? 5 : 0 },
    { name: 'past_keys', type: 'float32', dims: [layers, 1, 8, past, 32], fill: 'randn' },
    { name: 'past_values', type: 'float32', dims: [layers, 1, 8, past, 32], fill: 'randn' },
    { name: 'seqlens_k', type: 'int32', dims: [1], fill: 'const', value: past },
    { name: 'total_seq_len', type: 'int32', dims: [], fill: 'const', value: past + 1 },
  ];
}

function decoderFeeds(m) {
  const feeds = [
    { name: 'inputs_embeds', type: 'float32', dims: [1, SEQ, m.hidden], fill: 'randn' },
    { name: 'attention_mask', type: 'int64', dims: [1, SEQ], fill: 'ones' },
  ];
  m.layer_types.forEach((type, i) => {
    if (type === 'conv') {
      feeds.push({ name: `past_conv.${i}`, type: 'float32', dims: [1, m.hidden, m.conv_L] });
    } else {
      for (const kv of ['key', 'value']) {
        feeds.push({ name: `past_key_values.${i}.${kv}`, type: 'float32', dims: [1, m.num_kv_heads, 0, m.head_dim] });
      }
    }
  });
  return feeds;
}

// graph → [variant name, feed spec builder]
const CASES = {
  audio_encoder: {
    prefill: () => [
      { name: 'mel_spectrogram', type: 'float32', dims: [1, MEL_FRAMES, 128], fill: 'randn' },
      { name: 'mel_lengths', type: 'int64', dims: [1], fill: 'const', value: MEL_FRAMES },
    ],
  },
  audio_detokenizer: {
    frames: () => [{ name: 'audio_codes', type: 'int64', dims: [1, 8, DETOK_FRAMES], fill: 'randint', high: 2048 }],
  },
  vocoder_depthformer: {
    step0: (m) => depthformerFeeds(m, 0),
    step1: (m) => depthformerFeeds(m, 1),
  },
  decoder: {
    prefill: (m) => decoderFeeds(m),
  },
};

let ctx;
test.beforeAll(async () => {
  ctx = await startServer();
});
test.afterAll(async () => {
  ctx?.server.close();
});

test('report the WebGPU adapter', async ({ page }) => {
  // Diagnostic only. A bare requestAdapter() can return null on a cold GPU process even
  // though ORT goes on to acquire a device, so this reports and never fails the suite —
  // each case asserts the EP was actually used from ORT's own node-placement log.
  await page.goto(`${ctx.baseUrl}/harness/harness.html`);
  let info = null;
  for (let i = 0; i < 5 && info === null; i++) {
    info = await page.evaluate(() => window.webgpuAdapterInfo());
    if (info === null) await page.waitForTimeout(500);
  }
  test.info().annotations.push({ type: 'adapter', description: JSON.stringify(info) });
});

for (const [graph, variants] of Object.entries(CASES)) {
  for (const precision of ['fp32', 'q4']) {
    for (const [variant, feeds] of Object.entries(variants)) {
      test(`${graph} ${precision} ${variant} matches wasm on webgpu`, async ({ page }) => {
        const { manifest, baseUrl } = ctx;
        const entry = manifest.graphs[graph][precision];
        const logs = [];
        page.on('console', (msg) => logs.push(msg.text()));

        await page.goto(`${baseUrl}/harness/harness.html`);
        const result = await page.evaluate(
          (args) => window.runCase(args),
          {
            modelUrl: `${baseUrl}/model/${entry.file}`,
            externalData: entry.external_data,
            ep: 'webgpu',
            feedSpecs: feeds(manifest),
          },
        );

        // Only the EP-under-test session's logs; the wasm reference runs first.
        const epLogs = logs.slice(logs.findIndex((l) => l.includes(EP_SESSION_MARKER)) + 1);
        const placement = epLogs.filter((l) => l.includes('VerifyEachNodeIsAssignedToAnEp'));
        test.info().annotations.push(
          ...placement.map((l) => ({ type: 'placement', description: l.split('] ').pop() })),
        );
        expect(
          placement.some((l) => /All nodes placed on \[CPUExecutionProvider\]/.test(l)),
          'WebGPU EP took no nodes — the run silently fell back to CPU',
        ).toBe(false);

        const fallbackOps = [...new Set(
          epLogs.filter((l) => /kernel not found/i.test(l)).map((l) => l.match(/Op type: (\S+)/)?.[1]),
        )];
        test.info().annotations.push(
          { type: 'run_ms', description: result.runMs.toFixed(1) },
          { type: 'cpu_fallback', description: fallbackOps.join(' ') || 'none' },
        );
        const unexpected = fallbackOps.filter((op) => !CPU_FALLBACK_ALLOWED.has(op));
        expect(unexpected, 'compute ops without a WebGPU kernel').toEqual([]);

        const tol = TOLERANCE[precision];
        for (const out of result.outputs) {
          expect(out.error, out.name).toBeUndefined();
          expect(out.nonFinite, `${out.name} non-finite`).toBe(0);
          const bound = tol.atol + tol.rtol * out.maxAbsRef;
          test.info().annotations.push({
            type: 'output',
            description: `${out.name} ${out.type}[${out.dims}] max|diff|=${out.maxAbsDiff.toExponential(2)} bound=${bound.toExponential(2)}`,
          });
          expect(out.maxAbsDiff, `${out.name} max|diff|=${out.maxAbsDiff} (bound ${bound}, ref scale ${out.maxAbsRef})`).toBeLessThanOrEqual(bound);
        }
      });
    }
  }
}
