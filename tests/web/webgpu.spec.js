// Runs every synthetic LFM2.5-Audio graph on the WebGPU EP in headless Chromium and
// compares each output against the wasm (CPU) EP on identical inputs.
//
//   uv run python tests/web/export_synthetic.py   # once, writes tests/web/.synthetic
//   cd tests/web && npm test

import { checkWebGpuCase, test } from './checks.js';
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
  // onnxruntime-genai decoder layout: the conv state keeps the last L - 1 inputs
  m.layer_types.forEach((type, i) => {
    if (type === 'conv') {
      feeds.push({ name: `past.${i}.conv`, type: 'float32', dims: [1, m.hidden, m.conv_L - 1] });
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

test('report the WebGPU adapter', async ({ harnessPage: page }) => {
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
      test(`${graph} ${precision} ${variant} matches wasm on webgpu`, async ({ harnessPage: page }) => {
        const { manifest, baseUrl } = ctx;
        const entry = manifest.graphs[graph][precision];
        await checkWebGpuCase(page, baseUrl, {
          modelUrl: `${baseUrl}/model/${entry.file}`,
          externalData: entry.external_data,
          feedSpecs: feeds(manifest),
        }, TOLERANCE[precision]);
      });
    }
  }
}
