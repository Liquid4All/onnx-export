// Browser side of the WebGPU tests. Playwright calls window.runCase() and reads the result.
//
// Feeds are built here from a spec with a seeded PRNG so the wasm reference run and the
// WebGPU run see identical inputs. ort is the UMD global from ort.webgpu.min.js.

window.EP_SESSION_MARKER = '=== ep session ===';

function mulberry32(seed) {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randn(rand) {
  // Box-Muller
  const u = 1 - rand();
  const v = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function makeTensor(spec, rand) {
  const size = spec.dims.reduce((a, b) => a * b, 1);
  const fill = spec.fill ?? 'zeros';
  let data;
  switch (spec.type) {
    case 'float32': {
      data = new Float32Array(size);
      if (fill === 'randn') for (let i = 0; i < size; i++) data[i] = randn(rand) * (spec.scale ?? 1);
      else if (fill === 'ones') data.fill(1);
      break;
    }
    case 'int64': {
      data = new BigInt64Array(size);
      if (fill === 'ones') data.fill(1n);
      else if (fill === 'const') data.fill(BigInt(spec.value));
      else if (fill === 'randint') for (let i = 0; i < size; i++) data[i] = BigInt(Math.floor(rand() * spec.high));
      break;
    }
    case 'int32': {
      data = new Int32Array(size);
      if (fill === 'const') data.fill(spec.value);
      break;
    }
    default:
      throw new Error(`unsupported feed type ${spec.type}`);
  }
  return new ort.Tensor(spec.type, data, spec.dims);
}

function buildFeeds(feedSpecs, seed) {
  const rand = mulberry32(seed);
  const feeds = {};
  for (const spec of feedSpecs) feeds[spec.name] = makeTensor(spec, rand);
  return feeds;
}

async function createSession(modelUrl, externalData, ep, extra) {
  const options = {
    executionProviders: [ep],
    graphOptimizationLevel: 'all',
    logSeverityLevel: 0,
    logVerbosityLevel: 0,
    externalData: externalData.map((path) => ({ data: modelUrl.replace(/[^/]+$/, path), path })),
    extra,
  };
  return ort.InferenceSession.create(modelUrl, options);
}

function compare(name, ref, out) {
  if (ref.type !== out.type) return { name, error: `dtype ${ref.type} vs ${out.type}` };
  if (ref.dims.join('x') !== out.dims.join('x')) return { name, error: `dims ${ref.dims} vs ${out.dims}` };
  let maxAbsDiff = 0;
  let maxAbsRef = 0;
  let nonFinite = 0;
  const a = ref.data;
  const b = out.data;
  for (let i = 0; i < a.length; i++) {
    const x = Number(a[i]);
    const y = Number(b[i]);
    if (!Number.isFinite(y)) nonFinite++;
    maxAbsDiff = Math.max(maxAbsDiff, Math.abs(x - y));
    maxAbsRef = Math.max(maxAbsRef, Math.abs(x));
  }
  return { name, type: ref.type, dims: ref.dims, size: a.length, maxAbsDiff, maxAbsRef, nonFinite };
}

// Runs the graph on wasm (reference) and on `ep`, returns per-output comparison stats.
window.runCase = async ({ modelUrl, externalData = [], ep = 'webgpu', feedSpecs, seed = 0, extra = {} }) => {
  ort.env.wasm.wasmPaths = '/ort/';
  ort.env.wasm.numThreads = 1;
  ort.env.logLevel = 'verbose';

  const status = document.getElementById('status');
  status.textContent = `loading ${modelUrl} on ${ep}`;

  const reference = await createSession(modelUrl, externalData, 'wasm', {});
  const refOut = await reference.run(buildFeeds(feedSpecs, seed));

  // Everything logged after this marker belongs to the EP-under-test session.
  console.log(window.EP_SESSION_MARKER);
  const session = await createSession(modelUrl, externalData, ep, extra);
  const t0 = performance.now();
  const out = await session.run(buildFeeds(feedSpecs, seed));
  const runMs = performance.now() - t0;

  const outputs = Object.keys(refOut).map((name) => compare(name, refOut[name], out[name]));
  await session.release();
  await reference.release();
  status.textContent = 'done';
  return { outputs, runMs, inputNames: session.inputNames, outputNames: session.outputNames };
};

window.webgpuAdapterInfo = async () => {
  if (!navigator.gpu) return null;
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) return null;
  const info = adapter.info ?? {};
  return { vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description };
};
