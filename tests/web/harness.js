// Browser side of the WebGPU tests. Playwright calls window.runCase() and reads the result.
//
// Feeds are built here from a spec with a seeded PRNG, or read from export_lfm3.py's blobs, so
// the reference run and the WebGPU run see identical inputs. ort is the UMD global from
// ort.webgpu.min.js.

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

// float16 data is the bit patterns (ORT takes a Uint16Array, and returns one without Float16Array)
const TYPED_ARRAYS = {
  float32: Float32Array, float16: Uint16Array, int64: BigInt64Array, int32: Int32Array, bool: Uint8Array,
};

// Tensors of export_lfm3.py: one little-endian blob and its {name, type, dims, offset, bytes}
async function loadTensors({ url, tensors }) {
  const buffer = await (await fetch(url)).arrayBuffer();
  const out = {};
  for (const t of tensors) {
    const data = new TYPED_ARRAYS[t.type](buffer.slice(t.offset, t.offset + t.bytes));
    out[t.name] = new ort.Tensor(t.type, data, t.dims);
  }
  return out;
}

function halfToFloat(h) {
  const sign = h & 0x8000 ? -1 : 1;
  const exponent = (h >> 10) & 0x1f;
  const fraction = h & 0x3ff;
  if (exponent === 0) return sign * fraction * 2 ** -24;
  if (exponent === 31) return fraction ? NaN : sign * Infinity;
  return sign * (1 + fraction / 1024) * 2 ** (exponent - 15);
}

function values(tensor) {
  return tensor.type === 'float16' && tensor.data instanceof Uint16Array
    ? Array.from(tensor.data, halfToFloat)
    : tensor.data;
}

async function createSession(modelUrl, externalData, ep, extra) {
  const options = {
    // float32 partial sums for MatMulNBits with fp16 activations (onnxruntime-web 1.31+; earlier
    // releases ignore the option)
    executionProviders: [ep === 'webgpu' ? { name: ep, enableMatmulFp32Accumulation: true } : ep],
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
  const a = values(ref);
  const b = values(out);
  for (let i = 0; i < a.length; i++) {
    const x = Number(a[i]);
    const y = Number(b[i]);
    if (!Number.isFinite(y)) nonFinite++;
    maxAbsDiff = Math.max(maxAbsDiff, Math.abs(x - y));
    maxAbsRef = Math.max(maxAbsRef, Math.abs(x));
  }
  return { name, type: ref.type, dims: ref.dims, size: a.length, maxAbsDiff, maxAbsRef, nonFinite };
}

// Runs the graph on `ep` and returns per-output comparison stats against the reference: the
// `expected` tensors when given, else the wasm EP on the same feeds. Feeds come from a blob
// (`feeds`) or are built from `feedSpecs` with a seeded PRNG.
window.runCase = async ({ modelUrl, externalData = [], ep = 'webgpu', feedSpecs, feeds, expected, seed = 0, extra = {} }) => {
  ort.env.wasm.wasmPaths = '/ort/';
  ort.env.wasm.numThreads = 1;
  ort.env.logLevel = 'verbose';

  const status = document.getElementById('status');
  status.textContent = `loading ${modelUrl} on ${ep}`;
  const makeFeeds = () => (feeds ? loadTensors(feeds) : buildFeeds(feedSpecs, seed));

  let refOut = expected && (await loadTensors(expected));
  if (!refOut) {
    const reference = await createSession(modelUrl, externalData, 'wasm', {});
    refOut = await reference.run(await makeFeeds());
    await reference.release();
  }

  // Everything logged after this marker belongs to the EP-under-test session.
  console.log(window.EP_SESSION_MARKER);
  let t0 = performance.now();
  const session = await createSession(modelUrl, externalData, ep, extra);
  const createMs = performance.now() - t0;
  const input = await makeFeeds();
  t0 = performance.now();
  const out = await session.run(input);
  const runMs = performance.now() - t0;
  // The first run also compiles the shaders
  t0 = performance.now();
  await session.run(input);
  const warmMs = performance.now() - t0;

  const outputs = Object.keys(refOut).map((name) => compare(name, refOut[name], out[name]));
  await session.release();
  status.textContent = 'done';
  return { outputs, createMs, runMs, warmMs, inputNames: session.inputNames, outputNames: session.outputNames };
};

window.webgpuAdapterInfo = async () => {
  if (!navigator.gpu) return null;
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) return null;
  const info = adapter.info ?? {};
  return {
    vendor: info.vendor,
    architecture: info.architecture,
    device: info.device,
    description: info.description,
    // the WebGPU EP's fp16 kernels need it
    shaderF16: adapter.features.has('shader-f16'),
    subgroupMaxSize: info.subgroupMaxSize,
  };
};
