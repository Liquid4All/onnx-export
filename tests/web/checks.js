// Shared by the WebGPU specs: runs one case in the harness page on the webgpu EP, annotates the
// test with ORT's node placement, timings and each output's max|diff|, then checks them (soft, so
// a failing case still reports every number).

import { test as base, expect } from '@playwright/test';

// One page per worker, never closed: closing a page kills Chromium 1243 (SIGSEGV in the browser
// process) when it runs headless in an ssh session on macOS, and the next test then fails.
export const test = base.extend({
  harnessPage: [async ({ browser }, use) => {
    const context = await browser.newContext();
    await use(await context.newPage());
  }, { scope: 'worker' }],
});

const EP_SESSION_MARKER = '=== ep session ===';

// int64 shape/index plumbing has no WebGPU kernel unless enableInt64 is set, and ORT keeps
// it on the CPU anyway. Anything outside this list is compute silently leaving the GPU.
export const CPU_FALLBACK_ALLOWED = new Set([
  'Add', 'Sub', 'Mul', 'Cast', 'Clip', 'Concat', 'Equal', 'Gather', 'Greater', 'LessOrEqual',
  'Range', 'ReduceSum', 'Reshape', 'Shape', 'Slice', 'Transpose', 'Unsqueeze',
]);

// The LFM3 graphs' int64 / bool plumbing, which no float activation reaches: Neg of
// logits_to_keep; NonZero of the placeholder-token mask, the rows ScatterND writes the features to
// (on the GPU); Squeeze and Div of grid sizes and token counts; Div and Mod of the codec's FSQ
// digits of its int64 codes (whose Cast to float, [B, F, 12, 4], also runs there); ConstantOfShape
// and Expand of in-graph attention masks and position ids; Or of d1's bool set mask, from the
// int64 question ids (built on the GPU instead, from int32 ids with And, 1.30.0 gets it wrong:
// README 4.6). Allowed only with the case's floatNodes (lfm3.spec.js): without them a float Mod or
// NonZero on the CPU EP would pass.
const LFM3_PLUMBING = new Set([
  'Neg', 'NonZero', 'Squeeze', 'Div', 'Mod', 'ConstantOfShape', 'Expand', 'Or',
]);

// floatNodes (export_lfm3.py's float_nodes): the graph's nodes that read a float tensor, which
// never may run on the CPU EP, whatever their op
function allowedOnCpu(node, floatNodes) {
  if (!floatNodes) return CPU_FALLBACK_ALLOWED.has(node.op);
  return (CPU_FALLBACK_ALLOWED.has(node.op) || LFM3_PLUMBING.has(node.op)) && !floatNodes.has(node.name);
}

// The nodes ORT's placement log (VerifyEachNodeIsAssignedToAnEp, verbose) puts on the CPU EP
function cpuNodes(placement) {
  const nodes = [];
  let ep = null;
  for (const line of placement) {
    const header = line.match(/placed on \[(\w+)\]/);
    const node = line.match(/^\s+(\S+) \((.*)\)$/);
    if (header) ep = header[1];
    else if (node && ep === 'CPUExecutionProvider') nodes.push({ op: node[1], name: node[2] });
  }
  return nodes;
}

export async function checkWebGpuCase(page, baseUrl, args, tolerance, floatNodes = null) {
  const logs = [];
  const log = (msg) => logs.push(msg.text());
  page.on('console', log);
  let result;
  try {
    await page.goto(`${baseUrl}/harness/harness.html`);
    result = await page.evaluate((a) => window.runCase(a), { ep: 'webgpu', ...args });
  } finally {
    page.off('console', log);
  }

  // Only the EP-under-test session's logs; a wasm reference runs first.
  const epLogs = logs.slice(logs.findIndex((l) => l.includes(EP_SESSION_MARKER)) + 1);
  const placement = epLogs.filter((l) => l.includes('VerifyEachNodeIsAssignedToAnEp')).map((l) => l.split('] ').pop());
  const onCpu = cpuNodes(placement);
  const unexpected = onCpu.filter((n) => !allowedOnCpu(n, floatNodes)).map((n) => `${n.op}(${n.name})`);
  const noKernel = new Set(epLogs.map((l) => l.match(/kernel not found in registries for Op type: (\S+) node name: (.*)$/))
    .filter(Boolean).map((m) => `${m[1]}(${m[2]})`));
  const annotations = test.info().annotations;
  annotations.push(
    { type: 'cpu_nodes', description: onCpu.map((n) => `${n.op}(${n.name})`).join(' ') || 'none' },
    { type: 'cpu_fallback', description: unexpected.join(' ') || 'none' },
    { type: 'no_webgpu_kernel', description: [...noKernel].join(' ') || 'none' },
    { type: 'create_ms', description: result.createMs.toFixed(1) },
    { type: 'run_ms', description: result.runMs.toFixed(1) },
    { type: 'warm_run_ms', description: result.warmMs.toFixed(1) },
  );
  const bounds = result.outputs.map((out) => tolerance.atol + tolerance.rtol * out.maxAbsRef);
  result.outputs.forEach((out, i) => annotations.push({
    type: 'output',
    description: out.error
      ? `${out.name} ${out.error}`
      : `${out.name} ${out.type}[${out.dims}] max|diff|=${out.maxAbsDiff.toExponential(2)} bound=${bounds[i].toExponential(2)} nonfinite=${out.nonFinite}`,
  }));

  expect.soft(placement.length, 'no node placement in the ORT log: the CPU EP checks would pass vacuously')
    .toBeGreaterThan(0);
  expect.soft(
    placement.some((l) => /All nodes placed on \[CPUExecutionProvider\]/.test(l)),
    'WebGPU EP took no nodes — the run silently fell back to CPU',
  ).toBe(false);
  expect.soft(unexpected, 'compute nodes on the CPU EP').toEqual([]);
  result.outputs.forEach((out, i) => {
    expect.soft(out.error, out.name).toBeUndefined();
    if (out.error) return;
    expect.soft(out.nonFinite, `${out.name} non-finite`).toBe(0);
    expect.soft(out.maxAbsDiff, `${out.name} max|diff|=${out.maxAbsDiff} (bound ${bounds[i]}, ref scale ${out.maxAbsRef})`).toBeLessThanOrEqual(bounds[i]);
  });
}
