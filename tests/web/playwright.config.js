import { defineConfig } from '@playwright/test';

// SwiftShader (software Vulkan) lets WebGPU run on GPU-less Linux CI runners.
const linuxArgs = ['--enable-features=Vulkan', '--use-angle=vulkan', '--use-vulkan=swiftshader'];

export default defineConfig({
  testDir: '.',
  testMatch: '*.spec.js',
  timeout: 120_000,
  workers: 1,
  reporter: [['list']],
  use: {
    // Full Chromium (new headless) — the headless shell has no WebGPU.
    channel: 'chromium',
    headless: true,
    launchOptions: {
      args: [
        '--enable-unsafe-webgpu',
        '--ignore-gpu-blocklist',
        ...(process.platform === 'linux' ? linuxArgs : []),
      ],
    },
  },
});
