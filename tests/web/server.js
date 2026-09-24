// Minimal static server: /ort/ → onnxruntime-web dist, /harness/ → this dir, /model/ → synthetic onnx dir.
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));

const MIME = {
  '.html': 'text/html',
  '.js': 'text/javascript',
  '.mjs': 'text/javascript',
  '.wasm': 'application/wasm',
  '.json': 'application/json',
};

export function syntheticDir() {
  return process.env.LIQUIDONNX_SYNTHETIC_DIR ?? path.join(here, '.synthetic');
}

export function loadManifest() {
  const file = path.join(syntheticDir(), 'manifest.json');
  if (!fs.existsSync(file)) {
    throw new Error(`${file} not found — run: uv run python tests/web/export_synthetic.py`);
  }
  return JSON.parse(fs.readFileSync(file, 'utf8'));
}

export function startServer() {
  const manifest = loadManifest();
  const roots = {
    '/ort/': path.join(here, 'node_modules', 'onnxruntime-web', 'dist'),
    '/harness/': here,
    '/model/': path.join(syntheticDir(), manifest.model_dir, 'onnx'),
  };

  const server = http.createServer((req, res) => {
    const url = new URL(req.url, 'http://localhost');
    const prefix = Object.keys(roots).find((p) => url.pathname.startsWith(p));
    if (!prefix) return res.writeHead(404).end();
    const file = path.join(roots[prefix], url.pathname.slice(prefix.length));
    if (!file.startsWith(roots[prefix]) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) {
      return res.writeHead(404).end();
    }
    res.writeHead(200, {
      'Content-Type': MIME[path.extname(file)] ?? 'application/octet-stream',
      'Content-Length': fs.statSync(file).size,
    });
    fs.createReadStream(file).pipe(res);
  });

  return new Promise((resolve) => {
    server.listen(0, '127.0.0.1', () => {
      const { port } = server.address();
      resolve({ server, manifest, baseUrl: `http://127.0.0.1:${port}` });
    });
  });
}
