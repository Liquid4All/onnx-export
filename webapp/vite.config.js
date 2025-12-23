import { defineConfig } from 'vite';
import path from 'path';
import { fileURLToPath } from 'url';
import fs from 'fs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const modelsDir = path.resolve(__dirname, '..');

export default defineConfig({
  publicDir: false,

  server: {
    port: 3000,
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
    fs: {
      allow: ['..'],
      strict: false,
    },
  },

  preview: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },

  plugins: [
    {
      name: 'serve-models',
      configureServer(server) {
        server.middlewares.use('/models', (req, res, next) => {
          const filePath = path.join(modelsDir, req.url);
          console.log(`[serve-models] ${req.url} -> ${filePath}`);

          if (fs.existsSync(filePath) && fs.statSync(filePath).isFile()) {
            const stat = fs.statSync(filePath);
            const ext = path.extname(filePath).toLowerCase();

            const contentTypes = {
              '.json': 'application/json',
              '.onnx': 'application/octet-stream',
              '.onnx_data': 'application/octet-stream',
              '.bin': 'application/octet-stream',
              '.txt': 'text/plain',
              '.jinja': 'text/plain',
            };

            res.setHeader('Content-Type', contentTypes[ext] || 'application/octet-stream');
            res.setHeader('Content-Length', stat.size);
            res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
            res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');

            console.log(`[serve-models] Serving ${filePath} (${stat.size} bytes)`);
            const stream = fs.createReadStream(filePath);
            stream.pipe(res);
          } else {
            console.log(`[serve-models] NOT FOUND: ${filePath}`);
            next();
          }
        });
      },
    },
  ],

  optimizeDeps: {
    exclude: ['@huggingface/transformers', 'onnxruntime-web'],
  },

  build: {
    target: 'esnext',
  },
});
