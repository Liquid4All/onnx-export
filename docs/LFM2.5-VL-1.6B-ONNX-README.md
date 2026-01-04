---
license: other
license_name: lfm1.0
license_link: LICENSE
language:
- en
- ja
- ko
- fr
- es
- de
- it
- pt
- ar
- zh
pipeline_tag: image-text-to-text
tags:
- liquid
- edge
- lfm2.5-vl
- lfm2.5
- onnx
- onnxruntime
- transformers.js
- webgpu
base_model:
- LiquidAI/LFM2.5-VL-1.6B
---

<div align="center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/61b8e2ba285851687028d395/2b08LKpev0DNEk6DlnWkY.png" alt="Liquid AI" style="width: 100%; max-width: 100%;">

  <p>
    <a href="https://playground.liquid.ai/"><strong>Try LFM</strong></a> •
    <a href="https://docs.liquid.ai/lfm"><strong>Documentation</strong></a> •
    <a href="https://leap.liquid.ai/"><strong>LEAP</strong></a> •
    <a href="https://www.liquid.ai/blog/"><strong>Blog</strong></a>
  </p>
</div>

# LFM2.5-VL-1.6B-ONNX

ONNX export of [LFM2.5-VL-1.6B](https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B) for cross-platform inference with ONNX Runtime, WebGPU, and Transformers.js.

Find more details in the original model card: https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B

## Available Variants

### WebGPU Optimized

| Variant | Encoder | Decoder | Size | Use Case |
|---------|---------|---------|------|----------|
| `q4-fp16` | Q4 | FP16 | ~1.2GB | Balanced quality, smaller encoder |
| `fp16-fp16` | FP16 | FP16 | ~1.8GB | Higher quality, faster inference |

### All Precisions

| Format | Size vs FP32 | Use Case |
|--------|--------------|----------|
| FP32 | 100% | Maximum accuracy, debugging |
| FP16 | 50% | Balanced speed/accuracy |
| Q8 | ~35% | Good compression |
| Q4 | ~25% | Edge / memory-constrained |

## Python (ONNX Runtime)

### Installation

```bash
pip install onnxruntime-gpu transformers pillow
# or for CPU only:
pip install onnxruntime transformers pillow
```

### Interactive CLI

```bash
# Basic usage
lfm2-vl-infer --model LiquidAI/LFM2.5-VL-1.6B-ONNX

# With image
lfm2-vl-infer --model LiquidAI/LFM2.5-VL-1.6B-ONNX --images photo.jpg --prompt "What is this?"

# Compare two images
lfm2-vl-infer --model LiquidAI/LFM2.5-VL-1.6B-ONNX --images a.jpg b.jpg --prompt "Compare these"
```

### Python API

```python
from transformers import AutoProcessor
import onnxruntime as ort
from PIL import Image

# Load processor and ONNX sessions
processor = AutoProcessor.from_pretrained("LiquidAI/LFM2.5-VL-1.6B-ONNX", trust_remote_code=True)
embed_tokens = ort.InferenceSession("onnx/embed_tokens.onnx")
embed_images = ort.InferenceSession("onnx/embed_images_fp16.onnx")
decoder = ort.InferenceSession("onnx/decoder_q4.onnx")

# Prepare inputs
image = Image.open("photo.jpg")
messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe this image"}]}]
```

## WebGPU (Browser)

### Setup

```bash
npm install @huggingface/transformers onnxruntime-web
```

### Usage

```javascript
import { env } from "@huggingface/transformers";
import * as ort from "onnxruntime-web/webgpu";

// Configure for WebGPU
env.backends.onnx.wasm.proxy = false;
ort.env.wasm.numThreads = 1;

// Load model with FP16 precision (recommended for WebGPU)
const modelPath = "https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B-ONNX/resolve/main/onnx";

const embedTokens = await ort.InferenceSession.create(
  `${modelPath}/embed_tokens_fp16.onnx`,
  { executionProviders: ["webgpu"] }
);

const embedImages = await ort.InferenceSession.create(
  `${modelPath}/embed_images_fp16.onnx`,
  { executionProviders: ["webgpu"] }
);

const decoder = await ort.InferenceSession.create(
  `${modelPath}/decoder_fp16.onnx`,
  { executionProviders: ["webgpu"] }
);
```

### Notes

- **Q4 decoder is not supported on WebGPU** - use FP16 or Q8 for the decoder
- Q4 encoder works on WebGPU (smaller download)
- Recommended: `q4` encoder + `fp16` decoder for balance, or `fp16`/`fp16` for quality

## Model Files

```
onnx/
├── embed_tokens.onnx          # Token embeddings (FP32)
├── embed_tokens_fp16.onnx     # Token embeddings (FP16)
├── embed_images.onnx          # Vision encoder (FP32)
├── embed_images_fp16.onnx     # Vision encoder (FP16)
├── embed_images_q4.onnx       # Vision encoder (Q4)
├── embed_images_q8.onnx       # Vision encoder (Q8)
├── decoder.onnx               # Language decoder (FP32)
├── decoder_fp16.onnx          # Language decoder (FP16)
├── decoder_q4.onnx            # Language decoder (Q4)
└── decoder_q8.onnx            # Language decoder (Q8)
```

## Recommended Configurations

| Platform | Encoder | Decoder | Notes |
|----------|---------|---------|-------|
| WebGPU | fp16 | fp16 | Best quality for browser |
| WebGPU | q4 | fp16 | Smaller download, good quality |
| Server | fp16 | q4 | ONNX Runtime (CPU/CUDA) |

## License

This model is released under the [LFM 1.0 License](LICENSE).
