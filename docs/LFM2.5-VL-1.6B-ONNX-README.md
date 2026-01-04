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

ONNX export of [LFM2.5-VL-1.6B](https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B) for cross-platform inference.

## Available Variants

### WebGPU (Browser)

| Variant | Encoder | Decoder | Size | Use Case |
|---------|---------|---------|------|----------|
| `q4-fp16` | Q4 | FP16 | ~1.2GB | Balanced quality, smaller download |
| `fp16-fp16` | FP16 | FP16 | ~1.8GB | Higher quality |

### Server (ONNX Runtime)

| Variant | Encoder | Decoder | Size | Use Case |
|---------|---------|---------|------|----------|
| `fp16-q4` | FP16 | Q4 | ~1.5GB | Fast inference |
| `fp16-fp16` | FP16 | FP16 | ~1.8GB | Higher quality |

## Python

### Installation

```bash
pip install onnxruntime transformers pillow numpy
# or with GPU support:
pip install onnxruntime-gpu transformers pillow numpy
```

### Inference

```python
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from transformers import AutoProcessor
from PIL import Image

# Download model files
model_id = "LiquidAI/LFM2.5-VL-1.6B-ONNX"
embed_tokens_path = hf_hub_download(model_id, "onnx/embed_tokens_fp16.onnx")
embed_images_path = hf_hub_download(model_id, "onnx/embed_images_fp16.onnx")
decoder_path = hf_hub_download(model_id, "onnx/decoder_fp16.onnx")

# Load ONNX sessions
embed_tokens = ort.InferenceSession(embed_tokens_path)
embed_images = ort.InferenceSession(embed_images_path)
decoder = ort.InferenceSession(decoder_path)

# Load processor
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

# Prepare input
image = Image.open("photo.jpg")
messages = [{"role": "user", "content": [
    {"type": "image"},
    {"type": "text", "text": "What is in this image?"}
]}]

# Process inputs
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(images=[image], text=prompt, return_tensors="np")

# Get image embeddings
image_outputs = embed_images.run(None, {
    "pixel_values": inputs["pixel_values"],
    "pixel_attention_mask": inputs["pixel_attention_mask"],
    "spatial_shapes": inputs["spatial_shapes"],
})
image_embeds = image_outputs[0]

# Get token embeddings and merge with image embeddings
input_ids = inputs["input_ids"]
token_outputs = embed_tokens.run(None, {"input_ids": input_ids})
token_embeds = token_outputs[0]

# Replace <image> tokens with image embeddings
image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")
image_positions = np.where(input_ids[0] == image_token_id)[0]
for i, pos in enumerate(image_positions):
    if i < len(image_embeds):
        token_embeds[0, pos] = image_embeds[i]

# Generate (simplified single-token example)
decoder_output = decoder.run(None, {
    "inputs_embeds": token_embeds.astype(np.float32),
    "attention_mask": np.ones((1, token_embeds.shape[1]), dtype=np.int64),
})
logits = decoder_output[0]
next_token = np.argmax(logits[0, -1])
print(processor.tokenizer.decode([next_token]))
```

## WebGPU (Browser)

### Installation

```bash
npm install onnxruntime-web
```

### Inference

```javascript
import * as ort from "onnxruntime-web/webgpu";

// Configure WebGPU
ort.env.wasm.numThreads = 1;

const modelBase = "https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B-ONNX/resolve/main/onnx";

// Load sessions (use fp16 for WebGPU)
const embedTokens = await ort.InferenceSession.create(
  `${modelBase}/embed_tokens_fp16.onnx`,
  { executionProviders: ["webgpu"] }
);

const embedImages = await ort.InferenceSession.create(
  `${modelBase}/embed_images_fp16.onnx`,
  { executionProviders: ["webgpu"] }
);

const decoder = await ort.InferenceSession.create(
  `${modelBase}/decoder_fp16.onnx`,
  { executionProviders: ["webgpu"] }
);

// Run inference
const imageEmbeds = await embedImages.run({
  pixel_values: pixelValuesTensor,
  pixel_attention_mask: attentionMaskTensor,
  spatial_shapes: spatialShapesTensor,
});

const tokenEmbeds = await embedTokens.run({
  input_ids: inputIdsTensor,
});

// Merge embeddings and run decoder...
```

### WebGPU Notes

- **Use FP16 or Q8 for decoder** - Q4 decoder is not fully supported on WebGPU
- Q4 encoder works on WebGPU (recommended for smaller download)
- Best config: `embed_images_q4.onnx` + `decoder_fp16.onnx`

## Model Files

```
onnx/
├── embed_tokens.onnx           # Token embeddings (FP32)
├── embed_tokens_fp16.onnx      # Token embeddings (FP16)
├── embed_images.onnx           # Vision encoder (FP32)
├── embed_images_fp16.onnx      # Vision encoder (FP16)
├── embed_images_q4.onnx        # Vision encoder (Q4)
├── embed_images_q8.onnx        # Vision encoder (Q8)
├── decoder.onnx                # Language decoder (FP32)
├── decoder_fp16.onnx           # Language decoder (FP16)
├── decoder_q4.onnx             # Language decoder (Q4)
└── decoder_q8.onnx             # Language decoder (Q8)
```

## License

This model is released under the [LFM 1.0 License](LICENSE).
