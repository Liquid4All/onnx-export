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
| `fp16-q4` | FP16 | Q4 | ~1.5GB | Recommended |
| `fp16-fp16` | FP16 | FP16 | ~1.8GB | Higher quality |

### Server (ONNX Runtime)

| Variant | Encoder | Decoder | Size | Use Case |
|---------|---------|---------|------|----------|
| `fp16-q4` | FP16 | Q4 | ~1.5GB | Recommended |
| `fp16-fp16` | FP16 | FP16 | ~1.8GB | Higher quality |

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

## Python

### Installation

```bash
pip install onnxruntime transformers pillow torch huggingface_hub
# or with GPU support:
pip install onnxruntime-gpu transformers pillow torch huggingface_hub
```

### Inference

```python
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from transformers import AutoProcessor
from PIL import Image

# Download model files (fp16 encoder + q4 decoder recommended)
model_id = "LiquidAI/LFM2.5-VL-1.6B-ONNX"
embed_tokens_path = hf_hub_download(model_id, "onnx/embed_tokens_fp16.onnx")
embed_images_path = hf_hub_download(model_id, "onnx/embed_images_fp16.onnx")
decoder_path = hf_hub_download(model_id, "onnx/decoder_q4.onnx")

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
inputs = processor(images=[image], text=prompt, return_tensors="pt")

# Convert to numpy with correct dtypes
pixel_values = inputs["pixel_values"].numpy().astype(np.float32)
pixel_attention_mask = inputs["pixel_attention_mask"].numpy().astype(np.int64)
spatial_shapes = inputs["spatial_shapes"].numpy().astype(np.int64)
input_ids = inputs["input_ids"].numpy().astype(np.int64)

# Get image embeddings
image_outputs = embed_images.run(None, {
    "pixel_values": pixel_values,
    "pixel_attention_mask": pixel_attention_mask,
    "spatial_shapes": spatial_shapes,
})
image_embeds = image_outputs[0]

# Get token embeddings
token_outputs = embed_tokens.run(None, {"input_ids": input_ids})
token_embeds = token_outputs[0]

# Replace <image> tokens with image embeddings
image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")
image_positions = np.where(input_ids[0] == image_token_id)[0]
for i, pos in enumerate(image_positions):
    if i < len(image_embeds):
        token_embeds[0, pos] = image_embeds[i]

# Initialize KV cache for stateful decoding
ONNX_DTYPE = {"tensor(float)": np.float32, "tensor(float16)": np.float16, "tensor(int64)": np.int64}
cache = {}
for inp in decoder.get_inputs():
    if inp.name in {"inputs_embeds", "attention_mask", "position_ids"}:
        continue
    shape = [d if isinstance(d, int) else 1 for d in inp.shape]
    for i, d in enumerate(inp.shape):
        if isinstance(d, str) and "sequence" in d.lower():
            shape[i] = 0
    cache[inp.name] = np.zeros(shape, dtype=ONNX_DTYPE.get(inp.type, np.float32))

# Generate tokens
seq_len = token_embeds.shape[1]
generated_tokens = []

for step in range(100):  # max tokens
    if step == 0:
        embeds = token_embeds.astype(np.float32)
    else:
        last_token = np.array([[generated_tokens[-1]]], dtype=np.int64)
        embeds = embed_tokens.run(None, {"input_ids": last_token})[0].astype(np.float32)

    attn_mask = np.ones((1, seq_len + len(generated_tokens)), dtype=np.int64)
    feed = {"inputs_embeds": embeds, "attention_mask": attn_mask, **cache}

    outputs = decoder.run(None, feed)
    next_token = int(np.argmax(outputs[0][0, -1]))
    generated_tokens.append(next_token)

    # Update cache
    for i, out in enumerate(decoder.get_outputs()[1:], 1):
        name = out.name.replace("present_conv", "past_conv").replace("present.", "past_key_values.")
        if name in cache:
            cache[name] = outputs[i]

    if next_token == processor.tokenizer.eos_token_id:
        break

print(processor.tokenizer.decode(generated_tokens, skip_special_tokens=True))
```

## WebGPU (Browser)

### Installation

```bash
npm install onnxruntime-web @huggingface/transformers
```

### Inference

```javascript
import * as ort from "onnxruntime-web/webgpu";
import { AutoTokenizer } from "@huggingface/transformers";

ort.env.wasm.numThreads = 1;

const modelId = "LiquidAI/LFM2.5-VL-1.6B-ONNX";
const modelBase = `https://huggingface.co/${modelId}/resolve/main`;

// Load tokenizer
const tokenizer = await AutoTokenizer.from_pretrained(modelId);

// Load ONNX sessions with external data
async function loadSession(name) {
  const onnxPath = `${modelBase}/onnx/${name}.onnx`;
  const dataPath = `${modelBase}/onnx/${name}.onnx_data`;
  return ort.InferenceSession.create(onnxPath, {
    executionProviders: ["webgpu"],
    externalData: [{ path: `${name}.onnx_data`, data: dataPath }],
  });
}

const embedTokens = await loadSession("embed_tokens_fp16");
const embedImages = await loadSession("embed_images_fp16");
const decoder = await loadSession("decoder_q4");

// Get image embeddings (requires preprocessing - see notes below)
const imageOutputs = await embedImages.run({
  pixel_values: new ort.Tensor("float32", pixelValues, [numTiles, 1024, 768]),
  pixel_attention_mask: new ort.Tensor("int64", new BigInt64Array(mask), [numTiles, 1024]),
  spatial_shapes: new ort.Tensor("int64", new BigInt64Array(shapes), [numTiles, 2]),
});

// Get token embeddings
const inputIds = tokenizer.encode(prompt);
const tokenOutputs = await embedTokens.run({
  input_ids: new ort.Tensor("int64", new BigInt64Array(inputIds.map(BigInt)), [1, inputIds.length]),
});

// Merge image embeddings into token embeddings at <image> positions
// Then run decoder with KV cache for generation...
```

### WebGPU Notes

- Recommended: `embed_images_fp16.onnx` + `decoder_q4.onnx`
- For higher quality: `embed_images_fp16.onnx` + `decoder_fp16.onnx`
- Image preprocessing requires tiling (512×512), patch extraction (16×16), and normalization
- Models use external data files (`.onnx_data`) that are loaded automatically
- int64 tensors require `BigInt64Array`

## License

This model is released under the [LFM 1.0 License](LICENSE).
