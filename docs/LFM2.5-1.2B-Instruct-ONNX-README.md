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
pipeline_tag: text-generation
tags:
- liquid
- edge
- lfm2.5
- onnx
- onnxruntime
- webgpu
base_model:
- LiquidAI/LFM2.5-1.2B-Instruct
---

<div align="center">
  <img
    src="https://cdn-uploads.huggingface.co/production/uploads/61b8e2ba285851687028d395/2b08LKpev0DNEk6DlnWkY.png"
    alt="Liquid AI"
    style="width: 100%; max-width: 100%; height: auto; display: inline-block; margin-bottom: 0.5em; margin-top: 0.5em;"
  />
  <div style="display: flex; justify-content: center; gap: 0.5em; margin-bottom: 1em;">
    <a href="https://playground.liquid.ai/"><strong>Try LFM</strong></a> •
    <a href="https://docs.liquid.ai/lfm"><strong>Documentation</strong></a> •
    <a href="https://leap.liquid.ai/"><strong>LEAP</strong></a>
  </div>
</div>

# LFM2.5-1.2B-Instruct-ONNX

ONNX export of [LFM2.5-1.2B-Instruct](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct) for cross-platform inference.

LFM2.5 is a hybrid architecture combining multiplicative gates and short convolutions, optimized for edge deployment with fast inference on CPU, GPU, and NPU hardware.

## Recommended Variants

| Precision | Size | Platform | Use Case |
|-----------|------|----------|----------|
| Q4 | ~1.2GB | WebGPU, Server | Recommended for most uses |
| FP16 | ~2.4GB | WebGPU, Server | Higher quality |
| Q8 | ~1.7GB | Server only | Balance of quality and size |

- **WebGPU**: Use Q4 or FP16 (Q8 not supported)
- **Server**: All variants supported

## Model Files

```
onnx/
├── model.onnx              # FP32
├── model_fp16.onnx         # FP16
├── model_q4.onnx           # Q4 (recommended)
└── model_q8.onnx           # Q8
```

## Python

### Installation

```bash
pip install onnxruntime transformers numpy huggingface_hub
# or with GPU support:
pip install onnxruntime-gpu transformers numpy huggingface_hub
```

### Inference

```python
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

# Download model (Q4 recommended)
model_id = "LiquidAI/LFM2.5-1.2B-Instruct-ONNX"
model_path = hf_hub_download(model_id, "onnx/model_q4.onnx")
data_path = hf_hub_download(model_id, "onnx/model_q4.onnx_data")

# Load model and tokenizer
session = ort.InferenceSession(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# Prepare chat input
messages = [{"role": "user", "content": "What is the capital of France?"}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
input_ids = np.array([tokenizer.encode(prompt, add_special_tokens=False)], dtype=np.int64)

# Initialize KV cache
ONNX_DTYPE = {"tensor(float)": np.float32, "tensor(float16)": np.float16, "tensor(int64)": np.int64}
cache = {}
for inp in session.get_inputs():
    if inp.name in {"input_ids", "attention_mask", "position_ids"}:
        continue
    shape = [d if isinstance(d, int) else 1 for d in inp.shape]
    for i, d in enumerate(inp.shape):
        if isinstance(d, str) and "sequence" in d.lower():
            shape[i] = 0
    cache[inp.name] = np.zeros(shape, dtype=ONNX_DTYPE.get(inp.type, np.float32))

# Check if model uses position_ids
input_names = {inp.name for inp in session.get_inputs()}
use_position_ids = "position_ids" in input_names

# Generate tokens
seq_len = input_ids.shape[1]
generated_tokens = []

for step in range(100):  # max tokens
    if step == 0:
        ids = input_ids
        pos = np.arange(seq_len, dtype=np.int64).reshape(1, -1)
    else:
        ids = np.array([[generated_tokens[-1]]], dtype=np.int64)
        pos = np.array([[seq_len + len(generated_tokens) - 1]], dtype=np.int64)

    attn_mask = np.ones((1, seq_len + len(generated_tokens)), dtype=np.int64)
    feed = {"input_ids": ids, "attention_mask": attn_mask, **cache}
    if use_position_ids:
        feed["position_ids"] = pos

    outputs = session.run(None, feed)
    next_token = int(np.argmax(outputs[0][0, -1]))
    generated_tokens.append(next_token)

    # Update cache
    for i, out in enumerate(session.get_outputs()[1:], 1):
        name = out.name.replace("present_conv", "past_conv").replace("present.", "past_key_values.")
        if name in cache:
            cache[name] = outputs[i]

    if next_token == tokenizer.eos_token_id:
        break

print(tokenizer.decode(generated_tokens, skip_special_tokens=True))
```

## WebGPU (Browser)

### Installation

```bash
npm install @huggingface/transformers
```

### Inference

```javascript
import { AutoModelForCausalLM, AutoTokenizer, TextStreamer } from "@huggingface/transformers";

const modelId = "LiquidAI/LFM2.5-1.2B-Instruct-ONNX";

// Load model and tokenizer
const tokenizer = await AutoTokenizer.from_pretrained(modelId);
const model = await AutoModelForCausalLM.from_pretrained(modelId, {
  device: "webgpu",
  dtype: "q4",  // or "fp16"
});

// Prepare input
const messages = [{ role: "user", content: "What is the capital of France?" }];
const input = tokenizer.apply_chat_template(messages, {
  add_generation_prompt: true,
  return_dict: true,
});

// Generate with streaming
const streamer = new TextStreamer(tokenizer, { skip_prompt: true });
const output = await model.generate({
  ...input,
  max_new_tokens: 256,
  do_sample: false,
  streamer,
});

console.log(tokenizer.decode(output[0], { skip_special_tokens: true }));
```

### WebGPU Notes

- Enable WebGPU: `chrome://flags/#enable-unsafe-webgpu`
- Supported: Q4, FP16 (Q8 not supported on WebGPU)

## License

This model is released under the [LFM 1.0 License](LICENSE).
