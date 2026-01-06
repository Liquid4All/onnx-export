---
license: other
license_name: lfm1.0
license_link: LICENSE
language:
- en
pipeline_tag: text-generation
tags:
- liquid
- edge
- lfm2
- transcript
- meeting
- summarization
- onnx
- onnxruntime
- webgpu
base_model:
- LiquidAI/LFM2-2.6B-Transcript
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

# LFM2-2.6B-Transcript-ONNX

ONNX export of [LFM2-2.6B-Transcript](https://huggingface.co/LiquidAI/LFM2-2.6B-Transcript) for cross-platform inference.

LFM2-2.6B-Transcript is optimized for processing and summarizing meeting transcripts, extracting key points, action items, and decisions from conversational text.

## Recommended Variants

| Precision | Size | Platform | Use Case |
|-----------|------|----------|----------|
| Q4 | ~2.0GB | WebGPU, Server | Recommended for most uses |
| FP16 | ~4.8GB | WebGPU, Server | Higher quality |

- **WebGPU**: Use Q4 or FP16
- **Server**: All variants supported

## Model Files

```
onnx/
├── model.onnx              # FP32
├── model_fp16.onnx         # FP16
└── model_q4.onnx           # Q4 (recommended)
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
model_id = "LiquidAI/LFM2-2.6B-Transcript-ONNX"
model_path = hf_hub_download(model_id, "onnx/model_q4.onnx")
data_path = hf_hub_download(model_id, "onnx/model_q4.onnx_data")

# Load model and tokenizer
session = ort.InferenceSession(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# Prepare chat input
messages = [{"role": "user", "content": "Summarize this meeting transcript: ..."}]
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
npm install onnxruntime-web @huggingface/transformers
```

### Enable WebGPU

WebGPU is required for browser inference. To enable:

1. **Chrome/Edge**: Navigate to `chrome://flags/#enable-unsafe-webgpu`, enable, and restart
2. **Verify**: Check `chrome://gpu` for "WebGPU" status
3. **Test**: Run `navigator.gpu.requestAdapter()` in DevTools console

### Inference

```javascript
import * as ort from "onnxruntime-web/webgpu";
import { AutoTokenizer } from "@huggingface/transformers";

// Check WebGPU availability
if (!navigator.gpu) {
  throw new Error("WebGPU not available. Enable at chrome://flags/#enable-unsafe-webgpu");
}

ort.env.wasm.numThreads = 1;

const modelId = "LiquidAI/LFM2-2.6B-Transcript-ONNX";
const modelBase = `https://huggingface.co/${modelId}/resolve/main`;

// Load tokenizer and model
const tokenizer = await AutoTokenizer.from_pretrained(modelId);

const session = await ort.InferenceSession.create(`${modelBase}/onnx/model_q4.onnx`, {
  executionProviders: ["webgpu"],
  externalData: [{ path: "model_q4.onnx_data", data: `${modelBase}/onnx/model_q4.onnx_data` }],
});

// Model config
const hiddenSize = 2048;
const numKVHeads = 8;
const headDim = 256;

// Initialize KV cache
function initCache() {
  const cache = {};
  for (const name of session.inputNames) {
    if (name.startsWith("past_conv")) {
      cache[name] = new ort.Tensor("float32", new Float32Array(hiddenSize * 3), [1, hiddenSize, 3]);
    } else if (name.startsWith("past_key_values")) {
      cache[name] = new ort.Tensor("float32", new Float32Array(0), [1, numKVHeads, 0, headDim]);
    }
  }
  return cache;
}

// Update cache from outputs
function updateCache(cache, outputs) {
  for (const [name, tensor] of Object.entries(outputs)) {
    if (name.startsWith("present_conv")) {
      cache[name.replace("present_conv", "past_conv")] = tensor;
    } else if (name.startsWith("present.")) {
      cache[name.replace("present.", "past_key_values.")] = tensor;
    }
  }
}

// Prepare input
const messages = [{ role: "user", content: "Summarize this meeting transcript: ..." }];
const prompt = tokenizer.apply_chat_template(messages, { add_generation_prompt: true, tokenize: false });
const inputIds = tokenizer.encode(prompt);

// Generation loop
const cache = initCache();
const generatedTokens = [];
let curLen = inputIds.length;

for (let step = 0; step < 256; step++) {
  const ids = step === 0 ? inputIds : [generatedTokens[generatedTokens.length - 1]];
  const inputTensor = new ort.Tensor("int64", new BigInt64Array(ids.map(BigInt)), [1, ids.length]);
  const attentionMask = new ort.Tensor("int64", new BigInt64Array(curLen).fill(1n), [1, curLen]);

  const outputs = await session.run({ input_ids: inputTensor, attention_mask: attentionMask, ...cache });

  // Greedy decode
  const logits = outputs.logits;
  const vocabSize = logits.dims[2];
  const lastLogits = logits.data.slice((logits.dims[1] - 1) * vocabSize);
  const nextToken = lastLogits.indexOf(Math.max(...lastLogits));

  generatedTokens.push(nextToken);
  if (nextToken === tokenizer.eos_token_id) break;

  updateCache(cache, outputs);
  curLen++;
}

console.log(tokenizer.decode(generatedTokens, { skip_special_tokens: true }));
```

### WebGPU Notes

- Recommended: `model_q4.onnx` for smallest size
- For higher quality: `model_fp16.onnx`
- Models use external data files (`.onnx_data`) loaded automatically
- int64 tensors require `BigInt64Array`

## License

This model is released under the [LFM 1.0 License](LICENSE).
