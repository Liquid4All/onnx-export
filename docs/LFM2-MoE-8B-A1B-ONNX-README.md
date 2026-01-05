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
- moe
- mixture-of-experts
- onnx
- onnxruntime
base_model:
- LiquidAI/LFM2-8B-A1B
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

# LFM2-MoE-8B-A1B-ONNX

ONNX export of [LFM2-8B-A1B](https://huggingface.co/LiquidAI/LFM2-8B-A1B) for cross-platform inference.

LFM2-MoE is a Mixture of Experts model with 8B total parameters and ~1B active parameters per token. It uses 32 experts with 4 experts activated per token, combining the efficiency of sparse models with the quality of larger dense models.

## Recommended Variants

| Precision | Size | Use Case |
|-----------|------|----------|
| Q4F16 | ~15GB | Recommended (Q4 MoE + FP16 dense) |
| FP16 | ~16GB | Higher quality |
| Q4 | ~30GB | Full Q4 (larger due to expert weights) |

## Model Files

```
onnx/
├── model.onnx              # FP32
├── model_fp16.onnx         # FP16
├── model_q4.onnx           # Q4
└── model_q4f16.onnx        # Q4 MoE experts + FP16 dense (recommended)
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

# Download model (Q4F16 recommended)
model_id = "LiquidAI/LFM2-MoE-8B-A1B-ONNX"
model_path = hf_hub_download(model_id, "onnx/model_q4f16.onnx")
data_path = hf_hub_download(model_id, "onnx/model_q4f16.onnx_data")

# Load model and tokenizer
session = ort.InferenceSession(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# Prepare chat input
messages = [{"role": "user", "content": "Explain mixture of experts in one sentence."}]
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

## Model Architecture

- **Total Parameters**: 8B
- **Active Parameters**: ~1B per token
- **Experts**: 32 total, 4 active per token
- **Hidden Size**: 2048
- **Layers**: 24 (hybrid conv + attention)
- **Context Length**: 128K tokens

## License

This model is released under the [LFM 1.0 License](LICENSE).
