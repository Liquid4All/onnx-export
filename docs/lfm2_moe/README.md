# LFM2-MoE ONNX Inference

## Model Structure

```
LFM2-MoE-4x350M-ONNX/
├── config.json
├── tokenizer.json
└── onnx/
    └── model.onnx
```

## Inference Code

```python
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

# === Load model and tokenizer ===
model_dir = "LFM2-MoE-4x350M-ONNX"
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
session = ort.InferenceSession(f"{model_dir}/onnx/model.onnx")

input_names = {inp.name for inp in session.get_inputs()}
output_infos = session.get_outputs()

# === Initialize KV cache ===
def init_cache(session):
    cache = {}
    skip = {"input_ids", "inputs_embeds", "attention_mask", "position_ids"}
    for inp in session.get_inputs():
        if inp.name in skip:
            continue
        shape = [d if isinstance(d, int) else 1 for d in inp.shape]
        for i, d in enumerate(inp.shape):
            if isinstance(d, str) and "sequence" in d.lower():
                shape[i] = 0
        dtype = np.float16 if "float16" in inp.type else np.float32
        cache[inp.name] = np.zeros(shape, dtype=dtype)
    return cache

# === Update cache from outputs ===
def update_cache(cache, outputs, output_infos):
    for i, info in enumerate(output_infos[1:], 1):  # Skip logits
        name = info.name
        if "present_conv" in name:
            cache_name = name.replace("present_conv", "past_conv")
        elif "present." in name:
            cache_name = name.replace("present.", "past_key_values.")
        else:
            continue
        if cache_name in cache:
            cache[cache_name] = outputs[i]

# === Generate ===
messages = [{"role": "user", "content": "Hello, how are you?"}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
input_ids = np.array([tokenizer.encode(prompt, add_special_tokens=False)], dtype=np.int64)

cache = init_cache(session)
generated = []
cur_len = input_ids.shape[1]

for step in range(100):
    if step == 0:
        ids = input_ids
        pos = np.arange(cur_len, dtype=np.int64).reshape(1, -1)
    else:
        ids = np.array([[generated[-1]]], dtype=np.int64)
        pos = np.array([[cur_len - 1]], dtype=np.int64)

    feed = {"input_ids": ids, "attention_mask": np.ones((1, cur_len), dtype=np.int64)}
    if "position_ids" in input_names:
        feed["position_ids"] = pos
    feed.update(cache)

    outputs = session.run(None, feed)
    next_token = int(np.argmax(outputs[0][0, -1]))
    generated.append(next_token)

    update_cache(cache, outputs, output_infos)
    cur_len += 1

    print(tokenizer.decode([next_token]), end="", flush=True)
    if next_token == tokenizer.eos_token_id:
        break

print()
```

## Notes

MoE routing happens automatically inside the ONNX model. The inference loop is identical to LFM2 - expert selection is handled by the `MoE` operator during forward pass.

## CLI

```bash
uv run lfm2-moe-infer --model LFM2-MoE-4x350M-ONNX --prompt "Hello"
```
