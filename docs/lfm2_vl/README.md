# LFM2-VL ONNX Inference

## Model Structure

```
LFM2-VL-450M-ONNX/
├── config.json
├── tokenizer.json
├── preprocessor_config.json
└── onnx/
    ├── embed_tokens.onnx      # Text → embeddings
    ├── embed_images.onnx      # Image → embeddings
    └── decoder.onnx           # Embeddings → logits (with KV cache)
```

## Inference Code

```python
import numpy as np
import onnxruntime as ort
from PIL import Image
from transformers import AutoProcessor

# === Load models ===
model_dir = "LFM2-VL-450M-ONNX"
processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
tokenizer = processor.tokenizer

embed_tokens = ort.InferenceSession(f"{model_dir}/onnx/embed_tokens.onnx")
embed_images = ort.InferenceSession(f"{model_dir}/onnx/embed_images.onnx")
decoder = ort.InferenceSession(f"{model_dir}/onnx/decoder.onnx")

input_names = {inp.name for inp in decoder.get_inputs()}
output_infos = decoder.get_outputs()

# === KV cache functions ===
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

def update_cache(cache, outputs, output_infos):
    for i, info in enumerate(output_infos[1:], 1):
        name = info.name
        if "present_conv" in name:
            cache[name.replace("present_conv", "past_conv")] = outputs[i]
        elif "present." in name:
            cache[name.replace("present.", "past_key_values.")] = outputs[i]

# === Image preprocessing (Conv2D format) ===
def preprocess_image(image, patch_size=16, downsample=2, max_tokens=256):
    """Preprocess image for embed_images model."""
    import math
    w, h = image.size
    factor = patch_size * downsample  # 32

    # Smart resize to fit token budget
    max_pixels = max_tokens * (patch_size ** 2) * (downsample ** 2)
    new_w = max(factor, round(w / factor) * factor)
    new_h = max(factor, round(h / factor) * factor)

    if new_w * new_h > max_pixels:
        beta = math.sqrt((w * h) / max_pixels)
        new_w = max(factor, int(w / beta / factor) * factor)
        new_h = max(factor, int(h / beta / factor) * factor)

    image = image.resize((new_w, new_h), Image.BILINEAR)

    # Normalize (ImageNet mean/std)
    pixels = np.array(image).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    pixels = (pixels - mean) / std
    pixels = pixels.transpose(2, 0, 1)[np.newaxis]  # [1, 3, H, W]

    spatial_h = new_h // patch_size // downsample
    spatial_w = new_w // patch_size // downsample
    return pixels.astype(np.float32), spatial_h, spatial_w

# === Get image token ID ===
def get_image_token_id(tokenizer):
    for token in ["<image>", "<|image|>"]:
        tid = tokenizer.convert_tokens_to_ids(token)
        if tid != tokenizer.unk_token_id:
            return tid
    return tokenizer.image_token_id

# === Build combined embeddings ===
def build_inputs_embeds(text_embeds, image_embeds, image_token_id, input_ids):
    """Replace <image> tokens with image embeddings."""
    result = text_embeds.copy()
    positions = np.where(input_ids[0] == image_token_id)[0]
    for i, pos in enumerate(positions):
        if i < len(image_embeds):
            result[pos] = image_embeds[i]
    return result[np.newaxis].astype(np.float32)

# === Generate with image ===
image = Image.open("image.jpg").convert("RGB")
messages = [{"role": "user", "content": [
    {"type": "image"},
    {"type": "text", "text": "Describe this image."}
]}]

# Process with HuggingFace processor (handles token expansion)
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(images=[image], text=prompt, return_tensors="np")
input_ids = inputs["input_ids"].astype(np.int64)

# Get embeddings
text_embeds = embed_tokens.run(None, {"input_ids": input_ids})[0][0]

pixels, spatial_h, spatial_w = preprocess_image(image)
image_embeds = embed_images.run(None, {
    "pixel_values": pixels,
    "spatial_h": np.array(spatial_h, dtype=np.int64),
    "spatial_w": np.array(spatial_w, dtype=np.int64),
})[0][0]  # [num_patches, hidden]

image_token_id = get_image_token_id(tokenizer)
inputs_embeds = build_inputs_embeds(text_embeds, image_embeds, image_token_id, input_ids)

# Generate
cache = init_cache(decoder)
generated = []
cur_len = inputs_embeds.shape[1]
has_position_ids = "position_ids" in input_names

for step in range(100):
    if step == 0:
        embeds = inputs_embeds
        pos = np.arange(cur_len, dtype=np.int64).reshape(1, -1)
    else:
        token_ids = np.array([[generated[-1]]], dtype=np.int64)
        embeds = embed_tokens.run(None, {"input_ids": token_ids})[0]
        pos = np.array([[cur_len - 1]], dtype=np.int64)

    feed = {"inputs_embeds": embeds.astype(np.float32),
            "attention_mask": np.ones((1, cur_len), dtype=np.int64)}
    if has_position_ids:
        feed["position_ids"] = pos
    feed.update(cache)

    outputs = decoder.run(None, feed)
    next_token = int(np.argmax(outputs[0][0, -1]))
    generated.append(next_token)

    update_cache(cache, outputs, output_infos)
    cur_len += 1

    print(tokenizer.decode([next_token]), end="", flush=True)
    if next_token == tokenizer.eos_token_id:
        break

print()
```

## CLI

```bash
uv run lfm2-vl-infer --model LFM2-VL-450M-ONNX --image photo.jpg --prompt "What's in this image?"
```
