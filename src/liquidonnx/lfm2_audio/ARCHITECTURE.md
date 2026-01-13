# LFM2.5-Audio ONNX Architecture

## Current Structure (4 core models + 2 depthformer models)

```
exports/LFM2.5-Audio-1.5B-ONNX/onnx/
├── decoder.onnx           # 4.7 GB - LFM2 backbone
├── audio_encoder.onnx     # 480 MB - Conformer for ASR
├── audio_embedding.onnx   # 134 MB - Audio code embeddings
├── audio_detokenizer.onnx # 180 MB - Neural vocoder
└── depthformer/
    ├── depth_linear.onnx        # 67 MB - Called 1x per frame
    └── depthformer_unified.onnx # ~160 MB - Called 8x per frame
```

| Model | Input | Output | Used For |
|-------|-------|--------|----------|
| `decoder.onnx` | `inputs_embeds`, `attention_mask`, cache | `logits`, `hidden_states`, cache | All modes |
| `audio_encoder.onnx` | `mel_features`, `mel_lengths` | `audio_embeddings`, `output_lengths` | ASR only |
| `audio_embedding.onnx` | `audio_codes` [B, 8] | `audio_embeds` [B, 8, H] | TTS, Interleaved |
| `audio_detokenizer.onnx` | `audio_codes` [B, 8, T] | `stft_features` [B, T', 1282] | TTS, Interleaved |

### Data Flow

**Text Generation:**
```
input_ids → embed_tokens (numpy) → decoder → logits → sample → token
```

**ASR (Audio → Text):**
```
mel_spectrogram → audio_encoder → audio_embeddings → decoder → logits → text
```

**TTS (Text → Audio):**
```
text → decoder → hidden_states → depthformer (ONNX) → audio_codes → audio_detokenizer → ISTFT → waveform
```

**Interleaved:**
```
Mixed text/audio input → decoder → hidden_states → depthformer/sampling → mixed output
```

## Depthformer (TTS Audio Codebook Prediction)

The depthformer predicts 8 audio codebook tokens autoregressively using a 6-layer transformer.
Consolidated into 2 ONNX models (previously 18):

```
exports/LFM2.5-Audio-1.5B-ONNX/onnx/depthformer/
├── depth_linear.onnx        # [B, 2048] → [B, 8, 1024] (1x per frame)
└── depthformer_unified.onnx # All-in-one step (8x per frame)
```

| Model | Input | Output | Calls/Frame |
|-------|-------|--------|-------------|
| `depth_linear.onnx` | hidden_states [B, 2048] | depth_slices [B, 8, 1024] | 1 |
| `depthformer_unified.onnx` | depth_slices, step_idx, prev_token, cache | logits, cache | 8 |

### Unified Model Details

`depthformer_unified.onnx` consolidates:
- Transformer step with KV cache (was `depthformer_step.onnx`)
- 8 embedding tables (was `depth_embed_0..7.onnx`)
- 8 logits projections (was `depth_logits_0..7.onnx`)

**Inputs:**
- `depth_slices`: [B, 8, 1024] - All 8 depth slices from depth_linear
- `step_idx`: int64 scalar - Which codebook step (0-7)
- `prev_token`: [B] int64 - Previous step's sampled token
- `past_keys`: [6, B, seq_len, 8, 32] - KV cache keys
- `past_values`: [6, B, seq_len, 8, 32] - KV cache values

**Outputs:**
- `logits`: [B, 2049] - Codebook token probabilities
- `token_embed`: [B, 1024] - Placeholder (unused)
- `new_keys`, `new_values`: Updated KV cache

### Autoregressive Loop

```python
# 1. Project decoder output to depth space (1x per frame)
depth_slices = depth_linear(hidden_states)  # [B, 8, 1024]

# 2. Initialize
past_keys = zeros([6, B, 0, 8, 32])
past_values = zeros([6, B, 0, 8, 32])
prev_token = 0

# 3. Generate 8 codebook tokens
for step in range(8):
    logits, _, new_keys, new_values = depthformer_unified(
        depth_slices, step, prev_token, past_keys, past_values
    )
    token = sample(logits)  # Sample from [2049] logits
    prev_token = min(token, 2047)
    past_keys, past_values = new_keys, new_values
```

### ONNX-Compatible Implementation

The unified wrapper uses ONNX-compatible operations:
- **Rotary embeddings**: Decomposed from complex to real operations
  `(a + bi) * (cos + i*sin) = (a*cos - b*sin) + i*(a*sin + b*cos)`
- **GQA attention**: Manual head expansion (no `enable_gqa` flag)
- **Dynamic indexing**: Uses `torch.gather` for step-specific slice/weight selection
- **Step 0 handling**: Zeros previous embedding via mask multiplication

## Components Not Exported to ONNX

### Text Embeddings (embed_tokens)
Text token embeddings use **numpy lookup at runtime** instead of a separate ONNX model:
- Simple gather operation: `embeds = embed_weight[input_ids]`
- Weight loaded from model safetensors (134 MB)
- Avoids overhead of separate ONNX session for trivial operation

### ISTFT
Inverse Short-Time Fourier Transform is implemented in **scipy** instead of ONNX:
- `audio_detokenizer.onnx` outputs STFT features (log_magnitude + angle)
- ISTFT converts STFT features to waveform using scipy
- Configuration stored in `istft_config.json`

## Historical: Model Consolidation

### From 18 to 2 Depthformer Models

| Before (18 models) | After (2 models) |
|-------------------|------------------|
| `depth_linear.onnx` | `depth_linear.onnx` |
| `depthformer_step.onnx` | → merged into `depthformer_unified.onnx` |
| `depth_embed_0..7.onnx` (8) | → merged into `depthformer_unified.onnx` |
| `depth_logits_0..7.onnx` (8) | → merged into `depthformer_unified.onnx` |

**Benefits:**
- Fewer ONNX sessions to manage
- All weights in single model file
- Simpler deployment
- Same performance (depth_linear still called 1x, unified called 8x)

### Original 7-Model Structure

Before the depthformer ONNX export, the system had:

| Model | Status | Reason |
|-------|--------|--------|
| `embed_tokens.onnx` | Removed | Now numpy lookup |
| `decoder.onnx` | Kept | Core model |
| `audio_embedding.onnx` | Kept | Audio code embeddings |
| `audio_encoder.onnx` | Kept | ASR mode |
| `depthformer.onnx` | Removed | Now autoregressive ONNX |
| `audio_detokenizer.onnx` | Kept | Neural vocoder |
| `audio_lm_head.onnx` | Removed | Unused |

## Export Compatibility Notes

The `audio_detokenizer` wrapper uses ONNX-compatible operations:
- `.transpose(-2, -1)` instead of `.mT` (matrix transpose)
- `mode="nearest"` instead of `mode="nearest-exact"` for interpolation
- Weights loaded via `safetensors.torch` to handle bfloat16 → float32 conversion
