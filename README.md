# LiquidONNX

ONNX export and inference tools for [LFM2](https://www.liquid.ai/liquid-foundation-models) models.

## Supported Models

| Family | Source Models | Quant Formats |
|--------|---------------|---------------|
| **LFM2** (text) | LFM2.5-1.2B-Instruct, LFM2-2.6B-Transcript | fp32, fp16, q4, q8 |
| **LFM2-VL** (vision-language) | LFM2.5-VL-1.6B | fp32, fp16, q4, q8 |
| **LFM2-MoE** (mixture of experts) | LFM2-MoE-8B-A1B | fp32, fp16, q4, q4f16 |

## Installation

```bash
git clone https://github.com/Liquid4All/onnx-export.git
cd onnx-export
uv sync

# For GPU inference support
uv sync --extra gpu
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `lfm2-export` | Export LFM2 text models to ONNX |
| `lfm2-infer` | Interactive inference for LFM2 text models |
| `lfm2-bench` | Benchmark LFM2 ONNX model performance |
| `lfm2-vl-export` | Export LFM2-VL vision-language models to ONNX |
| `lfm2-vl-infer` | Interactive inference for LFM2-VL models |
| `lfm2-moe-export` | Export LFM2-MoE mixture-of-experts models to ONNX |
| `lfm2-moe-infer` | Interactive inference for LFM2-MoE models |

Run any command with `--help` for full options: `uv run lfm2-export --help`

## Export

### LFM2 Text Models

```bash
uv run lfm2-export LiquidAI/LFM2.5-1.2B-Instruct
uv run lfm2-export LiquidAI/LFM2-2.6B-Transcript

# With quantization
uv run lfm2-export LiquidAI/LFM2.5-1.2B-Instruct --precision q4
uv run lfm2-export LiquidAI/LFM2.5-1.2B-Instruct --precision fp16 q4 q8

# All precisions
uv run lfm2-export LiquidAI/LFM2.5-1.2B-Instruct --precision
```

### LFM2-VL Vision-Language Models

```bash
uv run lfm2-vl-export LiquidAI/LFM2.5-VL-1.6B

# With quantization
uv run lfm2-vl-export LiquidAI/LFM2.5-VL-1.6B --precision q4

# Conv2d vision format (alternative to default tiled)
uv run lfm2-vl-export LiquidAI/LFM2.5-VL-1.6B --vision-format conv2d
```

### LFM2-MoE Mixture of Experts

```bash
uv run lfm2-moe-export LiquidAI/LFM2-MoE-8B-A1B

# With Q4 quantization (uses QMoE operator)
uv run lfm2-moe-export LiquidAI/LFM2-MoE-8B-A1B --precision q4

# Q4 with FP16 non-expert layers
uv run lfm2-moe-export LiquidAI/LFM2-MoE-8B-A1B --precision q4f16
```

### Export Options

| Flag | Description |
|------|-------------|
| `--output-dir DIR` | Output directory (default: current directory) |
| `--output-name NAME` | Output folder name (default: `{model-name}-ONNX`) |
| `--precision [P ...]` | Precisions to export: `fp16`, `q4`, `q8` (or empty for all) |
| `--skip-export` | Skip FP32 export, only run quantization on existing export |
| `--block-size N` | Block size for quantization (default: 32) |
| `--q4-asymmetric` | Use asymmetric Q4 (default is symmetric for WebGPU) |

## Inference

All inference commands provide interactive multi-turn chat with streaming output. They automatically detect CUDA availability and fall back to CPU if needed.

### LFM2 Text Generation

```bash
# Interactive chat (starts conversation loop)
uv run lfm2-infer --model LFM2.5-1.2B-Instruct-ONNX/onnx/model_q4.onnx

# Single prompt (non-interactive)
uv run lfm2-infer --model LFM2.5-1.2B-Instruct-ONNX/onnx/model_q4.onnx \
    --prompt "Explain quantum computing"

# Force CPU execution
uv run lfm2-infer --model LFM2.5-1.2B-Instruct-ONNX/onnx/model_q4.onnx --cpu
```

### LFM2-VL Vision-Language

```bash
# Single image analysis
uv run lfm2-vl-infer --model LFM2.5-VL-1.6B-ONNX \
    --images photo.jpg \
    --prompt "What do you see in this image?"

# Multi-image comparison (up to 2 images)
uv run lfm2-vl-infer --model LFM2.5-VL-1.6B-ONNX \
    --images image1.jpg image2.jpg \
    --prompt "Compare these two images"

# Text-only (no images)
uv run lfm2-vl-infer --model LFM2.5-VL-1.6B-ONNX \
    --prompt "Hello, how are you?"
```

> **Note:** VL inference requires the model directory path (not a single .onnx file) since it loads multiple components: `embed_tokens.onnx`, `embed_images.onnx`, and `decoder.onnx`.

### LFM2-MoE

```bash
# Interactive chat
uv run lfm2-moe-infer --model LFM2-MoE-8B-A1B-ONNX/onnx/model_q4.onnx

# Force CPU (when model does not fit VRAM)
uv run lfm2-moe-infer --model LFM2-MoE-8B-A1B-ONNX/onnx/model_q4.onnx --cpu
```

### Inference Options

| Flag | Description |
|------|-------------|
| `--model PATH` | Path to ONNX model file or directory (required) |
| `--prompt TEXT` | Initial prompt (if omitted, starts interactive mode) |
| `--max-tokens N` | Maximum tokens to generate (default: 512) |
| `--no-stream` | Disable streaming output (print all at once) |
| `--cpu` | Force CPU execution (text/MoE only) |
| `--images PATH...` | Image paths for VL models (0-2 images) |

## Output Structure

### Text Models (LFM2, LFM2-MoE)

```
LFM2.5-1.2B-Instruct-ONNX/
├── config.json
├── tokenizer.json
└── onnx/
    ├── model.onnx           # FP32
    ├── model.onnx_data
    ├── model_fp16.onnx      # FP16
    ├── model_q4.onnx        # INT4
    └── model_q8.onnx        # INT8
```

### Vision-Language Models (LFM2-VL)

```
LFM2.5-VL-1.6B-ONNX/
├── config.json
├── tokenizer.json
├── tokenizer_config.json
└── onnx/
    ├── embed_tokens.onnx        # Token embeddings (FP32)
    ├── embed_images.onnx        # Vision encoder (FP32)
    ├── embed_images_q4.onnx     # Vision encoder (INT4)
    ├── decoder.onnx             # Language decoder (FP32)
    └── decoder_q4.onnx          # Language decoder (INT4)
```

## Benchmarking

```bash
# Text model benchmark
uv run lfm2-bench --model LiquidAI/LFM2.5-1.2B-Instruct \
    --onnx LFM2.5-1.2B-Instruct-ONNX/onnx/model_q4.onnx
```

## Testing

Tests verify ONNX exports against PyTorch reference models.

```bash
# Install dev dependencies
uv sync --extra dev

# LFM2 text model tests
uv run pytest tests/test_lfm2/test_decoder.py -v -k "q4"

# LFM2-VL vision-language tests
uv run pytest tests/test_lfm2_vl/test_decoder.py -v -k "450M"
uv run pytest tests/test_lfm2_vl/test_vision_encoder.py -v

# LFM2-MoE tests
uv run pytest tests/test_lfm2_moe/test_decoder.py -v
```

## Pre-exported Models

### LiquidAI (Official)

- [LiquidAI/LFM2.5-1.2B-Instruct-ONNX](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-ONNX)
- [LiquidAI/LFM2.5-VL-1.6B-ONNX](https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B-ONNX)

### onnx-community

- [onnx-community/LFM2-8B-A1B-ONNX](https://huggingface.co/onnx-community/LFM2-8B-A1B-ONNX)
- [onnx-community/LFM2-VL-450M-ONNX](https://huggingface.co/onnx-community/LFM2-VL-450M-ONNX)
- [onnx-community/LFM2-VL-1.6B-ONNX](https://huggingface.co/onnx-community/LFM2-VL-1.6B-ONNX)
- [onnx-community/LFM2-VL-3B-ONNX](https://huggingface.co/onnx-community/LFM2-VL-3B-ONNX)

> **Note:** The onnx-community models are exported using [Transformers.js](https://github.com/huggingface/transformers.js) tooling with a different export pipeline. This project aims to produce compatible graph structures and file naming conventions to ensure interoperability with Transformers.js and other ONNX consumers.

## Acknowledgements

Special thanks to [Joshua Lochner](https://huggingface.co/Xenova) for his work on [Transformers.js](https://github.com/huggingface/transformers.js) and the [onnx-community](https://huggingface.co/onnx-community) models, which inspired and informed this project's ONNX export approach.

## License

See [LICENSE](LICENSE) for details.
