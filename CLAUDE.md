# Python code

## Imports
* Prefer module imports for stdlib: `import pathlib` not `from pathlib import Path`
* Exception: `from dataclasses import dataclass` is fine (decorators)
* Don't use `typing` module - use builtin types: `list[int]`, `dict[str, int]`, `X | None`

## Style
* Use `logger` instead of `print` for output
* Use f-strings for formatting
* Prefer `pathlib` over `os.path` for path operations
* Don't re-throw or silence errors unless documented

## Comments
* Section headers: `# === Section Name ===`
* Numbered steps: `# === 1. Step description ===`
* Shape annotations with arrows: `# [B, S, H] → [B, S, 3H]`
* Keep ASCII diagrams in docstrings for architecture documentation
* Don't include comments that just restate the code
* Don't add redundant docstrings (e.g., `"""Configuration for X."""` when class is `XConfig`)

## Docstrings
* Executable examples must use `uv run`:
  - Scripts: `uv run lfm2-infer --help`
  - Tests: `uv run pytest tests/... -v`

# Code quality

```bash
uv run ruff check src tests        # Lint
uv run ruff format src tests       # Format
uv run ruff check --fix src tests  # Auto-fix
```

# Testing

The export pipelines have tests on tiny random checkpoints (no download):

```bash
uv run pytest tests/test_genai_export.py tests/test_genai_export_multimodal.py tests/test_export_cli.py -v
uv run pytest tests/test_lfm2_audio/test_modes_synthetic.py tests/test_lfm2_audio/test_reference_parity.py \
    tests/test_lfm2_audio/test_graph_structure.py -v
uv run pytest tests/test_compare.py tests/test_compare_wikitext.py -v
uv run pytest tests/test_lfm3/test_config.py tests/test_lfm3/test_rope.py tests/test_lfm3/test_tiny_export.py \
    tests/test_lfm3_vl/test_tiny_export.py tests/test_lfm3_vl/test_config.py tests/test_lfm3_vl/test_rope.py \
    tests/test_lfm3_audio/test_config.py tests/test_lfm3_audio/test_weights.py \
    tests/test_lfm3_audio/test_tiny_reference.py tests/test_lfm3_audio/test_tiny_decoder.py \
    tests/test_lfm3_audio/test_tiny_codec.py tests/test_lfm3_audio/test_gqa_window.py \
    tests/test_lfm3_audio/test_infer.py tests/test_lfm3_omni/test_config.py \
    tests/test_d1/test_config.py tests/test_d1/test_tiny_export.py tests/test_d1/test_tiny_session.py \
    tests/test_check_checkpoint.py -v  # in CI (check_python.yml)
uv run pytest tests/test_lfm3_audio/test_runtime.py tests/test_lfm3_audio/test_audio_encoder.py \
    tests/test_lfm3_audio/test_depth.py -v -k "(tiny or sample or sampling or orders) and not onnx"  # in CI
```

The other tests load large models and need the export in ./exports (README 5) - run specific tests
rather than the full suite:

```bash
uv run pytest tests/test_lfm2/test_decoder.py -v -k "350M and q4"
uv run pytest tests/test_lfm2_vl/test_decoder.py -v -k "450M and q4"
uv run pytest tests/test_lfm3/test_decoder.py -v -k "q4 and not q4f"  # LFM3-1.7B, README 3.1b
uv run pytest tests/test_lfm3_vl/test_decoder.py -v -k "q4 and not q4f16"  # LFM3-VL-2B, README 3.1c
uv run pytest tests/test_lfm3_audio/test_decoder.py -v -k "q4 and not q4f16"  # LFM3-Audio-2B, README 3.1d
uv run pytest tests/test_d1/test_answers.py -v --d1-model LiquidAI/d1-2B -k q4  # d1, README 3.1f
uv run pytest tests/test_lfm3_omni/test_decoder.py -v -k q4  # LFM3-Omni-2B, README 3.1e
```
