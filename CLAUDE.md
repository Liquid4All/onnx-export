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
```

The other tests load large models and need the export in ./exports (README 5) - run specific tests
rather than the full suite:

```bash
uv run pytest tests/test_lfm2/test_decoder.py -v -k "350M and q4"
uv run pytest tests/test_lfm2_vl/test_decoder.py -v -k "450M and q4"
```
