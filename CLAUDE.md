# Python code

## Imports
* Prefer module imports for stdlib: `import pathlib` not `from pathlib import Path`
* Exception: `from dataclasses import dataclass` is fine (decorators)
* Don't use `typing` module - use builtin types: `list[int]`, `dict[str, int]`, `X | None`

## Style
* Use `logger` instead of `print` for output
* Use f-strings for formatting
* Don't include comments that just restate the code
* Don't re-throw or silence errors unless documented

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

Tests load large models - run specific tests rather than full suite:

```bash
uv run pytest tests/test_lfm2/test_decoder.py -v -k "350M and q4"
uv run pytest tests/test_lfm2_vl/test_decoder.py -v -k "450M and q4"
```
