# Python code

* The executable scripts that have the main() function must list examples with `uv run`.
* Don't re-throw errors or silent them unless absolutely necessary and documented.
* Don't use typing package, use direct types
* Prefer importing modules instead of memebers for the std lib
* Use logger instead of print
* Use f-strings for formatting
* Do not include obvious comments

# Code quality

```bash
uv run ruff check src tests        # Lint
uv run ruff format src tests       # Format
uv run ruff check --fix src tests  # Auto-fix
uv run pytest tests                # Test
```
