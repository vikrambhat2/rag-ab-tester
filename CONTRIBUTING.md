# Contributing

1. `uv sync --group dev && uv run pre-commit install`
2. Create a branch, make your change, add or update tests in `tests/`.
3. Before opening a PR, make sure these pass (CI runs the same):
   ```bash
   uv run ruff check . && uv run ruff format --check .
   uv run mypy .
   uv run pytest
   ```
4. Update `CHANGELOG.md` under **Unreleased**.

## Guidelines

- Unit tests must not touch the network or read real credentials — use the fakes in `tests/conftest.py`.
- Wrap new network calls with `rag_ab_tester.retry.with_retry`, and log with `logging` (not `print`) inside `src/`.
- Never commit `.env`, certificates, or results containing private data.
- Built-in experiments live in `src/rag_ab_tester/experiments/`; user-style examples in `examples/`. See the README for the required module attributes.
- Library code must not read `.env`, configure logging or call `sys.exit` — only `cli.py` does that.
- Optional dependencies (OpenAI client, LangChain text splitters, BM25) must be imported lazily via `config.import_optional`, so the core install keeps working without them.
