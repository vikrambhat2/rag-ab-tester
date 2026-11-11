# Changelog

## Unreleased

### Added
- `uv` packaging (`pyproject.toml`, `uv.lock`); Streamlit deps moved to the `ui` extra.
- `.env.example`, upfront config validation (`ConfigError`), configurable timeouts.
- Retries with backoff, structured logging (`-v` / `LOG_LEVEL`), `--keep-indices`, `--seed`.
- Automatic cleanup of OpenSearch indices after each run.
- Unit tests, ruff + mypy, pre-commit, GitHub Actions CI, Dependabot.
- Dockerfile, `docker-compose.yml` (local OpenSearch), LICENSE, CONTRIBUTING.

- Optional `LLM_PROVIDER=openai` (any OpenAI-compatible chat endpoint; embeddings stay on WatsonX).
- Separate judge via `JUDGE_LLM_*`, with a warning when judge and generator are the same model.
- `WatsonxJudge` renamed `LLMJudge` (old names kept as aliases).

### Fixed
- `experiments/prompt.py` referenced undefined names and crashed with `NameError`.
- `experiments/custom/example_experiment.py` imported the removed `langchain_ollama`.
- Health checks no longer swallow errors silently.
- `compare_metric` now validates sample sizes and handles identical scores (NaN p-value).
- Hybrid retrieval rebuilt the BM25 index on every query; it is now built once.
- Failed queries no longer silently skew results: >50% failures aborts the experiment.
- `run_all.py` exits non-zero when any experiment fails.
- Removed stale Ollama/ChromaDB references from docs and `.gitignore`.
