# Changelog

## Unreleased

### Packaging (PyPI-ready SDK)
- Installable package `rag-ab-tester` (import `rag_ab_tester`, src layout, hatchling) with a
  `rag-ab` console command (`ingest`, `run`, `run-all`) replacing the root scripts.
- Light core install; providers live in optional extras (`openai`, `docs`, `all`) with a clear
  error naming the missing extra.
- Built-in experiments ship in the wheel (`rag-ab run chunking`); sample docs and the Streamlit
  app stay in the repo only.
- Library-safe: no `.env` loading at import, no logging configuration, no `sys.exit`;
  public API in `rag_ab_tester.__all__`; `py.typed`.
- Results location is configurable (`--results-dir` / `RAG_AB_RESULTS_DIR`); `docs_path` is an explicit parameter.
- Release workflow using PyPI Trusted Publishing (tag `v*`; manual run publishes to TestPyPI); CI builds and smoke-tests the wheel.

### Provider-neutral defaults
- The built-in provider is now any OpenAI-compatible endpoint for both chat and embeddings
  (`LLM_*`, `EMBED_*`, `JUDGE_LLM_*`); the default vector store is in-memory. The earlier
  provider-specific integrations, settings, extras and compose service are removed.
- Extras are now `openai` and `docs` (`all` = both); the heavier LangChain and
  provider-specific packages are no longer dependencies.
- The embedding experiment compares `EMBED_MODEL` with `EMBED_MODEL_CHALLENGER`.
- Removed the `--keep-indices` CLI flag (nothing persists by default); `keep_indices=` remains in the SDK.
- Markdown loading and BM25 no longer use LangChain loaders (own loader + `rank-bm25`).

### Provider-agnostic SDK
- New protocols `ChatModel`, `Embedder`, `VectorIndex` (`rag_ab_tester.protocols`); any LangChain
  chat model / embeddings object satisfies the first two, plain callables work as chat models.
- Vector index adapters: `InMemoryIndex` (default, no dependencies) and
  `LangChainVectorStoreIndex` (wraps any LangChain vector store).
- `RAGPipeline(index_name, *, llm=, embedder=, index=)`, `LLMJudge(llm=)`, and
  `run_experiment` / `run_all` / `generate_test_set` accept `llm`, `judge_llm`, `embedder`,
  `index_factory`; anything omitted falls back to the environment configuration.
- `RAGPipeline.ingest_texts()` for pre-chunked text; `get_embeddings()` is no longer abstract.
- **Breaking for custom pipelines:** `self.vectorstore` is replaced by `self.index` /
  `self.ready`; override `ingest_texts()` / `retrieve()` instead of touching the store directly.

### Added
- `uv` packaging (`pyproject.toml`, `uv.lock`); Streamlit deps moved to the `ui` extra.
- `.env.example`, upfront config validation (`ConfigError`), configurable timeouts.
- Retries with backoff, structured logging (`-v` / `LOG_LEVEL`), `--seed`.
- Automatic cleanup of each variant's index after a run.
- Unit tests, ruff + mypy, pre-commit, GitHub Actions CI, Dependabot.
- Dockerfile, `docker-compose.yml`, LICENSE, CONTRIBUTING.
- Separate judge via `JUDGE_LLM_*`, with a warning when judge and generator are the same model.
- Provider-specific judge classes replaced by a single `LLMJudge`.

### Fixed
- `experiments/prompt.py` referenced undefined names and crashed with `NameError`.
- `experiments/custom/example_experiment.py` imported the removed `langchain_ollama`.
- Health checks no longer swallow errors silently.
- `compare_metric` now validates sample sizes and handles identical scores (NaN p-value).
- Hybrid retrieval rebuilt the BM25 index on every query; it is now built once.
- Failed queries no longer silently skew results: >50% failures aborts the experiment.
- `run_all.py` exits non-zero when any experiment fails.
- Removed stale Ollama/ChromaDB references from docs and `.gitignore`.
