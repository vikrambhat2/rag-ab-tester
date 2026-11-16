# RAG A/B Tester

A reusable A/B testing framework for RAG pipelines.
Plug in any two variants, get statistical proof. Works with any LLM, embedding model and vector store:
the CLI talks to any OpenAI-compatible endpoint out of the box, and the SDK accepts your own objects.

## Install

```bash
pip install "rag-ab-tester[all]"      # what the bundled experiments use
```

The core install (statistics, reporting, judge, CLI) is deliberately light. Providers are extras:

| Extra | Adds |
|---|---|
| `openai` | Built-in chat + embeddings client for any OpenAI-compatible endpoint (`langchain-openai`) |
| `docs` | Markdown loading/chunking and BM25 hybrid retrieval (`langchain-text-splitters`, `rank-bm25`) |
| `all` | Both |

If something is missing you get an error naming the extra to install. Using your own models and
vector store (see below) needs neither extra, apart from `docs` if you load `.md` files from disk.

## Quick Start

Requires Python 3.10+. From a source checkout use [uv](https://docs.astral.sh/uv/) (`uv sync --all-extras`
then prefix commands with `uv run`); with pip, install as above.

```bash
# 1. Configure credentials (environment variables or a .env file in the working directory)
cp .env.example .env     # from the repo; set LLM_MODEL, LLM_API_KEY, EMBED_MODEL

# 2. Put your markdown docs in data/docs (or pass --docs-path), then generate a test set
rag-ab ingest

# 3. Run experiments in recommended order (each builds on the previous winner)
rag-ab run chunking --save-json
rag-ab run embedding --save-json
rag-ab run retrieval --save-json
rag-ab run prompt --save-json

# Or run all at once
rag-ab run-all --save-json
```

Configuration is validated up front: a missing or malformed variable produces one
error listing everything that needs fixing, before any network call is made.

Results and the champion config are written to `./results` (override with `--results-dir` or `RAG_AB_RESULTS_DIR`).

## Use as a Python SDK

```python
from rag_ab_tester import RAGPipeline, compare_metric, run_experiment

# Pure statistics — works with the core install, no services needed
cmp = compare_metric("faithfulness", control_scores, challenger_scores)
print(cmp.winner, cmp.p_value, cmp.cohens_d, (cmp.ci_low, cmp.ci_high))

# Run a built-in (or your own) experiment end to end
result = run_experiment("chunking", test_set_path="data/test_set.json", docs_path="data/docs")
print(result.overall_winner)
```

### Try it locally (no API keys)

[`examples/notebooks/quickstart_local.ipynb`](examples/notebooks/quickstart_local.ipynb) A/B tests chunk
size, embedding model, prompt and retrieval method using [Ollama](https://ollama.com) as the LLM and
Hugging Face `sentence-transformers` for embeddings. The experiment files it runs live in
[`examples/experiments/`](examples/experiments/) and also work from the CLI, e.g.
`rag-ab run examples/experiments/exp_retrieval.py`. Prerequisites are listed at the top of the notebook.

The library never reads `.env`, configures logging or exits the process; the `rag-ab` CLI does that.
Library users set environment variables themselves (see below). Public API: `RAGPipeline`, `LLMJudge`,
`compare_metric`, `run_experiment`, the result schemas, `Settings`/`get_settings`, the provider
protocols and index adapters (below), and the `ConfigError` / `ExperimentError` /
`MissingExtraError` exceptions.

### Bring your own LLM, embeddings and vector store

Nothing in the core is tied to a provider. Three small protocols
(`rag_ab_tester.protocols`) describe what the framework needs; anything that fits can be passed in:

| Protocol | Needs | Already satisfied by |
|---|---|---|
| `ChatModel` | `invoke(prompt)` returning text or an object with `.content` | any LangChain chat model (Anthropic, OpenAI, Ollama, Bedrock, Vertex, ...); a plain `str -> str` callable also works |
| `Embedder` | `embed_documents(texts)`, `embed_query(text)` | any LangChain `Embeddings` |
| `VectorIndex` | `add(texts, embedder)`, `search(query, k)`, `delete()` | `InMemoryIndex` (default), `LangChainVectorStoreIndex` (wraps Chroma, FAISS, pgvector, Qdrant, Pinecone, OpenSearch, ...) |

```python
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from rag_ab_tester import LangChainVectorStoreIndex, run_experiment


def chroma(index_name):  # called once per variant -> isolated collection each
    return LangChainVectorStoreIndex(
        build=lambda texts, emb: Chroma.from_texts(texts, emb, collection_name=index_name),
    )


result = run_experiment(
    "chunking",
    llm=ChatAnthropic(model="claude-sonnet-5-5"),  # generates answers
    judge_llm=ChatAnthropic(model="claude-opus-5-5"),  # use a different model to judge
    embedder=OllamaEmbeddings(model="nomic-embed-text"),  # used by both variants
    index_factory=chroma,
)
```

Every argument is optional and falls back to the environment configuration (an OpenAI-compatible
endpoint and an in-memory index). The same objects can be given to a single pipeline:
`MyPipeline("idx", llm=..., embedder=..., index=...)`. Notes:

- `embedder=` applies to both variants, so it is rejected for the built-in `embedding` experiment (which
  compares embedders); write your own experiment whose variants override `get_embeddings()` instead.
- `pipeline.ingest_texts(chunks)` indexes pre-chunked text, so documents can come from anywhere.
- Pass `delete=` to `LangChainVectorStoreIndex` if your store has no `delete_collection()`.
- Loading markdown files from `docs_path` (and BM25 hybrid retrieval) need the `docs` extra.
- The default in-memory index is rebuilt on every run, which is fine for test-set-sized corpora. Use a
  persistent store plus `keep_indices=True` if you want to inspect indices afterwards.

A runnable version is in `examples/sdk/bring_your_own_providers.py`.

## Configuration

The CLI (and any code that doesn't inject its own models) uses one kind of endpoint: an
**OpenAI-compatible API**. That covers OpenAI, Ollama, vLLM, LiteLLM, most gateways and hosted
inference services. Set these as environment variables or in a `.env` file:

```bash
# Generator: the model that answers questions (required)
LLM_MODEL=gpt-4o-mini
LLM_API_KEY=your-api-key
# LLM_BASE_URL=http://localhost:11434/v1    # unset = OpenAI; set for any other server (key then optional)

# Embeddings (required by the built-in experiments). Endpoint/key default to the LLM_* values.
EMBED_MODEL=text-embedding-3-small
# EMBED_BASE_URL=...  EMBED_API_KEY=...
# EMBED_MODEL_CHALLENGER=text-embedding-3-large   # compared against EMBED_MODEL by `rag-ab run embedding`

# Tuning (optional)
# LLM_TIMEOUT=120    LOG_LEVEL=INFO
```

### Separate judge

When judge and generator are the same model, LLM-as-judge scores can be biased toward that model's
own style. Set any of `JUDGE_LLM_MODEL`, `JUDGE_LLM_BASE_URL`, `JUDGE_LLM_API_KEY`; unset values fall
back to the generator's. For example, only the model differs:

```bash
JUDGE_LLM_MODEL=a-different-model
```

If the judge points at a different `JUDGE_LLM_BASE_URL`, it does not inherit the generator's API key
(so one provider's key is never sent to another's server). Each run prints both models and warns if
they are identical.

Configuration is validated lazily and fully: a missing or malformed variable produces one error
listing everything that needs fixing, before any network call is made.

## Project Structure

```
rag-ab-tester/
├── src/rag_ab_tester/
│   ├── cli.py               `rag-ab` command (ingest / run / run-all)
│   ├── runner.py            run_experiment(), experiment loading
│   ├── run_all.py           Run several experiments + summary
│   ├── ingest.py            Test-set generation from your docs
│   ├── config.py            Validated settings + OpenAI-compatible chat/embedding factories
│   ├── best_config.py       Champion config store (get / save / summary)
│   ├── retry.py             Exponential-backoff wrapper for network calls
│   ├── experiments/         Built-in experiments (chunking, embedding, retrieval, prompt)
│   ├── protocols.py         ChatModel / Embedder / VectorIndex interfaces
│   ├── stores/              Vector index adapters (in-memory, LangChain wrapper)
│   ├── pipeline/base.py     RAGPipeline base class
│   ├── evaluator/           LLM judge, 4 RAG metrics, t-test + Cohen's d + CI
│   ├── models/schemas.py    Pydantic data models
│   └── report/report.py     Rich terminal tables + JSON export
├── examples/                Template for your own experiment
├── tests/                   Unit tests (no network) + opt-in integration tests
├── data/docs/               Sample corpus (repo only — not shipped in the wheel)
└── streamlit_app/           Sample dashboard (repo only — not shipped in the wheel)
```

## Champion Config — How It Works

Each experiment saves the winner's hyperparameters to `results/best_config.json`.
Downstream experiments automatically read from this file so every run builds
on proven winners rather than hardcoded defaults.

```
chunking.py   → saves chunk_size + chunk_overlap
    ↓
embedding.py  → reads best chunk_size, saves embedding_model
    ↓
retrieval.py  → reads best chunk_size + embedding_model, saves retrieval_strategy
    ↓
prompt.py     → reads all three, saves prompt_template
```

After all four experiments the file looks like:

```json
{
  "chunk_size": 256,
  "chunk_overlap": 25,
  "embedding_model": "text-embedding-3-large",
  "retrieval_strategy": "hybrid",
  "prompt_template": "strict"
}
```

You can also read or write it manually:

```python
from rag_ab_tester.best_config import get, save

chunk_size = get("chunk_size")  # e.g. 256
embedding = get("embedding_model")  # "" means: use EMBED_MODEL

# Override a value manually
save({"embedding_model": "text-embedding-3-small"})
```

## Adding Your Own Experiment

Create an experiment file anywhere (see `examples/example_experiment.py`):

```python
from rag_ab_tester.pipeline.base import RAGPipeline
from rag_ab_tester.config import get_embeddings
from rag_ab_tester.best_config import get as best

_PROMPT = "Answer using only the context.\nContext: {context}\nQuestion: {query}\nAnswer:"


class MyControlPipeline(RAGPipeline):
    def get_chunk_size(self):
        return (best("chunk_size"), best("chunk_overlap"))

    def get_embeddings(self):
        return get_embeddings(best("embedding_model") or None)

    def get_prompt(self, query, context):
        return _PROMPT.format(context=context, query=query)


class MyChallengerPipeline(RAGPipeline):
    def get_chunk_size(self):
        return (best("chunk_size"), best("chunk_overlap"))

    def get_embeddings(self):
        return get_embeddings("text-embedding-3-large")

    def get_prompt(self, query, context):
        return _PROMPT.format(context=context, query=query)


EXPERIMENT_NAME = "My Experiment"
CONTROL = MyControlPipeline
CHALLENGER = MyChallengerPipeline
CONTROL_NAME = "baseline"
CHALLENGER_NAME = "embed-large"

# Optional: auto-save the winner to best_config.json
CHAMPION_CONFIG = {
    "baseline": {"embedding_model": "text-embedding-3-small"},
    "embed-large": {"embedding_model": "text-embedding-3-large"},
}
```

Run it:

```bash
rag-ab run path/to/my_experiment.py --save-json
```

Each experiment gets two isolated indices (`ctrl_<name>` and `chal_<name>`) so variants never share state.

## Metrics

| Metric | What it measures |
|---|---|
| Faithfulness | Answer grounded in retrieved context (no hallucination) |
| Answer Relevance | Answer addresses the question |
| Context Precision | Fraction of retrieved chunks that are relevant |
| Context Recall | Retrieved context covers the ground-truth answer |

## Statistical Output

Each metric comparison reports:
- **Δ** — raw score difference (challenger − control)
- **p-value** — paired t-test; p < 0.05 means the difference is statistically significant
- **Cohen's d** — effect size; |d| ≥ 0.2 means the difference is practically meaningful
- **95% CI** — confidence interval for the true mean difference
- **Winner** — "challenger", "control", or "no difference"

A change is worth acting on only if both `p < 0.05` **and** `|d| ≥ 0.2`.

## CLI Reference

```bash
rag-ab ingest [--docs-path data/docs] [--num-questions 20] [--output data/test_set.json] [--seed N]

# EXPERIMENT is a built-in name (chunking, embedding, retrieval, prompt) or a path to your .py file
rag-ab run EXPERIMENT [--test-set data/test_set.json] [--docs-path data/docs] [--results-dir DIR] \
                      [--save-json]

rag-ab run-all [--include chunking prompt] [--exclude embedding] [--experiments-dir DIR] \
               [--test-set ...] [--docs-path ...] [--save-json]

rag-ab --version        rag-ab -v <command>   # -v = debug logging
```

Exit codes: `0` success, `1` experiment/config failure (so CI and schedulers can detect it).

## Reliability Notes

- **Retries:** model and vector-store calls retry with exponential backoff on transient errors; bad input and configuration errors fail immediately.
- **Failed queries:** a query that still fails is scored 0.0 and logged. If more than 50% of a variant's queries fail, the experiment aborts rather than report skewed statistics.
- **Judge fallback:** if the LLM judge returns no parseable score it falls back to a neutral 0.5 and logs a warning.
- **Index hygiene:** each run clears any stale index first and removes its indices afterwards (`keep_indices=True` in the SDK keeps them).
- **Logging:** `-v` for debug output, or set `LOG_LEVEL`.

## Docker

The image runs the CLI only (the Streamlit app is a sample and is not containerised).

```bash
docker build -t rag-ab-tester .
docker run --rm --env-file .env -v "$PWD/results:/app/results" rag-ab-tester \
    rag-ab run chunking --save-json

# Or with compose (reads .env, mounts ./results and ./data)
docker compose run --rm app rag-ab ingest
docker compose run --rm app rag-ab run-all --save-json
```

Credentials are passed at runtime and never baked into the image. To use a model server on your host
(e.g. Ollama), set `LLM_BASE_URL=http://host.docker.internal:11434/v1`.

## Sample Dashboard (optional)

```bash
uv sync --group ui --all-extras      # from a source checkout
uv run streamlit run streamlit_app/app.py
```

It lives in the repo only (it is not part of the PyPI package). This is a local demo with no authentication; do not expose it publicly.

## Development

```bash
uv sync --group dev --all-extras
uv run pre-commit install
uv run ruff check . && uv run ruff format --check . && uv run mypy .
uv run pytest                          # unit tests, no network needed
RUN_INTEGRATION=1 uv run pytest -m integration   # live endpoint smoke tests
```

See [CONTRIBUTING.md](CONTRIBUTING.md). CI runs lint, type checks, tests (Python 3.10–3.12), a secret scan, a Docker build, and builds + smoke-tests the wheel.

## License

[MIT](LICENSE)
