# RAG A/B Tester

A reusable A/B testing framework for RAG pipelines.
Plug in any two variants, get statistical proof.
Powered by [IBM WatsonX AI](https://www.ibm.com/watsonx) (LLM + embeddings) and [OpenSearch](https://opensearch.org) (vector store).

## Quick Start

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
# 1. Install dependencies (creates .venv from uv.lock)
uv sync

# 2. Configure credentials
cp .env.example .env
# Edit .env — add WATSONX_* and OPENSEARCH_* values

# 3. Generate the test set (uses data/docs/ by default)
uv run python ingest.py

# 4. Run experiments in recommended order (each builds on the previous winner)
uv run python run_experiment.py --experiment experiments/chunking.py --save-json
uv run python run_experiment.py --experiment experiments/embedding.py --save-json
uv run python run_experiment.py --experiment experiments/retrieval.py --save-json
uv run python run_experiment.py --experiment experiments/prompt.py --save-json

# Or run all at once
uv run python run_all.py --save-json
```

Configuration is validated up front: a missing or malformed variable produces one
error listing everything that needs fixing, before any network call is made.

## Environment Variables

Create a `.env` file in the project root:

```bash
# WatsonX AI (required)
WATSONX_API_KEY=your-ibm-cloud-api-key
WATSONX_PROJECT_ID=your-watsonx-project-id
WATSONX_URL=https://us-south.ml.cloud.ibm.com

# WatsonX model overrides (optional)
WATSONX_LLM_MODEL_ID=ibm/granite-3-8b-instruct
WATSONX_EMBED_MODEL_ID=ibm/slate-125m-english-rtrvr-v2

# OpenSearch (required)
OPENSEARCH_URL=https://your-opensearch-host:9200
OPENSEARCH_USERNAME=your-username
OPENSEARCH_PASSWORD=your-password
OPENSEARCH_VERIFY_SSL=true     # false only for local dev; logs a warning

# Tuning (optional)
OPENSEARCH_TIMEOUT=60          # seconds
LOG_LEVEL=INFO
```

> **Note:** The OpenSearch vector store uses the `lucene` k-NN engine with `cosinesimil` space type — compatible with all standard OpenSearch deployments including IBM watsonx.data.

## LLM Providers & Separate Judge

By default both the RAG generator and the judge use WatsonX (`WATSONX_LLM_MODEL_ID`).
Two optional features build on that:

**1. Any OpenAI-compatible chat endpoint** (watsonx.ai Model Gateway, vLLM, OpenAI, …):

```bash
LLM_PROVIDER=openai
LLM_BASE_URL=https://your-endpoint/v1
LLM_API_KEY=...
LLM_MODEL=model-name
```

This replaces the chat LLM only. **Embeddings always use WatsonX**, so `WATSONX_*` is still
required to run experiments (`ingest.py` only needs the chat LLM).

**2. A judge that differs from the system under test.** When judge and generator are the same
model, LLM-as-judge scores can be biased toward that model's own style. Set any of
`JUDGE_LLM_PROVIDER`, `JUDGE_LLM_BASE_URL`, `JUDGE_LLM_API_KEY`, `JUDGE_LLM_MODEL`; unset values
fall back to the generator's. For example, only the model differs:

```bash
JUDGE_LLM_MODEL=ibm/granite-3-3-8b-instruct
```

If the judge switches *provider*, it inherits nothing from the generator (so one provider's key is
never sent to another's endpoint). Each run prints both models and warns if they are identical.

> **Is Granite available on the watsonx.ai Model Gateway?** Not confirmed from public docs — the
> gateway needs an administrator to register models. Check against your account (replace the host
> with your region; the key is your IBM Cloud API key):
>
> ```bash
> TOKEN=$(curl -s -X POST https://iam.cloud.ibm.com/identity/token \
>   -d "grant_type=urn:ibm:params:oauth:grant-type:apikey&apikey=$WATSONX_API_KEY" | jq -r .access_token)
> curl -s -H "Authorization: Bearer $TOKEN" https://us-south.ml.cloud.ibm.com/ml/gateway/v1/models
> ```
>
> The exact model-list path and auth are from third-party docs, so treat this as a starting point
> and confirm against IBM's API reference.

## Project Structure

```
rag-ab-tester/
├── tests/                   Unit tests (no network) + opt-in integration tests
├── experiments/
│   ├── chunking.py          Experiment 1: chunk size (256 vs 512)
│   ├── retrieval.py         Experiment 2: semantic vs hybrid BM25
│   ├── embedding.py         Experiment 3: slate-125m vs e5-large embeddings
│   ├── prompt.py            Experiment 4: open vs strict grounding
│   └── custom/              Drop your own experiments here
├── data/
│   ├── docs/                Markdown corpus (edit to use your own)
│   └── test_set.json        Auto-generated QA pairs (from ingest.py)
├── results/
│   ├── best_config.json     Champion config — auto-updated after each run
│   └── *.json               Saved experiment results
├── src/
│   ├── best_config.py       Champion config store (get / save / summary)
│   ├── config.py            Validated settings + WatsonX/OpenSearch client factories
│   ├── retry.py             Exponential-backoff wrapper for network calls
│   ├── logging_config.py    CLI logging setup
│   ├── models/schemas.py    Pydantic data models
│   ├── pipeline/base.py     RAGPipeline base class
│   ├── evaluator/
│   │   ├── judge.py         WatsonxJudge (LLM-as-judge scorer)
│   │   ├── metrics.py       4 RAG metrics
│   │   └── stats.py         Paired t-test + Cohen's d + CI
│   └── report/report.py     Rich terminal tables + JSON export
├── streamlit_app/           Optional Streamlit dashboard
├── ingest.py                Generate test_set.json from docs
├── run_experiment.py        Run a single experiment
└── run_all.py               Run all experiments/
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
  "embedding_model": "intfloat/multilingual-e5-large",
  "retrieval_strategy": "hybrid",
  "prompt_template": "strict"
}
```

You can also read or write it manually:

```python
from src.best_config import get, save

chunk_size = get("chunk_size")  # e.g. 256
embedding = get("embedding_model")  # e.g. "intfloat/multilingual-e5-large"

# Override a value manually
save({"embedding_model": "ibm/slate-125m-english-rtrvr-v2"})
```

## Adding Your Own Experiment

Create a file in `experiments/custom/`:

```python
from src.pipeline.base import RAGPipeline
from src.config import get_embeddings
from src.best_config import get as best

_PROMPT = "Answer using only the context.\nContext: {context}\nQuestion: {query}\nAnswer:"


class MyControlPipeline(RAGPipeline):
    def get_chunk_size(self):
        return (best("chunk_size"), best("chunk_overlap"))

    def get_embeddings(self):
        return get_embeddings(best("embedding_model"))

    def get_prompt(self, query, context):
        return _PROMPT.format(context=context, query=query)


class MyChallengerPipeline(RAGPipeline):
    def get_chunk_size(self):
        return (best("chunk_size"), best("chunk_overlap"))

    def get_embeddings(self):
        return get_embeddings("intfloat/multilingual-e5-large")

    def get_prompt(self, query, context):
        return _PROMPT.format(context=context, query=query)


EXPERIMENT_NAME = "My Experiment"
CONTROL = MyControlPipeline
CHALLENGER = MyChallengerPipeline
CONTROL_NAME = "baseline"
CHALLENGER_NAME = "e5-large"

# Optional: auto-save the winner to best_config.json
CHAMPION_CONFIG = {
    "baseline": {"embedding_model": "ibm/slate-125m-english-rtrvr-v2"},
    "e5-large": {"embedding_model": "intfloat/multilingual-e5-large"},
}
```

Run it:

```bash
uv run python run_experiment.py --experiment experiments/custom/my_experiment.py --save-json
```

Each experiment gets two isolated OpenSearch indices (`ctrl_<name>` and `chal_<name>`) so variants never share state.

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
# Generate test set
uv run python ingest.py [--docs-path data/docs] [--num-questions 20] [--output data/test_set.json] [--seed N] [-v]

# Single experiment
uv run python run_experiment.py --experiment <path> [--test-set data/test_set.json] [--save-json] [--keep-indices] [-v]

# All experiments
uv run python run_all.py [--include chunking prompt] [--exclude embedding] [--save-json] [--keep-indices] [-v]
```

Exit codes: `0` success, `1` experiment/config failure (so CI and schedulers can detect it).

## Reliability Notes

- **Retries:** WatsonX and OpenSearch calls retry with exponential backoff on transient errors; bad input and configuration errors fail immediately.
- **Failed queries:** a query that still fails is scored 0.0 and logged. If more than 50% of a variant's queries fail, the experiment aborts rather than report skewed statistics.
- **Judge fallback:** if the LLM judge returns no parseable score it falls back to a neutral 0.5 and logs a warning.
- **Index hygiene:** each run deletes any stale index first and removes its indices afterwards (pass `--keep-indices` to inspect them).
- **Logging:** `-v` for debug output, or set `LOG_LEVEL`.

## Docker

The image runs the CLI only (the Streamlit app is a sample and is not containerised).

```bash
docker build -t rag-ab-tester .
docker run --rm --env-file .env -v "$PWD/results:/app/results" rag-ab-tester \
    python run_experiment.py --experiment experiments/chunking.py --save-json

# Local OpenSearch for development (WatsonX credentials still come from .env)
docker compose up -d opensearch
docker compose run --rm app python ingest.py
docker compose run --rm app python run_all.py --save-json
```

Credentials are passed at runtime and never baked into the image. The compose file uses a
self-signed demo node with `OPENSEARCH_VERIFY_SSL=false` — do not do that against a real cluster.

## Sample Dashboard (optional)

```bash
uv sync --extra ui
uv run streamlit run streamlit_app/app.py
```

This is a local demo with no authentication; do not expose it publicly.

## Development

```bash
uv sync --group dev
uv run pre-commit install
uv run ruff check . && uv run ruff format --check . && uv run mypy .
uv run pytest                          # unit tests, no network needed
RUN_INTEGRATION=1 uv run pytest -m integration   # live WatsonX/OpenSearch smoke tests
```

See [CONTRIBUTING.md](CONTRIBUTING.md). CI runs lint, type checks, tests (Python 3.10–3.12), a secret scan and a Docker build.

## License

[MIT](LICENSE)
