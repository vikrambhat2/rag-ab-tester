# RAG A/B Tester

A reusable A/B testing framework for RAG pipelines.
Plug in any two variants, get statistical proof.
Powered by [IBM WatsonX AI](https://www.ibm.com/watsonx) (LLM + embeddings) and [OpenSearch](https://opensearch.org) (vector store).

## Quick Start

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Configure credentials
cp .env.example .env
# Edit .env — add WATSONX_* and OPENSEARCH_* values

# 3. Generate the test set (uses data/docs/ by default)
python ingest.py

# 4. Run experiments in recommended order (each builds on the previous winner)
python run_experiment.py --experiment experiments/chunking.py --save-json
python run_experiment.py --experiment experiments/embedding.py --save-json
python run_experiment.py --experiment experiments/retrieval.py --save-json
python run_experiment.py --experiment experiments/prompt.py --save-json

# Or run all at once
python run_all.py --save-json
```

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
OPENSEARCH_VERIFY_SSL=true
```

> **Note:** The OpenSearch vector store uses the `lucene` k-NN engine with `cosinesimil` space type — compatible with all standard OpenSearch deployments including IBM watsonx.data.

## Project Structure

```
rag-ab-tester/
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
│   ├── config.py            WatsonX AI + OpenSearch client helpers
│   ├── models/schemas.py    Pydantic data models
│   ├── pipeline/base.py     RAGPipeline base class
│   ├── evaluator/
│   │   ├── judge.py         WatsonxJudge (LLM-as-judge scorer)
│   │   ├── metrics.py       4 RAG metrics
│   │   └── stats.py         Paired t-test + Cohen's d + CI
│   └── report/report.py     Rich terminal tables + JSON export
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

chunk_size = get("chunk_size")       # e.g. 256
embedding  = get("embedding_model")  # e.g. "intfloat/multilingual-e5-large"

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
    def get_chunk_size(self):   return (best("chunk_size"), best("chunk_overlap"))
    def get_embeddings(self):   return get_embeddings(best("embedding_model"))
    def get_prompt(self, query, context): return _PROMPT.format(context=context, query=query)

class MyChallengerPipeline(RAGPipeline):
    def get_chunk_size(self):   return (best("chunk_size"), best("chunk_overlap"))
    def get_embeddings(self):   return get_embeddings("intfloat/multilingual-e5-large")
    def get_prompt(self, query, context): return _PROMPT.format(context=context, query=query)

EXPERIMENT_NAME = "My Experiment"
CONTROL         = MyControlPipeline
CHALLENGER      = MyChallengerPipeline
CONTROL_NAME    = "baseline"
CHALLENGER_NAME = "e5-large"

# Optional: auto-save the winner to best_config.json
CHAMPION_CONFIG = {
    "baseline": {"embedding_model": "ibm/slate-125m-english-rtrvr-v2"},
    "e5-large": {"embedding_model": "intfloat/multilingual-e5-large"},
}
```

Run it:

```bash
python run_experiment.py --experiment experiments/custom/my_experiment.py --save-json
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
python ingest.py [--docs-path data/docs] [--num-questions 20] [--output data/test_set.json]

# Single experiment
python run_experiment.py --experiment <path> [--test-set data/test_set.json] [--save-json]

# All experiments
python run_all.py [--include chunking prompt] [--exclude embedding] [--save-json]
```
