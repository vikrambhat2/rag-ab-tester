# RAG A/B Tester

A reusable A/B testing framework for RAG pipelines.
Plug in any two variants, get statistical proof.
Fully local — powered by [Ollama](https://ollama.com) and [ChromaDB](https://www.trychroma.com). No API keys required.

## Quick Start

```bash
# 1. Install Ollama and pull models
ollama pull llama3.2
ollama pull nomic-embed-text
ollama pull mxbai-embed-large   # only needed for the embedding experiment

# 2. Install Python dependencies
pip install -r requirements.txt

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

## Project Structure

```
rag-ab-tester/
├── experiments/
│   ├── chunking.py          Experiment 1: chunk size (256 vs 512)
│   ├── retrieval.py         Experiment 2: semantic vs hybrid BM25
│   ├── embedding.py         Experiment 3: nomic-embed-text vs mxbai-embed-large
│   ├── prompt.py            Experiment 4: conversational vs expert-QA
│   └── custom/              Drop your own experiments here
├── data/
│   ├── docs/                Markdown corpus (edit to use your own)
│   └── test_set.json        Auto-generated QA pairs (from ingest.py)
├── results/
│   ├── best_config.json     Champion config — auto-updated after each run
│   └── *.json               Saved experiment results
├── src/
│   ├── best_config.py       Champion config store (get / save / summary)
│   ├── config.py            Ollama client helpers (get_llm, get_embeddings)
│   ├── models/schemas.py    Pydantic data models
│   ├── pipeline/base.py     RAGPipeline base class
│   ├── evaluator/
│   │   ├── judge.py         OllamaJudge (LLM-as-judge scorer)
│   │   ├── metrics.py       4 RAG metrics
│   │   └── stats.py         Paired t-test + Cohen's d + CI
│   └── report/report.py     Rich terminal tables + JSON export
├── streamlit_app/           Optional Streamlit dashboard
├── ingest.py                Generate test_set.json from docs
├── run_experiment.py        Run a single experiment
└── run_all.py               Run all experiments/
```

## Champion Config — How It Works

Each experiment saves the winning hyperparameters to `results/best_config.json`.
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

After running all four, `results/best_config.json` looks like:

```json
{
  "chunk_size": 256,
  "chunk_overlap": 25,
  "embedding_model": "mxbai-embed-large",
  "retrieval_strategy": "hybrid",
  "prompt_template": "expert-qa"
}
```

You can also read or write it manually:

```python
from src.best_config import get, save

chunk_size = get("chunk_size")       # e.g. 256
embedding  = get("embedding_model")  # e.g. "mxbai-embed-large"

save({"embedding_model": "nomic-embed-text"})  # override manually
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
    def get_embeddings(self):   return get_embeddings("mxbai-embed-large")
    def get_prompt(self, query, context): return _PROMPT.format(context=context, query=query)

EXPERIMENT_NAME  = "My Experiment"
CONTROL          = MyControlPipeline
CHALLENGER       = MyChallengerPipeline
CONTROL_NAME     = "baseline"
CHALLENGER_NAME  = "mxbai"

# Optional: auto-save the winner to best_config.json
CHAMPION_CONFIG = {
    "baseline": {"embedding_model": "nomic-embed-text"},
    "mxbai":    {"embedding_model": "mxbai-embed-large"},
}
```

Run it:

```bash
python run_experiment.py --experiment experiments/custom/my_experiment.py --save-json
```

Each experiment gets two isolated ChromaDB collections (`ctrl_<name>` and `chal_<name>`)
so variants never share state.

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

## Environment Variables (optional overrides)

```bash
OLLAMA_LLM_MODEL=llama3.2          # default LLM
OLLAMA_EMBED_MODEL=nomic-embed-text # default embedding model
OLLAMA_BASE_URL=http://localhost:11434
```
