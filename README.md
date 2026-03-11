# RAG A/B Tester

A reusable A/B testing framework for RAG pipelines.
Plug in any two variants, get statistical proof.
Fully local — powered by [Ollama](https://ollama.com). No API keys required.

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

# 4. Run a single experiment
python run_experiment.py --experiment experiments/chunking.py

# 5. Run all experiments
python run_all.py
```

## Project Structure

```
rag-ab-tester/
├── experiments/
│   ├── chunking.py          Experiment 1: chunk size (256 vs 512)
│   ├── retrieval.py         Experiment 2: semantic vs hybrid BM25
│   ├── embedding.py         Experiment 3: nomic vs mxbai embeddings
│   ├── prompt.py            Experiment 4: open vs strict grounding
│   └── custom/              Drop your own experiments here
├── data/
│   ├── docs/                Markdown corpus (edit to use your own)
│   └── test_set.json        Auto-generated QA pairs (from ingest.py)
├── src/
│   ├── models/schemas.py    Pydantic data models
│   ├── pipeline/base.py     RAGPipeline base class
│   ├── evaluator/
│   │   ├── judge.py         OllamaJudge (LLM-as-judge scorer)
│   │   ├── metrics.py       4 RAG metrics
│   │   └── stats.py         Paired t-test + Cohen's d + CI
│   └── report/report.py     Rich terminal tables + JSON export
├── ingest.py                Generate test_set.json from docs
├── run_experiment.py        Run a single experiment
└── run_all.py               Run all experiments/
```

## Adding Your Own Experiment

Create a file in `experiments/custom/`:

```python
from src.pipeline.base import RAGPipeline
from langchain_ollama import OllamaEmbeddings

_PROMPT = "Answer using only the context.\nContext: {context}\nQuestion: {query}\nAnswer:"

class MyControlPipeline(RAGPipeline):
    def get_chunk_size(self): return (512, 50)
    def get_embeddings(self): return OllamaEmbeddings(model="nomic-embed-text")
    def get_prompt(self, query, context): return _PROMPT.format(context=context, query=query)

class MyChallengerPipeline(RAGPipeline):
    def get_chunk_size(self): return (1024, 100)
    def get_embeddings(self): return OllamaEmbeddings(model="nomic-embed-text")
    def get_prompt(self, query, context): return _PROMPT.format(context=context, query=query)

EXPERIMENT_NAME = "My Experiment"
CONTROL = MyControlPipeline
CHALLENGER = MyChallengerPipeline
CONTROL_NAME = "current-512"
CHALLENGER_NAME = "candidate-1024"
```

Run it:

```bash
python run_experiment.py --experiment experiments/custom/my_experiment.py
```

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
