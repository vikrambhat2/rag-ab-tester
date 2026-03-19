"""
Experiment 3 — Embedding Model
Hypothesis: intfloat/multilingual-e5-large produces richer semantic
            representations than ibm/slate-125m-english-rtrvr-v2,
            improving retrieval quality.

Chunk size is read from results/best_config.json (set by chunking.py).
Falls back to (512, 50) if the chunking experiment has not been run yet.

Both models are served via WatsonX AI.
"""
from src.pipeline.base import RAGPipeline
from src.config import get_embeddings
from src.best_config import get as best

_PROMPT = (
    "Answer using only the context below.\n"
    "Context: {context}\n"
    "Question: {query}\n"
    "Answer:"
)


class Slate125mPipeline(RAGPipeline):
    """ibm/slate-125m-english-rtrvr-v2 — control baseline."""

    def get_chunk_size(self):
        return (best("chunk_size"), best("chunk_overlap"))

    def get_embeddings(self):
        return get_embeddings("ibm/slate-125m-english-rtrvr-v2")

    def get_prompt(self, query, context):
        return _PROMPT.format(context=context, query=query)


class E5LargePipeline(RAGPipeline):
    """intfloat/multilingual-e5-large — multilingual, higher-capacity challenger."""

    def get_chunk_size(self):
        return (best("chunk_size"), best("chunk_overlap"))

    def get_embeddings(self):
        return get_embeddings("intfloat/multilingual-e5-large")

    def get_prompt(self, query, context):
        return _PROMPT.format(context=context, query=query)


# ── Experiment registration ─────────────────────────────────────────────── #
EXPERIMENT_NAME = "Embedding Model"
CONTROL = Slate125mPipeline
CHALLENGER = E5LargePipeline
CONTROL_NAME = "slate-125m-v2"
CHALLENGER_NAME = "e5-large"

CHAMPION_CONFIG = {
    "slate-125m-v2": {"embedding_model": "ibm/slate-125m-english-rtrvr-v2"},
    "e5-large":      {"embedding_model": "intfloat/multilingual-e5-large"},
}
