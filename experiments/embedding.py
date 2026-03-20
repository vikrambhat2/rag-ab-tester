"""
Experiment 3 — Embedding Model
Hypothesis: mxbai-embed-large produces richer semantic representations than
            nomic-embed-text, improving retrieval quality.

Prereq: ollama pull mxbai-embed-large

Chunk size is read from results/best_config.json (set by chunking experiment).
Falls back to 512/50 if chunking experiment has not been run yet.
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


class NomicEmbedPipeline(RAGPipeline):
    """nomic-embed-text — fast, lightweight 768-dim embeddings."""

    def get_chunk_size(self):
        return (best("chunk_size"), best("chunk_overlap"))

    def get_embeddings(self):
        return get_embeddings("nomic-embed-text")

    def get_prompt(self, query, context):
        return _PROMPT.format(context=context, query=query)


class MxbaiEmbedPipeline(RAGPipeline):
    """mxbai-embed-large — higher-capacity 1024-dim embeddings."""

    def get_chunk_size(self):
        return (best("chunk_size"), best("chunk_overlap"))

    def get_embeddings(self):
        return get_embeddings("mxbai-embed-large")

    def get_prompt(self, query, context):
        return _PROMPT.format(context=context, query=query)


# ── Experiment registration ─────────────────────────────────────────────── #
EXPERIMENT_NAME  = "Embedding Model"
CONTROL          = NomicEmbedPipeline
CHALLENGER       = MxbaiEmbedPipeline
CONTROL_NAME     = "nomic-embed-text"
CHALLENGER_NAME  = "mxbai-embed-large"

CHAMPION_CONFIG = {
    "nomic-embed-text":  {"embedding_model": "nomic-embed-text"},
    "mxbai-embed-large": {"embedding_model": "mxbai-embed-large"},
}
