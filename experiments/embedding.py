"""
Experiment 3 — Embedding Model
Hypothesis: mxbai-embed-large produces richer semantic representations than
            nomic-embed-text, improving retrieval quality.

Prereq: ollama pull mxbai-embed-large
"""
from langchain_ollama import OllamaEmbeddings
from src.pipeline.base import RAGPipeline

_PROMPT = (
    "Answer using only the context below.\n"
    "Context: {context}\n"
    "Question: {query}\n"
    "Answer:"
)


class NomicEmbedPipeline(RAGPipeline):
    """nomic-embed-text — fast, lightweight 768-dim embeddings."""

    def get_chunk_size(self):
        return (512, 50)

    def get_embeddings(self):
        return OllamaEmbeddings(model="nomic-embed-text")

    def get_prompt(self, query, context):
        return _PROMPT.format(context=context, query=query)


class MxbaiEmbedPipeline(RAGPipeline):
    """mxbai-embed-large — higher-capacity 1024-dim embeddings."""

    def get_chunk_size(self):
        return (512, 50)

    def get_embeddings(self):
        return OllamaEmbeddings(model="mxbai-embed-large")

    def get_prompt(self, query, context):
        return _PROMPT.format(context=context, query=query)


# ── Experiment registration ─────────────────────────────────────────────── #
EXPERIMENT_NAME = "Embedding Model"
CONTROL = NomicEmbedPipeline
CHALLENGER = MxbaiEmbedPipeline
CONTROL_NAME = "nomic-embed-text"
CHALLENGER_NAME = "mxbai-embed-large"
