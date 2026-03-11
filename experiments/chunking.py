"""
Experiment 1 — Chunk Size
Hypothesis: larger chunks provide more complete context, improving recall
            at the cost of retrieval precision.
"""
from langchain_ollama import OllamaEmbeddings
from src.pipeline.base import RAGPipeline

_PROMPT = (
    "Answer using only the context below.\n"
    "Context: {context}\n"
    "Question: {query}\n"
    "Answer:"
)


class SmallChunkPipeline(RAGPipeline):
    """256-token chunks, 25-token overlap."""

    def get_chunk_size(self):
        return (256, 25)

    def get_embeddings(self):
        return OllamaEmbeddings(model="nomic-embed-text")

    def get_prompt(self, query, context):
        return _PROMPT.format(context=context, query=query)


class LargeChunkPipeline(RAGPipeline):
    """512-token chunks, 50-token overlap."""

    def get_chunk_size(self):
        return (512, 50)

    def get_embeddings(self):
        return OllamaEmbeddings(model="nomic-embed-text")

    def get_prompt(self, query, context):
        return _PROMPT.format(context=context, query=query)


# ── Experiment registration ─────────────────────────────────────────────── #
EXPERIMENT_NAME = "Chunk Size"
CONTROL = SmallChunkPipeline
CHALLENGER = LargeChunkPipeline
CONTROL_NAME = "Small-256"
CHALLENGER_NAME = "Large-512"
