"""
Experiment 1 — Chunk Size
Hypothesis: larger chunks provide more complete context, improving recall
            at the cost of retrieval precision.
"""
from src.pipeline.base import RAGPipeline
from src.config import get_embeddings

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
        return get_embeddings()

    def get_prompt(self, query, context):
        return _PROMPT.format(context=context, query=query)


class LargeChunkPipeline(RAGPipeline):
    """512-token chunks, 50-token overlap."""

    def get_chunk_size(self):
        return (512, 50)

    def get_embeddings(self):
        return get_embeddings()

    def get_prompt(self, query, context):
        return _PROMPT.format(context=context, query=query)


# ── Experiment registration ─────────────────────────────────────────────── #
EXPERIMENT_NAME  = "Chunk Size"
CONTROL          = SmallChunkPipeline
CHALLENGER       = LargeChunkPipeline
CONTROL_NAME     = "small-256"
CHALLENGER_NAME  = "large-512"

CHAMPION_CONFIG = {
    "small-256": {"chunk_size": 256, "chunk_overlap": 25},
    "large-512": {"chunk_size": 512, "chunk_overlap": 50},
}
