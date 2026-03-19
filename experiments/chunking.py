"""
Experiment 1 — Chunk Size
Hypothesis: larger chunks provide more complete context, improving recall
            at the cost of retrieval precision.

The winner's chunk_size + chunk_overlap are saved to results/best_config.json
and used as the fixed baseline for all downstream experiments.
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
EXPERIMENT_NAME = "Chunk Size"
CONTROL = SmallChunkPipeline
CHALLENGER = LargeChunkPipeline
CONTROL_NAME = "Small-256"
CHALLENGER_NAME = "Large-512"

# Maps each variant name → the best_config values to save if it wins
CHAMPION_CONFIG = {
    "Small-256":  {"chunk_size": 256, "chunk_overlap": 25},
    "Large-512":  {"chunk_size": 512, "chunk_overlap": 50},
}
