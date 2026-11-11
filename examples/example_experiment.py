"""
Custom Experiment Template
Replace the pipeline definitions below with your own variants.

Run with:
    rag-ab run examples/example_experiment.py
"""

from rag_ab_tester.config import get_embeddings
from rag_ab_tester.pipeline.base import RAGPipeline

_PROMPT = "Answer using only the context below.\nContext: {context}\nQuestion: {query}\nAnswer:"


class ControlPipeline(RAGPipeline):
    """Your current production configuration."""

    def get_chunk_size(self):
        return (512, 50)

    def get_embeddings(self):
        return get_embeddings()

    def get_prompt(self, query, context):
        return _PROMPT.format(context=context, query=query)


class ChallengerPipeline(RAGPipeline):
    """Your proposed new configuration."""

    def get_chunk_size(self):
        return (1024, 100)

    def get_embeddings(self):
        return get_embeddings()

    def get_prompt(self, query, context):
        return _PROMPT.format(context=context, query=query)


# ── Experiment registration (required) ─────────────────────────────────── #
EXPERIMENT_NAME = "Custom Experiment"
CONTROL = ControlPipeline
CHALLENGER = ChallengerPipeline
CONTROL_NAME = "Current-512"
CHALLENGER_NAME = "Candidate-1024"
