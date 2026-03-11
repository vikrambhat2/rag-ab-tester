"""
Custom Experiment Template
Replace the pipeline definitions below with your own variants.

Run with:
    python run_experiment.py --experiment experiments/custom/example_experiment.py
"""
from langchain_ollama import OllamaEmbeddings
from src.pipeline.base import RAGPipeline

_PROMPT = (
    "Answer using only the context below.\n"
    "Context: {context}\n"
    "Question: {query}\n"
    "Answer:"
)


class ControlPipeline(RAGPipeline):
    """Your current production configuration."""

    def get_chunk_size(self):
        return (512, 50)

    def get_embeddings(self):
        return OllamaEmbeddings(model="nomic-embed-text")

    def get_prompt(self, query, context):
        return _PROMPT.format(context=context, query=query)


class ChallengerPipeline(RAGPipeline):
    """Your proposed new configuration."""

    def get_chunk_size(self):
        return (1024, 100)

    def get_embeddings(self):
        return OllamaEmbeddings(model="nomic-embed-text")

    def get_prompt(self, query, context):
        return _PROMPT.format(context=context, query=query)


# ── Experiment registration (required) ─────────────────────────────────── #
EXPERIMENT_NAME = "Custom Experiment"
CONTROL = ControlPipeline
CHALLENGER = ChallengerPipeline
CONTROL_NAME = "Current-512"
CHALLENGER_NAME = "Candidate-1024"
