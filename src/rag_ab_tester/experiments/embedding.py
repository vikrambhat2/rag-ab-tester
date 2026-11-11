"""
Experiment 3 — Embedding Model
Hypothesis: a higher-capacity embedding model produces richer semantic representations
            than the baseline, improving retrieval quality.

Control is ``EMBED_MODEL`` and challenger is ``EMBED_MODEL_CHALLENGER`` (both served by the
same OpenAI-compatible endpoint, e.g. text-embedding-3-small vs text-embedding-3-large).

Chunk size is read from results/best_config.json (set by chunking.py).
Falls back to (512, 50) if the chunking experiment has not been run yet.
"""

from rag_ab_tester.best_config import get as best
from rag_ab_tester.config import ConfigError, Settings, get_embeddings
from rag_ab_tester.pipeline.base import RAGPipeline

_PROMPT = "Answer using only the context below.\nContext: {context}\nQuestion: {query}\nAnswer:"

_SETTINGS = Settings.from_env()
_CONTROL_MODEL = _SETTINGS.embed_model
_CHALLENGER_MODEL = _SETTINGS.embed_model_challenger


class BaselineEmbeddingPipeline(RAGPipeline):
    """Control: the baseline embedding model (``EMBED_MODEL``)."""

    def get_chunk_size(self):
        return (best("chunk_size"), best("chunk_overlap"))

    def get_embeddings(self):
        return get_embeddings(_CONTROL_MODEL or None)

    def get_prompt(self, query, context):
        return _PROMPT.format(context=context, query=query)


class ChallengerEmbeddingPipeline(RAGPipeline):
    """Challenger: the model under test (``EMBED_MODEL_CHALLENGER``)."""

    def get_chunk_size(self):
        return (best("chunk_size"), best("chunk_overlap"))

    def get_embeddings(self):
        if not _CHALLENGER_MODEL:
            raise ConfigError(
                "EMBED_MODEL_CHALLENGER is not set; the embedding experiment needs a second "
                "embedding model to compare against EMBED_MODEL."
            )
        return get_embeddings(_CHALLENGER_MODEL)

    def get_prompt(self, query, context):
        return _PROMPT.format(context=context, query=query)


# ── Experiment registration ─────────────────────────────────────────────── #
EXPERIMENT_NAME = "Embedding Model"
CONTROL = BaselineEmbeddingPipeline
CHALLENGER = ChallengerEmbeddingPipeline
CONTROL_NAME = _CONTROL_MODEL or "baseline-embedder"
CHALLENGER_NAME = _CHALLENGER_MODEL or "challenger-embedder"

# This experiment compares embedders, so run_experiment(embedder=...) is rejected for it.
VARIES_EMBEDDER = True

CHAMPION_CONFIG = {
    CONTROL_NAME: {"embedding_model": _CONTROL_MODEL},
    CHALLENGER_NAME: {"embedding_model": _CHALLENGER_MODEL},
}
