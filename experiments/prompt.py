"""
Experiment 4 — Prompt Template
Hypothesis: strict grounding instructions reduce hallucination (faithfulness)
            at the cost of answer fluency (answer relevance).

Chunk size and embedding model are read from results/best_config.json.
Falls back to defaults if upstream experiments have not been run yet.
"""
from src.pipeline.base import RAGPipeline
from src.config import get_embeddings
from src.best_config import get as best

_OPEN_PROMPT = """\
Answer the question as helpfully as possible.

Context:
{context}

Question: {query}
Answer:"""

_STRICT_PROMPT = """\
Answer using ONLY the information in the context below.
Do not use prior knowledge. If the context is insufficient, say "I don't know."

Context:
{context}

Question: {query}
Answer:"""


class OpenPromptPipeline(RAGPipeline):
    """Helpful but unconstrained — may draw on parametric knowledge."""

    def get_chunk_size(self):
        return (best("chunk_size"), best("chunk_overlap"))

    def get_embeddings(self):
        return get_embeddings(best("embedding_model"))

    def get_prompt(self, query, context):
        return _OPEN_PROMPT.format(context=context, query=query)


class StrictPromptPipeline(RAGPipeline):
    """Strictly grounded — must stay within the retrieved context."""

    def get_chunk_size(self):
        return (best("chunk_size"), best("chunk_overlap"))

    def get_embeddings(self):
        return get_embeddings(best("embedding_model"))

    def get_prompt(self, query, context):
        return _STRICT_PROMPT.format(context=context, query=query)


# ── Experiment registration ─────────────────────────────────────────────── #
EXPERIMENT_NAME = "Prompt Template"
CONTROL = OpenPromptPipeline
CHALLENGER = StrictPromptPipeline
CONTROL_NAME = "Open Prompt"
CHALLENGER_NAME = "Strict Grounding"

CHAMPION_CONFIG = {
    "Open Prompt":      {"prompt_template": "open"},
    "Strict Grounding": {"prompt_template": "strict"},
}
