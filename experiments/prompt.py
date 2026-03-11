"""
Experiment 4 — Prompt Template
Hypothesis: strict grounding instructions reduce hallucination (faithfulness)
            at the cost of answer fluency (answer relevance).
"""
from langchain_ollama import OllamaEmbeddings
from src.pipeline.base import RAGPipeline

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
        return (512, 50)

    def get_embeddings(self):
        return OllamaEmbeddings(model="nomic-embed-text")

    def get_prompt(self, query, context):
        return _OPEN_PROMPT.format(context=context, query=query)


class StrictPromptPipeline(RAGPipeline):
    """Strictly grounded — must stay within the retrieved context."""

    def get_chunk_size(self):
        return (512, 50)

    def get_embeddings(self):
        return OllamaEmbeddings(model="nomic-embed-text")

    def get_prompt(self, query, context):
        return _STRICT_PROMPT.format(context=context, query=query)


# ── Experiment registration ─────────────────────────────────────────────── #
EXPERIMENT_NAME = "Prompt Template"
CONTROL = OpenPromptPipeline
CHALLENGER = StrictPromptPipeline
CONTROL_NAME = "Open Prompt"
CHALLENGER_NAME = "Strict Grounding"
