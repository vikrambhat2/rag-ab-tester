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

# ── Challenger ────────────────────────────────────────────────────────────── #
# Expert QA system — XML-delimited context, strict abstention string,
# no reasoning, verbatim grounding, no speculation.
_CHALLENGER_PROMPT = """\
You are an expert question-answering system.
You must answer the question using ONLY the information provided below.
Do NOT use prior knowledge, assumptions, or external facts.
If the information does not contain enough evidence to answer the question, \
respond exactly with: "Not answerable from the provided information."

Rules:
- Base every statement directly on the provided information.
- Do not add, infer, or speculate beyond the text.
- Do not rephrase the question.
- Be concise and factual.
- Do not explain your reasoning.

<information>
{context}
</information>

<q>{query}</q>
Answer:"""


class ConversationalPromptPipeline(RAGPipeline):
    """Step-by-step conversational assistant with NO_ANSWER abstention."""

    def get_chunk_size(self):
        return (best("chunk_size"), best("chunk_overlap"))

    def get_embeddings(self):
        return get_embeddings(best("embedding_model"))

    def get_prompt(self, query, context):
        return _CONTROL_PROMPT.format(context=context, query=query)


class ExpertQAPromptPipeline(RAGPipeline):
    """Strict XML-delimited expert QA with explicit abstention, no reasoning."""

    def get_chunk_size(self):
        return (best("chunk_size"), best("chunk_overlap"))

    def get_embeddings(self):
        return get_embeddings(best("embedding_model"))

    def get_prompt(self, query, context):
        return _CHALLENGER_PROMPT.format(context=context, query=query)


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
