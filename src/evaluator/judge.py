from __future__ import annotations
import re
from langchain_ollama import ChatOllama


class OllamaJudge:
    """
    LLM-as-judge using a local Ollama model.

    All scoring prompts are structured so the model must respond
    with a single float in [0, 1] on the final line.  If parsing
    fails the call is retried once; after that it returns 0.5
    (uncertain) rather than crashing the whole experiment run.
    """

    def __init__(self, model: str = "llama3.2", temperature: float = 0):
        self.llm = ChatOllama(model=model, temperature=temperature)

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _extract_score(self, text: str) -> float | None:
        """Pull the last float-like token from the model response."""
        matches = re.findall(r"\b(0(?:\.\d+)?|1(?:\.0+)?)\b", text)
        if matches:
            return float(matches[-1])
        return None

    def score(self, prompt: str) -> float:
        """
        Send a scoring prompt and return a float in [0, 1].
        Retries once on parse failure; falls back to 0.5.
        """
        for _ in range(2):
            try:
                response = self.llm.invoke(prompt)
                score = self._extract_score(response.content)
                if score is not None:
                    return max(0.0, min(1.0, score))
            except Exception:
                pass
        return 0.5

    # ------------------------------------------------------------------ #
    #  Named scoring methods used by metrics.py                           #
    # ------------------------------------------------------------------ #

    def score_faithfulness(self, answer: str, context: str) -> float:
        prompt = f"""You are evaluating whether an AI answer is faithful to the provided context.

Context:
{context}

Answer:
{answer}

Is every claim in the answer supported by the context?
- Score 1.0 if the answer contains no information beyond the context.
- Score 0.0 if the answer contradicts or ignores the context.
- Score between 0 and 1 for partial faithfulness.

Respond with only a single decimal number between 0 and 1."""
        return self.score(prompt)

    def score_answer_relevance(self, question: str, answer: str) -> float:
        prompt = f"""You are evaluating whether an AI answer addresses the question.

Question:
{question}

Answer:
{answer}

Does the answer directly address what was asked?
- Score 1.0 if the answer fully addresses the question.
- Score 0.0 if the answer is completely off-topic.
- Score between 0 and 1 for partial relevance.

Respond with only a single decimal number between 0 and 1."""
        return self.score(prompt)

    def score_context_precision(self, question: str, context_chunks: list[str]) -> float:
        context = "\n\n---\n\n".join(
            f"Chunk {i + 1}:\n{chunk}" for i, chunk in enumerate(context_chunks)
        )
        prompt = f"""You are evaluating the precision of retrieved context chunks.

Question:
{question}

Retrieved Chunks:
{context}

What fraction of these chunks are relevant to answering the question?
- Score 1.0 if all chunks are relevant.
- Score 0.0 if no chunks are relevant.
- Score between 0 and 1 proportional to how many are relevant.

Respond with only a single decimal number between 0 and 1."""
        return self.score(prompt)

    def score_context_recall(
        self, question: str, context_chunks: list[str], ground_truth: str
    ) -> float:
        context = "\n\n---\n\n".join(
            f"Chunk {i + 1}:\n{chunk}" for i, chunk in enumerate(context_chunks)
        )
        prompt = f"""You are evaluating whether retrieved context covers the ground truth answer.

Question:
{question}

Ground Truth Answer:
{ground_truth}

Retrieved Context:
{context}

Does the retrieved context contain enough information to derive the ground truth?
- Score 1.0 if the context fully covers the ground truth.
- Score 0.0 if the context is completely missing the needed information.
- Score between 0 and 1 for partial coverage.

Respond with only a single decimal number between 0 and 1."""
        return self.score(prompt)
