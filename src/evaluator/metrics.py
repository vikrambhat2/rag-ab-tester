from __future__ import annotations
from typing import List
from src.evaluator.judge import OllamaJudge


def faithfulness_score(answer: str, context_chunks: List[str], judge: OllamaJudge) -> float:
    """
    Measures whether the answer is grounded in the retrieved context.
    High score = no hallucination relative to the provided chunks.
    """
    context = "\n\n".join(context_chunks)
    return judge.score_faithfulness(answer, context)


def answer_relevance_score(question: str, answer: str, judge: OllamaJudge) -> float:
    """
    Measures whether the answer addresses the question.
    High score = directly answers what was asked.
    """
    return judge.score_answer_relevance(question, answer)


def context_precision_score(
    question: str, context_chunks: List[str], judge: OllamaJudge
) -> float:
    """
    Measures the fraction of retrieved chunks that are relevant.
    High score = retriever returned mostly useful chunks (low noise).
    """
    return judge.score_context_precision(question, context_chunks)


def context_recall_score(
    question: str,
    context_chunks: List[str],
    ground_truth: str,
    judge: OllamaJudge,
) -> float:
    """
    Measures whether the retrieved context covers the ground truth.
    High score = enough information was retrieved to answer correctly.
    """
    return judge.score_context_recall(question, context_chunks, ground_truth)
