"""
Generate a question/answer test set from a directory of markdown documents.

1. Loads all .md files under ``docs_path``
2. Samples random chunks as context
3. Asks the generator LLM for one question + ground-truth answer per chunk
4. Writes the Q&A pairs to a JSON file
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import track

from rag_ab_tester.config import get_llm
from rag_ab_tester.pipeline.base import load_and_split
from rag_ab_tester.protocols import as_chat_model, to_text
from rag_ab_tester.retry import with_retry

console = Console()
logger = logging.getLogger(__name__)


class IngestError(RuntimeError):
    """Raised when a test set cannot be generated."""


_QA_PROMPT = """\
You are creating a question-answering dataset for evaluating a RAG system.

Based on the following context, generate exactly ONE factual question and a concise ground-truth answer.
The question must be answerable solely from the context.

Context:
{context}

Respond in this exact format (two lines, nothing else):
QUESTION: <your question here>
ANSWER: <your answer here>"""


def generate_qa_pair(chunk: str, llm) -> dict | None:
    prompt = _QA_PROMPT.format(context=chunk)
    try:
        text = to_text(with_retry(llm.invoke, prompt))
        lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
        question = next(
            (ln[len("QUESTION:") :].strip() for ln in lines if ln.upper().startswith("QUESTION:")),
            None,
        )
        answer = next(
            (ln[len("ANSWER:") :].strip() for ln in lines if ln.upper().startswith("ANSWER:")), None
        )
        if question and answer:
            return {"query": question, "ground_truth": answer, "context_snippet": chunk[:200]}
    except Exception as e:
        logger.warning("QA generation failed: %s", e)
    return None


def generate_test_set(
    docs_path: str | Path,
    num_questions: int = 20,
    output: str | Path = "data/test_set.json",
    seed: int | None = None,
    llm: Any = None,
) -> list[dict]:
    """
    Generate ``num_questions`` Q&A pairs from ``docs_path`` and write them to ``output``.

    ``llm`` is any chat model with ``invoke(prompt)``; default is the configured generator.
    """
    console.print(f"[bold]Loading documents from:[/bold] {docs_path}")
    try:
        chunks = load_and_split(docs_path, chunk_size=512, overlap=50)
    except ValueError as exc:
        raise IngestError(str(exc)) from exc
    console.print(f"Split into {len(chunks)} chunks.")

    # get_llm raises ConfigError / MissingExtraError with an actionable message
    llm = as_chat_model(llm) if llm is not None else get_llm()

    # Sample without replacement (or use all if fewer chunks than requested)
    rng = random.Random(seed)
    sampled = rng.sample(chunks, min(num_questions, len(chunks)))

    test_cases: list[dict] = []
    for chunk in track(sampled, description="Generating QA pairs..."):
        result = generate_qa_pair(chunk, llm)
        if result:
            test_cases.append(result)

    if not test_cases:
        raise IngestError("The LLM produced no parseable question/answer pairs.")

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(test_cases, indent=2))

    console.print(
        f"\n[green]✓[/green] Generated [bold]{len(test_cases)}[/bold] test cases "
        f"→ [cyan]{output_path}[/cyan]"
    )
    if len(test_cases) < num_questions:
        console.print(
            f"[yellow]Note:[/yellow] Only {len(test_cases)}/{num_questions} generated "
            "(some chunks may have failed to parse). Add more docs for a larger test set."
        )
    return test_cases
