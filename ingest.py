"""
ingest.py — Generate a test set from the documents in data/docs/

Usage:
    python ingest.py [--docs-path data/docs] [--num-questions 20] [--output data/test_set.json]

What it does:
  1. Loads all .md files from data/docs/
  2. Samples random chunks as context
  3. Calls the WatsonX LLM to generate a question + ground-truth answer per chunk
  4. Writes the resulting Q&A pairs to data/test_set.json
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rich.console import Console
from rich.progress import track

from src.config import ConfigError, get_llm
from src.logging_config import setup_logging
from src.retry import with_retry

console = Console()
logger = logging.getLogger(__name__)

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
        response = with_retry(llm.invoke, prompt)
        lines = [ln.strip() for ln in response.content.strip().splitlines() if ln.strip()]
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


def main(docs_path: str, num_questions: int, output: str) -> None:
    console.print(f"[bold]Loading documents from:[/bold] {docs_path}")
    loader = DirectoryLoader(docs_path, glob="**/*.md", loader_cls=TextLoader)
    documents = loader.load()

    if not documents:
        console.print(
            "[red]No .md files found. Add markdown files to data/docs/ before running ingest.py.[/red]"
        )
        sys.exit(1)

    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    console.print(f"Split into {len(chunks)} chunks from {len(documents)} documents.")

    try:
        llm = get_llm()
    except ConfigError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(2)

    # Sample without replacement (or use all if fewer chunks than requested)
    sample_size = min(num_questions, len(chunks))
    sampled = random.sample(chunks, sample_size)

    test_cases: list[dict] = []
    for chunk in track(sampled, description="Generating QA pairs..."):
        result = generate_qa_pair(chunk.page_content, llm)
        if result:
            test_cases.append(result)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(test_cases, indent=2))

    console.print(
        f"\n[green]✓[/green] Generated [bold]{len(test_cases)}[/bold] test cases → [cyan]{output_path}[/cyan]"
    )

    if len(test_cases) < num_questions:
        console.print(
            f"[yellow]Note:[/yellow] Only {len(test_cases)}/{num_questions} generated "
            "(some chunks may have failed to parse). Add more docs for a larger test set."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a RAG test set from docs.")
    parser.add_argument("--docs-path", default="data/docs", help="Path to .md documents")
    parser.add_argument(
        "--num-questions", type=int, default=20, help="Number of QA pairs to generate"
    )
    parser.add_argument("--output", default="data/test_set.json", help="Output JSON path")
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducible sampling"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    args = parser.parse_args()

    setup_logging(args.verbose)
    if args.seed is not None:
        random.seed(args.seed)
    main(args.docs_path, args.num_questions, args.output)
