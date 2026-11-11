"""
Run an A/B experiment with your own LLM, embeddings and vector store - no hosted service or
credentials involved. This script uses tiny in-process stand-ins so it runs offline:

    python examples/sdk/bring_your_own_providers.py

Swap them for real objects, e.g. ``ChatAnthropic(...)``, ``OllamaEmbeddings(...)`` and
``LangChainVectorStoreIndex(...)`` (see the README).
"""

import hashlib
import json
import tempfile
from pathlib import Path

from rag_ab_tester import InMemoryIndex, RAGPipeline, run_experiment


class ToyEmbedder:
    """Bag-of-words hashing embedder (any object with these two methods works)."""

    def _vec(self, text):
        v = [0.0] * 64
        for word in text.lower().split():
            v[int(hashlib.md5(word.encode()).hexdigest(), 16) % 64] += 1.0
        return v

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


def generator(prompt: str) -> str:  # any str -> str callable is a valid chat model
    return "Cats purr when they are happy."


def judge(prompt: str) -> str:
    return "0.8"


class Small(RAGPipeline):
    def get_chunk_size(self):
        return (80, 0)

    def get_prompt(self, query, context):
        return f"Context: {context}\nQuestion: {query}\nAnswer:"


class Large(Small):
    def get_chunk_size(self):
        return (400, 0)


EXPERIMENT_NAME = "Toy Chunk Size"
CONTROL, CHALLENGER = Small, Large
CONTROL_NAME, CHALLENGER_NAME = "small", "large"

if __name__ == "__main__":
    work = Path(tempfile.mkdtemp())
    (work / "docs").mkdir()
    (work / "docs" / "cats.md").write_text("Cats purr when they are happy. " * 40)
    cases = [
        {"query": f"Why do cats purr? ({i})", "ground_truth": "They are happy."} for i in range(6)
    ]
    (work / "test_set.json").write_text(json.dumps(cases))

    result = run_experiment(
        Path(__file__),  # this file defines CONTROL / CHALLENGER
        test_set_path=work / "test_set.json",
        docs_path=work / "docs",
        llm=generator,
        judge_llm=judge,
        embedder=ToyEmbedder(),
        index_factory=lambda index_name: InMemoryIndex(),
    )
    print("winner:", result.overall_winner)
