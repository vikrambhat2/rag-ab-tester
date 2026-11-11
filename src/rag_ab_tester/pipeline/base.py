from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rag_ab_tester.config import import_optional
from rag_ab_tester.protocols import ChatModel, Embedder, VectorIndex, as_chat_model, to_text
from rag_ab_tester.retry import with_retry
from rag_ab_tester.stores.memory import InMemoryIndex

logger = logging.getLogger(__name__)


def load_and_split(docs_path: str | Path, chunk_size: int, overlap: int) -> list[str]:
    """Read every ``*.md`` under ``docs_path`` (UTF-8) and split it into text chunks."""
    root = Path(docs_path)
    documents = [
        text
        for f in (sorted(root.glob("**/*.md")) if root.is_dir() else [])
        if (text := f.read_text(encoding="utf-8")).strip()
    ]
    if not documents:
        raise ValueError(
            f"No markdown documents found in '{docs_path}'. Add .md files there first."
        )
    splitters = import_optional("langchain_text_splitters", "docs")
    splitter = splitters.RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap
    )
    return [chunk for text in documents for chunk in splitter.split_text(text)]


class RAGPipeline(ABC):
    """
    Base class for all RAG pipeline variants.

    Subclasses must implement two methods:
        get_chunk_size()  → (chunk_size, overlap) tuple
        get_prompt()      → format the final prompt string

    and may override ``get_embeddings()`` (defaults to the configured embeddings endpoint).

    Everything provider-specific is injectable, so the pipeline is LLM- and vector-store
    agnostic (see :mod:`rag_ab_tester.protocols`):

        llm       any chat model with ``invoke(prompt)`` (default: ``get_llm()``)
        embedder  any object with ``embed_documents``/``embed_query``; overrides
                  ``get_embeddings()`` (default: ``get_embeddings()``)
        index     any :class:`~rag_ab_tester.protocols.VectorIndex` (default: in-memory)

    Override ``ingest_texts()`` or ``retrieve()`` for structural changes (e.g. hybrid
    retrieval). Each pipeline owns an isolated index so control and challenger never
    share state.
    """

    def __init__(
        self,
        index_name: str,
        *,
        llm: Any = None,
        embedder: Embedder | None = None,
        index: VectorIndex | None = None,
    ):
        from rag_ab_tester.config import get_llm

        self.index_name = index_name
        self.llm: ChatModel = as_chat_model(llm) if llm is not None else get_llm()
        self._embedder = embedder
        self.index: VectorIndex = index if index is not None else InMemoryIndex()
        self.ready = False

    # ------------------------------------------------------------------ #
    #  Interface                                                           #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def get_chunk_size(self) -> tuple[int, int]:
        """Return (chunk_size, chunk_overlap)."""

    @abstractmethod
    def get_prompt(self, query: str, context: str) -> str:
        """Format and return the prompt string."""

    def get_embeddings(self) -> Embedder:
        """Embedding model for this variant (default: ``EMBED_*`` settings). Override to compare."""
        from rag_ab_tester.config import get_embeddings

        return get_embeddings()

    # ------------------------------------------------------------------ #
    #  Default implementations (override for structural experiments)       #
    # ------------------------------------------------------------------ #

    def ingest(self, docs_path: str | Path) -> None:
        """Load markdown docs, split into chunks, embed, and index."""
        chunk_size, overlap = self.get_chunk_size()
        self.ingest_texts(load_and_split(docs_path, chunk_size, overlap))

    def ingest_texts(self, texts: Sequence[str]) -> None:
        """
        Embed and index already-chunked text (skips loading and splitting, so any document
        source works; ``get_chunk_size`` is not applied).
        """
        chunks = list(texts)
        if not chunks:
            raise ValueError("No text chunks to index.")
        embedder = self._embedder if self._embedder is not None else self.get_embeddings()

        self.cleanup()  # drop any stale index from a previous (possibly aborted) run
        logger.info("Indexing %d chunks into '%s'", len(chunks), self.index_name)
        with_retry(self.index.add, chunks, embedder)
        self.ready = True

    def cleanup(self) -> None:
        """Delete this pipeline's index. Safe to call more than once; never raises."""
        try:
            self.index.delete()
        except Exception as exc:
            logger.warning("Could not delete index '%s': %s", self.index_name, exc)
        self.ready = False

    def retrieve(self, query: str, k: int = 3) -> list[str]:
        """Return top-k context chunks for a query."""
        if not self.ready:
            raise RuntimeError("Call ingest() before retrieve().")
        return with_retry(self.index.search, query, k)

    def query(self, question: str) -> tuple[str, list[str]]:
        """Run the full RAG pipeline and return (answer, context_chunks)."""
        chunks = self.retrieve(question)
        context = "\n\n".join(chunks)
        prompt = self.get_prompt(question, context)
        return to_text(with_retry(self.llm.invoke, prompt)), chunks
