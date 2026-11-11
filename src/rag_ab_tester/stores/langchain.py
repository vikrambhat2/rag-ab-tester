"""Adapter that turns any LangChain ``VectorStore`` into a :class:`VectorIndex`."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from rag_ab_tester.protocols import Embedder

logger = logging.getLogger(__name__)


class LangChainVectorStoreIndex:
    """
    Wrap any LangChain vector store (Chroma, FAISS, pgvector, Qdrant, Pinecone, ...).

    ``build(texts, embedder)`` must create the store *and* populate it. Give each index its own
    collection/table name so variants stay isolated. ``delete(store)`` removes it; if omitted,
    ``store.delete_collection()`` is used when the store has one (Chroma does), otherwise a
    warning is logged (fine for in-process stores such as FAISS).

    Example::

        from langchain_chroma import Chroma

        def factory(index_name):
            return LangChainVectorStoreIndex(
                build=lambda texts, emb: Chroma.from_texts(
                    texts, emb, collection_name=index_name
                ),
            )

        run_experiment("chunking", index_factory=factory, embedder=my_embeddings)
    """

    def __init__(
        self,
        build: Callable[[list[str], Embedder], Any],
        delete: Callable[[Any], None] | None = None,
    ):
        self._build = build
        self._delete = delete
        self._store: Any = None

    def add(self, texts: list[str], embedder: Embedder) -> None:
        self._store = self._build(texts, embedder)

    def search(self, query: str, k: int) -> list[str]:
        if self._store is None:
            raise RuntimeError("Index is empty; call add() first.")
        return [d.page_content for d in self._store.similarity_search(query, k=k)]

    def delete(self) -> None:
        store, self._store = self._store, None
        if store is None:
            return
        if self._delete is not None:
            self._delete(store)
        elif hasattr(store, "delete_collection"):
            store.delete_collection()
        else:
            logger.warning(
                "%s has no delete_collection(); pass delete= to clean it up.", type(store).__name__
            )
