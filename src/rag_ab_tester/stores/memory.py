"""Dependency-free in-process index (cosine similarity). Good for tests and quick trials."""

from __future__ import annotations

import numpy as np

from rag_ab_tester.protocols import Embedder


def _normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class InMemoryIndex:
    """Brute-force cosine-similarity index held in memory; needs no service or extra."""

    def __init__(self) -> None:
        self._texts: list[str] = []
        self._matrix: np.ndarray | None = None
        self._embedder: Embedder | None = None

    def add(self, texts: list[str], embedder: Embedder) -> None:
        if not texts:
            raise ValueError("No texts to index.")
        self._matrix = _normalise(np.asarray(embedder.embed_documents(texts), dtype=float))
        self._texts = list(texts)
        self._embedder = embedder

    def search(self, query: str, k: int) -> list[str]:
        if self._matrix is None or self._embedder is None:
            raise RuntimeError("Index is empty; call add() first.")
        q = _normalise(np.asarray(self._embedder.embed_query(query), dtype=float))
        order = np.argsort(-(self._matrix @ q))[:k]
        return [self._texts[i] for i in order]

    def delete(self) -> None:
        self._texts, self._matrix, self._embedder = [], None, None
