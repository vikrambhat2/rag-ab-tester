"""
Experiment 2 — Retrieval Strategy
Hypothesis: hybrid BM25+vector retrieval outperforms pure semantic search
            on terminology-heavy queries by adding keyword-matching signal.

Chunk size and embedding model are read from results/best_config.json.
Falls back to defaults if upstream experiments have not been run yet.
"""

from typing import Any

from rag_ab_tester.best_config import get as best
from rag_ab_tester.config import get_embeddings, import_optional
from rag_ab_tester.pipeline.base import RAGPipeline

_PROMPT = "Answer using only the context below.\nContext: {context}\nQuestion: {query}\nAnswer:"


class SemanticPipeline(RAGPipeline):
    """Pure vector (cosine similarity) retrieval."""

    def get_chunk_size(self):
        return (best("chunk_size"), best("chunk_overlap"))

    def get_embeddings(self):
        return get_embeddings(best("embedding_model") or None)

    def get_prompt(self, query, context):
        return _PROMPT.format(context=context, query=query)


class HybridPipeline(RAGPipeline):
    """
    Hybrid retrieval: vector results + BM25 keyword matching merged by
    deduplication (BM25 candidates first, vector fills remaining slots).
    """

    def get_chunk_size(self):
        return (best("chunk_size"), best("chunk_overlap"))

    def get_embeddings(self):
        return get_embeddings(best("embedding_model") or None)

    def get_prompt(self, query, context):
        return _PROMPT.format(context=context, query=query)

    _bm25: Any = None  # built lazily on first query, then reused

    def ingest_texts(self, texts):
        # Standard vector ingest, then keep the same chunks for the BM25 keyword index
        super().ingest_texts(texts)
        self.raw_chunks = list(texts)
        self._bm25 = None

    def retrieve(self, query: str, k: int = 3):
        vector_chunks = super().retrieve(query, k)  # semantic; checks ingest() was called
        if self._bm25 is None:
            bm25 = import_optional("rank_bm25", "docs")
            self._bm25 = bm25.BM25Okapi([chunk.split() for chunk in self.raw_chunks])
        bm25_chunks = self._bm25.get_top_n(query.split(), self.raw_chunks, n=k)
        # Merge: BM25 first (keyword signal), then fill with vector results
        merged = list(dict.fromkeys(bm25_chunks + vector_chunks))
        return merged[:k]


# ── Experiment registration ─────────────────────────────────────────────── #
EXPERIMENT_NAME = "Retrieval Strategy"
CONTROL = SemanticPipeline
CHALLENGER = HybridPipeline
CONTROL_NAME = "Semantic"
CHALLENGER_NAME = "Hybrid-BM25"

CHAMPION_CONFIG = {
    "Semantic": {"retrieval_strategy": "semantic"},
    "Hybrid-BM25": {"retrieval_strategy": "hybrid"},
}
