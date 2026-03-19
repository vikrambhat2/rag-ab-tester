"""
Experiment 2 — Retrieval Strategy
Hypothesis: hybrid BM25+vector retrieval outperforms pure semantic search
            on terminology-heavy queries by adding keyword-matching signal.

Chunk size and embedding model are read from results/best_config.json.
Falls back to defaults if upstream experiments have not been run yet.
"""
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.pipeline.base import RAGPipeline
from src.config import get_embeddings
from src.best_config import get as best

_PROMPT = (
    "Answer using only the context below.\n"
    "Context: {context}\n"
    "Question: {query}\n"
    "Answer:"
)


class SemanticPipeline(RAGPipeline):
    """Pure vector (cosine similarity) retrieval."""

    def get_chunk_size(self):
        return (best("chunk_size"), best("chunk_overlap"))

    def get_embeddings(self):
        return get_embeddings(best("embedding_model"))

    def get_prompt(self, query, context):
        return _PROMPT.format(context=context, query=query)


class HybridPipeline(RAGPipeline):
    """
    Hybrid retrieval: vector results + BM25 keyword matching merged by
    reciprocal rank fusion (BM25 candidates first, vector fills remaining).
    """

    def get_chunk_size(self):
        return (best("chunk_size"), best("chunk_overlap"))

    def get_embeddings(self):
        return get_embeddings(best("embedding_model"))

    def get_prompt(self, query, context):
        return _PROMPT.format(context=context, query=query)

    def ingest(self, docs_path: str = "data/docs"):
        # Standard vector ingest
        super().ingest(docs_path)
        # Also build BM25 index from the same chunks
        chunk_size, chunk_overlap = best("chunk_size"), best("chunk_overlap")
        loader = DirectoryLoader(docs_path, glob="**/*.md", loader_cls=TextLoader)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self.raw_chunks = splitter.split_documents(loader.load())

    def retrieve(self, query: str, k: int = 3):
        if self.vectorstore is None:
            raise RuntimeError("Call ingest() before retrieve().")
        # Vector results (semantic)
        vector_docs = self.vectorstore.similarity_search(query, k=k)
        # BM25 results (keyword)
        bm25_ret = BM25Retriever.from_documents(self.raw_chunks, k=k)
        bm25_docs = bm25_ret.invoke(query)
        # Merge: BM25 first (keyword signal), then fill with vector results
        # Deduplicate by content while preserving order
        seen: set[str] = set()
        merged = []
        for doc in bm25_docs + vector_docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                merged.append(doc)
        return [doc.page_content for doc in merged[:k]]


# ── Experiment registration ─────────────────────────────────────────────── #
EXPERIMENT_NAME = "Retrieval Strategy"
CONTROL = SemanticPipeline
CHALLENGER = HybridPipeline
CONTROL_NAME = "Semantic"
CHALLENGER_NAME = "Hybrid-BM25"

CHAMPION_CONFIG = {
    "Semantic":    {"retrieval_strategy": "semantic"},
    "Hybrid-BM25": {"retrieval_strategy": "hybrid"},
}
