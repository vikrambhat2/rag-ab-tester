"""
Experiment 2 — Retrieval Strategy
Hypothesis: hybrid BM25+vector retrieval outperforms pure semantic search
            on terminology-heavy queries by adding keyword-matching signal.
"""
from langchain_ollama import OllamaEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.pipeline.base import RAGPipeline

_PROMPT = (
    "Answer using only the context below.\n"
    "Context: {context}\n"
    "Question: {query}\n"
    "Answer:"
)


class SemanticPipeline(RAGPipeline):
    """Pure vector (cosine similarity) retrieval."""

    def get_chunk_size(self):
        return (512, 50)

    def get_embeddings(self):
        return OllamaEmbeddings(model="nomic-embed-text")

    def get_prompt(self, query, context):
        return _PROMPT.format(context=context, query=query)


class HybridPipeline(RAGPipeline):
    """
    Hybrid retrieval: 60% vector + 40% BM25 keyword matching.
    Overrides ingest() to also build a BM25 index and retrieve()
    to use EnsembleRetriever.
    """

    def get_chunk_size(self):
        return (512, 50)

    def get_embeddings(self):
        return OllamaEmbeddings(model="nomic-embed-text")

    def get_prompt(self, query, context):
        return _PROMPT.format(context=context, query=query)

    def ingest(self, docs_path: str = "data/docs"):
        # Standard vector ingest
        super().ingest(docs_path)
        # Also build BM25 index from the same chunks
        loader = DirectoryLoader(docs_path, glob="**/*.md")
        splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
        self.raw_chunks = splitter.split_documents(loader.load())

    def retrieve(self, query: str, k: int = 3):
        if self.vectorstore is None:
            raise RuntimeError("Call ingest() before retrieve().")
        vector_ret = self.vectorstore.as_retriever(search_kwargs={"k": k})
        bm25_ret = BM25Retriever.from_documents(self.raw_chunks, k=k)
        ensemble = EnsembleRetriever(
            retrievers=[bm25_ret, vector_ret],
            weights=[0.4, 0.6],
        )
        docs = ensemble.invoke(query)
        return [doc.page_content for doc in docs[:k]]


# ── Experiment registration ─────────────────────────────────────────────── #
EXPERIMENT_NAME = "Retrieval Strategy"
CONTROL = SemanticPipeline
CHALLENGER = HybridPipeline
CONTROL_NAME = "Semantic"
CHALLENGER_NAME = "Hybrid-BM25"
