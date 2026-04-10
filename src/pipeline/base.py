from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple, List

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class RAGPipeline(ABC):
    """
    Base class for all RAG pipeline variants.

    Subclasses must implement three methods:
        get_embeddings()  → which embedding model to use
        get_chunk_size()  → (chunk_size, overlap) tuple
        get_prompt()      → format the final prompt string

    Override `ingest()` or `retrieve()` for structural changes
    (e.g. hybrid retrieval).

    Each pipeline is scoped to an isolated OpenSearch index so control
    and challenger never share state.
    """

    def __init__(self, index_name: str):
        from src.config import get_llm
        self.index_name = index_name
        self.llm = get_llm()
        self.vectorstore = None

    # ------------------------------------------------------------------ #
    #  Abstract interface                                                  #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def get_embeddings(self):
        """Return a LangChain Embeddings instance for this variant."""

    @abstractmethod
    def get_chunk_size(self) -> Tuple[int, int]:
        """Return (chunk_size, chunk_overlap)."""

    @abstractmethod
    def get_prompt(self, query: str, context: str) -> str:
        """Format and return the prompt string."""

    # ------------------------------------------------------------------ #
    #  Default implementations (override for structural experiments)       #
    # ------------------------------------------------------------------ #

    def ingest(self, docs_path: str = "data/docs") -> None:
        """Load docs, split into chunks, embed, and index into OpenSearch."""
        from langchain_community.vectorstores import OpenSearchVectorSearch
        from src.config import get_opensearch_kwargs

        loader = DirectoryLoader(docs_path, glob="**/*.md", loader_cls=TextLoader)
        documents = loader.load()
        if not documents:
            raise ValueError(
                f"No markdown documents found in '{docs_path}'. "
                "Add .md files to data/docs/ before running experiments."
            )

        chunk_size, overlap = self.get_chunk_size()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
        )
        chunks = splitter.split_documents(documents)

        os_kwargs = get_opensearch_kwargs()

        self.vectorstore = OpenSearchVectorSearch.from_documents(
            chunks,
            embedding=self.get_embeddings(),
            index_name=self.index_name,
            engine="lucene",
            space_type="cosinesimil",
            **os_kwargs,
        )

    def retrieve(self, query: str, k: int = 3) -> List[str]:
        """Return top-k context chunks for a query."""
        if self.vectorstore is None:
            raise RuntimeError("Call ingest() before retrieve().")
        docs = self.vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]

    def query(self, question: str) -> Tuple[str, List[str]]:
        """Run the full RAG pipeline and return (answer, context_chunks)."""
        chunks = self.retrieve(question)
        context = "\n\n".join(chunks)
        prompt = self.get_prompt(question, context)
        response = self.llm.invoke(prompt)
        return response.content, chunks
