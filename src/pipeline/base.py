from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple, List

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader


class RAGPipeline(ABC):
    """
    Base class for all RAG pipeline variants.

    Subclasses must implement three methods:
        get_embeddings()  → which embedding model to use
        get_chunk_size()  → (chunk_size, overlap) tuple
        get_prompt()      → format the final prompt string

    Override `ingest()` or `retrieve()` for structural changes
    (e.g. hybrid retrieval).
    """

    def __init__(self, collection_name: str, persist_dir: str):
        self.llm = ChatOllama(model="llama3.2", temperature=0)
        self.collection_name = collection_name
        self.persist_dir = persist_dir
        self.vectorstore: Chroma | None = None

    # ------------------------------------------------------------------ #
    #  Abstract interface                                                  #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def get_embeddings(self) -> OllamaEmbeddings:
        """Return the embedding model to use for this variant."""

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
        """Load docs, split into chunks, embed, and persist to Chroma."""
        loader = DirectoryLoader(docs_path, glob="**/*.md")
        documents = loader.load()
        if not documents:
            raise ValueError(
                f"No markdown documents found in '{docs_path}'. "
                "Run ingest.py first or add .md files to data/docs/."
            )

        chunk_size, overlap = self.get_chunk_size()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
        )
        chunks = splitter.split_documents(documents)

        self.vectorstore = Chroma.from_documents(
            chunks,
            embedding=self.get_embeddings(),
            collection_name=self.collection_name,
            persist_directory=self.persist_dir,
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
