"""
Provider-neutral interfaces.

The framework only needs three capabilities, each described by a tiny Protocol so any
provider can be plugged in without subclassing anything:

* :class:`ChatModel`   - ``invoke(prompt)``. Every LangChain chat model already satisfies it
  (Anthropic, OpenAI, Ollama, Bedrock, Vertex, ...). A plain ``str -> str``
  callable works too (see :func:`as_chat_model`).
* :class:`Embedder`    - ``embed_documents`` / ``embed_query``. Every LangChain ``Embeddings``
  object already satisfies it.
* :class:`VectorIndex` - ``add`` / ``search`` / ``delete`` over plain text chunks. Ships with
  :class:`~rag_ab_tester.stores.InMemoryIndex` (the default) and
  :class:`~rag_ab_tester.stores.LangChainVectorStoreIndex` (wraps Chroma, FAISS, pgvector,
  Qdrant, Pinecone, OpenSearch, ...).

These are structural types: nothing in this module imports an optional dependency.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ChatModel(Protocol):
    """Anything with ``invoke(prompt)`` returning a string or an object with ``.content``."""

    def invoke(self, prompt: str) -> Any: ...


@runtime_checkable
class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


@runtime_checkable
class VectorIndex(Protocol):
    """
    One isolated, disposable index. A pipeline owns exactly one, so control and challenger
    never share state.

    ``add`` replaces nothing - the pipeline calls ``delete`` first to clear stale state.
    ``delete`` must be safe to call on an index that was never populated.
    """

    def add(self, texts: list[str], embedder: Embedder) -> None: ...

    def search(self, query: str, k: int) -> list[str]: ...

    def delete(self) -> None: ...


class _CallableChat:
    """Adapts a ``str -> str`` callable to :class:`ChatModel`."""

    def __init__(self, fn: Callable[[str], Any]):
        self._fn = fn

    def invoke(self, prompt: str) -> Any:
        return self._fn(prompt)


def as_chat_model(llm: Any) -> ChatModel:
    """Return ``llm`` if it has ``invoke``; wrap a bare callable; otherwise raise TypeError."""
    if hasattr(llm, "invoke"):
        return llm
    if callable(llm):
        return _CallableChat(llm)
    raise TypeError(
        f"llm must have an invoke(prompt) method or be a callable, got {type(llm).__name__}"
    )


def to_text(response: Any) -> str:
    """
    Normalise a model response to plain text.

    Handles ``str``, objects with a ``.content`` attribute, and LangChain "content blocks"
    (a list of strings / ``{"type": "text", "text": ...}`` dicts, as some providers return).
    """
    content = response if isinstance(response, str) else getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "".join(parts)
    return str(content)


def model_label(llm: Any) -> str:
    """Best-effort human-readable name for a model object (never includes secrets)."""
    for attr in ("model_name", "model", "model_id"):
        value = getattr(llm, attr, None)
        if isinstance(value, str) and value:
            return f"{type(llm).__name__}:{value}"
    return type(llm).__name__
