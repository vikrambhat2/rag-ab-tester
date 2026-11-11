import pytest

from rag_ab_tester import InMemoryIndex, LangChainVectorStoreIndex
from rag_ab_tester.protocols import ChatModel, Embedder, VectorIndex, model_label, to_text


class Emb:
    def embed_documents(self, texts):
        return [[float(len(t)), 1.0] for t in texts]

    def embed_query(self, text):
        return [float(len(text)), 1.0]


def test_builtin_types_satisfy_protocols():
    assert isinstance(InMemoryIndex(), VectorIndex)
    assert isinstance(LangChainVectorStoreIndex(lambda t, e: None), VectorIndex)
    assert isinstance(Emb(), Embedder)
    assert isinstance(type("L", (), {"invoke": lambda s, p: ""})(), ChatModel)


def test_memory_index_ranks_by_cosine_and_deletes():
    idx = InMemoryIndex()
    with pytest.raises(RuntimeError):
        idx.search("q", 1)
    idx.add(["aa", "bbbbbbbbbb"], Emb())
    assert idx.search("bbbbbbbbbb", 1) == ["bbbbbbbbbb"]
    idx.delete()
    idx.delete()  # idempotent
    with pytest.raises(RuntimeError):
        idx.search("q", 1)


def test_memory_index_rejects_empty():
    with pytest.raises(ValueError):
        InMemoryIndex().add([], Emb())


class Doc:
    def __init__(self, t):
        self.page_content = t


class FakeStore:
    deleted = False

    def __init__(self, texts):
        self.texts = texts

    def similarity_search(self, query, k):
        return [Doc(t) for t in self.texts[:k]]

    def delete_collection(self):
        FakeStore.deleted = True


def test_langchain_adapter_builds_searches_and_uses_delete_collection():
    idx = LangChainVectorStoreIndex(lambda texts, emb: FakeStore(texts))
    idx.delete()  # never populated: no-op
    idx.add(["a", "b", "c"], Emb())
    assert idx.search("q", 2) == ["a", "b"]
    idx.delete()
    assert FakeStore.deleted
    with pytest.raises(RuntimeError):
        idx.search("q", 1)


def test_langchain_adapter_custom_delete_and_warning(caplog):
    seen = []
    idx = LangChainVectorStoreIndex(lambda t, e: FakeStore(t), delete=seen.append)
    idx.add(["a"], Emb())
    idx.delete()
    assert len(seen) == 1 and isinstance(seen[0], FakeStore)

    idx = LangChainVectorStoreIndex(lambda t, e: object())
    idx.add(["a"], Emb())
    with caplog.at_level("WARNING"):
        idx.delete()
    assert "delete_collection" in caplog.text


def test_to_text_and_model_label():
    assert to_text("x") == "x"
    assert to_text(type("R", (), {"content": "y"})()) == "y"
    assert to_text(type("R", (), {"content": [{"text": "a"}, "b", {"type": "image"}]})()) == "ab"
    assert model_label(type("ChatFoo", (), {"model_name": "m1"})()) == "ChatFoo:m1"
    assert model_label(object()) == "object"
