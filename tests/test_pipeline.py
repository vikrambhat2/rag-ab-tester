import pytest

from src.pipeline.base import RAGPipeline
from tests.conftest import FakeLLM


class Doc:
    def __init__(self, text):
        self.page_content = text


class FakeStore:
    def similarity_search(self, query, k=3):
        return [Doc(f"chunk{i}") for i in range(k)]


class DummyPipeline(RAGPipeline):
    def get_embeddings(self):
        return object()

    def get_chunk_size(self):
        return (100, 10)

    def get_prompt(self, query, context):
        return f"{context}|{query}"


@pytest.fixture
def pipeline(monkeypatch):
    llm = FakeLLM(["the answer"])
    monkeypatch.setattr("src.config.get_llm", lambda model_id=None, role="generator": llm)
    return DummyPipeline(index_name="idx")


def test_retrieve_requires_ingest(pipeline):
    with pytest.raises(RuntimeError, match="ingest"):
        pipeline.retrieve("q")


def test_query_runs_full_pipeline(pipeline):
    pipeline.vectorstore = FakeStore()
    answer, chunks = pipeline.query("what?")
    assert answer == "the answer"
    assert chunks == ["chunk0", "chunk1", "chunk2"]


def test_ingest_without_docs_raises(pipeline, tmp_path, valid_env):
    with pytest.raises(ValueError, match="No markdown documents"):
        pipeline.ingest(str(tmp_path))


def test_cleanup_deletes_index_and_is_safe_on_error(pipeline, monkeypatch):
    deleted = []

    class Indices:
        def delete(self, index, ignore_unavailable):
            deleted.append(index)

    class Client:
        indices = Indices()

    monkeypatch.setattr("src.config.get_opensearch_client", lambda: Client())
    pipeline.vectorstore = FakeStore()
    pipeline.cleanup()
    assert deleted == ["idx"]
    assert pipeline.vectorstore is None

    def boom():
        raise ConnectionError("down")

    monkeypatch.setattr("src.config.get_opensearch_client", boom)
    pipeline.cleanup()  # must not raise
