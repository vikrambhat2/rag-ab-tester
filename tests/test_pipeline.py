import hashlib

import pytest
from tests.conftest import FakeLLM

from rag_ab_tester import InMemoryIndex
from rag_ab_tester.pipeline.base import RAGPipeline


class HashEmbedder:
    """Deterministic bag-of-words embedder: shared words => higher cosine similarity."""

    dim = 64

    def _vec(self, text):
        v = [0.0] * self.dim
        for word in text.lower().split():
            v[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1.0
        return v

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class DummyPipeline(RAGPipeline):
    def get_chunk_size(self):
        return (100, 10)

    def get_prompt(self, query, context):
        return f"{context}|{query}"


DOCS = [
    "cats purr when happy",
    "the moon orbits the earth",
    "python is a programming language",
]


@pytest.fixture
def pipeline():
    return DummyPipeline(
        index_name="idx",
        llm=FakeLLM(["the answer"]),
        embedder=HashEmbedder(),
        index=InMemoryIndex(),
    )


def test_retrieve_requires_ingest(pipeline):
    with pytest.raises(RuntimeError, match="ingest"):
        pipeline.retrieve("q")


def test_query_runs_full_pipeline_with_injected_providers(pipeline):
    pipeline.ingest_texts(DOCS)
    answer, chunks = pipeline.query("where does the moon orbit")
    assert answer == "the answer"
    assert chunks[0] == "the moon orbits the earth"
    assert len(chunks) == 3


def test_ingest_texts_rejects_empty(pipeline):
    with pytest.raises(ValueError, match="No text chunks"):
        pipeline.ingest_texts([])


def test_reingest_replaces_previous_content(pipeline):
    pipeline.ingest_texts(DOCS)
    pipeline.ingest_texts(["only this"])
    assert pipeline.retrieve("moon", k=3) == ["only this"]


def test_injected_embedder_overrides_get_embeddings(pipeline):
    class Overriding(DummyPipeline):
        def get_embeddings(self):
            raise AssertionError("must not be called when an embedder is injected")

    p = Overriding("i", llm=FakeLLM(), embedder=HashEmbedder(), index=InMemoryIndex())
    p.ingest_texts(DOCS)


def test_llm_may_be_a_plain_callable_or_return_strings():
    p = DummyPipeline(
        "i", llm=lambda prompt: "plain", embedder=HashEmbedder(), index=InMemoryIndex()
    )
    p.ingest_texts(DOCS)
    assert p.query("moon")[0] == "plain"


def test_llm_content_blocks_are_flattened():
    class Blocks:
        def invoke(self, prompt):
            return type("R", (), {"content": [{"type": "text", "text": "a"}, "b"]})()

    p = DummyPipeline("i", llm=Blocks(), embedder=HashEmbedder(), index=InMemoryIndex())
    p.ingest_texts(DOCS)
    assert p.query("moon")[0] == "ab"


def test_bad_llm_is_rejected():
    with pytest.raises(TypeError, match="invoke"):
        DummyPipeline("i", llm=42, embedder=HashEmbedder(), index=InMemoryIndex())


def test_default_llm_comes_from_config(monkeypatch):
    llm = FakeLLM(["x"])
    monkeypatch.setattr("rag_ab_tester.config.get_llm", lambda model_id=None, role="generator": llm)
    assert DummyPipeline("idx").llm is llm


def test_ingest_without_docs_raises(pipeline, tmp_path):
    with pytest.raises(ValueError, match="No markdown documents"):
        pipeline.ingest(str(tmp_path))


def test_ingest_from_markdown_files(pipeline, tmp_path):
    (tmp_path / "a.md").write_text("cats purr when happy. " * 20)
    pipeline.ingest(str(tmp_path))
    assert pipeline.ready


def test_cleanup_deletes_index_and_never_raises(pipeline):
    pipeline.ingest_texts(DOCS)
    pipeline.cleanup()
    assert not pipeline.ready
    with pytest.raises(RuntimeError):
        pipeline.index.search("q", 1)

    class Broken(InMemoryIndex):
        def delete(self):
            raise ConnectionError("down")

    pipeline.index = Broken()
    pipeline.cleanup()  # must not raise
