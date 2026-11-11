import pytest

from rag_ab_tester import best_config


@pytest.fixture(autouse=True)
def tmp_file(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_AB_RESULTS_DIR", str(tmp_path / "results"))


def test_defaults_when_nothing_saved():
    assert best_config.load() == best_config.DEFAULTS
    assert best_config.get("chunk_size") == 512


def test_save_and_get_roundtrip():
    best_config.save({"chunk_size": 256, "chunk_overlap": 25})
    assert best_config.get("chunk_size") == 256
    assert best_config.get("embedding_model") == best_config.DEFAULTS["embedding_model"]


def test_save_is_cumulative_and_drops_unknown_keys():
    best_config.save({"chunk_size": 256})
    best_config.save({"retrieval_strategy": "hybrid", "bogus": 1})
    cfg = best_config.load()
    assert cfg["chunk_size"] == 256
    assert cfg["retrieval_strategy"] == "hybrid"
    assert "bogus" not in cfg


def test_summary_mentions_all_fields():
    text = best_config.summary()
    assert "chunk=512/50" in text and "retrieval=semantic" in text
