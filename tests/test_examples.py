"""
Offline checks that the shipped example experiments (used by examples/notebooks) keep working
against the current API. They use stub providers, so no LLM, network or model download.
"""

import hashlib
import json
import re
import sys
from pathlib import Path

import pytest

from rag_ab_tester import InMemoryIndex, run_experiment

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "examples" / "experiments"
NOTEBOOK = ROOT / "examples" / "notebooks" / "quickstart_local.ipynb"
EXPERIMENT_FILES = sorted(EXPERIMENTS.glob("exp_*.py"))


class StubEmbedder:
    def _vec(self, text):
        v = [0.0] * 32
        for word in text.lower().split():
            v[int(hashlib.md5(word.encode()).hexdigest(), 16) % 32] += 1.0
        return v

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


@pytest.fixture(autouse=True)
def _experiments_on_path(monkeypatch):
    # experiment files do `from hf_base import ...`
    monkeypatch.syspath_prepend(str(EXPERIMENTS))
    sys.modules.pop("hf_base", None)


@pytest.fixture
def data(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text(
        "Cats purr when they are happy. " * 20 + "\n\nThe zeppelin flew over the harbour. " * 20
    )
    cases = [{"query": f"Why do cats purr? ({i})", "ground_truth": "Happiness."} for i in range(5)]
    (tmp_path / "test_set.json").write_text(json.dumps(cases))
    return tmp_path


def test_experiment_files_found():
    assert len(EXPERIMENT_FILES) == 4


@pytest.mark.parametrize("path", EXPERIMENT_FILES, ids=lambda p: p.stem)
def test_example_experiment_runs_offline(path, data):
    result = run_experiment(
        path,
        test_set_path=data / "test_set.json",
        docs_path=data / "docs",
        llm=lambda prompt: "Cats purr when happy.",
        judge_llm=lambda prompt: "0.8",
        embedder=StubEmbedder(),
        index_factory=lambda name: InMemoryIndex(),
    )
    assert result.control.scores and result.challenger.scores
    assert {c.metric for c in result.comparisons}


def test_hybrid_retrieval_adds_keyword_hits(data):
    import importlib

    exp = importlib.import_module("exp_retrieval")
    pipe = exp.Hybrid("hybrid", llm=lambda p: "x", embedder=StubEmbedder())
    pipe.ingest(data / "docs")
    hits = pipe.retrieve("zeppelin harbour", k=2)
    assert hits and any("zeppelin" in h for h in hits)


def test_notebook_is_valid_and_references_existing_files():
    nb = json.loads(NOTEBOOK.read_text())
    assert nb["nbformat"] == 4
    assert all(not c.get("outputs") for c in nb["cells"] if c["cell_type"] == "code")
    text = "".join("".join(c["source"]) for c in nb["cells"])
    referenced = set(re.findall(r"examples/experiments/(\w+\.py)", text))
    assert referenced, "notebook should reference the experiment files"
    for name in referenced:
        assert (EXPERIMENTS / name).exists(), name
