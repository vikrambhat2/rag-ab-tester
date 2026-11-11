import json

import pytest

import rag_ab_tester.runner as rx
from rag_ab_tester import best_config
from rag_ab_tester.pipeline.base import RAGPipeline


def test_safe_index_sanitises():
    assert rx._safe_index("ctrl", "Small-256") == "ctrl_small-256"
    assert rx._safe_index("chal", "Hybrid BM25!!") == "chal_hybrid_bm25"
    assert len(rx._safe_index("x", "a" * 500)) <= 255


def test_load_experiment_requires_attributes(tmp_path):
    f = tmp_path / "bad.py"
    f.write_text("EXPERIMENT_NAME = 'x'\n")
    with pytest.raises(AttributeError, match="missing required"):
        rx.load_experiment(str(f))


def test_load_experiment_ok():
    mod = rx.load_experiment(rx.resolve_experiment("chunking"))
    assert mod.EXPERIMENT_NAME == "Chunk Size"


@pytest.mark.parametrize(
    "ref", ["chunking", "embedding", "retrieval", "prompt", "examples/example_experiment.py"]
)
def test_every_bundled_experiment_is_loadable(ref):
    mod = rx.load_experiment(rx.resolve_experiment(ref))
    assert issubclass(mod.CONTROL, RAGPipeline) and issubclass(mod.CHALLENGER, RAGPipeline)
    assert mod.CONTROL_NAME != mod.CHALLENGER_NAME
    cc = getattr(mod, "CHAMPION_CONFIG", None)
    if cc:
        assert set(cc) == {mod.CONTROL_NAME, mod.CHALLENGER_NAME}


# ── end-to-end with fakes ─────────────────────────────────────────────────── #


class FakeJudge:
    def __init__(self, score):
        self._score = score

    def score_faithfulness(self, *a):
        return self._score

    score_answer_relevance = score_context_precision = score_context_recall = score_faithfulness


class FakePipeline:
    instances: list = []

    def __init__(self, index_name, fail=False):
        self.index_name = index_name
        self.fail = fail
        self.cleaned = False
        FakePipeline.instances.append(self)

    def ingest(self, docs_path):
        pass

    def cleanup(self):
        self.cleaned = True

    def query(self, q):
        if self.fail:
            raise ConnectionError("down")
        return "ans", ["c1"]


TEST_CASES = [{"query": f"q{i}", "ground_truth": "gt"} for i in range(4)]


def test_score_variant_aborts_when_most_queries_fail():
    with pytest.raises(rx.ExperimentError, match="unreliable"):
        rx.score_variant(FakePipeline("i", fail=True), TEST_CASES, FakeJudge(1.0), "v")


def test_score_variant_ok():
    res = rx.score_variant(FakePipeline("i"), TEST_CASES, FakeJudge(0.5), "v")
    assert len(res.scores) == 4 and res.avg("faithfulness") == 0.5


def test_run_missing_test_set(tmp_path):
    with pytest.raises(rx.ExperimentError, match="Test set not found"):
        rx.run_experiment("chunking", str(tmp_path / "nope.json"), False)


def test_run_cleans_up_indices_even_on_failure(tmp_path, monkeypatch):
    ts = tmp_path / "ts.json"
    ts.write_text(json.dumps(TEST_CASES))
    FakePipeline.instances.clear()

    exp = rx.load_experiment(rx.resolve_experiment("chunking"))
    exp.CONTROL = FakePipeline
    exp.CHALLENGER = lambda index_name: FakePipeline(index_name, fail=True)
    monkeypatch.setattr(rx, "load_experiment", lambda p: exp)
    monkeypatch.setattr(rx, "LLMJudge", lambda: FakeJudge(0.7))

    with pytest.raises(rx.ExperimentError):
        rx.run_experiment("chunking", str(ts), False)
    assert FakePipeline.instances and all(p.cleaned for p in FakePipeline.instances)


def test_run_keep_indices_skips_cleanup_and_updates_champion(tmp_path, monkeypatch):
    ts = tmp_path / "ts.json"
    ts.write_text(json.dumps(TEST_CASES))
    FakePipeline.instances.clear()
    monkeypatch.setenv("RAG_AB_RESULTS_DIR", str(tmp_path / "res"))

    class Better(FakePipeline):
        pass

    exp = rx.load_experiment(rx.resolve_experiment("chunking"))
    exp.CONTROL = FakePipeline
    exp.CHALLENGER = Better
    monkeypatch.setattr(rx, "load_experiment", lambda p: exp)

    # Judge is called 4 metrics x 4 queries = 16 times per variant (control first).
    # Use a cycle length coprime to 4 so each metric sees varying per-query scores.
    base = [0.2, 0.3, 0.25, 0.35, 0.22]
    seq = iter([base[i % 5] for i in range(16)] + [base[i % 5] + 0.5 for i in range(16)])

    class SeqJudge(FakeJudge):
        def score_faithfulness(self, *a):
            return next(seq)

        score_answer_relevance = score_context_precision = score_context_recall = score_faithfulness

    monkeypatch.setattr(rx, "LLMJudge", lambda: SeqJudge(0))

    result = rx.run_experiment("chunking", str(ts), False, keep_indices=True)
    assert result.overall_winner == exp.CHALLENGER_NAME
    assert not any(p.cleaned for p in FakePipeline.instances)
    assert best_config.get("chunk_size") == 512  # Large-512 champion saved


def test_run_warns_when_judge_equals_generator(tmp_path, monkeypatch, caplog):
    ts = tmp_path / "ts.json"
    ts.write_text(json.dumps(TEST_CASES))
    monkeypatch.setenv("RAG_AB_RESULTS_DIR", str(tmp_path / "res"))
    exp = rx.load_experiment(rx.resolve_experiment("chunking"))
    exp.CONTROL = exp.CHALLENGER = FakePipeline
    monkeypatch.setattr(rx, "load_experiment", lambda p: exp)
    monkeypatch.setattr(rx, "LLMJudge", lambda: FakeJudge(0.5))
    with caplog.at_level("WARNING"):
        rx.run_experiment("chunking", str(ts), False)
    assert "Judge and generator are the same model" in caplog.text


def test_run_no_warning_with_separate_judge(tmp_path, monkeypatch, caplog):
    ts = tmp_path / "ts.json"
    ts.write_text(json.dumps(TEST_CASES))
    monkeypatch.setenv("RAG_AB_RESULTS_DIR", str(tmp_path / "res"))
    monkeypatch.setenv("JUDGE_LLM_MODEL", "a-different-judge")
    exp = rx.load_experiment(rx.resolve_experiment("chunking"))
    exp.CONTROL = exp.CHALLENGER = FakePipeline
    monkeypatch.setattr(rx, "load_experiment", lambda p: exp)
    monkeypatch.setattr(rx, "LLMJudge", lambda: FakeJudge(0.5))
    with caplog.at_level("WARNING"):
        rx.run_experiment("chunking", str(ts), False)
    assert "same model" not in caplog.text


# ── provider-agnostic end-to-end ──────────────────────────────────────────── #

_EXPERIMENT_SRC = """
from rag_ab_tester import RAGPipeline

class Small(RAGPipeline):
    def get_chunk_size(self): return (80, 0)
    def get_prompt(self, query, context): return f"A {context} {query}"

class Large(Small):
    def get_chunk_size(self): return (300, 0)

EXPERIMENT_NAME = "Agnostic"
CONTROL, CHALLENGER = Small, Large
CONTROL_NAME, CHALLENGER_NAME = "small", "large"
"""


def test_run_experiment_with_injected_llm_judge_embedder_and_index(tmp_path, monkeypatch, caplog):
    from tests.conftest import FakeLLM
    from tests.test_pipeline import HashEmbedder

    from rag_ab_tester import InMemoryIndex

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("cats purr when happy. the moon orbits the earth. " * 30)
    ts = tmp_path / "ts.json"
    ts.write_text(json.dumps(TEST_CASES))
    exp_file = tmp_path / "exp.py"
    exp_file.write_text(_EXPERIMENT_SRC)
    monkeypatch.setenv("RAG_AB_RESULTS_DIR", str(tmp_path / "res"))

    # No LLM/embedding env at all: everything comes from the injected objects.
    made: list[tuple[str, InMemoryIndex]] = []

    def factory(index_name):
        idx = InMemoryIndex()
        made.append((index_name, idx))
        return idx

    gen, judge = FakeLLM(["an answer"]), FakeLLM(["0.8"])
    with caplog.at_level("WARNING"):
        result = rx.run_experiment(
            exp_file,
            str(ts),
            docs_path=docs,
            llm=gen,
            judge_llm=judge,
            embedder=HashEmbedder(),
            index_factory=factory,
        )

    assert [n for n, _ in made] == ["ctrl_small", "chal_large"]  # isolated per variant
    assert gen.calls == 8 and judge.calls == 8 * 4  # 2 variants x 4 queries; 4 metrics each
    assert result.control.avg("faithfulness") == pytest.approx(0.8)
    assert "same model" not in caplog.text  # distinct generator and judge objects


def test_embedder_override_rejected_for_embedding_experiment(tmp_path):
    ts = tmp_path / "ts.json"
    ts.write_text(json.dumps(TEST_CASES))
    with pytest.raises(rx.ExperimentError, match="compares embedding models"):
        rx.run_experiment("embedding", str(ts), embedder=object())


# ── built-in experiments ──────────────────────────────────────────────────── #


def test_hybrid_retrieval_merges_bm25_first_and_dedupes():
    from tests.conftest import FakeLLM
    from tests.test_pipeline import DOCS, HashEmbedder

    from rag_ab_tester import InMemoryIndex

    mod = rx.load_experiment(rx.resolve_experiment("retrieval"))
    p = mod.HybridPipeline("i", llm=FakeLLM(), embedder=HashEmbedder(), index=InMemoryIndex())
    with pytest.raises(RuntimeError, match="ingest"):
        p.retrieve("python")
    p.ingest_texts(DOCS)
    chunks = p.retrieve("programming", k=2)
    assert chunks[0] == "python is a programming language"  # keyword hit comes first
    assert len(chunks) == len(set(chunks)) == 2


def test_embedding_experiment_is_configured_from_env(monkeypatch):
    from rag_ab_tester.config import ConfigError

    monkeypatch.setenv("EMBED_MODEL", "small")
    monkeypatch.setenv("EMBED_MODEL_CHALLENGER", "large")
    mod = rx.load_experiment(rx.resolve_experiment("embedding"))
    assert (mod.CONTROL_NAME, mod.CHALLENGER_NAME) == ("small", "large")
    assert mod.CHAMPION_CONFIG["large"] == {"embedding_model": "large"}

    monkeypatch.delenv("EMBED_MODEL_CHALLENGER")
    mod = rx.load_experiment(rx.resolve_experiment("embedding"))
    assert mod.CONTROL_NAME != mod.CHALLENGER_NAME  # placeholders keep the two distinct
    with pytest.raises(ConfigError, match="EMBED_MODEL_CHALLENGER"):
        mod.ChallengerEmbeddingPipeline("i", llm=lambda p: "", index=None).get_embeddings()
