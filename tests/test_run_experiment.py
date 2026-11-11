import json

import pytest

import run_experiment as rx
from src import best_config
from src.pipeline.base import RAGPipeline


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
    mod = rx.load_experiment("experiments/chunking.py")
    assert mod.EXPERIMENT_NAME == "Chunk Size"


@pytest.mark.parametrize(
    "path", ["experiments/chunking.py", "experiments/embedding.py",
             "experiments/retrieval.py", "experiments/prompt.py",
             "experiments/custom/example_experiment.py"],
)  # fmt: skip
def test_every_bundled_experiment_is_loadable(path):
    mod = rx.load_experiment(path)
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

    def ingest(self):
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
        rx.run("experiments/chunking.py", str(tmp_path / "nope.json"), False)


def test_run_cleans_up_indices_even_on_failure(tmp_path, monkeypatch):
    ts = tmp_path / "ts.json"
    ts.write_text(json.dumps(TEST_CASES))
    FakePipeline.instances.clear()

    exp = rx.load_experiment("experiments/chunking.py")
    exp.CONTROL = FakePipeline
    exp.CHALLENGER = lambda index_name: FakePipeline(index_name, fail=True)
    monkeypatch.setattr(rx, "load_experiment", lambda p: exp)
    monkeypatch.setattr(rx, "LLMJudge", lambda: FakeJudge(0.7))

    with pytest.raises(rx.ExperimentError):
        rx.run("x", str(ts), False)
    assert FakePipeline.instances and all(p.cleaned for p in FakePipeline.instances)


def test_run_keep_indices_skips_cleanup_and_updates_champion(tmp_path, monkeypatch):
    ts = tmp_path / "ts.json"
    ts.write_text(json.dumps(TEST_CASES))
    FakePipeline.instances.clear()
    monkeypatch.setattr(best_config, "_FILE", tmp_path / "best.json")

    class Better(FakePipeline):
        pass

    exp = rx.load_experiment("experiments/chunking.py")
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

    result = rx.run("x", str(ts), False, keep_indices=True)
    assert result.overall_winner == exp.CHALLENGER_NAME
    assert not any(p.cleaned for p in FakePipeline.instances)
    assert best_config.get("chunk_size") == 512  # Large-512 champion saved


def test_run_warns_when_judge_equals_generator(tmp_path, monkeypatch, caplog):
    ts = tmp_path / "ts.json"
    ts.write_text(json.dumps(TEST_CASES))
    monkeypatch.setattr(best_config, "_FILE", tmp_path / "best.json")
    exp = rx.load_experiment("experiments/chunking.py")
    exp.CONTROL = exp.CHALLENGER = FakePipeline
    monkeypatch.setattr(rx, "load_experiment", lambda p: exp)
    monkeypatch.setattr(rx, "LLMJudge", lambda: FakeJudge(0.5))
    with caplog.at_level("WARNING"):
        rx.run("x", str(ts), False)
    assert "Judge and generator are the same model" in caplog.text


def test_run_no_warning_with_separate_judge(tmp_path, monkeypatch, caplog):
    ts = tmp_path / "ts.json"
    ts.write_text(json.dumps(TEST_CASES))
    monkeypatch.setattr(best_config, "_FILE", tmp_path / "best.json")
    monkeypatch.setenv("JUDGE_LLM_MODEL", "ibm/granite-3-3-8b-instruct")
    exp = rx.load_experiment("experiments/chunking.py")
    exp.CONTROL = exp.CHALLENGER = FakePipeline
    monkeypatch.setattr(rx, "load_experiment", lambda p: exp)
    monkeypatch.setattr(rx, "LLMJudge", lambda: FakeJudge(0.5))
    with caplog.at_level("WARNING"):
        rx.run("x", str(ts), False)
    assert "same model" not in caplog.text
