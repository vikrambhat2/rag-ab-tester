import pytest
from tests.conftest import FakeLLM

from rag_ab_tester.evaluator import metrics
from rag_ab_tester.evaluator.judge import LLMJudge


@pytest.fixture
def judge(monkeypatch, valid_env):
    def make(replies):
        llm = FakeLLM(replies)
        monkeypatch.setattr(
            "rag_ab_tester.config.get_llm", lambda model_id=None, role="generator": llm
        )
        return LLMJudge(), llm

    return make


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("0.85", 0.85),
        ("Score: 1.0", 1.0),
        ("I'd say 0", 0.0),
        ("0.2 then finally 0.9", 0.9),
        ("none", None),
    ],
)
def test_extract_score(judge, text, expected):
    j, _ = judge(["0.5"])
    assert j._extract_score(text) == expected


def test_score_parses_reply(judge):
    j, llm = judge(["0.7"])
    assert j.score("p") == 0.7
    assert llm.calls == 1


def test_score_retries_once_on_unparseable_then_succeeds(judge):
    j, llm = judge(["no idea", "0.6"])
    assert j.score("p") == 0.6
    assert llm.calls == 2


def test_score_falls_back_to_neutral(judge):
    j, _ = judge(["gibberish"])
    assert j.score("p") == 0.5


def test_score_survives_llm_errors(judge, monkeypatch):
    monkeypatch.setattr("rag_ab_tester.retry.wait_exponential", lambda **kw: lambda *a, **k: 0)
    j, _ = judge([ValueError("bad request")])  # non-retryable -> fast fail
    assert j.score("p") == 0.5


def test_metric_wrappers_delegate_to_judge(judge):
    j, _ = judge(["0.9"])
    assert metrics.faithfulness_score("a", ["c1", "c2"], j) == 0.9
    assert metrics.answer_relevance_score("q", "a", j) == 0.9
    assert metrics.context_precision_score("q", ["c"], j) == 0.9
    assert metrics.context_recall_score("q", ["c"], "gt", j) == 0.9
