import pytest

from src.retry import with_retry


def test_returns_value():
    assert with_retry(lambda x: x * 2, 4) == 8


def test_retries_transient_errors_then_succeeds():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("blip")
        return "ok"

    assert with_retry(flaky, min_wait=0, max_wait=0) == "ok"
    assert calls["n"] == 3


def test_gives_up_after_max_attempts():
    calls = {"n": 0}

    def always():
        calls["n"] += 1
        raise ConnectionError("down")

    with pytest.raises(ConnectionError):
        with_retry(always, attempts=3, min_wait=0, max_wait=0)
    assert calls["n"] == 3


def test_does_not_retry_programming_or_input_errors():
    calls = {"n": 0}

    def bad():
        calls["n"] += 1
        raise ValueError("bad input")

    with pytest.raises(ValueError):
        with_retry(bad, min_wait=0, max_wait=0)
    assert calls["n"] == 1
