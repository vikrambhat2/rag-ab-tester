import pytest

_ENV_VARS = [
    "LLM_MODEL", "LLM_BASE_URL", "LLM_API_KEY", "LLM_TIMEOUT",
    "JUDGE_LLM_MODEL", "JUDGE_LLM_BASE_URL", "JUDGE_LLM_API_KEY",
    "EMBED_MODEL", "EMBED_MODEL_CHALLENGER", "EMBED_BASE_URL", "EMBED_API_KEY",
]  # fmt: skip


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Tests must never see (or use) real credentials from the developer's shell/.env."""
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def valid_env(monkeypatch):
    """A complete, fake OpenAI-style configuration."""
    monkeypatch.setenv("LLM_MODEL", "gen-model")
    monkeypatch.setenv("LLM_API_KEY", "gen-key")
    monkeypatch.setenv("EMBED_MODEL", "embed-model")


class FakeResponse:
    def __init__(self, content: str):
        self.content = content


class FakeLLM:
    """Stand-in for a LangChain chat model. `replies` is consumed in order; the last one repeats."""

    def __init__(self, replies=("0.8",)):
        self.replies = list(replies)
        self.calls = 0

    def invoke(self, prompt):
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        if isinstance(reply, Exception):
            raise reply
        return FakeResponse(reply)


@pytest.fixture
def fake_llm():
    return FakeLLM
