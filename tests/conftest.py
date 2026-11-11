import pytest

_ENV_VARS = [
    "WATSONX_API_KEY", "WATSONX_PROJECT_ID", "WATSONX_URL",
    "WATSONX_LLM_MODEL_ID", "WATSONX_EMBED_MODEL_ID",
    "OPENSEARCH_URL", "OPENSEARCH_USERNAME", "OPENSEARCH_PASSWORD",
    "OPENSEARCH_VERIFY_SSL", "OPENSEARCH_TIMEOUT", "LLM_TIMEOUT",
    "LLM_PROVIDER", "LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL",
    "JUDGE_LLM_PROVIDER", "JUDGE_LLM_BASE_URL", "JUDGE_LLM_API_KEY", "JUDGE_LLM_MODEL",
]  # fmt: skip


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Tests must never see (or use) real credentials from the developer's shell/.env."""
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def valid_env(monkeypatch):
    monkeypatch.setenv("WATSONX_API_KEY", "key")
    monkeypatch.setenv("WATSONX_PROJECT_ID", "proj")
    monkeypatch.setenv("OPENSEARCH_URL", "https://os.example.com:9200")
    monkeypatch.setenv("OPENSEARCH_USERNAME", "user")
    monkeypatch.setenv("OPENSEARCH_PASSWORD", "pass")


class FakeResponse:
    def __init__(self, content: str):
        self.content = content


class FakeLLM:
    """Stand-in for ChatWatsonx. `replies` is consumed in order; the last one repeats."""

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
