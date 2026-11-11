"""
Live-endpoint smoke tests (a real OpenAI-compatible chat + embeddings server).

Skipped unless RUN_INTEGRATION=1 and LLM_MODEL / EMBED_MODEL (+ keys) are in the environment
or a .env file. Run with:  RUN_INTEGRATION=1 uv run pytest -m integration
"""

import os

import pytest
from dotenv import dotenv_values

pytestmark = pytest.mark.integration

# Captured at import time, before the autouse clean_env fixture strips the environment.
_REAL = {**dotenv_values(), **os.environ}
_VARS = (
    "LLM_MODEL", "LLM_BASE_URL", "LLM_API_KEY",
    "EMBED_MODEL", "EMBED_BASE_URL", "EMBED_API_KEY",
)  # fmt: skip


@pytest.fixture(autouse=True)
def live_env(monkeypatch):
    if _REAL.get("RUN_INTEGRATION") != "1":
        pytest.skip("set RUN_INTEGRATION=1 to run live-service tests")
    if not _REAL.get("LLM_MODEL") or not _REAL.get("EMBED_MODEL"):
        pytest.skip("set LLM_MODEL and EMBED_MODEL (and keys) to run live-service tests")
    for key in _VARS:
        if _REAL.get(key):
            monkeypatch.setenv(key, _REAL[key])


def test_chat_endpoint_reachable():
    from rag_ab_tester.config import get_llm
    from rag_ab_tester.protocols import to_text

    assert to_text(get_llm().invoke("Reply with the word: pong"))


def test_embeddings_endpoint_reachable():
    from rag_ab_tester.config import get_embeddings

    assert len(get_embeddings().embed_query("ping")) > 0
