"""
Live-service smoke tests (real WatsonX + OpenSearch).

Skipped unless RUN_INTEGRATION=1 and credentials are in the environment / .env.
Run with:  RUN_INTEGRATION=1 uv run pytest -m integration
"""

import os

import pytest
from dotenv import dotenv_values

pytestmark = pytest.mark.integration

# Captured at import time, before the autouse clean_env fixture strips the environment.
_REAL = {**dotenv_values(), **os.environ}
_NEEDED = (
    "WATSONX_API_KEY", "WATSONX_PROJECT_ID", "OPENSEARCH_URL",
    "OPENSEARCH_USERNAME", "OPENSEARCH_PASSWORD",
)  # fmt: skip


@pytest.fixture(autouse=True)
def live_env(monkeypatch):
    if _REAL.get("RUN_INTEGRATION") != "1":
        pytest.skip("set RUN_INTEGRATION=1 to run live-service tests")
    missing = [k for k in _NEEDED if not _REAL.get(k)]
    if missing:
        pytest.skip(f"missing credentials: {missing}")
    for key in (*_NEEDED, "WATSONX_URL", "OPENSEARCH_VERIFY_SSL"):
        if _REAL.get(key):
            monkeypatch.setenv(key, _REAL[key])


def test_opensearch_reachable():
    from src.config import check_opensearch

    assert check_opensearch()


def test_watsonx_reachable():
    from src.config import check_watsonx

    assert check_watsonx()
