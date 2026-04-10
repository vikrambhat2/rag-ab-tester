"""
src/config.py — Centralised client helpers for WatsonX AI and OpenSearch.

All credentials are read from environment variables (or a .env file).
Nothing in this file talks to any service at import time.
"""
from __future__ import annotations

import os
from urllib.parse import urlparse

from dotenv import load_dotenv

load_dotenv()

# ── WatsonX AI ────────────────────────────────────────────────────────────── #
WATSONX_API_KEY       = os.getenv("WATSONX_API_KEY", "")
WATSONX_PROJECT_ID    = os.getenv("WATSONX_PROJECT_ID", "")
WATSONX_URL           = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
WATSONX_LLM_MODEL_ID  = os.getenv("WATSONX_LLM_MODEL_ID",   "ibm/granite-3-8b-instruct")
WATSONX_EMBED_MODEL_ID = os.getenv("WATSONX_EMBED_MODEL_ID", "ibm/slate-125m-english-rtrvr-v2")

# ── OpenSearch ────────────────────────────────────────────────────────────── #
OPENSEARCH_URL        = os.getenv("OPENSEARCH_URL", "")          # e.g. https://host:9200
OPENSEARCH_USERNAME   = os.getenv("OPENSEARCH_USERNAME", "")
OPENSEARCH_PASSWORD   = os.getenv("OPENSEARCH_PASSWORD", "")
OPENSEARCH_VERIFY_SSL = os.getenv("OPENSEARCH_VERIFY_SSL", "true").lower() not in ("false", "0", "no")


# ── WatsonX helpers ───────────────────────────────────────────────────────── #

def get_llm(model_id: str | None = None):
    """Return a ChatWatsonx instance (LangChain-compatible)."""
    from langchain_ibm import ChatWatsonx
    return ChatWatsonx(
        model_id=model_id or WATSONX_LLM_MODEL_ID,
        url=WATSONX_URL,
        apikey=WATSONX_API_KEY,
        project_id=WATSONX_PROJECT_ID,
        params={"temperature": 0, "max_new_tokens": 512},
    )


def get_embeddings(model_id: str | None = None):
    """Return a WatsonxEmbeddings instance (LangChain-compatible)."""
    from langchain_ibm import WatsonxEmbeddings
    return WatsonxEmbeddings(
        model_id=model_id or WATSONX_EMBED_MODEL_ID,
        url=WATSONX_URL,
        apikey=WATSONX_API_KEY,
        project_id=WATSONX_PROJECT_ID,
    )


def check_watsonx() -> bool:
    """Quick connectivity check — returns True if WatsonX responds."""
    try:
        llm = get_llm()
        llm.invoke("ping")
        return True
    except Exception:
        return False


# ── OpenSearch helpers ────────────────────────────────────────────────────── #

def _parsed_url():
    if not OPENSEARCH_URL:
        raise ValueError(
            "OPENSEARCH_URL is not set. Add it to your .env file.\n"
            "Example: OPENSEARCH_URL=https://your-host:9200"
        )
    parsed = urlparse(OPENSEARCH_URL)
    if not parsed.hostname:
        raise ValueError(f"Could not parse hostname from OPENSEARCH_URL='{OPENSEARCH_URL}'")
    return parsed


def get_opensearch_client():
    """Return a raw opensearch-py client for admin operations."""
    from opensearchpy import OpenSearch
    parsed = _parsed_url()
    return OpenSearch(
        hosts=[{"host": parsed.hostname, "port": parsed.port or 9200}],
        http_auth=(OPENSEARCH_USERNAME, OPENSEARCH_PASSWORD),
        use_ssl=parsed.scheme == "https",
        verify_certs=OPENSEARCH_VERIFY_SSL,
        ssl_show_warn=False,
        timeout=60,
    )


def get_opensearch_kwargs() -> dict:
    """
    Return kwargs suitable for OpenSearchVectorSearch.from_documents()
    and OpenSearchVectorSearch() constructors.
    """
    parsed = _parsed_url()
    return {
        "opensearch_url": OPENSEARCH_URL,
        "http_auth": (OPENSEARCH_USERNAME, OPENSEARCH_PASSWORD),
        "use_ssl": parsed.scheme == "https",
        "verify_certs": OPENSEARCH_VERIFY_SSL,
        "ssl_show_warn": False,
        "timeout": 60,
    }


def check_opensearch() -> bool:
    """Quick connectivity check — returns True if OpenSearch cluster responds."""
    try:
        client = get_opensearch_client()
        info = client.info()
        return bool(info.get("version"))
    except Exception:
        return False
