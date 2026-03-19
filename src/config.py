"""
src/config.py — Centralised configuration for WatsonX AI

Reads credentials and model IDs from environment variables.
A .env file in the project root is loaded automatically via python-dotenv.

Required env vars:
    WATSONX_APIKEY       — IBM Cloud API key
    WATSONX_PROJECT_ID   — WatsonX project ID

Optional env vars (with defaults):
    WATSONX_URL              — WatsonX endpoint (default: us-south)
    WATSONX_LLM_MODEL_ID     — LLM model ID
    WATSONX_EMBED_MODEL_ID   — Default embedding model ID
"""
from __future__ import annotations
import os
from pathlib import Path

from dotenv import load_dotenv

# Auto-load .env from project root (no-op if absent)
load_dotenv(Path(__file__).parent.parent / ".env")

# ──────────────────────────────────────────────────────────────────────────── #
#  Constants / defaults                                                        #
# ──────────────────────────────────────────────────────────────────────────── #

_WATSONX_URL_DEFAULT = "https://us-south.ml.cloud.ibm.com"
_LLM_MODEL_DEFAULT = "ibm/granite-3-8b-instruct"
_EMBED_MODEL_DEFAULT = "ibm/slate-125m-english-rtrvr-v2"


# ──────────────────────────────────────────────────────────────────────────── #
#  Internal helpers                                                            #
# ──────────────────────────────────────────────────────────────────────────── #

def _require(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise EnvironmentError(
            f"Required environment variable '{name}' is not set. "
            "Copy .env.example to .env and fill in your credentials."
        )
    return val


def _get_llm_params(temperature: float = 0) -> dict:
    return {
        "decoding_method": "greedy" if temperature == 0 else "sample",
        "max_new_tokens": 512,
        "temperature": temperature,
        "repetition_penalty": 1.0,
    }


# ──────────────────────────────────────────────────────────────────────────── #
#  Public factories                                                            #
# ──────────────────────────────────────────────────────────────────────────── #

def get_llm(temperature: float = 0):
    """Return a configured ChatWatsonx instance."""
    from langchain_ibm import ChatWatsonx  # noqa: PLC0415
    return ChatWatsonx(
        model_id=os.getenv("WATSONX_LLM_MODEL_ID", _LLM_MODEL_DEFAULT),
        url=os.getenv("WATSONX_URL", _WATSONX_URL_DEFAULT),
        apikey=_require("WATSONX_APIKEY"),
        project_id=_require("WATSONX_PROJECT_ID"),
        params=_get_llm_params(temperature),
    )


def get_embeddings(model_id: str | None = None):
    """Return a configured WatsonxEmbeddings instance."""
    from langchain_ibm import WatsonxEmbeddings  # noqa: PLC0415
    return WatsonxEmbeddings(
        model_id=model_id or os.getenv("WATSONX_EMBED_MODEL_ID", _EMBED_MODEL_DEFAULT),
        url=os.getenv("WATSONX_URL", _WATSONX_URL_DEFAULT),
        apikey=_require("WATSONX_APIKEY"),
        project_id=_require("WATSONX_PROJECT_ID"),
    )


def check_watsonx() -> tuple[bool, str]:
    """Lightweight check: returns (ok, message)."""
    try:
        _require("WATSONX_APIKEY")
        _require("WATSONX_PROJECT_ID")
        return True, "Credentials set"
    except EnvironmentError as e:
        return False, str(e)
