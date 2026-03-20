"""
src/config.py — Centralised client helpers for Ollama and ChromaDB.

All configuration is read from environment variables (or a .env file).
Nothing in this file talks to any service at import time.
"""
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

# ── Ollama ────────────────────────────────────────────────────────────────── #
OLLAMA_LLM_MODEL   = os.getenv("OLLAMA_LLM_MODEL",   "llama3.2")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL",  "nomic-embed-text")
OLLAMA_BASE_URL    = os.getenv("OLLAMA_BASE_URL",     "http://localhost:11434")


# ── Ollama helpers ─────────────────────────────────────────────────────────── #

def get_llm(model: str | None = None):
    """Return a ChatOllama instance (LangChain-compatible)."""
    from langchain_ollama import ChatOllama
    return ChatOllama(
        model=model or OLLAMA_LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
    )


def get_embeddings(model: str | None = None):
    """Return an OllamaEmbeddings instance (LangChain-compatible)."""
    from langchain_ollama import OllamaEmbeddings
    return OllamaEmbeddings(
        model=model or OLLAMA_EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
    )


def check_ollama() -> bool:
    """Quick connectivity check — returns True if Ollama responds."""
    try:
        llm = get_llm()
        llm.invoke("ping")
        return True
    except Exception:
        return False
