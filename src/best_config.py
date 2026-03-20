"""
src/best_config.py — Champion config store.

Persists the winning hyperparameters from each experiment to
results/best_config.json so downstream experiments automatically
build on proven winners rather than hardcoded defaults.

Cascade order:
    chunking  → saves chunk_size + chunk_overlap
    embedding → reads chunk_size, saves embedding_model
    retrieval → reads chunk_size + embedding_model, saves retrieval_strategy
    prompt    → reads all three, saves prompt_template
"""
from __future__ import annotations

import json
from pathlib import Path

_FILE = Path("results/best_config.json")

_DEFAULTS: dict = {
    "chunk_size": 512,
    "chunk_overlap": 50,
    "embedding_model": "nomic-embed-text",
    "retrieval_strategy": "semantic",
    "prompt_template": "default",
}


def load() -> dict:
    """Load config from disk, falling back to defaults for missing keys."""
    if _FILE.exists():
        return {**_DEFAULTS, **json.loads(_FILE.read_text())}
    return _DEFAULTS.copy()


def save(updates: dict) -> None:
    """Merge updates into the persisted config and write to disk."""
    current = load()
    current.update(updates)
    _FILE.parent.mkdir(exist_ok=True)
    _FILE.write_text(json.dumps(current, indent=2))


def get(key: str):
    """Return a single config value (falls back to default if not set)."""
    return load().get(key, _DEFAULTS[key])


def summary() -> str:
    """One-line summary of the current champion config."""
    cfg = load()
    return " | ".join(f"{k}={v}" for k, v in cfg.items())
