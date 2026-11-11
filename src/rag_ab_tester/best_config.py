"""
src/best_config.py — Champion configuration store.

Persists the best-performing hyperparameter from each experiment to
results/best_config.json so subsequent experiments automatically build
on proven winners rather than hardcoded defaults.

Usage:
    from rag_ab_tester.best_config import get, save

    # Read a value (falls back to default if not yet saved)
    chunk_size, chunk_overlap = get("chunk_size"), get("chunk_overlap")

    # Persist a winner after an experiment
    save({"chunk_size": 256, "chunk_overlap": 25})
"""

from __future__ import annotations

import json
from pathlib import Path

from rag_ab_tester.paths import results_dir


def _file() -> Path:
    return results_dir() / "best_config.json"


# Defaults are used when an experiment has not been run yet.
DEFAULTS: dict = {
    "chunk_size": 512,
    "chunk_overlap": 50,
    "embedding_model": "",  # "" = use EMBED_MODEL from the environment
    "retrieval_strategy": "semantic",  # "semantic" | "hybrid"
    "prompt_template": "open",  # "open" | "strict"
}

# Keys each experiment is responsible for saving
EXPERIMENT_KEYS: dict[str, list[str]] = {
    "Chunk Size": ["chunk_size", "chunk_overlap"],
    "Embedding Model": ["embedding_model"],
    "Retrieval Strategy": ["retrieval_strategy"],
    "Prompt Template": ["prompt_template"],
}


def load() -> dict:
    """Return current best config merged over defaults."""
    if _file().exists():
        stored = json.loads(_file().read_text())
        return {**DEFAULTS, **stored}
    return DEFAULTS.copy()


def get(key: str):
    """Return a single best-config value, falling back to the default."""
    return load().get(key, DEFAULTS[key])


def save(updates: dict) -> None:
    """Persist one or more key-value pairs into the best config store."""
    current = load()
    current.update(updates)
    _file().parent.mkdir(parents=True, exist_ok=True)
    # Write only keys that are in DEFAULTS to avoid cruft accumulation
    _file().write_text(json.dumps({k: current[k] for k in DEFAULTS if k in current}, indent=2))


def summary() -> str:
    """Return a human-readable one-liner of the current champion config."""
    cfg = load()
    return (
        f"chunk={cfg['chunk_size']}/{cfg['chunk_overlap']}  "
        f"embed={cfg['embedding_model'] or 'default'}  "
        f"retrieval={cfg['retrieval_strategy']}  "
        f"prompt={cfg['prompt_template']}"
    )
