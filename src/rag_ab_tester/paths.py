"""Filesystem locations used by the framework (all overridable)."""

from __future__ import annotations

import os
from pathlib import Path

RESULTS_DIR_ENV = "RAG_AB_RESULTS_DIR"


def results_dir() -> Path:
    """Where results and the champion config are stored (default: ./results)."""
    return Path(os.environ.get(RESULTS_DIR_ENV, "results"))
