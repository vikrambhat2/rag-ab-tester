"""Shared logging setup for the CLI entry points."""

from __future__ import annotations

import logging
import os

from rich.logging import RichHandler


def setup_logging(verbose: bool = False) -> None:
    """Configure root logging. LOG_LEVEL env var overrides the default level."""
    level_name = "DEBUG" if verbose else os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
        force=True,
    )
    # Third-party libraries are noisy at DEBUG.
    for noisy in ("urllib3", "opensearch", "httpx", "httpcore", "ibm_watsonx_ai"):
        logging.getLogger(noisy).setLevel(max(level, logging.WARNING))
