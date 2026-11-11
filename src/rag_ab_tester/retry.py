"""Retry helper for flaky network calls (model and vector-store APIs)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TypeVar

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Errors that will never succeed on retry (bad credentials, bad input, bad config).
_NON_RETRYABLE = (ValueError, TypeError, KeyError, AttributeError, PermissionError)


def _is_retryable(exc: BaseException) -> bool:
    return not isinstance(exc, _NON_RETRYABLE)


def with_retry(
    fn: Callable[..., T],
    *args,
    attempts: int = 4,
    min_wait: float = 1.0,
    max_wait: float = 20.0,
    **kwargs,
) -> T:
    """Call ``fn(*args, **kwargs)`` with exponential backoff on transient errors."""

    @retry(
        stop=stop_after_attempt(attempts),
        wait=wait_exponential(multiplier=min_wait, max=max_wait),
        retry=retry_if_exception(_is_retryable),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _call() -> T:
        return fn(*args, **kwargs)

    return _call()
