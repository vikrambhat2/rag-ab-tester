"""
Settings and client factories for the built-in OpenAI-compatible provider.

The CLI (and any code that does not inject its own models) talks to one kind of endpoint: an
OpenAI-compatible HTTP API, which covers OpenAI itself plus Ollama, vLLM, LiteLLM, Azure/OpenAI
gateways and most hosted inference services. Use the SDK's injection points
(``llm=``, ``embedder=``, ``index_factory=``) to plug in anything else.

All values come from environment variables. Nothing here talks to a service at import time, and
settings are validated lazily so a missing variable produces one clear error.
"""

from __future__ import annotations

import importlib
import logging
import os
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import SecretStr

logger = logging.getLogger(__name__)


class MissingExtraError(ImportError):
    """An optional dependency group is not installed."""


def import_optional(module: str, extra: str) -> Any:
    """Import an optional dependency, or raise an error naming the extra to install."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise MissingExtraError(
            f"'{module}' is required for this feature. "
            f"Install it with: pip install 'rag-ab-tester[{extra}]'"
        ) from exc


class ConfigError(ValueError):
    """Raised when required configuration is missing or malformed."""


LLMRole = Literal["generator", "judge"]

# Placeholder for local servers (Ollama, vLLM) that accept any key; the OpenAI client rejects "".
_NO_KEY = "not-needed"


@dataclass(frozen=True)
class LLMConfig:
    """Resolved chat-LLM settings for one role (generator or judge)."""

    model: str
    base_url: str = ""  # empty = OpenAI's own endpoint
    api_key: str = ""

    @property
    def label(self) -> str:
        host = urlparse(self.base_url).hostname
        return f"{self.model}@{host}" if host else self.model


@dataclass(frozen=True)
class Settings:
    # Generator (the model that answers questions)
    llm_model: str = ""
    llm_base_url: str = ""
    llm_api_key: str = ""
    # Judge overrides - each falls back to the generator value when unset
    judge_llm_model: str = ""
    judge_llm_base_url: str = ""
    judge_llm_api_key: str = ""
    # Embeddings - base URL / key fall back to the generator's (typical: one provider)
    embed_model: str = ""
    embed_model_challenger: str = ""  # only the built-in embedding experiment uses this
    embed_base_url: str = ""
    embed_api_key: str = ""
    llm_timeout: int = 120

    @classmethod
    def from_env(cls) -> Settings:
        env = os.environ.get
        try:
            timeout = int(env("LLM_TIMEOUT", "120"))
        except ValueError as exc:
            raise ConfigError("LLM_TIMEOUT must be an integer number of seconds") from exc
        return cls(
            llm_model=env("LLM_MODEL", ""),
            llm_base_url=env("LLM_BASE_URL", ""),
            llm_api_key=env("LLM_API_KEY", ""),
            judge_llm_model=env("JUDGE_LLM_MODEL", ""),
            judge_llm_base_url=env("JUDGE_LLM_BASE_URL", ""),
            judge_llm_api_key=env("JUDGE_LLM_API_KEY", ""),
            embed_model=env("EMBED_MODEL", ""),
            embed_model_challenger=env("EMBED_MODEL_CHALLENGER", ""),
            embed_base_url=env("EMBED_BASE_URL", ""),
            embed_api_key=env("EMBED_API_KEY", ""),
            llm_timeout=timeout,
        )

    def llm_config(self, role: LLMRole = "generator", model_id: str | None = None) -> LLMConfig:
        """
        Resolve the chat-LLM config for a role.

        The judge inherits the generator's settings field by field. If the judge points at a
        *different* endpoint it inherits no API key, so one provider's key is never sent to
        another provider's server.
        """
        base_url, api_key, model = self.llm_base_url, self.llm_api_key, self.llm_model
        if role == "judge":
            if self.judge_llm_base_url and self.judge_llm_base_url != base_url:
                api_key = ""
            base_url = self.judge_llm_base_url or base_url
            api_key = self.judge_llm_api_key or api_key
            model = self.judge_llm_model or model
        return LLMConfig(model=model_id or model, base_url=base_url, api_key=api_key)

    def embed_config(self, model_id: str | None = None) -> LLMConfig:
        """Resolve the embeddings endpoint (falls back to the generator's endpoint and key)."""
        return LLMConfig(
            model=model_id or self.embed_model,
            base_url=self.embed_base_url or self.llm_base_url,
            api_key=self.embed_api_key or self.llm_api_key,
        )

    @staticmethod
    def _endpoint_problems(prefix: str, cfg: LLMConfig, *, key_var: str, url_var: str) -> list[str]:
        issues = []
        if cfg.base_url and not urlparse(cfg.base_url).hostname:
            issues.append(f"{url_var} is not a valid URL (e.g. https://host/v1): {cfg.base_url!r}")
        if not cfg.model:
            issues.append(f"{prefix}_MODEL is not set")
        if not cfg.api_key and not cfg.base_url:
            issues.append(f"{key_var} is not set (or set {url_var} for a local server)")
        return issues

    def llm_problems(self, role: LLMRole, model_id: str | None = None) -> list[str]:
        """Problems with the chat-LLM config for a role (empty if OK)."""
        prefix = "JUDGE_LLM" if role == "judge" else "LLM"
        return self._endpoint_problems(
            prefix,
            self.llm_config(role, model_id),
            key_var=f"{prefix}_API_KEY",
            url_var=f"{prefix}_BASE_URL",
        )

    def embed_problems(self, model_id: str | None = None) -> list[str]:
        """Problems with the embeddings config (empty if OK)."""
        return self._endpoint_problems(
            "EMBED",
            self.embed_config(model_id),
            key_var="EMBED_API_KEY (or LLM_API_KEY)",
            url_var="EMBED_BASE_URL",
        )


def get_settings(
    *,
    llm: LLMRole | None = None,
    embeddings: bool = False,
    model_id: str | None = None,
) -> Settings:
    """
    Load settings from the environment and validate the pieces requested.

    ``llm`` validates that role's chat-LLM config; ``embeddings=True`` validates the embeddings
    config (``model_id`` overrides the model for either).
    """
    settings = Settings.from_env()
    issues: list[str] = []
    if llm is not None:
        issues += settings.llm_problems(llm, model_id)
    if embeddings:
        issues += settings.embed_problems(model_id)
    if issues:
        raise ConfigError(
            "Invalid configuration:\n  - "
            + "\n  - ".join(issues)
            + "\nSet them in your environment or a .env file (see .env.example in the repository)."
        )
    return settings


def get_llm(model_id: str | None = None, role: LLMRole = "generator") -> Any:
    """Return a LangChain chat model for the role (OpenAI-compatible endpoint)."""
    s = get_settings(llm=role, model_id=model_id)
    cfg = s.llm_config(role, model_id)
    ChatOpenAI = import_optional("langchain_openai", "openai").ChatOpenAI

    return ChatOpenAI(
        model=cfg.model,
        base_url=cfg.base_url or None,
        api_key=SecretStr(cfg.api_key or _NO_KEY),
        temperature=0,
        max_completion_tokens=512,
        timeout=s.llm_timeout,
        max_retries=0,  # retries are handled by rag_ab_tester.retry.with_retry
    )


def describe_llm(role: LLMRole = "generator") -> str:
    """Human-readable ``model@host`` label for a role (no validation, no secrets)."""
    return Settings.from_env().llm_config(role).label


def get_embeddings(model_id: str | None = None) -> Any:
    """Return a LangChain ``OpenAIEmbeddings`` for the configured OpenAI-compatible endpoint."""
    s = get_settings(embeddings=True, model_id=model_id)
    cfg = s.embed_config(model_id)
    OpenAIEmbeddings = import_optional("langchain_openai", "openai").OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=cfg.model,
        base_url=cfg.base_url or None,
        api_key=SecretStr(cfg.api_key or _NO_KEY),
        timeout=s.llm_timeout,
        max_retries=0,
        # Non-OpenAI servers expect raw strings, not the client-side tokenised input.
        check_embedding_ctx_length=not cfg.base_url,
    )
