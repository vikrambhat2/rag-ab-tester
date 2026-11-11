"""
src/config.py — Settings and client factories for WatsonX AI and OpenSearch.

All credentials come from environment variables (or a .env file).
Nothing in this module talks to a service at import time, and settings are
validated lazily so a missing variable produces one clear error listing
everything that is wrong.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlparse

from dotenv import load_dotenv
from pydantic import SecretStr

load_dotenv()

logger = logging.getLogger(__name__)

_FALSE_VALUES = ("false", "0", "no", "off")


class ConfigError(ValueError):
    """Raised when required configuration is missing or malformed."""


LLMRole = Literal["generator", "judge"]
_PROVIDERS = ("watsonx", "openai")


@dataclass(frozen=True)
class LLMConfig:
    """Resolved chat-LLM settings for one role (generator or judge)."""

    provider: str  # "watsonx" | "openai" (any OpenAI-compatible server)
    model: str
    base_url: str = ""
    api_key: str = ""

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}"


@dataclass(frozen=True)
class Settings:
    watsonx_api_key: str
    watsonx_project_id: str
    watsonx_url: str
    watsonx_llm_model_id: str
    watsonx_embed_model_id: str
    opensearch_url: str
    opensearch_username: str
    opensearch_password: str
    opensearch_verify_ssl: bool
    opensearch_timeout: int
    llm_timeout: int
    # Chat-LLM provider for the generator (RAG answers). Embeddings are always WatsonX.
    llm_provider: str = "watsonx"
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = ""
    # Judge overrides — each falls back to the generator value when unset.
    judge_llm_provider: str = ""
    judge_llm_base_url: str = ""
    judge_llm_api_key: str = ""
    judge_llm_model: str = ""

    @classmethod
    def from_env(cls) -> Settings:
        env = os.environ.get
        return cls(
            watsonx_api_key=env("WATSONX_API_KEY", ""),
            watsonx_project_id=env("WATSONX_PROJECT_ID", ""),
            watsonx_url=env("WATSONX_URL", "https://us-south.ml.cloud.ibm.com"),
            watsonx_llm_model_id=env("WATSONX_LLM_MODEL_ID", "ibm/granite-3-8b-instruct"),
            watsonx_embed_model_id=env("WATSONX_EMBED_MODEL_ID", "ibm/slate-125m-english-rtrvr-v2"),
            opensearch_url=env("OPENSEARCH_URL", ""),
            opensearch_username=env("OPENSEARCH_USERNAME", ""),
            opensearch_password=env("OPENSEARCH_PASSWORD", ""),
            opensearch_verify_ssl=env("OPENSEARCH_VERIFY_SSL", "true").lower() not in _FALSE_VALUES,
            opensearch_timeout=int(env("OPENSEARCH_TIMEOUT", "60")),
            llm_timeout=int(env("LLM_TIMEOUT", "120")),
            llm_provider=env("LLM_PROVIDER", "watsonx").strip().lower(),
            llm_base_url=env("LLM_BASE_URL", ""),
            llm_api_key=env("LLM_API_KEY", ""),
            llm_model=env("LLM_MODEL", ""),
            judge_llm_provider=env("JUDGE_LLM_PROVIDER", "").strip().lower(),
            judge_llm_base_url=env("JUDGE_LLM_BASE_URL", ""),
            judge_llm_api_key=env("JUDGE_LLM_API_KEY", ""),
            judge_llm_model=env("JUDGE_LLM_MODEL", ""),
        )

    def llm_config(self, role: LLMRole = "generator", model_id: str | None = None) -> LLMConfig:
        """
        Resolve the chat-LLM config for a role.

        The judge inherits the generator's settings field-by-field when it uses the same
        provider; if it switches provider it inherits nothing (so it never sends one
        provider's API key to another provider's endpoint).
        """
        provider = self.llm_provider
        base_url, api_key, model = self.llm_base_url, self.llm_api_key, self.llm_model

        if role == "judge":
            if self.judge_llm_provider and self.judge_llm_provider != provider:
                provider, base_url, api_key, model = self.judge_llm_provider, "", "", ""
            base_url = self.judge_llm_base_url or base_url
            api_key = self.judge_llm_api_key or api_key
            model = self.judge_llm_model or model

        if provider == "watsonx":
            # WatsonX models are named by WATSONX_LLM_MODEL_ID unless a judge override is set.
            judge_override = self.judge_llm_model if role == "judge" else ""
            model = judge_override or self.watsonx_llm_model_id
            base_url = api_key = ""
        if model_id:
            model = model_id
        return LLMConfig(provider=provider, model=model, base_url=base_url, api_key=api_key)

    def llm_problems(self, role: LLMRole, model_id: str | None = None) -> list[str]:
        """Problems with the chat-LLM config for a role (WatsonX creds checked separately)."""
        cfg = self.llm_config(role, model_id)
        prefix = "JUDGE_LLM" if role == "judge" else "LLM"
        if cfg.provider not in _PROVIDERS:
            return [f"{prefix}_PROVIDER must be one of {_PROVIDERS}, got {cfg.provider!r}"]
        if cfg.provider == "watsonx":
            return []
        issues = []
        if not urlparse(cfg.base_url).hostname:
            issues.append(f"{prefix}_BASE_URL is not set or invalid (e.g. https://host/v1)")
        if not cfg.model:
            issues.append(f"{prefix}_MODEL is not set")
        if not cfg.api_key:
            issues.append(f"{prefix}_API_KEY is not set")
        return issues

    def problems(self, *, watsonx: bool = True, opensearch: bool = True) -> list[str]:
        """Return a list of human-readable configuration problems (empty if OK)."""
        issues: list[str] = []
        if watsonx:
            if not self.watsonx_api_key:
                issues.append("WATSONX_API_KEY is not set")
            if not self.watsonx_project_id:
                issues.append("WATSONX_PROJECT_ID is not set")
            if not urlparse(self.watsonx_url).hostname:
                issues.append(f"WATSONX_URL is not a valid URL: {self.watsonx_url!r}")
        if opensearch:
            parsed = urlparse(self.opensearch_url)
            if not self.opensearch_url:
                issues.append("OPENSEARCH_URL is not set (e.g. https://host:9200)")
            elif parsed.scheme not in ("http", "https") or not parsed.hostname:
                issues.append(f"OPENSEARCH_URL is not a valid http(s) URL: {self.opensearch_url!r}")
            if not self.opensearch_username or not self.opensearch_password:
                issues.append("OPENSEARCH_USERNAME / OPENSEARCH_PASSWORD are not set")
        return issues


def get_settings(
    *,
    watsonx: bool = True,
    opensearch: bool = True,
    llm: LLMRole | None = None,
    model_id: str | None = None,
) -> Settings:
    """
    Load settings from the environment and validate the requested services.

    ``llm`` additionally validates that role's chat-LLM config; WatsonX credentials are then
    required only if that role actually uses the WatsonX provider.
    """
    settings = Settings.from_env()
    issues: list[str] = []
    if llm is not None:
        issues += settings.llm_problems(llm, model_id)
        if settings.llm_config(llm, model_id).provider == "watsonx":
            watsonx = True
    issues = settings.problems(watsonx=watsonx, opensearch=opensearch) + issues
    if issues:
        raise ConfigError(
            "Invalid configuration:\n  - "
            + "\n  - ".join(issues)
            + "\nSee .env.example for the full list of variables."
        )
    if opensearch and not settings.opensearch_verify_ssl:
        logger.warning(
            "OPENSEARCH_VERIFY_SSL is disabled — TLS certificates are NOT verified. "
            "Only use this for local development."
        )
    return settings


# ── WatsonX helpers ───────────────────────────────────────────────────────── #


def get_llm(model_id: str | None = None, role: LLMRole = "generator"):
    """
    Return a LangChain chat model for the given role.

    Provider is chosen by LLM_PROVIDER (generator) / JUDGE_LLM_PROVIDER (judge):
    ``watsonx`` (default, langchain-ibm) or ``openai`` (any OpenAI-compatible endpoint,
    e.g. the watsonx.ai Model Gateway, vLLM, OpenAI).
    """
    s = get_settings(watsonx=False, opensearch=False, llm=role, model_id=model_id)
    cfg = s.llm_config(role, model_id)

    if cfg.provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=cfg.model,
            base_url=cfg.base_url,
            api_key=SecretStr(cfg.api_key),
            temperature=0,
            max_completion_tokens=512,
            timeout=s.llm_timeout,
            max_retries=0,  # retries are handled by src.retry.with_retry
        )

    from langchain_ibm import ChatWatsonx

    return ChatWatsonx(
        model_id=cfg.model,
        url=SecretStr(s.watsonx_url),
        apikey=SecretStr(s.watsonx_api_key),
        project_id=s.watsonx_project_id,
        params={"temperature": 0, "max_new_tokens": 512},
    )


def describe_llm(role: LLMRole = "generator") -> str:
    """Human-readable ``provider:model`` label for a role (no validation, no secrets)."""
    return Settings.from_env().llm_config(role).label


def get_embeddings(model_id: str | None = None):
    """Return a WatsonxEmbeddings instance (LangChain-compatible)."""
    from langchain_ibm import WatsonxEmbeddings

    s = get_settings(opensearch=False)
    return WatsonxEmbeddings(
        model_id=model_id or s.watsonx_embed_model_id,
        url=SecretStr(s.watsonx_url),
        apikey=SecretStr(s.watsonx_api_key),
        project_id=s.watsonx_project_id,
    )


def check_watsonx() -> bool:
    """Connectivity check for the generator LLM. Failures are logged."""
    try:
        get_llm().invoke("ping")
        return True
    except Exception as exc:
        logger.error("WatsonX health check failed: %s", exc)
        return False


# ── OpenSearch helpers ────────────────────────────────────────────────────── #


def get_opensearch_client():
    """Return a raw opensearch-py client for admin operations."""
    from opensearchpy import OpenSearch

    s = get_settings(watsonx=False)
    parsed = urlparse(s.opensearch_url)
    return OpenSearch(
        hosts=[{"host": parsed.hostname, "port": parsed.port or 9200}],
        http_auth=(s.opensearch_username, s.opensearch_password),
        use_ssl=parsed.scheme == "https",
        verify_certs=s.opensearch_verify_ssl,
        ssl_show_warn=False,
        timeout=s.opensearch_timeout,
    )


def get_opensearch_kwargs() -> dict:
    """Kwargs for the OpenSearchVectorSearch constructors."""
    s = get_settings(watsonx=False)
    parsed = urlparse(s.opensearch_url)
    return {
        "opensearch_url": s.opensearch_url,
        "http_auth": (s.opensearch_username, s.opensearch_password),
        "use_ssl": parsed.scheme == "https",
        "verify_certs": s.opensearch_verify_ssl,
        "ssl_show_warn": False,
        "timeout": s.opensearch_timeout,
    }


def check_opensearch() -> bool:
    """Connectivity check — True if the cluster responds. Failures are logged."""
    try:
        info = get_opensearch_client().info()
        return bool(info.get("version"))
    except Exception as exc:
        logger.error("OpenSearch health check failed: %s", exc)
        return False
