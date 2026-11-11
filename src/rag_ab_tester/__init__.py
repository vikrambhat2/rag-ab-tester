"""rag-ab-tester: statistically sound A/B testing for RAG pipelines."""

import logging
from importlib import metadata

try:
    __version__ = metadata.version("rag-ab-tester")
except metadata.PackageNotFoundError:  # running from a source tree that isn't installed
    __version__ = "0+unknown"

# Libraries must not configure logging; the CLI does that. This silences "no handler" warnings.
logging.getLogger(__name__).addHandler(logging.NullHandler())

from rag_ab_tester.config import ConfigError, MissingExtraError, Settings, get_settings  # noqa: E402
from rag_ab_tester.evaluator.judge import LLMJudge  # noqa: E402
from rag_ab_tester.evaluator.stats import compare_metric  # noqa: E402
from rag_ab_tester.models.schemas import (  # noqa: E402
    ABReport,
    ExperimentResult,
    MetricComparison,
    QueryScore,
    VariantResult,
)
from rag_ab_tester.pipeline.base import RAGPipeline  # noqa: E402
from rag_ab_tester.protocols import ChatModel, Embedder, VectorIndex  # noqa: E402
from rag_ab_tester.runner import ExperimentError, run_experiment  # noqa: E402
from rag_ab_tester.stores import InMemoryIndex, LangChainVectorStoreIndex  # noqa: E402

__all__ = [
    "ABReport",
    "ChatModel",
    "ConfigError",
    "Embedder",
    "ExperimentError",
    "ExperimentResult",
    "InMemoryIndex",
    "LLMJudge",
    "LangChainVectorStoreIndex",
    "MetricComparison",
    "MissingExtraError",
    "QueryScore",
    "RAGPipeline",
    "Settings",
    "VariantResult",
    "VectorIndex",
    "__version__",
    "compare_metric",
    "get_settings",
    "run_experiment",
]
