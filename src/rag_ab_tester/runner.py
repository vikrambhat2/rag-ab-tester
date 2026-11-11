"""Run A/B experiments: ingest both variants, score them with an LLM judge, compare statistically."""

from __future__ import annotations

import importlib.util
import json
import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rich.console import Console

from rag_ab_tester.best_config import save as save_best
from rag_ab_tester.best_config import summary as best_summary
from rag_ab_tester.config import describe_llm
from rag_ab_tester.evaluator.judge import LLMJudge
from rag_ab_tester.evaluator.metrics import (
    answer_relevance_score,
    context_precision_score,
    context_recall_score,
    faithfulness_score,
)
from rag_ab_tester.evaluator.stats import compare_metric
from rag_ab_tester.models.schemas import ExperimentResult, QueryScore, VariantResult
from rag_ab_tester.protocols import Embedder, VectorIndex, model_label
from rag_ab_tester.report.report import print_experiment, save_experiment_json

console = Console()
logger = logging.getLogger(__name__)

# Abort the experiment if more than this fraction of queries fail — scoring
# failed queries as 0.0 would otherwise silently skew the statistics.
MAX_FAILURE_RATE = 0.5


class ExperimentError(RuntimeError):
    """Raised when an experiment cannot produce trustworthy results."""


METRICS = ["faithfulness", "answer_relevance", "context_precision", "context_recall"]


# ──────────────────────────────────────────────────────────────────────────── #
#  Helpers                                                                     #
# ──────────────────────────────────────────────────────────────────────────── #


BUILTIN_DIR = Path(__file__).parent / "experiments"


def builtin_experiments() -> dict[str, Path]:
    """Names and paths of the experiments that ship with the package."""
    return {p.stem: p for p in sorted(BUILTIN_DIR.glob("*.py")) if not p.name.startswith("_")}


def resolve_experiment(ref: str | Path) -> str:
    """Resolve a built-in experiment name or a file path to an experiment file path."""
    path = Path(ref)
    if path.suffix == ".py" and path.exists():
        return str(path)
    builtin = builtin_experiments()
    name = path.stem if path.suffix == ".py" else str(ref)
    if name in builtin:
        return str(builtin[name])
    raise ExperimentError(
        f"Unknown experiment '{ref}'. Built-ins: {', '.join(builtin)}; "
        "or pass the path to your own experiment .py file."
    )


def load_experiment(path: str):
    """Dynamically load an experiment module from a file path."""
    spec = importlib.util.spec_from_file_location("experiment", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load experiment from '{path}'")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    required = ["EXPERIMENT_NAME", "CONTROL", "CHALLENGER", "CONTROL_NAME", "CHALLENGER_NAME"]
    missing = [attr for attr in required if not hasattr(mod, attr)]
    if missing:
        raise AttributeError(f"Experiment module is missing required attributes: {missing}")
    return mod


def _safe_index(prefix: str, name: str) -> str:
    """
    Build a store-safe index name.
    Rules: lowercase, alphanumeric + hyphens/underscores, no leading - or _.
    """
    slug = re.sub(r"[^a-z0-9_-]", "_", name.lower())
    slug = re.sub(r"_+", "_", slug).strip("_-")
    return f"{prefix}_{slug}"[:255]


def score_variant(
    pipeline,
    test_cases: list[dict],
    judge: LLMJudge,
    name: str,
) -> VariantResult:
    """Run all test cases through a pipeline variant and score each one."""
    scores: list[QueryScore] = []
    failures = 0
    for i, tc in enumerate(test_cases):
        console.print(f"  [{name}] {i + 1}/{len(test_cases)}: {tc['query'][:60]}...")
        try:
            answer, chunks = pipeline.query(tc["query"])
            scores.append(
                QueryScore(
                    query=tc["query"],
                    faithfulness=faithfulness_score(answer, chunks, judge),
                    answer_relevance=answer_relevance_score(tc["query"], answer, judge),
                    context_precision=context_precision_score(tc["query"], chunks, judge),
                    context_recall=context_recall_score(
                        tc["query"], chunks, tc["ground_truth"], judge
                    ),
                )
            )
        except Exception as e:
            failures += 1
            logger.error("[%s] query %d failed: %s", name, i + 1, e)
            scores.append(
                QueryScore(
                    query=tc["query"],
                    faithfulness=0.0,
                    answer_relevance=0.0,
                    context_precision=0.0,
                    context_recall=0.0,
                )
            )

    if test_cases and failures / len(test_cases) > MAX_FAILURE_RATE:
        raise ExperimentError(
            f"[{name}] {failures}/{len(test_cases)} queries failed — results would be unreliable."
        )
    if failures:
        logger.warning(
            "[%s] %d/%d queries failed and were scored 0.0", name, failures, len(test_cases)
        )

    return VariantResult(variant_name=name, scores=scores)


# ──────────────────────────────────────────────────────────────────────────── #
#  Main                                                                        #
# ──────────────────────────────────────────────────────────────────────────── #


def run_experiment(
    experiment: str | Path,
    test_set_path: str | Path = "data/test_set.json",
    save_json: bool = False,
    keep_indices: bool = False,
    docs_path: str | Path = "data/docs",
    *,
    llm: Any = None,
    judge_llm: Any = None,
    embedder: Embedder | None = None,
    index_factory: Callable[[str], VectorIndex] | None = None,
) -> ExperimentResult:
    """
    Run one experiment end to end.

    ``experiment`` is a built-in name (``"chunking"``), or a path to an experiment ``.py`` file.

    Providers are pluggable; anything left as ``None`` falls back to the environment config:

    * ``llm``           chat model for generating answers (any object with ``invoke(prompt)``)
    * ``judge_llm``     chat model for the LLM judge (use a different model than ``llm``)
    * ``embedder``      embedding model used by *both* variants (not allowed for experiments
                        that compare embedders)
    * ``index_factory`` ``index_name -> VectorIndex``, called once per variant so each gets an
                        isolated index (default: in-memory)
    """
    experiment_path = resolve_experiment(experiment)
    exp = load_experiment(experiment_path)
    if embedder is not None and getattr(exp, "VARIES_EMBEDDER", False):
        raise ExperimentError(
            f"'{exp.EXPERIMENT_NAME}' compares embedding models, so a single embedder= "
            "override would make both variants identical."
        )

    test_set_file = Path(test_set_path)
    if not test_set_file.exists():
        raise ExperimentError(
            f"Test set not found at '{test_set_path}'. Run `rag-ab ingest` first."
        )

    test_cases: list[dict] = json.loads(test_set_file.read_text())
    if not test_cases:
        raise ExperimentError("Test set is empty. Re-run `rag-ab ingest`.")

    console.print(
        f"\n[bold cyan]{'=' * 60}[/bold cyan]\n"
        f"[bold]Experiment:[/bold] {exp.EXPERIMENT_NAME}\n"
        f"[bold]Control:[/bold]    {exp.CONTROL_NAME}\n"
        f"[bold]Challenger:[/bold] {exp.CHALLENGER_NAME}\n"
        f"[bold]Test cases:[/bold] {len(test_cases)}\n"
        f"[bold cyan]{'=' * 60}[/bold cyan]"
    )

    judge = LLMJudge(llm=judge_llm) if judge_llm is not None else LLMJudge()
    gen_label = model_label(llm) if llm is not None else describe_llm("generator")
    judge_label = model_label(judge_llm) if judge_llm is not None else describe_llm("judge")
    console.print(
        f"[bold]Generator LLM:[/bold] {gen_label}\n[bold]Judge LLM:[/bold]     {judge_label}"
    )
    injected = llm is not None or judge_llm is not None
    # Injected objects may share a class name without sharing a model, so only trust labels
    # that include a model name (provider:model), or identity.
    same_model = (llm is not None and llm is judge_llm) or (
        gen_label == judge_label and (not injected or ":" in gen_label)
    )
    if same_model:
        logger.warning(
            "Judge and generator are the same model (%s); LLM-as-judge scores can be biased "
            "toward the model's own outputs. Set JUDGE_LLM_* to use a different judge.",
            gen_label,
        )

    def build(cls, prefix: str, name: str):
        """Instantiate a variant with its own isolated index; only pass what was overridden."""
        index_name = _safe_index(prefix, name)
        kwargs: dict[str, Any] = {}
        if llm is not None:
            kwargs["llm"] = llm
        if embedder is not None:
            kwargs["embedder"] = embedder
        if index_factory is not None:
            kwargs["index"] = index_factory(index_name)
        return cls(index_name=index_name, **kwargs)

    control = build(exp.CONTROL, "ctrl", exp.CONTROL_NAME)
    challenger = build(exp.CHALLENGER, "chal", exp.CHALLENGER_NAME)

    try:
        console.print(f"\n[bold]Ingesting: {exp.CONTROL_NAME}[/bold]")
        control.ingest(docs_path)

        console.print(f"[bold]Ingesting: {exp.CHALLENGER_NAME}[/bold]")
        challenger.ingest(docs_path)

        console.print(f"\n[bold]Scoring: {exp.CONTROL_NAME}[/bold]")
        control_result = score_variant(control, test_cases, judge, exp.CONTROL_NAME)

        console.print(f"\n[bold]Scoring: {exp.CHALLENGER_NAME}[/bold]")
        challenger_result = score_variant(challenger, test_cases, judge, exp.CHALLENGER_NAME)
    finally:
        if not keep_indices:
            control.cleanup()
            challenger.cleanup()

    # Statistical comparison
    comparisons = [
        compare_metric(
            m,
            [getattr(s, m) for s in control_result.scores],
            [getattr(s, m) for s in challenger_result.scores],
        )
        for m in METRICS
    ]

    challenger_wins = sum(1 for c in comparisons if c.winner == "challenger")
    control_wins = sum(1 for c in comparisons if c.winner == "control")

    if challenger_wins > control_wins:
        overall = exp.CHALLENGER_NAME
    elif control_wins > challenger_wins:
        overall = exp.CONTROL_NAME
    else:
        overall = "no clear winner"

    result = ExperimentResult(
        experiment_name=exp.EXPERIMENT_NAME,
        control_name=exp.CONTROL_NAME,
        challenger_name=exp.CHALLENGER_NAME,
        control=control_result,
        challenger=challenger_result,
        comparisons=comparisons,
        overall_winner=overall,
    )

    print_experiment(result)

    if save_json:
        save_experiment_json(result)

    # ── Auto-save champion config ────────────────────────────────────────── #
    champion_config = getattr(exp, "CHAMPION_CONFIG", None)
    if champion_config and overall != "no clear winner":
        updates = champion_config.get(overall)
        if updates:
            save_best(updates)
            console.print(f"\n[bold green]✓ Champion config updated:[/bold green] {best_summary()}")
    elif champion_config and overall == "no clear winner":
        console.print(f"\n[dim]Champion config unchanged (no clear winner): {best_summary()}[/dim]")

    return result
