"""
run_experiment.py — Run a single A/B experiment

Usage:
    python run_experiment.py --experiment experiments/chunking.py
    python run_experiment.py --experiment experiments/custom/my_experiment.py
    python run_experiment.py --experiment experiments/chunking.py --test-set data/test_set.json
    python run_experiment.py --experiment experiments/chunking.py --save-json
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import sys
from pathlib import Path

from rich.console import Console

from src.best_config import save as save_best_config
from src.evaluator.judge import OllamaJudge
from src.evaluator.metrics import (
    faithfulness_score,
    answer_relevance_score,
    context_precision_score,
    context_recall_score,
)
from src.evaluator.stats import compare_metric
from src.models.schemas import ExperimentResult, QueryScore, VariantResult
from src.report.report import print_experiment, save_experiment_json

console = Console()

METRICS = ["faithfulness", "answer_relevance", "context_precision", "context_recall"]


# ──────────────────────────────────────────────────────────────────────────── #
#  Helpers                                                                     #
# ──────────────────────────────────────────────────────────────────────────── #

def load_experiment(path: str):
    """Dynamically load an experiment module from a file path."""
    spec = importlib.util.spec_from_file_location("experiment", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load experiment from '{path}'")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]

    required = ["EXPERIMENT_NAME", "CONTROL", "CHALLENGER", "CONTROL_NAME", "CHALLENGER_NAME"]
    missing = [attr for attr in required if not hasattr(mod, attr)]
    if missing:
        raise AttributeError(
            f"Experiment module is missing required attributes: {missing}"
        )
    return mod


def score_variant(
    pipeline,
    test_cases: list[dict],
    judge: OllamaJudge,
    name: str,
) -> VariantResult:
    """Run all test cases through a pipeline variant and score each one."""
    scores: list[QueryScore] = []
    for i, tc in enumerate(test_cases):
        console.print(
            f"  [{name}] {i + 1}/{len(test_cases)}: {tc['query'][:60]}..."
        )
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
            console.print(f"  [red]Error on query {i + 1}: {e}[/red]")
            scores.append(
                QueryScore(
                    query=tc["query"],
                    faithfulness=0.0,
                    answer_relevance=0.0,
                    context_precision=0.0,
                    context_recall=0.0,
                )
            )

    return VariantResult(variant_name=name, scores=scores)


# ──────────────────────────────────────────────────────────────────────────── #
#  Main                                                                        #
# ──────────────────────────────────────────────────────────────────────────── #

def run(experiment_path: str, test_set_path: str, save_json: bool) -> ExperimentResult:
    exp = load_experiment(experiment_path)

    test_set_file = Path(test_set_path)
    if not test_set_file.exists():
        console.print(
            f"[red]Test set not found at '{test_set_path}'. Run python ingest.py first.[/red]"
        )
        sys.exit(1)

    test_cases: list[dict] = json.loads(test_set_file.read_text())
    if not test_cases:
        console.print("[red]Test set is empty. Re-run python ingest.py.[/red]")
        sys.exit(1)

    console.print(
        f"\n[bold cyan]{'=' * 60}[/bold cyan]\n"
        f"[bold]Experiment:[/bold] {exp.EXPERIMENT_NAME}\n"
        f"[bold]Control:[/bold]    {exp.CONTROL_NAME}\n"
        f"[bold]Challenger:[/bold] {exp.CHALLENGER_NAME}\n"
        f"[bold]Test cases:[/bold] {len(test_cases)}\n"
        f"[bold cyan]{'=' * 60}[/bold cyan]"
    )

    judge = OllamaJudge()

    # Instantiate pipelines with isolated Chroma collections
    control = exp.CONTROL(
        collection_name=f"ctrl_{exp.CONTROL_NAME.replace(' ', '_')}",
        persist_dir=f"./.chroma/ctrl_{exp.CONTROL_NAME.replace(' ', '_')}",
    )
    challenger = exp.CHALLENGER(
        collection_name=f"chal_{exp.CHALLENGER_NAME.replace(' ', '_')}",
        persist_dir=f"./.chroma/chal_{exp.CHALLENGER_NAME.replace(' ', '_')}",
    )

    console.print(f"\n[bold]Ingesting: {exp.CONTROL_NAME}[/bold]")
    control.ingest()

    console.print(f"[bold]Ingesting: {exp.CHALLENGER_NAME}[/bold]")
    challenger.ingest()

    console.print(f"\n[bold]Scoring: {exp.CONTROL_NAME}[/bold]")
    control_result = score_variant(control, test_cases, judge, exp.CONTROL_NAME)

    console.print(f"\n[bold]Scoring: {exp.CHALLENGER_NAME}[/bold]")
    challenger_result = score_variant(challenger, test_cases, judge, exp.CHALLENGER_NAME)

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

    # Auto-update champion config if the experiment defines CHAMPION_CONFIG
    champion_config = getattr(exp, "CHAMPION_CONFIG", None)
    if champion_config and overall in champion_config:
        updates = champion_config[overall]
        save_best_config(updates)
        console.print(
            f"[bold green]Champion config updated:[/bold green] {updates}"
        )

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a single RAG A/B experiment.")
    parser.add_argument("--experiment", required=True, help="Path to experiment .py file")
    parser.add_argument(
        "--test-set", default="data/test_set.json", help="Path to test_set.json"
    )
    parser.add_argument(
        "--save-json", action="store_true", help="Save result to results/<name>.json"
    )
    args = parser.parse_args()

    run(args.experiment, args.test_set, args.save_json)
