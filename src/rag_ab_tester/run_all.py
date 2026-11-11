"""Run several experiments in sequence and summarise the results."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.markup import escape
from rich.rule import Rule

from rag_ab_tester.models.schemas import ABReport
from rag_ab_tester.protocols import Embedder, VectorIndex
from rag_ab_tester.report.report import save_report_json
from rag_ab_tester.runner import builtin_experiments, run_experiment

console = Console()


def select_experiments(
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    extra_dir: str | Path | None = None,
) -> list[Path]:
    """Built-in experiments (plus any ``*.py`` in ``extra_dir``), filtered by name substrings."""
    files = list(builtin_experiments().values())
    if extra_dir:
        files += sorted(p for p in Path(extra_dir).glob("*.py") if not p.name.startswith("_"))
    if include:
        files = [p for p in files if any(inc in p.stem for inc in include)]
    if exclude:
        files = [p for p in files if not any(exc in p.stem for exc in exclude)]
    return files


def run_all(
    experiment_files: list[Path],
    test_set_path: str | Path = "data/test_set.json",
    save_json: bool = False,
    keep_indices: bool = False,
    docs_path: str | Path = "data/docs",
    *,
    llm: Any = None,
    judge_llm: Any = None,
    embedder: Embedder | None = None,
    index_factory: Callable[[str], VectorIndex] | None = None,
) -> tuple[ABReport, list[str]]:
    """
    Run each experiment; return the combined report and the names that failed.

    ``llm``, ``judge_llm``, ``embedder`` and ``index_factory`` are passed to every experiment
    (see :func:`rag_ab_tester.run_experiment`).
    """
    console.print(Rule("[bold]RAG A/B Testing Framework[/bold]"))
    console.print(f"Found [bold]{len(experiment_files)}[/bold] experiment(s) to run:\n")
    for p in experiment_files:
        console.print(f"  • {p.stem}")
    console.print()

    report = ABReport()
    failed: list[str] = []

    for experiment_path in experiment_files:
        console.print(Rule(f"[cyan]{experiment_path.stem}[/cyan]"))
        try:
            report.experiments.append(
                run_experiment(
                    experiment_path,
                    test_set_path=test_set_path,
                    save_json=save_json,
                    keep_indices=keep_indices,
                    docs_path=docs_path,
                    llm=llm,
                    judge_llm=judge_llm,
                    embedder=embedder,
                    index_factory=index_factory,
                )
            )
        except Exception as e:
            console.print(f"[red]Failed to run {experiment_path.name}: {escape(str(e))}[/red]")
            failed.append(experiment_path.stem)

    console.print(Rule("[bold]Summary[/bold]"))
    for exp_result in report.experiments:
        ow = exp_result.overall_winner
        colour = (
            "green"
            if ow == exp_result.challenger_name
            else "yellow"
            if ow == exp_result.control_name
            else "dim"
        )
        console.print(f"  {exp_result.experiment_name:30s}  → [{colour}]{ow}[/{colour}]")

    if failed:
        console.print(f"\n[red]Failed experiments:[/red] {', '.join(failed)}")
    if save_json and report.experiments:
        save_report_json(report)
    return report, failed
