"""
run_all.py — Discover and run every experiment in experiments/

Usage:
    python run_all.py                          # runs all experiments/
    python run_all.py --include chunking prompt # only those two
    python run_all.py --exclude embedding       # skip embedding
    python run_all.py --save-json              # persist all results to results/

The runner skips __init__.py and the custom/ subdirectory by default.
To run custom experiments, use run_experiment.py directly.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.rule import Rule

from run_experiment import run
from src.models.schemas import ABReport
from src.report.report import print_report, save_report_json

console = Console()


def discover_experiments(experiments_dir: str = "experiments") -> list[Path]:
    """Return all .py files in experiments/ (excluding __init__ and custom/)."""
    root = Path(experiments_dir)
    return sorted(
        p
        for p in root.glob("*.py")
        if p.name != "__init__.py"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all RAG A/B experiments.")
    parser.add_argument(
        "--experiments-dir",
        default="experiments",
        help="Directory containing experiment .py files",
    )
    parser.add_argument(
        "--test-set",
        default="data/test_set.json",
        help="Path to test_set.json",
    )
    parser.add_argument(
        "--include",
        nargs="+",
        metavar="NAME",
        help="Only run experiments whose filename contains these substrings",
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        metavar="NAME",
        help="Skip experiments whose filename contains these substrings",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Save individual results + combined report to results/",
    )
    args = parser.parse_args()

    experiment_files = discover_experiments(args.experiments_dir)

    if not experiment_files:
        console.print(f"[red]No experiment files found in '{args.experiments_dir}'.[/red]")
        sys.exit(1)

    # Apply include/exclude filters
    if args.include:
        experiment_files = [
            p for p in experiment_files
            if any(inc in p.stem for inc in args.include)
        ]
    if args.exclude:
        experiment_files = [
            p for p in experiment_files
            if not any(exc in p.stem for exc in args.exclude)
        ]

    if not experiment_files:
        console.print("[yellow]No experiments matched the given filters.[/yellow]")
        sys.exit(0)

    console.print(Rule("[bold]RAG A/B Testing Framework[/bold]"))
    console.print(f"Found [bold]{len(experiment_files)}[/bold] experiment(s) to run:\n")
    for p in experiment_files:
        console.print(f"  • {p}")
    console.print()

    report = ABReport()
    failed: list[str] = []

    for experiment_path in experiment_files:
        console.print(Rule(f"[cyan]{experiment_path.stem}[/cyan]"))
        try:
            result = run(
                str(experiment_path),
                test_set_path=args.test_set,
                save_json=args.save_json,
            )
            report.experiments.append(result)
        except Exception as e:
            console.print(f"[red]Failed to run {experiment_path.name}: {e}[/red]")
            failed.append(str(experiment_path))

    # Summary
    console.print(Rule("[bold]Summary[/bold]"))
    for exp_result in report.experiments:
        ow = exp_result.overall_winner
        colour = (
            "green" if ow == exp_result.challenger_name
            else "yellow" if ow == exp_result.control_name
            else "dim"
        )
        console.print(
            f"  {exp_result.experiment_name:30s}  → [{colour}]{ow}[/{colour}]"
        )

    if failed:
        console.print(f"\n[red]Failed experiments:[/red] {', '.join(failed)}")

    if args.save_json:
        save_report_json(report)


if __name__ == "__main__":
    main()
