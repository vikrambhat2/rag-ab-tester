from __future__ import annotations
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from src.models.schemas import ExperimentResult, ABReport, MetricComparison
from src.evaluator.stats import effect_size_label

console = Console()

METRICS = ["faithfulness", "answer_relevance", "context_precision", "context_recall"]

# colour helpers
WINNER_COLOUR = {"challenger": "green", "control": "yellow", "no difference": "dim"}


def _winner_cell(cmp: MetricComparison, name: str) -> str:
    colour = WINNER_COLOUR.get(cmp.winner, "white")
    label = name if cmp.winner in ("challenger", "control") else "no diff"
    return f"[{colour}]{label}[/{colour}]"


def _tick(flag: bool) -> str:
    return "[green]✓[/green]" if flag else "[red]✗[/red]"


def print_experiment(result: ExperimentResult) -> None:
    """Render a rich table for a single ExperimentResult."""
    title = (
        f"[bold cyan]Experiment: {result.experiment_name}[/bold cyan]  |  "
        f"[yellow]{result.control_name}[/yellow] vs "
        f"[green]{result.challenger_name}[/green]"
    )
    console.print(Panel(title, expand=False))

    table = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="bold", min_width=20)
    table.add_column(f"Control\n({result.control_name})", justify="right")
    table.add_column(f"Challenger\n({result.challenger_name})", justify="right")
    table.add_column("Δ", justify="right")
    table.add_column("p-value", justify="right")
    table.add_column("Cohen's d", justify="right")
    table.add_column("95% CI", justify="right")
    table.add_column("Winner", justify="center")

    for cmp in result.comparisons:
        delta_str = f"+{cmp.delta:.3f}" if cmp.delta >= 0 else f"{cmp.delta:.3f}"
        p_str = f"{cmp.p_value:.4f} {_tick(cmp.significant)}"
        d_str = f"{cmp.cohens_d:+.3f} {_tick(cmp.meaningful)} ({effect_size_label(cmp.cohens_d)})"
        ci_str = f"[{cmp.ci_low:.3f}, {cmp.ci_high:.3f}]"

        table.add_row(
            cmp.metric.replace("_", " ").title(),
            f"{cmp.control_avg:.3f}",
            f"{cmp.challenger_avg:.3f}",
            delta_str,
            p_str,
            d_str,
            ci_str,
            _winner_cell(cmp, result.challenger_name if cmp.winner == "challenger" else result.control_name),
        )

    console.print(table)

    # overall winner
    ow_colour = "green" if result.overall_winner == result.challenger_name else \
                "yellow" if result.overall_winner == result.control_name else "dim"
    console.print(
        f"  → Overall winner: [{ow_colour}]{result.overall_winner}[/{ow_colour}]\n"
    )


def print_report(report: ABReport) -> None:
    """Render all experiments in an ABReport."""
    console.rule("[bold]RAG A/B Test Report[/bold]")
    for experiment in report.experiments:
        print_experiment(experiment)
    console.rule()


def save_report_json(report: ABReport, path: str = "results/report.json") -> None:
    """Persist the full ABReport as JSON."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report.model_dump(), indent=2))
    console.print(f"[dim]Report saved → {output}[/dim]")


def save_experiment_json(result: ExperimentResult, results_dir: str = "results") -> None:
    """Persist a single ExperimentResult as JSON."""
    output = Path(results_dir) / f"{result.experiment_name.lower().replace(' ', '_')}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result.model_dump(), indent=2))
    console.print(f"[dim]Result saved → {output}[/dim]")
