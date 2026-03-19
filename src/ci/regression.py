"""
src/ci/regression.py — Regression detection for CI/CD.

A regression is defined as: the challenger (PR branch) scores significantly
and meaningfully *worse* than the control (main branch) on at least one metric.

Uses the same MetricComparison data already produced by run_experiment.py —
no extra computation needed.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.models.schemas import ExperimentResult, MetricComparison


@dataclass
class RegressionReport:
    regression_detected: bool
    regressed_metrics: list[str]       # metrics where control beat challenger
    improved_metrics: list[str]        # metrics where challenger beat control
    neutral_metrics: list[str]         # no significant difference
    overall_winner: str
    summary_line: str                  # one-liner for logs / PR title


def check_regression(result: ExperimentResult) -> RegressionReport:
    """
    Inspect an ExperimentResult and return a RegressionReport.

    Regression logic:
      - A metric has regressed if winner == "control"
        (challenger is statistically + meaningfully worse).
      - A metric has improved if winner == "challenger".
      - Everything else is neutral (no significant difference).

    Overall regression is flagged only when at least one metric regresses
    AND no metric improves — a mixed result is treated as neutral to avoid
    false positives.
    """
    regressed: list[str] = []
    improved: list[str] = []
    neutral: list[str] = []

    for comp in result.comparisons:
        if comp.winner == "control":
            regressed.append(comp.metric)
        elif comp.winner == "challenger":
            improved.append(comp.metric)
        else:
            neutral.append(comp.metric)

    # Only block CI if there are regressions with NO compensating improvements
    regression_detected = len(regressed) > 0 and len(improved) == 0

    if regression_detected:
        summary = (
            f"❌ Regression detected in {result.experiment_name}: "
            f"{', '.join(regressed)} degraded significantly."
        )
    elif improved:
        summary = (
            f"✅ {result.experiment_name}: {result.overall_winner} improves "
            f"{', '.join(improved)}."
        )
    else:
        summary = (
            f"➡️  {result.experiment_name}: No significant difference detected."
        )

    return RegressionReport(
        regression_detected=regression_detected,
        regressed_metrics=regressed,
        improved_metrics=improved,
        neutral_metrics=neutral,
        overall_winner=result.overall_winner,
        summary_line=summary,
    )


def exit_code(report: RegressionReport, fail_on_regression: bool) -> int:
    """Return shell exit code: 1 to block CI, 0 to pass."""
    if fail_on_regression and report.regression_detected:
        return 1
    return 0
