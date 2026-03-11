from __future__ import annotations
import numpy as np
from scipy import stats
from src.models.schemas import MetricComparison


def cohens_d(a: list[float], b: list[float]) -> float:
    """
    Compute Cohen's d effect size between two paired samples.
    Uses pooled standard deviation.
    Positive value means b > a (challenger beats control).
    """
    mean_diff = np.mean(b) - np.mean(a)
    pooled_std = np.sqrt(
        (
            (len(a) - 1) * np.std(a, ddof=1) ** 2
            + (len(b) - 1) * np.std(b, ddof=1) ** 2
        )
        / (len(a) + len(b) - 2)
    )
    return float(mean_diff / pooled_std) if pooled_std > 0 else 0.0


def confidence_interval(
    a: list[float], b: list[float], confidence: float = 0.95
) -> tuple[float, float]:
    """
    95% confidence interval for the mean difference (b - a).
    Uses paired differences and the t-distribution.
    """
    diffs = np.array(b) - np.array(a)
    se = stats.sem(diffs)
    alpha = 1 - confidence
    margin = se * stats.t.ppf(1 - alpha / 2, df=len(diffs) - 1)
    mean = float(np.mean(diffs))
    return mean - float(margin), mean + float(margin)


def compare_metric(
    metric: str,
    control_scores: list[float],
    challenger_scores: list[float],
) -> MetricComparison:
    """
    Run a paired t-test and compute Cohen's d + CI for one metric.

    Decision logic:
        significant AND meaningful → challenger/control wins
        significant but negligible → "no difference" (not worth acting on)
        not significant            → "no difference" (could be noise)
    """
    control_avg = float(np.mean(control_scores))
    challenger_avg = float(np.mean(challenger_scores))
    delta = challenger_avg - control_avg

    _, p_value = stats.ttest_rel(control_scores, challenger_scores)
    d = cohens_d(control_scores, challenger_scores)
    ci_low, ci_high = confidence_interval(control_scores, challenger_scores)

    significant = bool(p_value < 0.05)
    meaningful = bool(abs(d) >= 0.2)

    if significant and meaningful:
        winner = "challenger" if delta > 0 else "control"
    else:
        winner = "no difference"

    return MetricComparison(
        metric=metric,
        control_avg=round(control_avg, 3),
        challenger_avg=round(challenger_avg, 3),
        delta=round(delta, 3),
        p_value=round(float(p_value), 4),
        cohens_d=round(d, 3),
        ci_low=round(ci_low, 3),
        ci_high=round(ci_high, 3),
        significant=significant,
        meaningful=meaningful,
        winner=winner,
    )


def effect_size_label(d: float) -> str:
    """Human-readable label for Cohen's d magnitude."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    if abs_d < 0.5:
        return "small"
    if abs_d < 0.8:
        return "medium"
    return "large"
