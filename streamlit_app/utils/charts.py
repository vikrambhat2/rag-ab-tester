"""
utils/charts.py — Plotly chart builders for ExperimentResult visualisation
"""
from __future__ import annotations
import sys
from pathlib import Path

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.schemas import ExperimentResult  # noqa: E402
from src.evaluator.stats import effect_size_label  # noqa: E402

METRICS = ["faithfulness", "answer_relevance", "context_precision", "context_recall"]
METRIC_LABELS = {
    "faithfulness": "Faithfulness",
    "answer_relevance": "Answer Relevance",
    "context_precision": "Context Precision",
    "context_recall": "Context Recall",
}

CONTROL_COLOR = "#636EFA"
CHALLENGER_COLOR = "#EF553B"
WIN_COLORS = {
    "challenger": "#00CC96",
    "control": "#FFA15A",
    "no difference": "#B0B0B0",
}


def metric_bar_chart(result: ExperimentResult) -> go.Figure:
    """Grouped bar chart comparing 4 metrics across control and challenger."""
    labels = [METRIC_LABELS[m] for m in METRICS]
    control_avgs = [result.control.avg(m) for m in METRICS]
    challenger_avgs = [result.challenger.avg(m) for m in METRICS]

    fig = go.Figure(data=[
        go.Bar(name=result.control_name, x=labels, y=control_avgs,
               marker_color=CONTROL_COLOR, text=[f"{v:.3f}" for v in control_avgs],
               textposition="outside"),
        go.Bar(name=result.challenger_name, x=labels, y=challenger_avgs,
               marker_color=CHALLENGER_COLOR, text=[f"{v:.3f}" for v in challenger_avgs],
               textposition="outside"),
    ])
    fig.update_layout(
        barmode="group",
        title="Average Scores by Metric",
        yaxis=dict(title="Score", range=[0, 1.15]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=380,
        margin=dict(t=60, b=40),
    )
    return fig


def ci_chart(result: ExperimentResult) -> go.Figure:
    """Horizontal error-bar chart showing 95% CI of challenger − control per metric."""
    rows = []
    for cmp in result.comparisons:
        rows.append({
            "metric": METRIC_LABELS[cmp.metric],
            "delta": cmp.delta,
            "ci_low": cmp.ci_low,
            "ci_high": cmp.ci_high,
            "winner": cmp.winner,
            "significant": cmp.significant,
            "meaningful": cmp.meaningful,
        })

    fig = go.Figure()
    for row in rows:
        colour = WIN_COLORS.get(row["winner"], "#B0B0B0")
        fig.add_trace(go.Scatter(
            x=[row["delta"]],
            y=[row["metric"]],
            mode="markers",
            marker=dict(color=colour, size=12, symbol="diamond"),
            error_x=dict(
                type="data",
                symmetric=False,
                array=[row["ci_high"] - row["delta"]],
                arrayminus=[row["delta"] - row["ci_low"]],
                color=colour,
                thickness=2,
                width=6,
            ),
            name=row["winner"],
            showlegend=False,
            hovertemplate=(
                f"<b>{row['metric']}</b><br>"
                f"Δ = {row['delta']:+.3f}<br>"
                f"95% CI [{row['ci_low']:.3f}, {row['ci_high']:.3f}]<br>"
                f"Winner: {row['winner']}<extra></extra>"
            ),
        ))

    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.6)
    fig.update_layout(
        title="95% Confidence Interval of Δ (Challenger − Control)",
        xaxis_title="Δ Score",
        height=300,
        margin=dict(t=50, b=40, l=160),
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(autorange="reversed"),
    )
    return fig


def scatter_chart(result: ExperimentResult, metric: str = "faithfulness") -> go.Figure:
    """Per-query scatter: x=control score, y=challenger score. Dots above diagonal = challenger wins."""
    control_scores = [getattr(s, metric) for s in result.control.scores]
    challenger_scores = [getattr(s, metric) for s in result.challenger.scores]
    queries = [s.query[:60] + "…" if len(s.query) > 60 else s.query for s in result.control.scores]

    colours = [
        CHALLENGER_COLOR if c > ct else CONTROL_COLOR if ct > c else "#B0B0B0"
        for c, ct in zip(challenger_scores, control_scores)
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=control_scores,
        y=challenger_scores,
        mode="markers",
        marker=dict(color=colours, size=8, opacity=0.8),
        text=queries,
        hovertemplate="<b>%{text}</b><br>Control: %{x:.3f}<br>Challenger: %{y:.3f}<extra></extra>",
        name="Query",
    ))
    # Diagonal reference line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(dash="dash", color="gray", width=1),
        name="Equal",
        showlegend=False,
    ))
    fig.update_layout(
        title=f"Per-Query Scores — {METRIC_LABELS.get(metric, metric)}",
        xaxis_title=f"{result.control_name}",
        yaxis_title=f"{result.challenger_name}",
        xaxis=dict(range=[-0.05, 1.05]),
        yaxis=dict(range=[-0.05, 1.05]),
        height=380,
        margin=dict(t=50, b=40),
    )
    return fig


def per_query_heatmap(result: ExperimentResult) -> go.Figure:
    """Heatmap of per-query scores for both variants across all 4 metrics."""
    rows = []
    for s in result.control.scores:
        for m in METRICS:
            rows.append({
                "query": s.query[:50] + "…",
                "variant": result.control_name,
                "metric": METRIC_LABELS[m],
                "score": getattr(s, m),
            })
    for s in result.challenger.scores:
        for m in METRICS:
            rows.append({
                "query": s.query[:50] + "…",
                "variant": result.challenger_name,
                "metric": METRIC_LABELS[m],
                "score": getattr(s, m),
            })

    df = pd.DataFrame(rows)
    df_pivot = df.groupby(["variant", "metric"])["score"].mean().unstack()

    fig = px.imshow(
        df_pivot,
        color_continuous_scale="RdYlGn",
        zmin=0, zmax=1,
        text_auto=".3f",
        title="Average Score Heatmap (Variant × Metric)",
        aspect="auto",
    )
    fig.update_layout(height=250, margin=dict(t=50, b=40))
    return fig


def stats_dataframe(result: ExperimentResult) -> pd.DataFrame:
    """Build a styled DataFrame of the statistical comparison table."""
    rows = []
    for cmp in result.comparisons:
        delta_str = f"{cmp.delta:+.3f}"
        rows.append({
            "Metric": METRIC_LABELS[cmp.metric],
            "Control": f"{cmp.control_avg:.3f}",
            "Challenger": f"{cmp.challenger_avg:.3f}",
            "Δ": delta_str,
            "p-value": f"{cmp.p_value:.4f}",
            "Sig. (p<0.05)": "✓" if cmp.significant else "✗",
            "Cohen's d": f"{cmp.cohens_d:+.3f}",
            "Effect": effect_size_label(cmp.cohens_d),
            "Meaningful": "✓" if cmp.meaningful else "✗",
            "Winner": cmp.winner,
            "_winner_raw": cmp.winner,
        })
    return pd.DataFrame(rows)


def color_winner_rows(df: pd.DataFrame) -> pd.DataFrame.style:
    """Apply row background colours based on the winner column."""
    def row_color(row):
        w = row["_winner_raw"]
        if w == "challenger":
            return ["background-color: #d4edda"] * len(row)
        if w == "control":
            return ["background-color: #fff3cd"] * len(row)
        return ["background-color: #f8f9fa"] * len(row)

    return (
        df.drop(columns=["_winner_raw"])
        .style.apply(row_color, axis=1)
        .set_properties(**{"text-align": "center"})
        .set_table_styles([
            {"selector": "th", "props": [("text-align", "center"), ("font-weight", "bold")]},
        ])
    )
