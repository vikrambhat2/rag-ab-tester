"""
pages/3_Results.py — Browse and visualise saved experiment results
"""
import sys
from pathlib import Path

import streamlit as st
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
STREAMLIT_ROOT = Path(__file__).parent.parent
for _p in [str(PROJECT_ROOT), str(STREAMLIT_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.results_loader import list_result_files, load_result, load_all_results
from utils.charts import (
    metric_bar_chart,
    ci_chart,
    scatter_chart,
    per_query_heatmap,
    stats_dataframe,
    color_winner_rows,
    METRICS,
    METRIC_LABELS,
)

st.set_page_config(page_title="Results", page_icon="📊", layout="wide")
st.title("📊 Results")
st.divider()

result_files = list_result_files()

if not result_files:
    st.info(
        "No saved results yet. Run an experiment with **Save result to results/** enabled "
        "on the **Run Experiment** page."
    )
    st.stop()

# ── Sidebar: result selector ─────────────────────────────────────────────── #
with st.sidebar:
    st.header("Saved Results")
    file_labels = {p.stem.replace("_", " ").title(): p for p in result_files}
    chosen_label = st.radio(
        "Select result",
        options=list(file_labels.keys()),
        label_visibility="collapsed",
    )
    selected_file = file_labels[chosen_label]
    st.caption(f"`{selected_file.name}`")

    st.divider()
    show_all = st.checkbox("Compare all results", value=False)

# ── All-results comparison view ───────────────────────────────────────────── #
if show_all:
    st.subheader("All Experiments — Overview")
    all_results = load_all_results()

    if not all_results:
        st.warning("Could not load any results.")
    else:
        rows = []
        for r in all_results:
            row = {
                "Experiment": r.experiment_name,
                "Control": r.control_name,
                "Challenger": r.challenger_name,
                "Overall Winner": r.overall_winner,
            }
            for m in METRICS:
                ctrl_avg = r.control.avg(m)
                chal_avg = r.challenger.avg(m)
                row[f"{METRIC_LABELS[m]} Δ"] = f"{chal_avg - ctrl_avg:+.3f}"
            rows.append(row)

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Multi-experiment metric bar (overall averages)
        import plotly.graph_objects as go
        fig = go.Figure()
        for r in all_results:
            fig.add_trace(go.Bar(
                name=r.experiment_name,
                x=[METRIC_LABELS[m] for m in METRICS],
                y=[r.challenger.avg(m) - r.control.avg(m) for m in METRICS],
                text=[
                    f"{r.challenger.avg(m) - r.control.avg(m):+.3f}"
                    for m in METRICS
                ],
                textposition="outside",
            ))
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.6)
        fig.update_layout(
            title="Challenger − Control Δ Across All Experiments",
            barmode="group",
            yaxis_title="Δ Score",
            height=400,
            margin=dict(t=50, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

# ── Single result view ────────────────────────────────────────────────────── #
result = load_result(selected_file)

if result is None:
    st.error(f"Could not load result from `{selected_file.name}`. The file may be malformed.")
    st.stop()

st.markdown(f"## {result.experiment_name}")
st.caption(f"**Control:** {result.control_name}   |   **Challenger:** {result.challenger_name}")

# KPI cards
k1, k2, k3, k4, k5 = st.columns(5)
overall_ctrl = sum(result.control.avg(m) for m in METRICS) / 4
overall_chal = sum(result.challenger.avg(m) for m in METRICS) / 4
delta_overall = overall_chal - overall_ctrl
wins_chal = sum(1 for c in result.comparisons if c.winner == "challenger")
wins_ctrl = sum(1 for c in result.comparisons if c.winner == "control")

k1.metric("Control Overall", f"{overall_ctrl:.3f}")
k2.metric("Challenger Overall", f"{overall_chal:.3f}", delta=f"{delta_overall:+.3f}")
k3.metric("Queries Evaluated", len(result.control.scores))
k4.metric("Challenger Metric Wins", f"{wins_chal}/4")
k5.metric("Overall Winner", result.overall_winner)

st.divider()

# ── Charts — row 1 ────────────────────────────────────────────────────────── #
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(metric_bar_chart(result), use_container_width=True)
with col2:
    st.plotly_chart(ci_chart(result), use_container_width=True)

# ── Charts — row 2: heatmap ───────────────────────────────────────────────── #
st.plotly_chart(per_query_heatmap(result), use_container_width=True)

# ── Statistical table ─────────────────────────────────────────────────────── #
st.markdown("### Statistical Comparison")
st.caption(
    "**Green rows** = challenger wins (p < 0.05 and |d| ≥ 0.2). "
    "**Yellow rows** = control wins. **Grey rows** = no significant difference."
)
df_stats = stats_dataframe(result)
st.dataframe(color_winner_rows(df_stats), use_container_width=True, hide_index=True)

# Decision framework callout
with st.expander("How to interpret these results"):
    st.markdown(
        """
| p < 0.05 | |d| ≥ 0.2 | Decision |
|:---:|:---:|---|
| ✓ | ✓ | **Act on it** — real and large enough to matter. Switch to the winner. |
| ✓ | ✗ | **Log it, don't ship it** — real but negligible effect. |
| ✗ | any | **Noise** — run more queries or move on. |

**Cohen's d effect sizes:**
- < 0.2 → negligible
- 0.2–0.5 → small
- 0.5–0.8 → medium
- ≥ 0.8 → large
        """
    )

st.divider()

# ── Per-query scatter ─────────────────────────────────────────────────────── #
st.markdown("### Per-Query Scatter")
st.caption("Dots above the diagonal = challenger outperformed control on that query.")

metric_choice = st.selectbox(
    "Metric",
    options=METRICS,
    format_func=lambda m: METRIC_LABELS[m],
    key="results_scatter_metric",
)
st.plotly_chart(scatter_chart(result, metric_choice), use_container_width=True)

# ── Raw data ──────────────────────────────────────────────────────────────── #
with st.expander("Raw Per-Query Data"):
    rows = []
    ctrl_map = {s.query: s for s in result.control.scores}
    chal_map = {s.query: s for s in result.challenger.scores}

    for q, cs in ctrl_map.items():
        ch = chal_map.get(q)
        row = {"Query": q}
        for m in METRICS:
            row[f"Ctrl — {METRIC_LABELS[m]}"] = round(getattr(cs, m), 3)
            if ch:
                row[f"Chal — {METRIC_LABELS[m]}"] = round(getattr(ch, m), 3)
                row[f"Δ {METRIC_LABELS[m]}"] = round(getattr(ch, m) - getattr(cs, m), 3)
        rows.append(row)

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ── Download ──────────────────────────────────────────────────────────────── #
st.divider()
st.download_button(
    label="Download result JSON",
    data=selected_file.read_text(),
    file_name=selected_file.name,
    mime="application/json",
)
