"""
pages/1_Run_Experiment.py — Run a single A/B experiment with live log output
"""
import sys
import time
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent.parent
STREAMLIT_ROOT = Path(__file__).parent.parent
for _p in [str(PROJECT_ROOT), str(STREAMLIT_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.results_loader import (
    discover_experiments,
    load_test_set,
    list_result_files,
    load_result,
)
from utils.runner import stream_subprocess, PROJECT_ROOT
from utils.charts import (
    metric_bar_chart,
    ci_chart,
    scatter_chart,
    stats_dataframe,
    color_winner_rows,
    METRICS,
    METRIC_LABELS,
)

st.set_page_config(page_title="Run Experiment", page_icon="🚀", layout="wide")
st.title("🚀 Run Experiment")
st.markdown("Select an experiment, configure options, and run it with live log output.")
st.divider()

# ── Experiment selector ───────────────────────────────────────────────────── #
experiments = discover_experiments()

if not experiments:
    st.error("No experiment files found in `experiments/`. Check your project layout.")
    st.stop()

exp_options = {
    str(p.relative_to(PROJECT_ROOT)): p for p in experiments
}
selected_label = st.selectbox(
    "Experiment",
    options=list(exp_options.keys()),
    help="Experiments in `experiments/` and `experiments/custom/`",
)
selected_path = exp_options[selected_label]

# Preview experiment metadata
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("_exp_preview", selected_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    exp_name = getattr(mod, "EXPERIMENT_NAME", selected_path.stem)
    ctrl_name = getattr(mod, "CONTROL_NAME", "—")
    chal_name = getattr(mod, "CHALLENGER_NAME", "—")

    ci1, ci2, ci3 = st.columns(3)
    ci1.metric("Experiment", exp_name)
    ci2.metric("Control", ctrl_name)
    ci3.metric("Challenger", chal_name)
except Exception:
    st.caption(f"Could not parse metadata from `{selected_label}`")

# ── Options ───────────────────────────────────────────────────────────────── #
st.divider()
with st.expander("Options", expanded=False):
    test_set_path = st.text_input(
        "Test set path",
        value="data/test_set.json",
        help="Relative to project root",
    )
    save_json = st.checkbox("Save result to results/", value=True)

test_cases = load_test_set(PROJECT_ROOT / test_set_path)
if not test_cases:
    st.warning(
        f"No test cases found at `{test_set_path}`. "
        "Go to the **Test Set** page to generate them first."
    )

# ── Run button ────────────────────────────────────────────────────────────── #
st.divider()
run_col, _ = st.columns([1, 3])
run_clicked = run_col.button(
    "▶ Run Experiment",
    type="primary",
    disabled=not test_cases,
    use_container_width=True,
)

if run_clicked:
    st.session_state.pop("run_result", None)

    cmd = [
        "python", "run_experiment.py",
        "--experiment", str(selected_path.relative_to(PROJECT_ROOT)),
        "--test-set", test_set_path,
    ]
    if save_json:
        cmd.append("--save-json")

    st.markdown("### Live Output")
    log_placeholder = st.empty()
    start = time.time()

    with st.spinner("Running experiment…"):
        returncode = stream_subprocess(cmd, log_placeholder)

    elapsed = time.time() - start
    if returncode == 0:
        st.success(f"Experiment completed in {elapsed:.1f}s")
    else:
        st.error(f"Experiment failed (exit code {returncode}). Check the log above.")

    # Load the freshest result file
    if save_json:
        result_files = list_result_files()
        if result_files:
            latest = result_files[0]
            result = load_result(latest)
            if result and result.experiment_name == exp_name:
                st.session_state["run_result"] = result
                st.session_state["run_result_name"] = exp_name

# ── Inline results ────────────────────────────────────────────────────────── #
result = st.session_state.get("run_result")
if result:
    st.divider()
    st.markdown(f"## Results — {result.experiment_name}")
    st.caption(f"{result.control_name}  vs  {result.challenger_name}")

    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    overall_avg_ctrl = sum(result.control.avg(m) for m in METRICS) / 4
    overall_avg_chal = sum(result.challenger.avg(m) for m in METRICS) / 4
    delta_overall = overall_avg_chal - overall_avg_ctrl

    k1.metric("Control Overall", f"{overall_avg_ctrl:.3f}")
    k2.metric("Challenger Overall", f"{overall_avg_chal:.3f}", delta=f"{delta_overall:+.3f}")
    wins_c = sum(1 for c in result.comparisons if c.winner == "challenger")
    wins_ctrl = sum(1 for c in result.comparisons if c.winner == "control")
    k3.metric("Challenger Metric Wins", f"{wins_c}/4")
    winner_colour = "green" if result.overall_winner == result.challenger_name else "orange"
    k4.metric("Overall Winner", result.overall_winner)

    # Charts
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.plotly_chart(metric_bar_chart(result), use_container_width=True)
    with chart_col2:
        st.plotly_chart(ci_chart(result), use_container_width=True)

    # Statistical table
    st.markdown("### Statistical Comparison")
    df = stats_dataframe(result)
    st.dataframe(color_winner_rows(df), use_container_width=True, hide_index=True)

    # Decision guide
    st.markdown("### How to Read This")
    st.markdown(
        """
| p < 0.05 | |d| ≥ 0.2 | Decision |
|---|---|---|
| ✓ | ✓ | Real and meaningful — switch to winner |
| ✓ | ✗ | Real but negligible — log it, don't act |
| ✗ | any | Noise — run more queries or move on |
        """
    )

    # Per-query scatter
    st.markdown("### Per-Query Scores")
    metric_choice = st.selectbox(
        "Metric for scatter",
        options=METRICS,
        format_func=lambda m: METRIC_LABELS[m],
        key="scatter_metric",
    )
    st.plotly_chart(scatter_chart(result, metric_choice), use_container_width=True)

    # Raw data table
    with st.expander("Raw Per-Query Data"):
        import pandas as pd
        rows = []
        ctrl_map = {s.query: s for s in result.control.scores}
        chal_map = {s.query: s for s in result.challenger.scores}
        for q in ctrl_map:
            cs = ctrl_map[q]
            ch = chal_map.get(q)
            row = {"Query": q[:80]}
            for m in METRICS:
                row[f"{result.control_name} — {METRIC_LABELS[m]}"] = f"{getattr(cs, m):.3f}"
                if ch:
                    row[f"{result.challenger_name} — {METRIC_LABELS[m]}"] = f"{getattr(ch, m):.3f}"
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
