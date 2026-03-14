"""
pages/2_Run_All.py — Run all experiments in batch with live logs and summary
"""
import sys
import time
from pathlib import Path

import streamlit as st
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
STREAMLIT_ROOT = Path(__file__).parent.parent
for _p in [str(PROJECT_ROOT), str(STREAMLIT_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.results_loader import discover_experiments, load_all_results, load_test_set
from utils.runner import stream_subprocess

st.set_page_config(page_title="Run All Experiments", page_icon="⚡", layout="wide")
st.title("⚡ Run All Experiments")
st.markdown("Run every experiment in `experiments/` in sequence and see a combined summary.")
st.divider()

# ── Experiment list ───────────────────────────────────────────────────────── #
experiments = discover_experiments()

if not experiments:
    st.error("No experiment files found in `experiments/`.")
    st.stop()

# Load metadata for display
exp_meta = []
for p in experiments:
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("_ep", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        exp_meta.append({
            "label": getattr(mod, "EXPERIMENT_NAME", p.stem),
            "path": p,
            "control": getattr(mod, "CONTROL_NAME", "—"),
            "challenger": getattr(mod, "CHALLENGER_NAME", "—"),
        })
    except Exception:
        exp_meta.append({"label": p.stem, "path": p, "control": "—", "challenger": "—"})

all_labels = [m["label"] for m in exp_meta]
selected_labels = st.multiselect(
    "Select experiments to run",
    options=all_labels,
    default=all_labels,
    help="Deselect any experiments you want to skip",
)

# ── Options ───────────────────────────────────────────────────────────────── #
with st.expander("Options", expanded=False):
    test_set_path = st.text_input("Test set path", value="data/test_set.json")
    save_json = st.checkbox("Save results to results/", value=True)

test_cases = load_test_set(PROJECT_ROOT / test_set_path)
if not test_cases:
    st.warning(
        f"No test cases found at `{test_set_path}`. Generate them on the **Test Set** page first."
    )

st.divider()

# ── Run button ────────────────────────────────────────────────────────────── #
selected_meta = [m for m in exp_meta if m["label"] in selected_labels]
run_col, info_col = st.columns([1, 3])
run_clicked = run_col.button(
    f"▶ Run {len(selected_meta)} Experiments",
    type="primary",
    disabled=not test_cases or not selected_meta,
    use_container_width=True,
)
info_col.markdown(
    f"Will run **{len(selected_meta)}** experiment(s) sequentially. "
    f"Each experiment ingests two variants and scores **{len(test_cases)} queries**."
)

if run_clicked:
    st.session_state.pop("all_results_summary", None)

    all_summaries = []

    for i, meta in enumerate(selected_meta, 1):
        rel_path = str(meta["path"].relative_to(PROJECT_ROOT))
        st.markdown(f"### [{i}/{len(selected_meta)}] {meta['label']}")
        st.caption(f"`{rel_path}` — {meta['control']} vs {meta['challenger']}")

        log_placeholder = st.empty()
        cmd = [
            "python", "run_experiment.py",
            "--experiment", rel_path,
            "--test-set", test_set_path,
        ]
        if save_json:
            cmd.append("--save-json")

        start = time.time()
        with st.spinner(f"Running {meta['label']}…"):
            returncode = stream_subprocess(cmd, log_placeholder)
        elapsed = time.time() - start

        if returncode == 0:
            st.success(f"Completed in {elapsed:.1f}s")
            all_summaries.append({
                "label": meta["label"], "status": "ok", "elapsed": elapsed
            })
        else:
            st.error(f"Failed (exit code {returncode})")
            all_summaries.append({
                "label": meta["label"], "status": "error", "elapsed": elapsed
            })

    st.session_state["all_results_summary"] = all_summaries

# ── Summary table ─────────────────────────────────────────────────────────── #
if "all_results_summary" in st.session_state:
    st.divider()
    st.markdown("## Batch Summary")

    summaries = st.session_state["all_results_summary"]
    ok_count = sum(1 for s in summaries if s["status"] == "ok")
    st.metric("Completed", f"{ok_count}/{len(summaries)}")

    # Load saved results and show winners
    saved = load_all_results()
    saved_map = {r.experiment_name: r for r in saved}

    rows = []
    for s in summaries:
        r = saved_map.get(s["label"])
        rows.append({
            "Experiment": s["label"],
            "Status": "✓" if s["status"] == "ok" else "✗",
            "Time (s)": f"{s['elapsed']:.1f}",
            "Overall Winner": r.overall_winner if r else "—",
            "Challenger Wins": (
                f"{sum(1 for c in r.comparisons if c.winner == 'challenger')}/4"
                if r else "—"
            ),
        })

    df = pd.DataFrame(rows)

    def highlight_status(row):
        if row["Status"] == "✓":
            return [""] * len(row)
        return ["color: red"] * len(row)

    st.dataframe(
        df.style.apply(highlight_status, axis=1),
        use_container_width=True,
        hide_index=True,
    )
    st.info("Open the **Results** page for detailed charts and statistical analysis.")
