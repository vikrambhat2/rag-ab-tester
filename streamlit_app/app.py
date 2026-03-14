"""
app.py — Home dashboard for the RAG A/B Testing Framework
"""
import sys
from pathlib import Path

import streamlit as st

# ── Path bootstrap ──────────────────────────────────────────────────────── #
PROJECT_ROOT = Path(__file__).parent.parent
STREAMLIT_ROOT = Path(__file__).parent
for _p in [str(PROJECT_ROOT), str(STREAMLIT_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.results_loader import (  # noqa: E402
    discover_experiments,
    list_result_files,
    load_result,
    load_test_set,
)

# ── Page config ─────────────────────────────────────────────────────────── #
st.set_page_config(
    page_title="RAG A/B Tester",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Header ───────────────────────────────────────────────────────────────── #
st.title("🔬 RAG A/B Testing Framework")
st.markdown(
    "Plug in any two RAG pipeline variants, get statistical proof. "
    "Fully local — powered by **Ollama**. No API keys required."
)
st.divider()

# ── Status cards ─────────────────────────────────────────────────────────── #
col_ollama, col_testset, col_results, col_experiments = st.columns(4)

# Ollama status
import urllib.request, urllib.error
def check_ollama() -> bool:
    try:
        urllib.request.urlopen("http://localhost:11434", timeout=2)
        return True
    except Exception:
        return False

ollama_ok = check_ollama()
with col_ollama:
    status_icon = "🟢" if ollama_ok else "🔴"
    status_text = "Running" if ollama_ok else "Not reachable"
    st.metric("Ollama", status_text, delta=None)
    if not ollama_ok:
        st.caption("Start with: `ollama serve`")

# Test set
test_cases = load_test_set()
with col_testset:
    st.metric("Test Cases", len(test_cases), delta=None)
    if not test_cases:
        st.caption("Run **Test Set** page to generate")

# Saved results
result_files = list_result_files()
with col_results:
    st.metric("Saved Results", len(result_files), delta=None)

# Available experiments
experiments = discover_experiments()
with col_experiments:
    st.metric("Experiments", len(experiments), delta=None)

st.divider()

# ── Available experiments ────────────────────────────────────────────────── #
st.subheader("Available Experiments")

if experiments:
    exp_data = []
    for p in experiments:
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("_exp", p)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            exp_data.append({
                "File": p.name,
                "Experiment": getattr(mod, "EXPERIMENT_NAME", p.stem),
                "Control": getattr(mod, "CONTROL_NAME", "—"),
                "Challenger": getattr(mod, "CHALLENGER_NAME", "—"),
                "Location": str(p.relative_to(PROJECT_ROOT)),
            })
        except Exception:
            exp_data.append({
                "File": p.name,
                "Experiment": p.stem,
                "Control": "—",
                "Challenger": "—",
                "Location": str(p.relative_to(PROJECT_ROOT)),
            })

    import pandas as pd
    st.dataframe(
        pd.DataFrame(exp_data),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("No experiment files found in `experiments/`.")

# ── Latest results ───────────────────────────────────────────────────────── #
if result_files:
    st.divider()
    st.subheader("Recent Results")

    recent_rows = []
    for p in result_files[:10]:
        r = load_result(p)
        if r:
            challenger_wins = sum(1 for c in r.comparisons if c.winner == "challenger")
            control_wins = sum(1 for c in r.comparisons if c.winner == "control")
            recent_rows.append({
                "Experiment": r.experiment_name,
                "Control": r.control_name,
                "Challenger": r.challenger_name,
                "Overall Winner": r.overall_winner,
                "Challenger Wins": f"{challenger_wins}/4 metrics",
                "File": p.name,
            })

    if recent_rows:
        import pandas as pd
        df = pd.DataFrame(recent_rows)

        def highlight_winner(row):
            if row["Overall Winner"] == row["Challenger"]:
                return [""] * (len(row) - 2) + ["color: green; font-weight: bold", ""]
            if row["Overall Winner"] == row["Control"]:
                return [""] * (len(row) - 2) + ["color: orange; font-weight: bold", ""]
            return [""] * len(row)

        st.dataframe(df.drop(columns=["File"]), use_container_width=True, hide_index=True)

# ── Quick-start guide ────────────────────────────────────────────────────── #
st.divider()
st.subheader("Quick Start")

cols = st.columns(4)
steps = [
    ("1. Generate Test Set", "Navigate to **Test Set** to generate Q&A pairs from your docs.", "📋"),
    ("2. Run an Experiment", "Go to **Run Experiment**, pick a variant pair, and click Run.", "🚀"),
    ("3. View Results", "Open **Results** for charts, statistical tables, and per-query analysis.", "📊"),
    ("4. Add Your Own", "Drop a `.py` file in `experiments/custom/` — 10 lines of code.", "🧩"),
]
for col, (title, body, icon) in zip(cols, steps):
    with col:
        st.markdown(f"### {icon} {title}")
        st.markdown(body)
