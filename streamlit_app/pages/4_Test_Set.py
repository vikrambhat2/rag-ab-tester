"""
pages/4_Test_Set.py — View and regenerate the test set
"""
import sys
import json
from pathlib import Path

import streamlit as st
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
STREAMLIT_ROOT = Path(__file__).parent.parent
for _p in [str(PROJECT_ROOT), str(STREAMLIT_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.results_loader import load_test_set
from utils.runner import stream_subprocess

st.set_page_config(page_title="Test Set", page_icon="📋", layout="wide")
st.title("📋 Test Set")
st.markdown(
    "The test set is a JSON file of question + ground-truth answer pairs "
    "auto-generated from your documents. "
    "Both variants in an experiment are evaluated against the same test set."
)
st.divider()

TEST_SET_PATH = PROJECT_ROOT / "data" / "test_set.json"
DOCS_PATH = PROJECT_ROOT / "data" / "docs"

# ── Current test set stats ────────────────────────────────────────────────── #
test_cases = load_test_set(TEST_SET_PATH)

col_count, col_docs, col_path = st.columns(3)
col_count.metric("QA Pairs", len(test_cases))

doc_files = list(DOCS_PATH.glob("**/*.md")) if DOCS_PATH.exists() else []
col_docs.metric("Source Documents", len(doc_files))
col_path.metric("Path", str(TEST_SET_PATH.relative_to(PROJECT_ROOT)))

# ── Test set table ────────────────────────────────────────────────────────── #
if test_cases:
    st.divider()
    st.subheader("Current Test Cases")

    search = st.text_input("Filter questions", placeholder="Type to search…")
    df = pd.DataFrame([
        {
            "#": i + 1,
            "Question": tc["query"],
            "Ground Truth": tc["ground_truth"],
            "Context Snippet": tc.get("context_snippet", "—")[:120],
        }
        for i, tc in enumerate(test_cases)
    ])
    if search:
        mask = df["Question"].str.contains(search, case=False, na=False)
        df = df[mask]

    st.dataframe(df, use_container_width=True, hide_index=True, height=400)

    st.download_button(
        label="Download test_set.json",
        data=TEST_SET_PATH.read_text(),
        file_name="test_set.json",
        mime="application/json",
    )
else:
    st.info("No test set found. Generate one below.")

# ── Regenerate ────────────────────────────────────────────────────────────── #
st.divider()
st.subheader("Generate / Regenerate Test Set")
st.markdown(
    "This calls `ingest.py` to load your docs, sample random chunks, "
    "and ask the local Ollama LLM to generate one Q&A pair per chunk."
)

if not doc_files:
    st.warning(
        f"No `.md` files found in `{DOCS_PATH.relative_to(PROJECT_ROOT)}`. "
        "Add markdown documents there before generating the test set."
    )

with st.form("ingest_form"):
    col_docs_path, col_num_q, col_output = st.columns(3)

    docs_path_input = col_docs_path.text_input(
        "Docs path",
        value="data/docs",
        help="Relative to project root",
    )
    num_questions = col_num_q.number_input(
        "Number of questions",
        min_value=5,
        max_value=200,
        value=20,
        step=5,
        help="More questions → stronger statistical power but longer runtime",
    )
    output_path = col_output.text_input(
        "Output path",
        value="data/test_set.json",
        help="Will overwrite existing test set",
    )

    if test_cases:
        st.warning(
            f"This will **overwrite** the existing {len(test_cases)}-question test set. "
            "Download a backup above if you want to keep it."
        )

    submitted = st.form_submit_button(
        "Generate Test Set",
        type="primary",
        disabled=not doc_files,
    )

if submitted:
    st.markdown("### Generation Log")
    log_placeholder = st.empty()

    import time
    start = time.time()

    cmd = [
        "python", "ingest.py",
        "--docs-path", docs_path_input,
        "--num-questions", str(int(num_questions)),
        "--output", output_path,
    ]

    with st.spinner("Generating test set…"):
        returncode = stream_subprocess(cmd, log_placeholder)

    elapsed = time.time() - start

    if returncode == 0:
        st.success(f"Done in {elapsed:.1f}s — reload the page to see the updated test set.")
        st.rerun()
    else:
        st.error("Generation failed. Check the log above.")

# ── Source documents ──────────────────────────────────────────────────────── #
if doc_files:
    st.divider()
    with st.expander(f"Source Documents ({len(doc_files)} files)"):
        for p in sorted(doc_files):
            rel = p.relative_to(PROJECT_ROOT)
            size_kb = p.stat().st_size / 1024
            st.markdown(f"- `{rel}` — {size_kb:.1f} KB")
