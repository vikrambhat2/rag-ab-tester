"""
utils/runner.py — Subprocess streaming helper for Streamlit

Runs CLI scripts from the project root and streams stdout/stderr
line-by-line into a Streamlit placeholder so users see live output.
"""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path
from subprocess import PIPE, STDOUT

# Project root is two levels up from this file (streamlit_app/utils/ → root)
PROJECT_ROOT = Path(__file__).parent.parent.parent


def stream_subprocess(
    cmd: list[str],
    log_placeholder,
    cwd: Path | None = None,
) -> int:
    """
    Run a command and stream its output into a Streamlit code placeholder.

    Args:
        cmd: Command list, e.g. ["python", "run_experiment.py", "--experiment", "..."]
        log_placeholder: A st.empty() object to update with cumulative log text.
        cwd: Working directory. Defaults to PROJECT_ROOT.

    Returns:
        Process return code (0 = success).
    """
    working_dir = cwd or PROJECT_ROOT
    proc = subprocess.Popen(
        [sys.executable] + cmd[1:] if cmd[0] == "python" else cmd,
        stdout=PIPE,
        stderr=STDOUT,
        text=True,
        cwd=str(working_dir),
        bufsize=1,
    )

    log_lines: list[str] = []
    # Strip ANSI colour codes emitted by Rich so the Streamlit code block is clean
    import re
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

    for line in proc.stdout:
        clean = ansi_escape.sub("", line)
        log_lines.append(clean)
        # Show the last 200 lines so very long runs don't explode the DOM
        visible = "".join(log_lines[-200:])
        log_placeholder.code(visible, language="")

    proc.wait()
    return proc.returncode
