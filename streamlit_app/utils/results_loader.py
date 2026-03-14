"""
utils/results_loader.py — Load ExperimentResult / ABReport from JSON
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

# Ensure the project root is on the path so we can import src.*
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.schemas import ABReport, ExperimentResult  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "results"


def list_result_files() -> list[Path]:
    """Return all .json files in results/, sorted newest first."""
    if not RESULTS_DIR.exists():
        return []
    return sorted(RESULTS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)


def load_result(path: Path) -> ExperimentResult | None:
    """Load a single ExperimentResult from a JSON file. Returns None on error."""
    try:
        data = json.loads(path.read_text())
        # ABReport has an "experiments" key; single result does not
        if "experiments" in data:
            return None  # it's a full report, not a single result
        return ExperimentResult.model_validate(data)
    except Exception:
        return None


def load_report(path: Path) -> ABReport | None:
    """Load an ABReport from a JSON file. Returns None on error."""
    try:
        data = json.loads(path.read_text())
        if "experiments" not in data:
            # Wrap a single experiment into a report
            result = ExperimentResult.model_validate(data)
            return ABReport(experiments=[result])
        return ABReport.model_validate(data)
    except Exception:
        return None


def load_all_results() -> list[ExperimentResult]:
    """Load every single-experiment JSON in results/."""
    results = []
    for path in list_result_files():
        r = load_result(path)
        if r is not None:
            results.append(r)
    return results


def discover_experiments(experiments_dir: Path | None = None) -> list[Path]:
    """Return all .py experiment files (root + custom/), sorted."""
    root = experiments_dir or (PROJECT_ROOT / "experiments")
    files = sorted(p for p in root.glob("*.py") if p.name != "__init__.py")
    custom = sorted(p for p in (root / "custom").glob("*.py") if p.name != "__init__.py")
    return files + custom


def load_test_set(path: Path | None = None) -> list[dict]:
    """Load data/test_set.json. Returns empty list if missing."""
    p = path or (PROJECT_ROOT / "data" / "test_set.json")
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text())
    except Exception:
        return []
