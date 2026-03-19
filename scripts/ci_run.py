"""
scripts/ci_run.py — CI entrypoint for the RAG A/B Tester GitHub Action.

Runs a single experiment, checks for regressions, posts a PR comment,
and exits with code 1 if a regression is detected (blocking the merge).

Usage (called by action.yml):
    python scripts/ci_run.py \\
        --experiment experiments/chunking.py \\
        --test-set data/test_set.json \\
        --fail-on-regression true \\
        --github-token ${{ secrets.GITHUB_TOKEN }} \\
        --repo owner/repo \\
        --pr-number 42

Environment variable equivalents (for local testing):
    RAG_EXPERIMENT, RAG_TEST_SET, RAG_FAIL_ON_REGRESSION,
    GITHUB_TOKEN, GITHUB_REPOSITORY, PR_NUMBER
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure project root is on the path regardless of where this is invoked from
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from run_experiment import run as run_experiment
from src.ci.regression import check_regression, exit_code
from src.ci.github_comment import format_comment, post_or_update_comment


def _bool(value: str) -> bool:
    return value.lower() in ("true", "1", "yes")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RAG A/B Tester — CI runner")
    p.add_argument("--experiment",          default=os.getenv("RAG_EXPERIMENT", "experiments/chunking.py"))
    p.add_argument("--test-set",            default=os.getenv("RAG_TEST_SET", "data/test_set.json"))
    p.add_argument("--fail-on-regression",  default=os.getenv("RAG_FAIL_ON_REGRESSION", "true"))
    p.add_argument("--save-json",           action="store_true", default=True)
    p.add_argument("--github-token",        default=os.getenv("GITHUB_TOKEN", ""))
    p.add_argument("--repo",                default=os.getenv("GITHUB_REPOSITORY", ""))
    p.add_argument("--pr-number",           default=os.getenv("PR_NUMBER", ""), type=str)
    return p.parse_args()


def set_github_output(key: str, value: str) -> None:
    """Write a key=value pair to $GITHUB_OUTPUT (Actions step output)."""
    output_file = os.getenv("GITHUB_OUTPUT")
    if output_file:
        with open(output_file, "a") as f:
            f.write(f"{key}={value}\n")


def main() -> int:
    args = parse_args()
    fail_on_regression = _bool(args.fail_on_regression)

    print(f"\n{'=' * 60}")
    print(f"RAG CI Run")
    print(f"  Experiment : {args.experiment}")
    print(f"  Test set   : {args.test_set}")
    print(f"  Fail on reg: {fail_on_regression}")
    print(f"{'=' * 60}\n")

    # ── 1. Run the experiment ──────────────────────────────────────────────
    try:
        result = run_experiment(
            experiment_path=args.experiment,
            test_set_path=args.test_set,
            save_json=args.save_json,
        )
    except SystemExit as e:
        # run_experiment calls sys.exit on missing test set etc.
        print(f"Experiment failed with exit code {e.code}")
        return int(e.code or 1)
    except Exception as e:
        print(f"Experiment raised an exception: {e}")
        return 1

    # ── 2. Regression check ───────────────────────────────────────────────
    report = check_regression(result)

    print(f"\n{report.summary_line}")
    if report.regressed_metrics:
        print(f"  Regressed : {', '.join(report.regressed_metrics)}")
    if report.improved_metrics:
        print(f"  Improved  : {', '.join(report.improved_metrics)}")

    # ── 3. GitHub Actions outputs ─────────────────────────────────────────
    set_github_output("overall-winner",       result.overall_winner)
    set_github_output("regression-detected",  str(report.regression_detected).lower())
    set_github_output("improved-metrics",     ",".join(report.improved_metrics))
    set_github_output("regressed-metrics",    ",".join(report.regressed_metrics))

    # ── 4. Post PR comment ────────────────────────────────────────────────
    has_token  = bool(args.github_token)
    has_repo   = bool(args.repo)
    has_pr     = bool(args.pr_number)

    if has_token and has_repo and has_pr:
        try:
            body = format_comment(result, report)
            url  = post_or_update_comment(
                body=body,
                token=args.github_token,
                repo=args.repo,
                pr_number=int(args.pr_number),
            )
            print(f"\nPR comment posted: {url}")
        except Exception as e:
            # Never let a comment failure block CI
            print(f"Warning: failed to post PR comment: {e}")
    else:
        missing = [
            k for k, v in {
                "--github-token": has_token,
                "--repo": has_repo,
                "--pr-number": has_pr,
            }.items() if not v
        ]
        print(f"\nSkipping PR comment (missing: {', '.join(missing)})")
        # Print formatted comment to stdout so it appears in CI logs
        print("\n" + format_comment(result, report))

    # ── 5. Exit code ──────────────────────────────────────────────────────
    code = exit_code(report, fail_on_regression)
    if code != 0:
        print(f"\n🚫 Blocking merge — regression detected in {result.experiment_name}.")
    return code


if __name__ == "__main__":
    sys.exit(main())
