"""
src/ci/github_comment.py — Format experiment results as a GitHub PR comment
and post/update it via the GitHub REST API.

The comment is idempotent: if a previous RAG comment exists on the PR it is
updated in-place rather than creating a new one on every push.
"""
from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
from typing import Optional

from src.models.schemas import ExperimentResult, MetricComparison
from src.ci.regression import RegressionReport

# Marker embedded in every comment so we can find and update it later
_COMMENT_MARKER = "<!-- rag-ab-tester -->"

METRIC_LABELS = {
    "faithfulness":      "Faithfulness",
    "answer_relevance":  "Answer Relevance",
    "context_precision": "Context Precision",
    "context_recall":    "Context Recall",
}


# ── Formatting ────────────────────────────────────────────────────────────── #

def _winner_badge(winner: str, control_name: str, challenger_name: str) -> str:
    if winner == "control":
        return f"🔵 {control_name}"
    if winner == "challenger":
        return f"🟢 {challenger_name}"
    return "➖ no diff"


def _sig(comp: MetricComparison) -> str:
    sig = "✓" if comp.significant else "✗"
    mng = "✓" if comp.meaningful else "✗"
    return f"p={comp.p_value} {sig} / d={comp.cohens_d} {mng}"


def format_comment(
    result: ExperimentResult,
    report: RegressionReport,
) -> str:
    ctrl  = result.control_name
    chal  = result.challenger_name

    # Header
    status_icon = "❌" if report.regression_detected else ("✅" if report.improved_metrics else "➡️")
    lines = [
        _COMMENT_MARKER,
        f"## {status_icon} RAG Experiment: {result.experiment_name}",
        "",
        f"> {report.summary_line}",
        "",
        f"| Metric | {ctrl} | {chal} | Δ | Stats | Winner |",
        f"|--------|{'---' * 3}|{'---' * 3}|---|-------|--------|",
    ]

    for comp in result.comparisons:
        label  = METRIC_LABELS.get(comp.metric, comp.metric)
        delta  = f"+{comp.delta:.3f}" if comp.delta >= 0 else f"{comp.delta:.3f}"
        badge  = _winner_badge(comp.winner, ctrl, chal)
        stats  = _sig(comp)
        lines.append(
            f"| {label} | {comp.control_avg:.3f} | {comp.challenger_avg:.3f} "
            f"| {delta} | {stats} | {badge} |"
        )

    lines += [
        "",
        f"**Overall winner:** {result.overall_winner}",
        "",
    ]

    # Regression / improvement callout
    if report.regression_detected:
        lines += [
            "### ⚠️ Regression detected — merge blocked",
            "",
            f"The following metrics degraded significantly: "
            f"**{', '.join(report.regressed_metrics)}**",
            "",
            "Fix the regression or set `fail-on-regression: false` to override.",
        ]
    elif report.improved_metrics:
        lines += [
            "### 🎉 Improvement confirmed",
            "",
            f"**{', '.join(report.improved_metrics)}** improved with "
            f"statistical significance (p < 0.05) and practical effect (|d| ≥ 0.2).",
        ]
    else:
        lines += [
            "_No statistically significant difference detected. "
            "Merge is safe but the change provides no measurable RAG improvement._",
        ]

    lines += ["", "---", "_Powered by [RAG A/B Tester](https://github.com/your-org/rag-ab-tester)_"]
    return "\n".join(lines)


# ── GitHub API ────────────────────────────────────────────────────────────── #

def _gh_request(method: str, url: str, token: str, body: Optional[dict] = None):
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def _find_existing_comment(token: str, repo: str, pr_number: int) -> Optional[int]:
    """Return the comment ID of an existing RAG comment on this PR, or None."""
    url = f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments?per_page=100"
    try:
        comments = _gh_request("GET", url, token)
        for c in comments:
            if _COMMENT_MARKER in c.get("body", ""):
                return c["id"]
    except urllib.error.HTTPError:
        pass
    return None


def post_or_update_comment(
    body: str,
    token: str,
    repo: str,
    pr_number: int,
) -> str:
    """Post a new comment or update the existing RAG comment. Returns the comment URL."""
    existing_id = _find_existing_comment(token, repo, pr_number)

    if existing_id:
        url = f"https://api.github.com/repos/{repo}/issues/comments/{existing_id}"
        result = _gh_request("PATCH", url, token, {"body": body})
    else:
        url = f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments"
        result = _gh_request("POST", url, token, {"body": body})

    return result.get("html_url", "")
