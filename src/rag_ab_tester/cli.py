"""Command-line interface: ``rag-ab ingest | run | run-all``."""

from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv
from rich.console import Console
from rich.markup import escape

from rag_ab_tester import __version__
from rag_ab_tester.config import ConfigError, MissingExtraError
from rag_ab_tester.ingest import IngestError, generate_test_set
from rag_ab_tester.logging_config import setup_logging
from rag_ab_tester.paths import RESULTS_DIR_ENV
from rag_ab_tester.run_all import run_all, select_experiments
from rag_ab_tester.runner import ExperimentError, builtin_experiments, run_experiment

console = Console()

_EXPECTED_ERRORS = (ConfigError, MissingExtraError, ExperimentError, IngestError)


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--test-set", default="data/test_set.json", help="Path to test_set.json")
    p.add_argument("--docs-path", default="data/docs", help="Directory of .md documents")
    p.add_argument("--results-dir", help=f"Where results are written (env: {RESULTS_DIR_ENV})")
    p.add_argument("--save-json", action="store_true", help="Save results as JSON")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag-ab", description="A/B testing for RAG pipelines.")
    parser.add_argument("--version", action="version", version=f"rag-ab-tester {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    ing = sub.add_parser("ingest", help="Generate a Q&A test set from your documents")
    ing.add_argument("--docs-path", default="data/docs", help="Directory of .md documents")
    ing.add_argument("--num-questions", type=int, default=20)
    ing.add_argument("--output", default="data/test_set.json")
    ing.add_argument("--seed", type=int, default=None, help="Seed for reproducible sampling")

    run = sub.add_parser("run", help="Run one experiment")
    run.add_argument(
        "experiment",
        help=f"Built-in name ({', '.join(builtin_experiments())}) or path to an experiment .py file",
    )
    _common(run)

    allp = sub.add_parser("run-all", help="Run every built-in experiment")
    allp.add_argument("--include", nargs="+", metavar="NAME", help="Only names containing these")
    allp.add_argument("--exclude", nargs="+", metavar="NAME", help="Skip names containing these")
    allp.add_argument("--experiments-dir", help="Also run every *.py experiment in this directory")
    _common(allp)
    return parser


def main(argv: list[str] | None = None) -> None:
    load_dotenv()  # the CLI (not the library) reads .env from the current directory
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)
    if getattr(args, "results_dir", None):
        os.environ[RESULTS_DIR_ENV] = args.results_dir

    try:
        if args.command == "ingest":
            generate_test_set(args.docs_path, args.num_questions, args.output, args.seed)
        elif args.command == "run":
            run_experiment(
                args.experiment,
                args.test_set,
                args.save_json,
                docs_path=args.docs_path,
            )
        else:
            files = select_experiments(args.include, args.exclude, args.experiments_dir)
            if not files:
                console.print("[yellow]No experiments matched the given filters.[/yellow]")
                return
            _, failed = run_all(files, args.test_set, args.save_json, docs_path=args.docs_path)
            if failed:
                sys.exit(1)
    except _EXPECTED_ERRORS as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
