import pytest

from rag_ab_tester import cli


def test_version(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    assert "rag-ab-tester" in capsys.readouterr().out


def test_run_unknown_experiment_exits_1(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["run", "does-not-exist"])
    assert exc.value.code == 1


def test_run_missing_test_set_exits_1(tmp_path):
    with pytest.raises(SystemExit) as exc:
        cli.main(["run", "chunking", "--test-set", str(tmp_path / "nope.json")])
    assert exc.value.code == 1


def test_run_all_exits_nonzero_when_an_experiment_fails(tmp_path, monkeypatch):

    monkeypatch.setattr(cli, "run_all", lambda files, *a, **k: (None, ["chunking"]))
    with pytest.raises(SystemExit) as exc:
        cli.main(["run-all", "--include", "chunking", "--test-set", str(tmp_path / "t.json")])
    assert exc.value.code == 1


def test_run_all_with_no_matches_is_not_an_error(capsys):
    cli.main(["run-all", "--include", "zzz-no-match"])
    assert "No experiments matched" in capsys.readouterr().out


def test_results_dir_flag_sets_env(tmp_path, monkeypatch):
    monkeypatch.delenv("RAG_AB_RESULTS_DIR", raising=False)
    monkeypatch.setattr(cli, "run_experiment", lambda *a, **k: None)
    cli.main(["run", "chunking", "--results-dir", str(tmp_path)])
    import os

    assert os.environ["RAG_AB_RESULTS_DIR"] == str(tmp_path)
    monkeypatch.delenv("RAG_AB_RESULTS_DIR", raising=False)


def test_error_message_keeps_bracketed_extra_name(capsys, monkeypatch):
    # Rich would swallow "[docs]" as markup if the text weren't escaped.
    from rag_ab_tester.config import MissingExtraError

    def boom(*a, **k):
        raise MissingExtraError("install: pip install 'rag-ab-tester[docs]'")

    monkeypatch.setattr(cli, "run_experiment", boom)
    with pytest.raises(SystemExit):
        cli.main(["run", "chunking"])
    assert "[docs]" in capsys.readouterr().out


def test_builtin_retrieval_experiment_loads_without_langchain_community(monkeypatch):
    import sys

    from rag_ab_tester import runner

    monkeypatch.setitem(sys.modules, "langchain_community.retrievers", None)
    mod = runner.load_experiment(runner.resolve_experiment("retrieval"))
    assert mod.EXPERIMENT_NAME == "Retrieval Strategy"
