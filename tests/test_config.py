import pytest

from rag_ab_tester import config
from rag_ab_tester.config import ConfigError, Settings, get_settings


def test_missing_everything_lists_all_problems():
    with pytest.raises(ConfigError) as exc:
        get_settings(llm="generator", embeddings=True)
    msg = str(exc.value)
    for name in ("LLM_MODEL", "LLM_API_KEY", "EMBED_MODEL"):
        assert name in msg


def test_valid_env_passes(valid_env):
    s = get_settings(llm="generator", embeddings=True)
    assert s.llm_model == "gen-model" and s.embed_model == "embed-model"


def test_nothing_is_required_unless_requested():
    assert get_settings().llm_model == ""


def test_base_url_makes_api_key_optional(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "llama3")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:11434/v1")
    assert get_settings(llm="generator").llm_config().label == "llama3@localhost"


def test_invalid_base_url(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "m")
    monkeypatch.setenv("LLM_BASE_URL", "not a url")
    with pytest.raises(ConfigError, match="LLM_BASE_URL"):
        get_settings(llm="generator")


def test_invalid_timeout(monkeypatch):
    monkeypatch.setenv("LLM_TIMEOUT", "soon")
    with pytest.raises(ConfigError, match="LLM_TIMEOUT"):
        Settings.from_env()


def test_error_text_points_at_env_file(monkeypatch):
    with pytest.raises(ConfigError, match=r"\.env"):
        get_settings(llm="generator")


# ── judge separation ──────────────────────────────────────────────────────── #


@pytest.fixture
def gen_env(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "https://gw.example.com/v1")
    monkeypatch.setenv("LLM_API_KEY", "gen-key")
    monkeypatch.setenv("LLM_MODEL", "gen-model")


def test_judge_defaults_to_generator(gen_env):
    s = Settings.from_env()
    assert s.llm_config("judge") == s.llm_config("generator")


def test_judge_model_override_inherits_endpoint_and_key(gen_env, monkeypatch):
    monkeypatch.setenv("JUDGE_LLM_MODEL", "judge-model")
    cfg = Settings.from_env().llm_config("judge")
    assert cfg.model == "judge-model"
    assert (cfg.base_url, cfg.api_key) == ("https://gw.example.com/v1", "gen-key")


def test_judge_on_another_endpoint_never_gets_generator_key(gen_env, monkeypatch):
    monkeypatch.setenv("JUDGE_LLM_BASE_URL", "http://localhost:11434/v1")
    cfg = Settings.from_env().llm_config("judge")
    assert cfg.api_key == "" and cfg.base_url == "http://localhost:11434/v1"
    monkeypatch.setenv("JUDGE_LLM_API_KEY", "judge-key")
    assert Settings.from_env().llm_config("judge").api_key == "judge-key"


def test_model_id_override_wins(gen_env):
    assert Settings.from_env().llm_config("judge", "other").model == "other"


def test_describe_llm_has_no_secrets(gen_env):
    label = config.describe_llm("generator")
    assert label == "gen-model@gw.example.com" and "gen-key" not in label


# ── embeddings ────────────────────────────────────────────────────────────── #


def test_embeddings_fall_back_to_generator_endpoint(gen_env, monkeypatch):
    monkeypatch.setenv("EMBED_MODEL", "e")
    cfg = Settings.from_env().embed_config()
    assert (cfg.model, cfg.base_url, cfg.api_key) == ("e", "https://gw.example.com/v1", "gen-key")


def test_embeddings_can_use_their_own_endpoint(gen_env, monkeypatch):
    monkeypatch.setenv("EMBED_MODEL", "e")
    monkeypatch.setenv("EMBED_BASE_URL", "http://embed.local/v1")
    monkeypatch.setenv("EMBED_API_KEY", "ek")
    cfg = Settings.from_env().embed_config()
    assert (cfg.base_url, cfg.api_key) == ("http://embed.local/v1", "ek")


def test_get_embeddings_requires_model(gen_env):
    with pytest.raises(ConfigError, match="EMBED_MODEL"):
        config.get_embeddings()


# ── client construction (needs the `openai` extra; no network) ────────────── #


def test_get_llm_builds_chat_client(gen_env):
    llm = config.get_llm()
    assert llm.model_name == "gen-model" and llm.max_retries == 0


def test_get_llm_without_extra_names_the_extra(gen_env, monkeypatch):
    def missing(module, extra):
        raise config.MissingExtraError(f"pip install 'rag-ab-tester[{extra}]'")

    monkeypatch.setattr(config, "import_optional", missing)
    with pytest.raises(config.MissingExtraError, match=r"\[openai\]"):
        config.get_llm()


def test_get_embeddings_builds_client_and_skips_ctx_check_for_custom_servers(gen_env, monkeypatch):
    monkeypatch.setenv("EMBED_MODEL", "nomic")
    emb = config.get_embeddings()
    assert emb.model == "nomic" and emb.check_embedding_ctx_length is False


def test_import_optional_reports_extra():
    with pytest.raises(config.MissingExtraError, match=r"rag-ab-tester\[docs\]"):
        config.import_optional("definitely_not_a_module", "docs")
