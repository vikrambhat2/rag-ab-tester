import pytest

from src import config
from src.config import ConfigError, Settings, get_settings


def test_missing_everything_lists_all_problems():
    with pytest.raises(ConfigError) as exc:
        get_settings()
    msg = str(exc.value)
    for name in ("WATSONX_API_KEY", "WATSONX_PROJECT_ID", "OPENSEARCH_URL", "OPENSEARCH_USERNAME"):
        assert name in msg


def test_valid_env_passes(valid_env):
    s = get_settings()
    assert s.opensearch_verify_ssl is True
    assert s.opensearch_url == "https://os.example.com:9200"


def test_service_scoping(monkeypatch):
    monkeypatch.setenv("WATSONX_API_KEY", "k")
    monkeypatch.setenv("WATSONX_PROJECT_ID", "p")
    assert get_settings(opensearch=False).watsonx_api_key == "k"
    with pytest.raises(ConfigError):
        get_settings(watsonx=False)


@pytest.mark.parametrize("value", ["false", "0", "No", "OFF"])
def test_ssl_verification_can_be_disabled_with_warning(valid_env, monkeypatch, caplog, value):
    monkeypatch.setenv("OPENSEARCH_VERIFY_SSL", value)
    with caplog.at_level("WARNING"):
        assert get_settings().opensearch_verify_ssl is False
    assert "NOT verified" in caplog.text


def test_invalid_opensearch_url(valid_env, monkeypatch):
    monkeypatch.setenv("OPENSEARCH_URL", "not-a-url")
    with pytest.raises(ConfigError, match="OPENSEARCH_URL"):
        get_settings()


def test_opensearch_kwargs(valid_env):
    kw = config.get_opensearch_kwargs()
    assert kw["use_ssl"] is True
    assert kw["http_auth"] == ("user", "pass")
    assert kw["timeout"] == 60


def test_health_checks_return_false_and_log(caplog):
    with caplog.at_level("ERROR"):
        assert config.check_opensearch() is False
        assert config.check_watsonx() is False
    assert "health check failed" in caplog.text


def test_settings_problems_are_pure():
    assert Settings.from_env().problems(watsonx=False, opensearch=False) == []


# ── LLM provider / judge separation ───────────────────────────────────────── #


@pytest.fixture
def openai_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_BASE_URL", "https://gw.example.com/v1")
    monkeypatch.setenv("LLM_API_KEY", "gen-key")
    monkeypatch.setenv("LLM_MODEL", "gen-model")


def test_default_provider_is_watsonx(valid_env):
    cfg = Settings.from_env().llm_config()
    assert cfg.provider == "watsonx"
    assert cfg.model == "ibm/granite-3-8b-instruct"


def test_openai_provider_needs_no_watsonx_credentials(openai_env):
    # e.g. ingest.py: only the chat LLM is used
    s = get_settings(watsonx=False, opensearch=False, llm="generator")
    assert s.llm_config().label == "openai:gen-model"


def test_openai_provider_validation_lists_missing_vars(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    with pytest.raises(ConfigError) as exc:
        get_settings(watsonx=False, opensearch=False, llm="generator")
    for name in ("LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY"):
        assert name in str(exc.value)


def test_unknown_provider_rejected(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "bogus")
    with pytest.raises(ConfigError, match="LLM_PROVIDER"):
        get_settings(watsonx=False, opensearch=False, llm="generator")


def test_watsonx_generator_still_requires_watsonx_creds():
    with pytest.raises(ConfigError, match="WATSONX_API_KEY"):
        get_settings(watsonx=False, opensearch=False, llm="generator")


def test_judge_inherits_generator_by_default(openai_env):
    s = Settings.from_env()
    assert s.llm_config("judge") == s.llm_config("generator")


def test_judge_model_override_keeps_generator_endpoint(openai_env, monkeypatch):
    monkeypatch.setenv("JUDGE_LLM_MODEL", "judge-model")
    cfg = Settings.from_env().llm_config("judge")
    assert (cfg.model, cfg.base_url, cfg.api_key) == (
        "judge-model",
        "https://gw.example.com/v1",
        "gen-key",
    )


def test_judge_provider_switch_does_not_leak_generator_key(openai_env, monkeypatch, valid_env):
    monkeypatch.setenv("JUDGE_LLM_PROVIDER", "watsonx")
    cfg = Settings.from_env().llm_config("judge")
    assert cfg.provider == "watsonx" and cfg.api_key == "" and cfg.base_url == ""
    assert cfg.model == "ibm/granite-3-8b-instruct"


def test_judge_switching_to_openai_requires_its_own_settings(valid_env, monkeypatch):
    monkeypatch.setenv("JUDGE_LLM_PROVIDER", "openai")
    with pytest.raises(ConfigError) as exc:
        get_settings(watsonx=False, opensearch=False, llm="judge")
    assert "JUDGE_LLM_BASE_URL" in str(exc.value)


def test_get_llm_builds_chat_openai(openai_env):
    llm = config.get_llm()
    assert type(llm).__name__ == "ChatOpenAI"
    assert llm.model_name == "gen-model"
    assert str(llm.openai_api_base) == "https://gw.example.com/v1"
    assert llm.max_retries == 0


def test_get_llm_judge_uses_judge_model(openai_env, monkeypatch):
    monkeypatch.setenv("JUDGE_LLM_MODEL", "judge-model")
    assert config.get_llm(role="judge").model_name == "judge-model"
    assert config.get_llm().model_name == "gen-model"


def test_describe_llm_has_no_secrets(openai_env):
    label = config.describe_llm("generator")
    assert label == "openai:gen-model" and "gen-key" not in label
