import pytest

from deepresearch.config import (
    load_settings,
)


def test_load_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    monkeypatch.setenv(
        "LLM_API_KEY",
        "test-api-key",
    )

    monkeypatch.setenv(
        "LLM_MODEL",
        "test-model",
    )

    monkeypatch.delenv(
        "LLM_BASE_URL",
        raising=False,
    )

    settings = load_settings()

    assert (
        settings.llm_api_key
        == "test-api-key"
    )

    assert (
        settings.llm_model
        == "test-model"
    )

    assert (
        settings.llm_base_url
        is None
    )


def test_missing_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    monkeypatch.setenv(
        "LLM_API_KEY",
        "",
    )

    monkeypatch.setenv(
        "LLM_MODEL",
        "test-model",
    )

    with pytest.raises(
        RuntimeError,
        match="LLM_API_KEY",
    ):
        load_settings()


def test_missing_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    monkeypatch.setenv(
        "LLM_API_KEY",
        "test-api-key",
    )

    monkeypatch.setenv(
        "LLM_MODEL",
        "",
    )

    with pytest.raises(
        RuntimeError,
        match="LLM_MODEL",
    ):
        load_settings()