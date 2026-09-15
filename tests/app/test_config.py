from __future__ import annotations

import pytest

from app.core.config import AppSettings


def test_settings_load_and_expose_only_public_values(monkeypatch, tmp_path):
    monkeypatch.setenv("WORKING_DIR", str(tmp_path / "storage"))
    monkeypatch.setenv("INPUT_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("MAX_UPLOAD_MB", "7")
    monkeypatch.setenv("MAX_QUERY_CONCURRENCY", "3")
    monkeypatch.setenv("LLM_BINDING_API_KEY", "must-not-appear")

    settings = AppSettings.from_env()
    public = settings.public_dict()

    assert settings.max_upload_bytes == 7 * 1024 * 1024
    assert public["max_query_concurrency"] == 3
    assert "key" not in " ".join(public).lower()
    assert "must-not-appear" not in str(public)


@pytest.mark.parametrize(
    ("name", "value"),
    [("APP_PORT", "zero"), ("MAX_UPLOAD_MB", "0"), ("PARSER_TIMEOUT", "-1")],
)
def test_settings_reject_invalid_positive_integers(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        AppSettings.from_env()


def test_settings_reject_unknown_extension(monkeypatch):
    monkeypatch.setenv("SUPPORTED_FILE_EXTENSIONS", ".pdf,.exe")
    with pytest.raises(ValueError, match="unsupported configured extensions"):
        AppSettings.from_env()
