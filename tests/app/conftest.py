from __future__ import annotations

from pathlib import Path

import pytest

from app.core.config import AppSettings


@pytest.fixture
def app_settings(tmp_path: Path) -> AppSettings:
    return AppSettings(
        host="127.0.0.1",
        port=8000,
        working_dir=tmp_path / "storage",
        upload_dir=tmp_path / "uploads",
        output_dir=tmp_path / "output",
        max_upload_bytes=1024,
        supported_extensions=frozenset({".pdf", ".md", ".png", ".jpg", ".jpeg"}),
        parser_backend="pipeline",
        parser_timeout=30,
        max_query_concurrency=2,
    )
