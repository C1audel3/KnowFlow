from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_streamlit_page_survives_offline_backend(monkeypatch):
    monkeypatch.setenv("API_BASE_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("UI_REQUEST_TIMEOUT", "0.1")
    app_path = Path(__file__).resolve().parents[2] / "ui" / "streamlit_app.py"

    app = AppTest.from_file(str(app_path)).run(timeout=10)

    assert not app.exception
    assert app.title[0].value == "KnowFlow 多模态知识库"
    assert any("无法连接后端服务" in item.value for item in app.sidebar.error)
