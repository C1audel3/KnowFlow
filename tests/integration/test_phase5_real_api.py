from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from app.core.config import PROJECT_ROOT, AppSettings
from app.main import create_app

pytestmark = pytest.mark.integration


def _real_integration_enabled() -> bool:
    return os.getenv("RUN_REAL_INTEGRATION", "").lower() in {"1", "true", "yes"}


@pytest.mark.skipif(
    not _real_integration_enabled(),
    reason="set RUN_REAL_INTEGRATION=1 to call configured model services",
)
def test_real_deepseek_ollama_query():
    key = os.getenv("LLM_BINDING_API_KEY", "").strip().lower()
    if key in {"", "replace-with-your-api-key", "your_api_key"}:
        pytest.skip("LLM_BINDING_API_KEY is not configured")

    working_dir = PROJECT_ROOT / "rag_storage_phase2"
    if not (working_dir / "graph_chunk_entity_relation.graphml").is_file():
        pytest.skip("phase-two knowledge base is not available")

    embedding_host = os.getenv("EMBEDDING_BINDING_HOST", "http://127.0.0.1:11434/v1")
    try:
        response = httpx.get(
            embedding_host.removesuffix("/v1") + "/api/tags", timeout=3
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        pytest.fail(f"Ollama is unavailable ({type(exc).__name__})")

    settings = replace(AppSettings.from_env(), working_dir=Path(working_dir))
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/query",
            json={"question": "Atlas 项目计划在哪个季度发布？"},
        )

    assert response.status_code == 200
    assert "2027" in response.json()["answer"]
    assert "Q3" in response.json()["answer"]
