from __future__ import annotations

from dataclasses import replace

from fastapi.testclient import TestClient

from app.main import create_app
from fakes import FakeRAG


def client_for(app_settings, rag: FakeRAG) -> TestClient:
    return TestClient(create_app(app_settings, rag_factory=lambda _: rag))


def test_health_public_config_and_openapi_are_available(app_settings):
    rag = FakeRAG()
    with client_for(app_settings, rag) as client:
        health = client.get("/health")
        config = client.get("/config/public")
        openapi = client.get("/openapi.json")

    assert health.json() == {
        "status": "ok",
        "rag_initialized": True,
        "parser": "mineru",
        "vision_enabled": True,
    }
    assert config.status_code == 200
    assert config.json()["supported_extensions"] == [
        ".jpeg",
        ".jpg",
        ".md",
        ".pdf",
        ".png",
    ]
    assert "api_key" not in config.text.lower()
    assert set(openapi.json()["paths"]) == {
        "/health",
        "/config/public",
        "/documents",
        "/documents/{task_id}",
        "/query",
    }
    assert rag.finalize_calls == 1


def test_upload_and_status_endpoint(app_settings):
    rag = FakeRAG()
    with client_for(app_settings, rag) as client:
        response = client.post(
            "/documents",
            files={"file": ("report.pdf", b"%PDF-1.4\n", "application/pdf")},
        )
        assert response.status_code == 202
        task_id = response.json()["task_id"]

        status_response = client.get(f"/documents/{task_id}")
        while status_response.json()["status"] in {"pending", "processing"}:
            status_response = client.get(f"/documents/{task_id}")

    body = status_response.json()
    assert body["status"] == "completed"
    assert body["document_id"].startswith("doc-")
    assert body["duration_ms"] >= 0


def test_upload_errors_use_stable_safe_shape(app_settings):
    with client_for(app_settings, FakeRAG()) as client:
        unsupported = client.post("/documents", files={"file": ("bad.exe", b"payload")})
        mismatch = client.post("/documents", files={"file": ("bad.png", b"payload")})
        missing = client.get("/documents/not-found")

    assert unsupported.status_code == 415
    assert unsupported.json()["error"]["code"] == "INVALID_FILE_TYPE"
    assert mismatch.status_code == 400
    assert mismatch.json()["error"]["code"] == "INVALID_FILE"
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "DOCUMENT_NOT_FOUND"
    assert all(
        "request_id" in item.json()["error"]
        for item in (unsupported, mismatch, missing)
    )


def test_oversized_upload_returns_413(app_settings):
    settings = replace(app_settings, max_upload_bytes=8)
    with client_for(settings, FakeRAG()) as client:
        response = client.post(
            "/documents",
            files={"file": ("large.pdf", b"%PDF-1.4-too-large")},
        )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "FILE_TOO_LARGE"


def test_failed_background_task_is_visible_without_details(app_settings):
    rag = FakeRAG(process_error=RuntimeError("private stack detail"))
    with client_for(app_settings, rag) as client:
        created = client.post("/documents", files={"file": ("bad.pdf", b"%PDF-1.4\n")})
        task_id = created.json()["task_id"]
        response = client.get(f"/documents/{task_id}")
        while response.json()["status"] in {"pending", "processing"}:
            response = client.get(f"/documents/{task_id}")

    assert response.json()["status"] == "failed"
    assert response.json()["error"] == "文档处理失败"
    assert "private stack detail" not in response.text


def test_query_endpoint_and_validation(app_settings):
    rag = FakeRAG()
    with client_for(app_settings, rag) as client:
        response = client.post("/query", json={"question": "Atlas 何时发布？"})
        blank = client.post("/query", json={"question": "   "})
        wrong_mode = client.post("/query", json={"question": "hello", "mode": "global"})

    assert response.status_code == 200
    assert response.json()["answer"] == "Atlas 将在 2027 Q3 发布。"
    assert response.json()["sources"] == []
    assert blank.status_code == 400
    assert blank.json()["error"]["code"] == "INVALID_REQUEST"
    assert wrong_mode.status_code == 400


def test_query_failure_does_not_leak_internal_error(app_settings, caplog):
    rag = FakeRAG(query_error=RuntimeError("secret-api-key-value"))
    with client_for(app_settings, rag) as client:
        response = client.post("/query", json={"question": "hello"})

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "MODEL_UNAVAILABLE"
    assert "secret-api-key-value" not in response.text
    assert "secret-api-key-value" not in caplog.text
