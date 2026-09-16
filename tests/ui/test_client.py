from __future__ import annotations

import json

import httpx
import pytest

from ui.client import (
    APIClient,
    UIError,
    UISettings,
    normalize_api_base_url,
    sources_text,
    validate_upload,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (" http://127.0.0.1:8000/ ", "http://127.0.0.1:8000"),
        ("https://example.com/", "https://example.com"),
    ],
)
def test_normalize_api_base_url(raw, expected):
    assert normalize_api_base_url(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "localhost:8000",
        "ftp://example.com",
        "http://x/?a=1",
        "http://x/api",
        "http://user:password@x",
    ],
)
def test_normalize_api_base_url_rejects_unsafe_values(raw):
    with pytest.raises(ValueError, match="API_BASE_URL"):
        normalize_api_base_url(raw)


def test_ui_settings_from_environment(monkeypatch):
    monkeypatch.setenv("API_BASE_URL", "http://api.internal:9000/")
    monkeypatch.setenv("UI_REQUEST_TIMEOUT", "4.5")
    monkeypatch.setenv("UI_POLL_INTERVAL", "0.25")
    monkeypatch.setenv("UI_TASK_TIMEOUT", "60")

    settings = UISettings.from_env()

    assert settings.api_base_url == "http://api.internal:9000"
    assert settings.request_timeout == 4.5
    assert settings.poll_interval == 0.25
    assert settings.task_timeout == 60


def test_ui_settings_reject_non_positive_timeout(monkeypatch):
    monkeypatch.setenv("UI_TASK_TIMEOUT", "0")
    with pytest.raises(ValueError, match="UI_TASK_TIMEOUT"):
        UISettings.from_env()


def test_upload_preflight_validation():
    validate_upload("report.PDF", 10, max_upload_mb=1)

    with pytest.raises(UIError) as extension:
        validate_upload("report.exe", 10)
    with pytest.raises(UIError) as empty:
        validate_upload("report.pdf", 0)
    with pytest.raises(UIError) as large:
        validate_upload("report.pdf", 2 * 1024 * 1024, max_upload_mb=1)

    assert extension.value.code == "INVALID_FILE_TYPE"
    assert empty.value.code == "INVALID_FILE"
    assert large.value.code == "FILE_TOO_LARGE"


def test_health_config_upload_status_and_query(api_client):
    health = api_client.health()
    config = api_client.public_config()
    created = api_client.upload_document("../demo.pdf", b"%PDF-1.4", "application/pdf")
    task = api_client.document_status(created.task_id)
    answer = api_client.query(" Atlas 何时发布？ ")

    assert health.rag_initialized is True
    assert config.parser_backend == "pipeline"
    assert created.status == "pending"
    assert task.document_id == "doc-1"
    assert answer.answer == "2027 Q3"
    assert answer.mode == "mix"


def test_upload_sends_only_safe_basename():
    observed = {}

    def handler(request: httpx.Request) -> httpx.Response:
        observed["body"] = request.content
        return httpx.Response(
            202,
            content=json.dumps(
                {"task_id": "1", "status": "pending", "file_name": "safe.pdf"}
            ).encode(),
            headers={"content-type": "application/json"},
        )

    client = APIClient("http://test", transport=httpx.MockTransport(handler))
    client.upload_document("../../safe.pdf", b"%PDF", "application/pdf")
    client.close()

    assert b"safe.pdf" in observed["body"]
    assert b"../../safe.pdf" not in observed["body"]


def test_backend_error_uses_safe_contract_without_raw_body():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            503,
            content=json.dumps(
                {
                    "error": {
                        "code": "MODEL_UNAVAILABLE",
                        "message": "模型服务暂时不可用",
                        "request_id": "id",
                    },
                    "debug": "secret-api-key",
                }
            ).encode(),
        )

    client = APIClient("http://test", transport=httpx.MockTransport(handler))
    with pytest.raises(UIError) as exc_info:
        client.query("hello")
    client.close()

    assert exc_info.value.code == "MODEL_UNAVAILABLE"
    assert exc_info.value.message == "模型服务暂时不可用"
    assert "secret-api-key" not in str(exc_info.value)


def test_non_json_and_invalid_success_responses_are_safe():
    def bad_gateway(request: httpx.Request) -> httpx.Response:
        return httpx.Response(502, text="private reverse proxy stack")

    first = APIClient("http://test", transport=httpx.MockTransport(bad_gateway))
    with pytest.raises(UIError, match="后端服务请求失败"):
        first.health()
    first.close()

    second = APIClient(
        "http://test",
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={})),
    )
    with pytest.raises(UIError) as invalid:
        second.health()
    second.close()
    assert invalid.value.code == "INVALID_RESPONSE"


@pytest.mark.parametrize(
    ("raised", "code"),
    [
        (httpx.ConnectError("private host"), "API_OFFLINE"),
        (httpx.ReadTimeout("private timeout"), "REQUEST_TIMEOUT"),
    ],
)
def test_network_errors_are_sanitized(raised, code):
    def handler(request: httpx.Request) -> httpx.Response:
        raise raised

    client = APIClient("http://test", transport=httpx.MockTransport(handler))
    with pytest.raises(UIError) as exc_info:
        client.health()
    client.close()

    assert exc_info.value.code == code
    assert "private" not in str(exc_info.value)


def test_poll_document_reports_updates_until_completed():
    statuses = iter(["pending", "processing", "completed"])

    def handler(request: httpx.Request) -> httpx.Response:
        status = next(statuses)
        return httpx.Response(
            200,
            json={"task_id": "1", "status": status, "file_name": "demo.pdf"},
        )

    client = APIClient("http://test", transport=httpx.MockTransport(handler))
    updates = []
    result = client.poll_document(
        "1",
        timeout=5,
        interval=0,
        on_update=lambda task: updates.append(task.status),
        sleep=lambda _: None,
    )
    client.close()

    assert result.status == "completed"
    assert updates == ["pending", "processing", "completed"]


def test_poll_document_timeout_and_failed_terminal_state():
    def pending(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"task_id": "1", "status": "pending", "file_name": "demo.pdf"},
        )

    client = APIClient("http://test", transport=httpx.MockTransport(pending))
    with pytest.raises(UIError) as timeout:
        client.poll_document("1", timeout=0.001, interval=0.01)
    client.close()
    assert timeout.value.code == "TASK_TIMEOUT"

    failed = APIClient(
        "http://test",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    "task_id": "1",
                    "status": "failed",
                    "file_name": "demo.pdf",
                    "error": "文档处理失败",
                },
            )
        ),
    )
    assert failed.poll_document("1", timeout=1, interval=0).status == "failed"
    failed.close()


def test_empty_question_and_source_fallback(api_client):
    with pytest.raises(UIError) as exc_info:
        api_client.query("   ")

    assert exc_info.value.code == "EMPTY_QUESTION"
    assert sources_text([]) == "当前版本暂未提供结构化来源"
    assert sources_text(["a.pdf", "b.md"]) == "a.pdf、b.md"
