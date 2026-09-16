from __future__ import annotations

import json

import httpx
import pytest

from ui.client import APIClient


def json_response(status_code: int, payload: dict) -> httpx.Response:
    return httpx.Response(
        status_code,
        content=json.dumps(payload).encode(),
        headers={"content-type": "application/json"},
    )


@pytest.fixture
def success_handler():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return json_response(
                200,
                {
                    "status": "ok",
                    "rag_initialized": True,
                    "parser": "mineru",
                    "vision_enabled": True,
                },
            )
        if request.url.path == "/config/public":
            return json_response(
                200,
                {
                    "max_upload_mb": 20,
                    "supported_extensions": [".pdf", ".md", ".png"],
                    "parser": "mineru",
                    "parser_backend": "pipeline",
                    "query_mode": "mix",
                    "max_query_concurrency": 2,
                },
            )
        if request.url.path == "/documents" and request.method == "POST":
            return json_response(
                202,
                {
                    "task_id": "task-1",
                    "status": "pending",
                    "file_name": "demo.pdf",
                },
            )
        if request.url.path == "/documents/task-1":
            return json_response(
                200,
                {
                    "task_id": "task-1",
                    "status": "completed",
                    "file_name": "demo.pdf",
                    "document_id": "doc-1",
                    "content_blocks": 7,
                    "content_types": {"text": 4, "image": 1},
                    "duration_ms": 1200,
                },
            )
        if request.url.path == "/query":
            return json_response(
                200,
                {
                    "answer": "2027 Q3",
                    "sources": [],
                    "duration_ms": 42,
                    "mode": "mix",
                },
            )
        return json_response(404, {"error": {"message": "not found"}})

    return handler


@pytest.fixture
def api_client(success_handler):
    client = APIClient(
        "http://testserver/",
        transport=httpx.MockTransport(success_handler),
    )
    yield client
    client.close()
