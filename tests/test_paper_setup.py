from types import SimpleNamespace

import httpx

from scripts.check_paper_setup import api_checks, local_checks
from ui.client import APIClient


def test_setup_does_not_echo_credentials_or_invalid_configuration(monkeypatch):
    monkeypatch.setenv("LLM_BINDING_API_KEY", "SECRET_SENTINEL")
    monkeypatch.setenv("SUPPORTED_FILE_EXTENSIONS", "SECRET_SENTINEL")
    monkeypatch.setenv("EMBEDDING_BINDING_API_KEY", "replace-with-key")
    results = local_checks()
    assert "SECRET_SENTINEL" not in str(results)
    assert (
        next(item for item in results if item["name"] == "configuration")["status"]
        == "fail"
    )
    assert (
        next(
            item
            for item in results
            if item["name"] == "configured:EMBEDDING_BINDING_API_KEY"
        )["status"]
        == "fail"
    )


def test_api_checks_are_read_only_and_handle_unavailable_without_leaking():
    seen = []

    def handler(request):
        seen.append((request.method, request.url.path))
        return httpx.Response(503, json={"error": {"message": "PRIVATE_SENTINEL"}})

    client = APIClient("http://example.test", transport=httpx.MockTransport(handler))
    try:
        result = api_checks(client)
    finally:
        client.close()
    assert seen == [("GET", "/health")]
    assert result[0]["status"] == "fail"
    assert "PRIVATE_SENTINEL" not in str(result)


def test_api_checks_validate_readiness_and_do_not_generate_or_upload():
    class Client:
        def health(self):
            return SimpleNamespace(status="ok", rag_initialized=True)

        def public_config(self):
            return SimpleNamespace(supported_extensions=[".pdf", ".md"])

        def papers(self):
            return []

    assert all(item["status"] == "pass" for item in api_checks(Client()))
