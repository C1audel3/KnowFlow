from pathlib import Path
from types import SimpleNamespace

import httpx
from streamlit.testing.v1 import AppTest

from app.schemas.papers import Paper, PaperEvidence, PaperReadingResponse
from ui.client import APIClient


def test_paper_client_routes_and_serializes_reading_request():
    from app.schemas.papers import PaperReadingRequest

    seen = []

    def handler(request):
        seen.append((request.method, request.url.path))
        if request.url.path == "/papers/read":
            return httpx.Response(
                200,
                json={
                    "answer": "Answer [E1]",
                    "evidence": [],
                    "citations": ["E1"],
                    "status": "answered",
                    "duration_ms": 12,
                    "vision_used": False,
                    "retrieval": "bm25",
                    "warnings": [],
                },
            )
        return httpx.Response(200, json=[])

    client = APIClient("http://testserver", transport=httpx.MockTransport(handler))
    assert client.papers() == []
    assert client.paper_content("paper-a") == []
    result = client.paper_read(
        PaperReadingRequest(paper_id="paper-a", question="method")
    )
    assert result.answer == "Answer [E1]"
    assert seen == [
        ("GET", "/papers"),
        ("GET", "/papers/paper-a/content"),
        ("POST", "/papers/read"),
    ]
    client.close()


def test_paper_workspace_can_browse_and_submit_a_question(monkeypatch):
    import ui.client

    evidence = PaperEvidence(
        evidence_id="paper-a-1",
        paper_id="paper-a",
        page=2,
        section="Methods",
        modality="text",
        text="Retrieval pipeline",
    )
    calls = []

    class FakeClient:
        def health(self):
            return SimpleNamespace(rag_initialized=True)

        def public_config(self):
            return SimpleNamespace(max_upload_mb=20)

        def papers(self):
            return [
                Paper(
                    paper_id="paper-a",
                    title="Example Paper",
                    file_name="demo.pdf",
                    created_at="2026-11-10",
                    block_count=1,
                    modalities={"text": 1},
                    vector_indexed=False,
                )
            ]

        def paper_content(self, paper_id, offset):
            assert paper_id == "paper-a"
            return [evidence]

        def paper_readings(self, paper_id, offset):
            return []

        def paper_read(self, request):
            calls.append(request)
            return PaperReadingResponse(
                answer="方法使用检索。[E1]",
                evidence=[evidence],
                citations=["E1"],
                status="answered",
                duration_ms=10,
                vision_used=False,
                retrieval="bm25",
            )

    monkeypatch.setenv("API_BASE_URL", "http://paper-ui-test:8123")
    monkeypatch.setattr(ui.client, "APIClient", lambda *args: FakeClient())
    path = Path(__file__).resolve().parents[2] / "ui" / "streamlit_app.py"
    app = AppTest.from_file(str(path)).run(timeout=15)
    assert not app.exception
    assert app.selectbox[0].value == "paper-a"
    app.text_area[0].set_value("What is the method?")
    next(button for button in app.button if button.label == "开始阅读").click()
    app.run(timeout=15)
    assert not app.exception
    assert calls[0].paper_id == "paper-a"
    assert calls[0].question == "What is the method?"
    assert any("方法使用检索" in item.value for item in app.markdown)
    assert any("[E1] 已引用" in item.label for item in app.expander)
    saved = app.session_state["paper_answer"]
    assert saved[2]["question"] == "What is the method?"
    assert saved[1]["evidence"][0]["evidence_id"] == "paper-a-1"
    assert any(
        button.label == "导出本次阅读记录" for button in app.get("download_button")
    )
