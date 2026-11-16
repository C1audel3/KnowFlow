from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app.schemas.papers import (
    PaperReadingHistory,
    PaperReadingRequest,
    PaperReadingResponse,
)
from ui.client import DocumentTaskView, UIError


@pytest.mark.parametrize(
    "outcome", ["completed", "failed", "expired", "offline", "timeout"]
)
def test_upload_tracking_stops_on_terminal_state_and_keeps_transient_errors(
    monkeypatch, outcome
):
    import ui.paper_tasks as tasks

    calls = []

    class Client:
        def document_status(self, task_id):
            calls.append(task_id)
            if len(calls) == 1:
                return DocumentTaskView(
                    task_id=task_id, file_name="demo.pdf", status="processing"
                )
            if outcome in {"expired", "offline"}:
                raise UIError(
                    "temporary",
                    "DOCUMENT_NOT_FOUND" if outcome == "expired" else "API_OFFLINE",
                )
            return DocumentTaskView(
                task_id=task_id,
                file_name="demo.pdf",
                status=outcome,
                error="解析失败" if outcome == "failed" else None,
            )

    monkeypatch.setattr(tasks, "test_client", Client(), raising=False)
    clock = [0.0]
    monkeypatch.setattr(tasks.time, "monotonic", lambda: clock[0])
    app = AppTest.from_string(
        "import ui.paper_tasks as tasks\ntasks.render_upload_tracker(tasks.test_client, 10)\ntasks.render_task_result()"
    )
    app.session_state["paper_task_id"] = "task-a"
    app.run()
    assert not app.exception and len(calls) == 1
    if outcome == "timeout":
        clock[0] = 11.0
    app.run()
    assert not app.exception
    if outcome == "offline":
        assert app.session_state["paper_task_id"] == "task-a"
        assert app.warning
    else:
        expected = "tracking_paused" if outcome == "timeout" else outcome
        assert app.session_state["paper_task_result"]["status"] == expected
        before = len(calls)
        app.run()
        assert len(calls) == before
        if outcome == "timeout":
            next(
                button for button in app.button if button.label == "继续跟踪任务"
            ).click()
            app.run()
            assert len(calls) == before + 1


def test_history_can_be_loaded_in_new_session_without_generating(monkeypatch):
    from types import SimpleNamespace
    import ui.client
    from app.schemas.papers import Paper

    request = PaperReadingRequest(paper_id="paper-a", question="历史问题")
    response = PaperReadingResponse(
        history_id="reading-a",
        answer="历史回答",
        evidence=[],
        citations=[],
        status="uncited",
        duration_ms=1,
        vision_used=False,
        retrieval="bm25",
    )
    record = PaperReadingHistory(
        history_id="reading-a",
        paper_id="paper-a",
        created_at="2026-11-12T00:00:00+00:00",
        question=request.question,
        task=request.task,
        status=response.status,
        vision_used=False,
        request=request,
        response=response,
    )
    calls = []

    class Client:
        def health(self):
            return SimpleNamespace(rag_initialized=True)

        def public_config(self):
            return SimpleNamespace(max_upload_mb=20)

        def papers(self):
            return [
                Paper(
                    paper_id="paper-a",
                    title="Paper",
                    file_name="a.pdf",
                    created_at="today",
                    block_count=1,
                    modalities={"text": 1},
                    vector_indexed=True,
                )
            ]

        def paper_content(self, *args):
            return []

        def paper_readings(self, paper_id, offset):
            return [record]

        def paper_reading(self, paper_id, history_id):
            calls.append((paper_id, history_id))
            return record

        def paper_read(self, *args):
            raise AssertionError("history must not call the model")

    monkeypatch.setenv("API_BASE_URL", "http://history-ui-test:8123")
    monkeypatch.setattr(ui.client, "APIClient", lambda *args: Client())
    path = Path(__file__).resolve().parents[2] / "ui/streamlit_app.py"
    app = AppTest.from_file(str(path)).run()
    assert not app.exception
    next(item for item in app.selectbox if item.label == "已保存的阅读").select(
        "reading-a"
    )
    app.run()
    next(button for button in app.button if button.label == "载入记录").click()
    app.run()
    assert not app.exception
    assert calls == [("paper-a", "reading-a")]
    assert any("历史回答" in item.value for item in app.markdown)
    assert app.session_state["paper_answer"][2]["question"] == "历史问题"
