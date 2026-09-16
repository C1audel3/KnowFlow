from __future__ import annotations

import asyncio
from dataclasses import replace
from io import BytesIO

import pytest

from app.schemas.models import QueryRequest, TaskStatus
from app.services.knowledge_base import KnowledgeBaseService
from fakes import FakeRAG


class MemoryUpload:
    def __init__(self, name: str, content: bytes) -> None:
        self.filename = name
        self._content = BytesIO(content)

    async def read(self, size: int = -1) -> bytes:
        return self._content.read(size)

    async def close(self) -> None:
        self._content.close()


def pdf_upload(name: str = "demo.pdf") -> MemoryUpload:
    return MemoryUpload(name, b"%PDF-1.4\n")


class ConcurrencyRAG(FakeRAG):
    def __init__(self) -> None:
        super().__init__()
        self.query_active = 0
        self.query_max_active = 0
        self.query_started = asyncio.Event()
        self.query_release = asyncio.Event()

    async def aquery(self, question, **kwargs):
        self.query_calls.append((question, kwargs))
        self.query_active += 1
        self.query_max_active = max(self.query_max_active, self.query_active)
        self.query_started.set()
        await self.query_release.wait()
        self.query_active -= 1
        return "answer"


@pytest.mark.asyncio
async def test_query_concurrency_limit_is_enforced(app_settings):
    settings = replace(app_settings, max_query_concurrency=1)
    rag = ConcurrencyRAG()
    service = KnowledgeBaseService(settings, rag_factory=lambda _: rag)
    await service.initialize()

    first = asyncio.create_task(service.query(QueryRequest(question="one")))
    second = asyncio.create_task(service.query(QueryRequest(question="two")))
    await rag.query_started.wait()
    await asyncio.sleep(0)

    assert rag.query_active == 1
    assert len(rag.query_calls) == 1
    rag.query_release.set()
    await asyncio.gather(first, second)
    assert rag.query_max_active == 1
    await service.close()


@pytest.mark.asyncio
async def test_query_can_finish_while_document_processing_waits(app_settings):
    gate = asyncio.Event()
    rag = FakeRAG(process_gate=gate)
    service = KnowledgeBaseService(app_settings, rag_factory=lambda _: rag)
    await service.initialize()

    document = await asyncio.wait_for(service.submit_document(pdf_upload()), timeout=1)
    await asyncio.sleep(0)
    assert service.get_document(document.task_id).status is TaskStatus.PROCESSING

    query = await asyncio.wait_for(
        service.query(QueryRequest(question="question")), timeout=1
    )
    assert query.answer
    gate.set()
    assert (
        await asyncio.wait_for(service.wait_for_document(document.task_id), timeout=1)
    ).status is TaskStatus.COMPLETED
    await asyncio.wait_for(service.close(), timeout=1)


@pytest.mark.asyncio
async def test_close_waits_for_active_document_before_finalizing(app_settings):
    gate = asyncio.Event()
    rag = FakeRAG(process_gate=gate)
    service = KnowledgeBaseService(app_settings, rag_factory=lambda _: rag)
    await service.initialize()
    document = await service.submit_document(pdf_upload())
    await asyncio.sleep(0)

    closing = asyncio.create_task(service.close())
    await asyncio.sleep(0)
    assert closing.done() is False
    assert rag.finalize_calls == 0

    gate.set()
    await asyncio.wait_for(closing, timeout=1)
    assert service.get_document(document.task_id).status is TaskStatus.COMPLETED
    assert rag.finalize_calls == 1


@pytest.mark.asyncio
async def test_repeated_uploads_get_isolated_tasks_and_paths(app_settings):
    rag = FakeRAG()
    service = KnowledgeBaseService(app_settings, rag_factory=lambda _: rag)
    await service.initialize()

    first = await service.submit_document(pdf_upload("same.pdf"))
    second = await service.submit_document(pdf_upload("same.pdf"))
    await asyncio.gather(
        service.wait_for_document(first.task_id),
        service.wait_for_document(second.task_id),
    )

    assert first.task_id != second.task_id
    assert rag.process_calls[0]["file_path"] != rag.process_calls[1]["file_path"]
    await asyncio.wait_for(service.close(), timeout=1)
