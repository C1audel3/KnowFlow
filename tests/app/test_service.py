from __future__ import annotations

import asyncio
from dataclasses import replace
from io import BytesIO

import pytest
from fastapi import UploadFile

from app.core.errors import (
    FileTooLargeError,
    InvalidFileError,
    InvalidFileTypeError,
    RAGNotReadyError,
    TaskNotFoundError,
)
from app.schemas.models import QueryRequest, TaskStatus
from app.services.knowledge_base import KnowledgeBaseService
from fakes import FakeRAG


def upload(name: str, content: bytes) -> UploadFile:
    return UploadFile(file=BytesIO(content), filename=name)


@pytest.mark.asyncio
async def test_service_lifecycle_creates_directories_and_finalizes(app_settings):
    rag = FakeRAG()
    service = KnowledgeBaseService(app_settings, rag_factory=lambda _: rag)

    await service.initialize()
    assert service.initialized is True
    assert app_settings.upload_dir.is_dir()

    await service.close()
    assert service.initialized is False
    assert rag.finalize_calls == 1


@pytest.mark.asyncio
async def test_service_rejects_calls_before_initialization(app_settings):
    service = KnowledgeBaseService(app_settings, rag_factory=lambda _: FakeRAG())

    with pytest.raises(RAGNotReadyError):
        await service.query(QueryRequest(question="hello"))


@pytest.mark.asyncio
async def test_document_moves_from_pending_to_completed(app_settings):
    gate = asyncio.Event()
    rag = FakeRAG(process_gate=gate)
    service = KnowledgeBaseService(app_settings, rag_factory=lambda _: rag)
    await service.initialize()

    submitted = await service.submit_document(upload("../report.pdf", b"%PDF-1.4\n"))
    assert submitted.status is TaskStatus.PENDING
    await asyncio.sleep(0)
    assert service.get_document(submitted.task_id).status is TaskStatus.PROCESSING

    gate.set()
    result = await service.wait_for_document(submitted.task_id)
    assert result.status is TaskStatus.COMPLETED
    assert result.file_name == "report.pdf"
    assert result.document_id
    assert result.content_blocks == 2
    assert result.content_types == {"image": 1, "text": 1}
    stored_path = rag.process_calls[0]["file_path"]
    assert str(app_settings.upload_dir) in stored_path
    assert "report.pdf" not in stored_path
    assert rag.process_calls[0]["backend"] == "pipeline"
    await service.close()


@pytest.mark.asyncio
async def test_markdown_skips_binary_parser_options(app_settings):
    rag = FakeRAG()
    service = KnowledgeBaseService(app_settings, rag_factory=lambda _: rag)
    await service.initialize()

    task = await service.submit_document(upload("note.md", b"# valid markdown\n"))
    result = await service.wait_for_document(task.task_id)

    assert result.status is TaskStatus.COMPLETED
    assert "backend" not in rag.process_calls[0]
    await service.close()


@pytest.mark.asyncio
async def test_failed_document_has_safe_status_and_finalizes(app_settings):
    rag = FakeRAG(process_error=RuntimeError("secret parser detail"))
    service = KnowledgeBaseService(app_settings, rag_factory=lambda _: rag)
    await service.initialize()

    task = await service.submit_document(upload("bad.pdf", b"%PDF-1.4\n"))
    result = await service.wait_for_document(task.task_id)

    assert result.status is TaskStatus.FAILED
    assert result.error == "文档处理失败"
    assert "secret" not in result.error
    await service.close()
    assert rag.finalize_calls == 1


@pytest.mark.asyncio
async def test_upload_validation_removes_partial_files(app_settings):
    service = KnowledgeBaseService(app_settings, rag_factory=lambda _: FakeRAG())
    await service.initialize()

    with pytest.raises(InvalidFileTypeError):
        await service.submit_document(upload("malware.exe", b"payload"))
    with pytest.raises(InvalidFileError):
        await service.submit_document(upload("empty.pdf", b""))
    with pytest.raises(InvalidFileError):
        await service.submit_document(upload("fake.png", b"not png"))

    assert list(app_settings.upload_dir.iterdir()) == []
    await service.close()


@pytest.mark.asyncio
async def test_markdown_validates_content_beyond_header(app_settings):
    settings = replace(app_settings, max_upload_bytes=8192)
    service = KnowledgeBaseService(settings, rag_factory=lambda _: FakeRAG())
    await service.initialize()

    with pytest.raises(InvalidFileError, match="Markdown"):
        await service.submit_document(upload("bad.md", b"a" * 4096 + b"\x00"))

    assert list(settings.upload_dir.iterdir()) == []
    await service.close()


@pytest.mark.asyncio
async def test_large_upload_is_rejected_and_removed(app_settings):
    tiny_settings = replace(app_settings, max_upload_bytes=8)
    service = KnowledgeBaseService(tiny_settings, rag_factory=lambda _: FakeRAG())
    await service.initialize()

    with pytest.raises(FileTooLargeError):
        await service.submit_document(upload("large.pdf", b"%PDF-1.4-too-large"))

    assert list(tiny_settings.upload_dir.iterdir()) == []
    await service.close()


@pytest.mark.asyncio
async def test_query_uses_fixed_mix_options(app_settings):
    rag = FakeRAG()
    service = KnowledgeBaseService(app_settings, rag_factory=lambda _: rag)
    await service.initialize()

    result = await service.query(QueryRequest(question=" Atlas 何时发布？ "))

    assert "2027 Q3" in result.answer
    assert result.sources == []
    assert result.mode == "mix"
    assert rag.query_calls == [
        (
            "Atlas 何时发布？",
            {"mode": "mix", "vlm_enhanced": False, "enable_rerank": False},
        )
    ]
    await service.close()


def test_unknown_task_is_rejected(app_settings):
    service = KnowledgeBaseService(app_settings, rag_factory=lambda _: FakeRAG())
    with pytest.raises(TaskNotFoundError):
        service.get_document("missing")
