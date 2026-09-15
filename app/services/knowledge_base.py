"""RAG lifecycle, upload validation, in-memory tasks, and query orchestration."""

from __future__ import annotations

import asyncio
import codecs
import logging
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import UploadFile

from app.core.config import AppSettings
from app.core.errors import (
    FileTooLargeError,
    InvalidFileError,
    InvalidFileTypeError,
    ModelUnavailableError,
    RAGNotReadyError,
    TaskNotFoundError,
)
from app.schemas.models import (
    DocumentTaskResponse,
    QueryRequest,
    QueryResponse,
    TaskStatus,
)
from examples.simple_multimodal_rag import build_rag

logger = logging.getLogger(__name__)
UPLOAD_CHUNK_SIZE = 1024 * 1024


@dataclass(slots=True)
class _DocumentTask:
    task_id: str
    status: TaskStatus
    file_name: str
    stored_path: Path
    created_at: datetime
    document_id: str | None = None
    content_blocks: int | None = None
    content_types: dict[str, int] | None = None
    completed_at: datetime | None = None
    duration_ms: int | None = None
    error: str | None = None

    def public(self) -> DocumentTaskResponse:
        return DocumentTaskResponse.model_validate(self)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _content_types(content: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(
        str(item.get("type", "unknown")) for item in content if isinstance(item, dict)
    )
    return dict(sorted(counts.items()))


def _valid_signature(extension: str, header: bytes) -> bool:
    if extension == ".pdf":
        return header.startswith(b"%PDF-")
    if extension == ".png":
        return header.startswith(b"\x89PNG\r\n\x1a\n")
    if extension in {".jpg", ".jpeg"}:
        return header.startswith(b"\xff\xd8\xff")
    if extension == ".md":
        return True
    return False


class KnowledgeBaseService:
    """Own one RAG instance and expose safe application operations."""

    def __init__(
        self,
        settings: AppSettings,
        rag_factory: Callable[[Path], Any] | None = None,
    ) -> None:
        self.settings = settings
        self._rag_factory = rag_factory or build_rag
        self.rag: Any | None = None
        self.initialized = False
        self._tasks: dict[str, _DocumentTask] = {}
        self._workers: set[asyncio.Task[None]] = set()
        self._worker_by_task: dict[str, asyncio.Task[None]] = {}
        self._ingest_lock = asyncio.Semaphore(1)
        self._query_lock = asyncio.Semaphore(settings.max_query_concurrency)

    async def initialize(self) -> None:
        self.settings.upload_dir.mkdir(parents=True, exist_ok=True)
        self.settings.output_dir.mkdir(parents=True, exist_ok=True)
        self.settings.working_dir.mkdir(parents=True, exist_ok=True)
        self.rag = self._rag_factory(self.settings.working_dir)
        self.initialized = True

    async def close(self) -> None:
        workers = tuple(self._workers)
        if workers:
            await asyncio.gather(*workers, return_exceptions=True)
        if self.rag is not None:
            await self.rag.finalize_storages()
        self.initialized = False

    def _require_ready(self) -> Any:
        if not self.initialized or self.rag is None:
            raise RAGNotReadyError()
        return self.rag

    async def submit_document(self, upload: UploadFile) -> DocumentTaskResponse:
        self._require_ready()
        display_name = Path(upload.filename or "").name
        if not display_name or display_name in {".", ".."}:
            raise InvalidFileError("文件名无效")
        extension = Path(display_name).suffix.lower()
        if extension not in self.settings.supported_extensions:
            raise InvalidFileTypeError()

        task_id = str(uuid4())
        stored_path = (self.settings.upload_dir / f"{task_id}{extension}").resolve()
        if stored_path.parent != self.settings.upload_dir:
            raise InvalidFileError("文件路径无效")

        size = 0
        header = b""
        markdown_decoder = (
            codecs.getincrementaldecoder("utf-8")() if extension == ".md" else None
        )
        try:
            with stored_path.open("xb") as output:
                while chunk := await upload.read(UPLOAD_CHUNK_SIZE):
                    size += len(chunk)
                    if size > self.settings.max_upload_bytes:
                        raise FileTooLargeError()
                    if len(header) < 4096:
                        header += chunk[: 4096 - len(header)]
                    if markdown_decoder is not None:
                        if b"\x00" in chunk:
                            raise InvalidFileError("Markdown 文件内容无效")
                        try:
                            markdown_decoder.decode(chunk)
                        except UnicodeDecodeError as exc:
                            raise InvalidFileError(
                                "Markdown 文件必须使用 UTF-8"
                            ) from exc
                    output.write(chunk)
                if markdown_decoder is not None:
                    try:
                        markdown_decoder.decode(b"", final=True)
                    except UnicodeDecodeError as exc:
                        raise InvalidFileError("Markdown 文件必须使用 UTF-8") from exc
        except Exception:
            stored_path.unlink(missing_ok=True)
            raise
        finally:
            await upload.close()

        if size == 0:
            stored_path.unlink(missing_ok=True)
            raise InvalidFileError("文件不能为空")
        if not _valid_signature(extension, header):
            stored_path.unlink(missing_ok=True)
            raise InvalidFileError("文件扩展名与内容不匹配")

        record = _DocumentTask(
            task_id=task_id,
            status=TaskStatus.PENDING,
            file_name=display_name,
            stored_path=stored_path,
            created_at=_utc_now(),
        )
        self._tasks[task_id] = record
        worker = asyncio.create_task(self._process_document(record))
        self._workers.add(worker)
        self._worker_by_task[task_id] = worker
        worker.add_done_callback(
            lambda completed, current_task_id=task_id: self._forget_worker(
                current_task_id, completed
            )
        )
        return record.public()

    def _forget_worker(self, task_id: str, worker: asyncio.Task[None]) -> None:
        self._workers.discard(worker)
        if self._worker_by_task.get(task_id) is worker:
            self._worker_by_task.pop(task_id, None)

    async def _process_document(self, record: _DocumentTask) -> None:
        started_at = time.perf_counter()
        record.status = TaskStatus.PROCESSING
        try:
            rag = self._require_ready()
            async with self._ingest_lock:
                options: dict[str, Any] = {}
                if record.stored_path.suffix != ".md":
                    options = {
                        "backend": self.settings.parser_backend,
                        "timeout": self.settings.parser_timeout,
                    }
                await rag.process_document_complete(
                    file_path=str(record.stored_path),
                    output_dir=str(self.settings.output_dir),
                    parse_method="auto",
                    display_stats=False,
                    **options,
                )
                content, document_id = await rag.parse_document(
                    file_path=str(record.stored_path),
                    output_dir=str(self.settings.output_dir),
                    parse_method="auto",
                    display_stats=False,
                    **options,
                )
            record.document_id = document_id
            record.content_blocks = len(content)
            record.content_types = _content_types(content)
            record.status = TaskStatus.COMPLETED
        except asyncio.CancelledError:
            record.status = TaskStatus.FAILED
            record.error = "文档处理已取消"
            raise
        except Exception as exc:
            logger.error(
                "Document processing failed for %s (error_type=%s)",
                record.file_name,
                type(exc).__name__,
            )
            record.status = TaskStatus.FAILED
            record.error = "文档处理失败"
        finally:
            record.completed_at = _utc_now()
            record.duration_ms = round((time.perf_counter() - started_at) * 1000)

    def get_document(self, task_id: str) -> DocumentTaskResponse:
        record = self._tasks.get(task_id)
        if record is None:
            raise TaskNotFoundError()
        return record.public()

    async def wait_for_document(self, task_id: str) -> DocumentTaskResponse:
        """Wait for one task; intended for tests and controlled integrations."""
        if task_id not in self._tasks:
            raise TaskNotFoundError()
        worker = self._worker_by_task.get(task_id)
        if worker is not None:
            await asyncio.shield(worker)
        return self.get_document(task_id)

    async def query(self, request: QueryRequest) -> QueryResponse:
        rag = self._require_ready()
        started_at = time.perf_counter()
        try:
            async with self._query_lock:
                answer = await rag.aquery(
                    request.question,
                    mode="mix",
                    vlm_enhanced=request.vlm_enhanced,
                    enable_rerank=False,
                )
        except Exception as exc:
            logger.error("RAG query failed (error_type=%s)", type(exc).__name__)
            raise ModelUnavailableError() from exc
        if not isinstance(answer, str) or not answer.strip():
            raise ModelUnavailableError()
        return QueryResponse(
            answer=answer.strip(),
            sources=[],
            duration_ms=round((time.perf_counter() - started_at) * 1000),
            mode="mix",
        )
