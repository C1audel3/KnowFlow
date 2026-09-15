"""FastAPI entry point for the KnowFlow multimodal RAG service."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
import logging
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, File, Request, UploadFile, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.core.config import PROJECT_ROOT, AppSettings
from app.core.errors import AppError
from app.schemas import (
    DocumentTaskResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
)
from app.services import KnowledgeBaseService

logger = logging.getLogger(__name__)


def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "message": message,
                "request_id": str(uuid4()),
            }
        },
    )


def create_app(
    settings: AppSettings | None = None,
    rag_factory: Callable[[Any], Any] | None = None,
) -> FastAPI:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    resolved_settings = settings or AppSettings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        service = KnowledgeBaseService(
            resolved_settings,
            rag_factory=rag_factory,
        )
        app.state.service = service
        await service.initialize()
        try:
            yield
        finally:
            await service.close()

    app = FastAPI(
        title="KnowFlow Multimodal RAG API",
        version="0.3.0",
        lifespan=lifespan,
    )

    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        return _error_response(exc.status_code, exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return _error_response(400, "INVALID_REQUEST", "请求参数无效")

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        logger.error("Unhandled API error (error_type=%s)", type(exc).__name__)
        return _error_response(500, "INTERNAL_ERROR", "服务内部错误")

    @app.get("/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        service: KnowledgeBaseService = request.app.state.service
        return HealthResponse(
            status="ok" if service.initialized else "unavailable",
            rag_initialized=service.initialized,
        )

    @app.get("/config/public")
    async def public_config() -> dict[str, object]:
        return resolved_settings.public_dict()

    @app.post(
        "/documents",
        response_model=DocumentTaskResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def upload_document(
        request: Request, file: UploadFile = File(...)
    ) -> DocumentTaskResponse:
        service: KnowledgeBaseService = request.app.state.service
        return await service.submit_document(file)

    @app.get("/documents/{task_id}", response_model=DocumentTaskResponse)
    async def document_status(task_id: str, request: Request) -> DocumentTaskResponse:
        service: KnowledgeBaseService = request.app.state.service
        return service.get_document(task_id)

    @app.post("/query", response_model=QueryResponse)
    async def query(payload: QueryRequest, request: Request) -> QueryResponse:
        service: KnowledgeBaseService = request.app.state.service
        return await service.query(payload)

    return app


app = create_app()
