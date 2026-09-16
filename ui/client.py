"""Safe, testable HTTP client and validation helpers for the demo UI."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

DEFAULT_EXTENSIONS = (".pdf", ".md", ".png", ".jpg", ".jpeg")
TERMINAL_TASK_STATES = {"completed", "failed"}


class UIError(Exception):
    """An error message that is safe to display to end users."""

    def __init__(self, message: str, code: str = "UI_ERROR") -> None:
        super().__init__(message)
        self.message = message
        self.code = code


def normalize_api_base_url(value: str) -> str:
    candidate = value.strip().rstrip("/")
    parsed = urlsplit(candidate)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("API_BASE_URL must be an http(s) URL")
    if parsed.username or parsed.password:
        raise ValueError("API_BASE_URL cannot contain credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("API_BASE_URL cannot contain a query or fragment")
    if parsed.path not in {"", "/"}:
        raise ValueError("API_BASE_URL cannot contain a path")
    return urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))


def _positive_float(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)).strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


@dataclass(frozen=True, slots=True)
class UISettings:
    api_base_url: str
    request_timeout: float
    poll_interval: float
    task_timeout: float

    @classmethod
    def from_env(cls) -> UISettings:
        return cls(
            api_base_url=normalize_api_base_url(
                os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
            ),
            request_timeout=_positive_float("UI_REQUEST_TIMEOUT", 30),
            poll_interval=_positive_float("UI_POLL_INTERVAL", 1),
            task_timeout=_positive_float("UI_TASK_TIMEOUT", 900),
        )


class HealthView(BaseModel):
    model_config = ConfigDict(extra="ignore")

    status: str
    rag_initialized: bool
    parser: str
    vision_enabled: bool


class PublicConfigView(BaseModel):
    model_config = ConfigDict(extra="ignore")

    max_upload_mb: int = Field(gt=0)
    supported_extensions: list[str]
    parser: str
    parser_backend: str
    query_mode: str
    max_query_concurrency: int = Field(gt=0)


class DocumentTaskView(BaseModel):
    model_config = ConfigDict(extra="ignore")

    task_id: str
    status: str
    file_name: str
    document_id: str | None = None
    content_blocks: int | None = None
    content_types: dict[str, int] | None = None
    duration_ms: int | None = None
    error: str | None = None


class QueryView(BaseModel):
    model_config = ConfigDict(extra="ignore")

    answer: str = Field(min_length=1)
    sources: list[str]
    duration_ms: int = Field(ge=0)
    mode: str


def validate_upload(
    file_name: str,
    size_bytes: int,
    *,
    supported_extensions: list[str] | tuple[str, ...] = DEFAULT_EXTENSIONS,
    max_upload_mb: int = 20,
) -> None:
    extension = Path(file_name).suffix.lower()
    if extension not in {item.lower() for item in supported_extensions}:
        raise UIError("不支持该文件类型", "INVALID_FILE_TYPE")
    if size_bytes <= 0:
        raise UIError("文件不能为空", "INVALID_FILE")
    if size_bytes > max_upload_mb * 1024 * 1024:
        raise UIError("文件超过大小限制", "FILE_TOO_LARGE")


def sources_text(sources: list[str]) -> str:
    if not sources:
        return "当前版本暂未提供结构化来源"
    return "、".join(sources)


class APIClient:
    def __init__(
        self,
        base_url: str,
        timeout: float = 30,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = normalize_api_base_url(base_url)
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    @staticmethod
    def _error_from_response(response: httpx.Response) -> UIError:
        messages = {
            400: "请求内容无效",
            404: "任务不存在或已经失效",
            413: "文件超过大小限制",
            415: "不支持该文件类型",
            503: "模型或 RAG 服务暂时不可用",
        }
        code = f"HTTP_{response.status_code}"
        message = messages.get(response.status_code, "后端服务请求失败")
        try:
            payload = response.json()
            error = payload.get("error") if isinstance(payload, dict) else None
            if isinstance(error, dict):
                safe_code = error.get("code")
                safe_message = error.get("message")
                if isinstance(safe_code, str) and safe_code:
                    code = safe_code[:80]
                if isinstance(safe_message, str) and 0 < len(safe_message) <= 200:
                    message = safe_message
        except ValueError:
            pass
        return UIError(message, code)

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            response = self._client.request(method, path, **kwargs)
        except httpx.TimeoutException as exc:
            raise UIError("请求超时，请稍后重试", "REQUEST_TIMEOUT") from exc
        except httpx.RequestError as exc:
            raise UIError(
                "无法连接后端服务，请确认 FastAPI 已启动", "API_OFFLINE"
            ) from exc
        if response.is_error:
            raise self._error_from_response(response)
        return response

    @staticmethod
    def _validated_json(response: httpx.Response, model: type[BaseModel]) -> BaseModel:
        try:
            return model.model_validate(response.json())
        except (ValueError, ValidationError) as exc:
            raise UIError("后端返回了无法识别的数据", "INVALID_RESPONSE") from exc

    def health(self) -> HealthView:
        response = self._request("GET", "/health")
        return self._validated_json(response, HealthView)  # type: ignore[return-value]

    def public_config(self) -> PublicConfigView:
        response = self._request("GET", "/config/public")
        return self._validated_json(response, PublicConfigView)  # type: ignore[return-value]

    def upload_document(
        self, file_name: str, content: bytes, content_type: str
    ) -> DocumentTaskView:
        response = self._request(
            "POST",
            "/documents",
            files={"file": (Path(file_name).name, content, content_type)},
        )
        return self._validated_json(response, DocumentTaskView)  # type: ignore[return-value]

    def document_status(self, task_id: str) -> DocumentTaskView:
        response = self._request("GET", f"/documents/{task_id}")
        return self._validated_json(response, DocumentTaskView)  # type: ignore[return-value]

    def poll_document(
        self,
        task_id: str,
        *,
        timeout: float,
        interval: float,
        on_update: Callable[[DocumentTaskView], None] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> DocumentTaskView:
        deadline = clock() + timeout
        while True:
            task = self.document_status(task_id)
            if on_update is not None:
                on_update(task)
            if task.status in TERMINAL_TASK_STATES:
                return task
            if clock() >= deadline:
                raise UIError("文档处理等待超时，可稍后重新查询任务", "TASK_TIMEOUT")
            sleep(interval)

    def query(self, question: str, *, vlm_enhanced: bool = False) -> QueryView:
        question = question.strip()
        if not question:
            raise UIError("请输入问题", "EMPTY_QUESTION")
        response = self._request(
            "POST",
            "/query",
            json={
                "question": question,
                "mode": "mix",
                "vlm_enhanced": vlm_enhanced,
            },
        )
        return self._validated_json(response, QueryView)  # type: ignore[return-value]
