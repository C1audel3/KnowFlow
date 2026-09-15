"""Pydantic request and response models."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class HealthResponse(BaseModel):
    status: Literal["ok", "unavailable"]
    rag_initialized: bool
    parser: str = "mineru"
    vision_enabled: bool = True


class DocumentTaskResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    task_id: str
    status: TaskStatus
    file_name: str
    document_id: str | None = None
    content_blocks: int | None = None
    content_types: dict[str, int] | None = None
    created_at: datetime
    completed_at: datetime | None = None
    duration_ms: int | None = None
    error: str | None = None


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    mode: Literal["mix"] = "mix"
    vlm_enhanced: bool = False

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be blank")
        return value


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    duration_ms: int
    mode: Literal["mix"]
