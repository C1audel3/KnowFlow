"""Contracts for the paper library and evidence-grounded reading."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class Paper(BaseModel):
    paper_id: str
    title: str
    file_name: str
    created_at: str
    block_count: int
    modalities: dict[str, int]
    vector_indexed: bool


class PaperEvidence(BaseModel):
    evidence_id: str
    paper_id: str
    page: int | None
    page_end: int | None = None
    section: str
    modality: str
    text: str
    image_url: str | None = None


class PaperReadingRequest(BaseModel):
    paper_id: str = Field(min_length=1, max_length=200)
    question: str = Field(default="", max_length=2000)
    task: Literal["question", "overview"] = "question"
    use_images: bool = True
    retrieval_mode: Literal["retrieved", "none"] = "retrieved"
    top_k: int = Field(default=5, ge=2, le=10)

    @model_validator(mode="after")
    def require_question(self) -> "PaperReadingRequest":
        self.question = self.question.strip()
        if self.task == "question" and not self.question:
            raise ValueError("a question is required")
        if self.retrieval_mode == "none" and self.task != "question":
            raise ValueError("no-retrieval baseline only supports questions")
        return self


class PaperReadingResponse(BaseModel):
    history_id: str | None = None
    answer: str
    evidence: list[PaperEvidence]
    citations: list[str]
    status: Literal["answered", "no_evidence", "uncited"]
    duration_ms: int
    vision_used: bool
    retrieval: Literal[
        "bm25", "hybrid_rrf", "section_coverage", "figure_lookup", "none"
    ]
    warnings: list[str] = Field(default_factory=list)


class PaperReadingHistoryItem(BaseModel):
    history_id: str
    paper_id: str
    created_at: str
    question: str
    task: Literal["question", "overview"]
    status: Literal["answered", "no_evidence", "uncited"]
    vision_used: bool


class PaperReadingHistory(PaperReadingHistoryItem):
    request: PaperReadingRequest
    response: PaperReadingResponse
