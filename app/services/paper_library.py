"""SQLite paper catalogue and stable, page-aware source evidence."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any
from uuid import uuid4

from app.core.errors import AppError
from app.schemas.papers import (
    Paper,
    PaperEvidence,
    PaperReadingRequest,
    PaperReadingResponse,
    PaperReadingHistory,
    PaperReadingHistoryItem,
)


TEXT_FIELDS = (
    "text",
    "image_caption",
    "image_footnote",
    "table_caption",
    "table_body",
    "table_footnote",
    "latex",
    "equation_caption",
    "chart_caption",
    "chart_footnote",
    "content",
    "list_items",
)


def flatten(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(flatten(item) for item in value)
    return ""


class PaperLibrary:
    def __init__(self, working_dir: Path, upload_dir: Path, output_dir: Path):
        working_dir.mkdir(parents=True, exist_ok=True)
        self.path = working_dir / "papers.sqlite3"
        self.upload_dir = upload_dir.resolve()
        self.output_dir = output_dir.resolve()
        with self.connect() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS papers (
                    id TEXT PRIMARY KEY, title TEXT NOT NULL, file_name TEXT NOT NULL,
                    stored_path TEXT NOT NULL, created_at TEXT NOT NULL,
                    vector_space TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS paper_evidence (
                    id TEXT PRIMARY KEY, paper_id TEXT NOT NULL,
                    position INTEGER NOT NULL, source_index INTEGER NOT NULL,
                    page INTEGER, section TEXT NOT NULL, modality TEXT NOT NULL,
                    text TEXT NOT NULL, asset TEXT, vector TEXT
                );
                CREATE INDEX IF NOT EXISTS evidence_paper ON paper_evidence(paper_id);
                CREATE TABLE IF NOT EXISTS paper_readings (
                    id TEXT PRIMARY KEY, paper_id TEXT NOT NULL,
                    created_at TEXT NOT NULL, question TEXT NOT NULL,
                    task TEXT NOT NULL, status TEXT NOT NULL,
                    vision_used INTEGER NOT NULL,
                    request_json TEXT NOT NULL, response_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS readings_paper
                    ON paper_readings(paper_id, created_at DESC, id DESC);
            """)
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(paper_evidence)")
            }
            if "page_end" not in columns:
                connection.execute(
                    "ALTER TABLE paper_evidence ADD COLUMN page_end INTEGER"
                )

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def safe_asset(self, value: str) -> Path | None:
        if not value:
            return None
        raw = Path(value)
        candidate = (raw if raw.is_absolute() else self.output_dir / raw).resolve()
        if (
            any(
                candidate.is_relative_to(root)
                for root in (self.output_dir, self.upload_dir)
            )
            and candidate.is_file()
        ):
            return candidate
        return None

    def prepare(
        self, paper_id: str, content: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        rows = []
        section = ""
        for index, item in enumerate(content):
            if not isinstance(item, dict):
                continue
            if (
                item.get("type")
                in {
                    "page_number",
                    "page_header",
                    "page_footer",
                    "header",
                    "footer",
                    "page_footnote",
                    "page_aside_text",
                }
                or item.get("sub_type") == "ref_text"
            ):
                continue
            text = "\n".join(
                part for key in TEXT_FIELDS if (part := flatten(item.get(key))).strip()
            ).strip()
            if item.get("text_level") and text:
                section = text[:250]
            page_idx = item.get("page_idx")
            page = page_idx + 1 if type(page_idx) is int and page_idx >= 0 else None
            page_end_idx = item.get("_page_end_idx")
            page_end = (
                page_end_idx + 1
                if page is not None
                and type(page_end_idx) is int
                and page_end_idx > page_idx
                else None
            )
            asset = self.safe_asset(str(item.get("img_path") or ""))
            if not text and not asset:
                continue
            for part, start in enumerate(range(0, max(1, len(text)), 1650)):
                rows.append(
                    {
                        "id": f"{paper_id}-{index}-{part}",
                        "paper_id": paper_id,
                        "position": len(rows),
                        "source_index": index,
                        "page": page,
                        "page_end": page_end,
                        "section": section,
                        "modality": str(item.get("type", "text")),
                        "text": text[start : start + 1800],
                        "asset": str(asset) if asset else None,
                        "vector": None,
                    }
                )
                if start + 1800 >= len(text):
                    break
        if not rows:
            raise AppError("EMPTY_PAPER", "未解析出可阅读的论文内容", 400)
        return rows

    def save(
        self,
        paper_id: str,
        file_name: str,
        stored_path: Path,
        rows: list[dict[str, Any]],
        vector_space: str,
    ) -> None:
        title = next(
            (row["text"][:250] for row in rows if row["section"] == row["text"]),
            Path(file_name).stem,
        )
        with self.connect() as connection:
            existing = connection.execute(
                "SELECT created_at FROM papers WHERE id = ?", (paper_id,)
            ).fetchone()
            connection.execute(
                "INSERT OR REPLACE INTO papers VALUES (?, ?, ?, ?, ?, ?)",
                (
                    paper_id,
                    title,
                    file_name,
                    str(stored_path.resolve()),
                    existing["created_at"]
                    if existing
                    else datetime.now(timezone.utc).isoformat(),
                    vector_space,
                ),
            )
            connection.execute(
                "DELETE FROM paper_evidence WHERE paper_id = ?", (paper_id,)
            )
            connection.executemany(
                "INSERT INTO paper_evidence (id, paper_id, position, source_index, page, section, modality, text, asset, vector, page_end) VALUES (:id, :paper_id, :position, :source_index, :page, :section, :modality, :text, :asset, :vector, :page_end)",
                [
                    {
                        **row,
                        "page_end": row.get("page_end"),
                        "vector": json.dumps(row["vector"])
                        if row["vector"] is not None
                        else None,
                    }
                    for row in rows
                ],
            )

    def get(self, paper_id: str) -> Paper:
        with self.connect() as connection:
            paper = connection.execute(
                "SELECT * FROM papers WHERE id = ?", (paper_id,)
            ).fetchone()
            counts = connection.execute(
                "SELECT modality, COUNT(*) AS n, COUNT(vector) AS vectors FROM paper_evidence WHERE paper_id = ? GROUP BY modality",
                (paper_id,),
            ).fetchall()
        if paper is None:
            raise AppError("PAPER_NOT_FOUND", "论文不存在", 404)
        modalities = {row["modality"]: row["n"] for row in counts}
        return Paper(
            paper_id=paper_id,
            title=paper["title"],
            file_name=paper["file_name"],
            created_at=paper["created_at"],
            block_count=sum(modalities.values()),
            modalities=modalities,
            vector_indexed=bool(counts)
            and all(row["vectors"] == row["n"] for row in counts),
        )

    def list(self) -> list[Paper]:
        with self.connect() as connection:
            ids = connection.execute(
                "SELECT id FROM papers ORDER BY created_at DESC"
            ).fetchall()
        return [self.get(row["id"]) for row in ids]

    def save_reading(
        self, request: PaperReadingRequest, response: PaperReadingResponse
    ) -> str:
        self.get(request.paper_id)
        if any(item.paper_id != request.paper_id for item in response.evidence):
            raise ValueError("reading evidence belongs to another paper")
        identifier = uuid4().hex
        snapshot = response.model_copy(update={"history_id": identifier})
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO paper_readings VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    identifier,
                    request.paper_id,
                    datetime.now(timezone.utc).isoformat(),
                    request.question,
                    request.task,
                    response.status,
                    int(response.vision_used),
                    request.model_dump_json(),
                    snapshot.model_dump_json(),
                ),
            )
        return identifier

    def readings(
        self, paper_id: str, offset: int = 0, limit: int = 20
    ) -> list[PaperReadingHistoryItem]:
        self.get(paper_id)
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT id AS history_id, paper_id, created_at, question, task, status, vision_used "
                "FROM paper_readings WHERE paper_id = ? ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?",
                (paper_id, limit, offset),
            ).fetchall()
        return [PaperReadingHistoryItem.model_validate(dict(row)) for row in rows]

    def reading(self, paper_id: str, history_id: str) -> PaperReadingHistory:
        self.get(paper_id)
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM paper_readings WHERE paper_id = ? AND id = ?",
                (paper_id, history_id),
            ).fetchone()
        if row is None:
            raise AppError("READING_NOT_FOUND", "阅读记录不存在", 404)
        request = PaperReadingRequest.model_validate_json(row["request_json"])
        response = PaperReadingResponse.model_validate_json(row["response_json"])
        return PaperReadingHistory(
            history_id=row["id"],
            paper_id=row["paper_id"],
            created_at=row["created_at"],
            question=row["question"],
            task=row["task"],
            status=row["status"],
            vision_used=bool(row["vision_used"]),
            request=request,
            response=response,
        )

    def vector_space(self, paper_id: str) -> str:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT vector_space FROM papers WHERE id = ?", (paper_id,)
            ).fetchone()
        return row["vector_space"] if row else ""

    def blocks(self, paper_id: str) -> list[dict[str, Any]]:
        self.get(paper_id)
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM paper_evidence WHERE paper_id = ? ORDER BY position",
                (paper_id,),
            ).fetchall()
        return [
            {
                **dict(row),
                "vector": json.loads(row["vector"]) if row["vector"] else None,
            }
            for row in rows
        ]

    @staticmethod
    def public(row: dict[str, Any]) -> PaperEvidence:
        return PaperEvidence(
            evidence_id=row["id"],
            paper_id=row["paper_id"],
            page=row["page"],
            page_end=row.get("page_end"),
            section=row["section"],
            modality=row["modality"],
            text=row["text"],
            image_url=f"/papers/{row['paper_id']}/assets/{row['id']}"
            if row["asset"]
            else None,
        )

    def asset(self, paper_id: str, evidence_id: str) -> Path:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT asset FROM paper_evidence WHERE paper_id = ? AND id = ?",
                (paper_id, evidence_id),
            ).fetchone()
        path = self.safe_asset(row["asset"] or "") if row else None
        if path is None:
            raise AppError("ASSET_NOT_FOUND", "证据图片不存在", 404)
        return path

    def original(self, paper_id: str) -> Path:
        self.get(paper_id)
        with self.connect() as connection:
            row = connection.execute(
                "SELECT stored_path FROM papers WHERE id = ?", (paper_id,)
            ).fetchone()
        path = self.safe_asset(row["stored_path"])
        if path is None or not path.is_relative_to(self.upload_dir):
            raise AppError("ASSET_NOT_FOUND", "原文文件不存在", 404)
        return path
