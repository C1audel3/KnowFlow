from __future__ import annotations

import asyncio
from pathlib import Path

from raganything.callbacks import CallbackManager


class FakeRAG:
    def __init__(
        self,
        *,
        process_error: Exception | None = None,
        query_error: Exception | None = None,
        process_gate: asyncio.Event | None = None,
    ) -> None:
        self.callback_manager = CallbackManager()
        self.process_error = process_error
        self.query_error = query_error
        self.process_gate = process_gate
        self.process_calls: list[dict] = []
        self.parse_calls: list[dict] = []
        self.query_calls: list[tuple[str, dict]] = []
        self.finalize_calls = 0

    async def process_document_complete(self, **kwargs):
        self.process_calls.append(kwargs)
        if self.process_gate is not None:
            await self.process_gate.wait()
        if self.process_error is not None:
            raise self.process_error

    async def parse_document(self, **kwargs):
        self.parse_calls.append(kwargs)
        path = Path(kwargs["file_path"])
        return [
            {"type": "text", "text": "Project Atlas"},
            {"type": "image", "img_path": "fixture.png"},
        ], f"doc-{path.stem}"

    async def aquery(self, question, **kwargs):
        self.query_calls.append((question, kwargs))
        if self.query_error is not None:
            raise self.query_error
        return "Atlas 将在 2027 Q3 发布。"

    async def finalize_storages(self):
        self.finalize_calls += 1
