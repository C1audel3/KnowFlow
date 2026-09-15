from __future__ import annotations

from pathlib import Path

import pytest

from raganything.callbacks import CallbackManager
from raganything.parser import MineruParser

from examples.process_real_documents import (
    DEFAULT_DOCUMENTS,
    FIXED_QUESTIONS,
    count_content_types,
    parser_options,
    run_document_pipeline,
    validate_document_paths,
)


class FakeDocumentRAG:
    def __init__(self, content_by_suffix=None, process_error=None, query_error=None):
        self.callback_manager = CallbackManager()
        self.content_by_suffix = content_by_suffix or {
            ".pdf": [
                {"type": "text", "text": "Atlas 2027 Q3", "page_idx": 0},
                {"type": "image", "img_path": "/tmp/image.png", "page_idx": 0},
                {"type": "table", "table_body": "Sensor-B 18 ms", "page_idx": 0},
                {"type": "equation", "latex": "F_1", "page_idx": 0},
            ],
            ".md": [{"type": "text", "text": "Lin Qiao", "page_idx": 0}],
            ".png": [{"type": "text", "text": "PIXEL-42", "page_idx": 0}],
        }
        self.process_error = process_error
        self.query_error = query_error
        self.process_calls = []
        self.parse_calls = []
        self.query_calls = []
        self.finalize_calls = 0
        self.answers = {
            "pdf_text": "发布时间是 2027 Q3。",
            "pdf_table": "Sensor-B 的延迟是 18 ms。",
            "pdf_image": "视觉网关名为 ORION。",
            "pdf_equation": "F1 综合精确率和召回率。",
            "markdown": "负责人是 Lin Qiao。",
            "image": "IMAGE CODE 是 PIXEL-42。",
        }

    async def process_document_complete(self, **kwargs):
        self.process_calls.append(kwargs)
        path = kwargs["file_path"]
        if self.process_error:
            self.callback_manager.dispatch(
                "on_document_error",
                file_path=path,
                error=self.process_error,
                stage="parse",
            )
            raise self.process_error
        content = self.content_by_suffix[Path(path).suffix]
        self.callback_manager.dispatch(
            "on_parse_complete",
            file_path=path,
            content_blocks=len(content),
            doc_id=f"doc-{Path(path).stem}",
            duration_seconds=0.1,
        )
        self.callback_manager.dispatch(
            "on_text_insert_complete", file_path=path, duration_seconds=0.2
        )
        self.callback_manager.dispatch(
            "on_multimodal_complete",
            file_path=path,
            processed_count=max(len(content) - 1, 0),
            duration_seconds=0.3,
        )
        self.callback_manager.dispatch("on_document_complete", file_path=path)

    async def parse_document(self, **kwargs):
        self.parse_calls.append(kwargs)
        path = Path(kwargs["file_path"])
        return self.content_by_suffix[path.suffix], f"doc-{path.stem}"

    async def aquery(self, question, **kwargs):
        self.query_calls.append((question, kwargs))
        if self.query_error:
            raise self.query_error
        case = next(case for case in FIXED_QUESTIONS if case["question"] == question)
        return self.answers[case["id"]]

    async def finalize_storages(self):
        self.finalize_calls += 1


def test_phase2_sample_files_are_small_and_valid():
    pdf, markdown, image = validate_document_paths(list(DEFAULT_DOCUMENTS))

    assert pdf.read_bytes().startswith(b"%PDF-")
    assert image.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert "Lin Qiao" in markdown.read_text(encoding="utf-8")
    assert all(path.stat().st_size < 1_000_000 for path in (pdf, markdown, image))


def test_markdown_uses_real_parser_and_resolves_image():
    content = MineruParser().parse_text_file(DEFAULT_DOCUMENTS[1])

    assert {item["type"] for item in content} == {"text", "image"}
    image = next(item for item in content if item["type"] == "image")
    assert Path(image["img_path"]).is_absolute()
    assert Path(image["img_path"]).is_file()


def test_document_validation_rejects_missing_and_unsupported_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="document not found"):
        validate_document_paths([tmp_path / "missing.pdf"])

    unsupported = tmp_path / "sample.exe"
    unsupported.write_bytes(b"not an executable")
    with pytest.raises(ValueError, match="unsupported document type"):
        validate_document_paths([unsupported])


def test_content_type_counts_are_stable():
    assert count_content_types(
        [{"type": "text"}, {"type": "image"}, {"type": "text"}]
    ) == {"image": 1, "text": 2}


def test_markdown_skips_mineru_backend_options():
    assert parser_options(Path("note.md"), "pipeline", 900) == {}
    assert parser_options(Path("report.pdf"), "pipeline", 900) == {
        "backend": "pipeline",
        "timeout": 900,
    }


@pytest.mark.asyncio
async def test_real_document_orchestration_reports_stats_and_queries(tmp_path):
    rag = FakeDocumentRAG()

    result = await run_document_pipeline(
        rag, list(DEFAULT_DOCUMENTS), tmp_path / "output"
    )

    assert result["passed"] is True
    assert result["documents_passed"] == 3
    assert result["questions_passed"] == 6
    assert result["documents"][0]["content_types"] == {
        "equation": 1,
        "image": 1,
        "table": 1,
        "text": 1,
    }
    assert all(document["passed"] for document in result["documents"])
    assert all(answer["passed"] for answer in result["answers"])
    assert len(rag.process_calls) == len(rag.parse_calls) == 3
    assert rag.process_calls[0]["backend"] == "pipeline"
    assert "backend" not in rag.process_calls[1]
    assert len(rag.query_calls) == 6
    assert all(options["mode"] == "mix" for _, options in rag.query_calls)
    assert all(options["enable_rerank"] is False for _, options in rag.query_calls)
    assert rag.finalize_calls == 1


@pytest.mark.asyncio
async def test_document_failure_is_raised_and_storage_is_finalized(tmp_path):
    rag = FakeDocumentRAG(process_error=RuntimeError("parser failed"))

    with pytest.raises(RuntimeError, match="parser failed"):
        await run_document_pipeline(rag, [DEFAULT_DOCUMENTS[0]], tmp_path / "output")

    assert rag.finalize_calls == 1


@pytest.mark.asyncio
async def test_query_failure_is_raised_and_storage_is_finalized(tmp_path):
    rag = FakeDocumentRAG(query_error=RuntimeError("model timeout"))

    with pytest.raises(RuntimeError, match="model timeout"):
        await run_document_pipeline(rag, list(DEFAULT_DOCUMENTS), tmp_path / "output")

    assert rag.finalize_calls == 1
