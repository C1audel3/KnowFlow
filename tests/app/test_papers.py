from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from app.core.errors import AppError, ModelUnavailableError
from app.main import create_app
from app.schemas.papers import PaperReadingRequest
from app.services.paper_library import PaperLibrary
from app.services.paper_reading import index_paper, read_paper, reference_keys, retrieve
from fakes import FakeRAG


class PaperRAG(FakeRAG):
    def __init__(self):
        super().__init__()
        self.generations = []
        self.response = "方法包含检索与生成两个阶段。[E1]"
        self.fail_embedding = False

    async def embedding_func(self, texts):
        if self.fail_embedding:
            raise RuntimeError("private embedding error")
        # Deterministic bilingual vectors test semantic matching without any network.
        return [
            [1.0, 0.1]
            if any(word in text.lower() for word in ("method", "方法"))
            else [0.1, 1.0]
            for text in texts
        ]

    async def llm_model_func(self, prompt, **kwargs):
        self.generations.append((prompt, kwargs))
        if self.query_error:
            raise self.query_error
        return self.response

    async def vision_model_func(self, prompt, **kwargs):
        return await self.llm_model_func(prompt, **kwargs)

    async def parse_document(self, **kwargs):
        if getattr(self, "parse_error", None):
            raise self.parse_error
        path = Path(kwargs["file_path"])
        output = Path(kwargs["output_dir"])
        image = output / "figure.png"
        Image.new("RGB", (30, 20), "white").save(image)
        return [
            {
                "type": "text",
                "text": "Example Research",
                "text_level": 1,
                "page_idx": 0,
            },
            {"type": "text", "text": "Methods", "text_level": 1, "page_idx": 0},
            {
                "type": "text",
                "text": "Our method uses retrieval. Figure 2 shows the pipeline.",
                "page_idx": 0,
            },
            {
                "type": "image",
                "image_caption": ["Figure 2: Method pipeline."],
                "img_path": str(image),
                "page_idx": 1,
            },
            {
                "type": "table",
                "table_caption": ["Table 1: Ablation results"],
                "table_body": "Baseline 70, Proposed 80",
                "page_idx": 2,
            },
            {
                "type": "equation",
                "text": "Equation 1 defines the objective",
                "latex": "L = a + b",
                "page_idx": 2,
            },
            {"type": "text", "text": "Limitations", "text_level": 1, "page_idx": 2},
            {
                "type": "text",
                "text": "Only synthetic data was evaluated.",
                "page_idx": 2,
            },
        ], f"doc-{path.stem}"


@pytest.fixture
def library(app_settings):
    app_settings.upload_dir.mkdir(parents=True)
    app_settings.output_dir.mkdir(parents=True)
    return PaperLibrary(
        app_settings.working_dir, app_settings.upload_dir, app_settings.output_dir
    )


async def add_paper(
    library, rag, identifier="paper-a", text="Our method uses retrieval."
):
    original = library.upload_dir / f"{identifier}.pdf"
    original.write_bytes(b"%PDF-1.4\n")
    await index_paper(
        library,
        rag,
        identifier,
        f"{identifier}.pdf",
        original,
        [
            {"type": "text", "text": "Methods", "text_level": 1, "page_idx": 0},
            {"type": "text", "text": text, "page_idx": 1},
        ],
    )


@pytest.mark.asyncio
async def test_catalogue_persists_and_reimport_replaces_atomically(library):
    rag = PaperRAG()
    await add_paper(library, rag)
    await add_paper(library, rag, text="A revised method.")
    reopened = PaperLibrary(library.path.parent, library.upload_dir, library.output_dir)
    assert len(reopened.list()) == 1
    assert reopened.get("paper-a").vector_indexed
    blocks = reopened.blocks("paper-a")
    assert len(blocks) == 2
    assert blocks[1]["page"] == 2
    assert blocks[1]["text"] == "A revised method."
    assert "asset" not in reopened.public(blocks[1]).model_dump()


@pytest.mark.asyncio
async def test_hybrid_retrieval_is_strictly_scoped_to_selected_paper(library):
    rag = PaperRAG()
    await add_paper(library, rag)
    await add_paper(library, rag, "paper-b", "SECRET_OTHER_PAPER method")
    result = await read_paper(
        library, rag, PaperReadingRequest(paper_id="paper-a", question="解释方法")
    )
    assert result.retrieval == "hybrid_rrf"
    assert {item.paper_id for item in result.evidence} == {"paper-a"}
    assert "SECRET_OTHER_PAPER" not in rag.generations[-1][0]
    assert result.citations == ["E1"]


@pytest.mark.asyncio
async def test_figure_lookup_passes_original_to_vision_and_keeps_prose(library):
    rag = PaperRAG()
    original = library.upload_dir / "demo.pdf"
    original.write_bytes(b"%PDF-1.4\n")
    content, _ = await rag.parse_document(
        file_path=str(original), output_dir=str(library.output_dir)
    )
    await index_paper(library, rag, "demo", "demo.pdf", original, content)
    result = await read_paper(
        library, rag, PaperReadingRequest(paper_id="demo", question="解释图2中的流程")
    )
    assert result.retrieval == "figure_lookup"
    assert result.vision_used
    assert result.evidence[0].modality == "image"
    assert any("shows the pipeline" in item.text for item in result.evidence)
    messages = rag.generations[-1][1]["messages"]
    image_parts = [
        item for item in messages[1]["content"] if item["type"] == "image_url"
    ]
    assert len(image_parts) == 1
    assert image_parts[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    no_images = await read_paper(
        library,
        rag,
        PaperReadingRequest(paper_id="demo", question="Figure 2", use_images=False),
    )
    assert not no_images.vision_used
    assert "messages" not in rag.generations[-1][1]


@pytest.mark.asyncio
async def test_missing_figure_does_not_answer_with_different_figure(library):
    rag = PaperRAG()
    await add_paper(library, rag)
    result = await read_paper(
        library, rag, PaperReadingRequest(paper_id="paper-a", question="Figure 20")
    )
    assert result.status == "no_evidence"
    assert not rag.generations


@pytest.mark.asyncio
async def test_embedding_failure_falls_back_without_leaking_error(library):
    rag = PaperRAG()
    await add_paper(library, rag)
    rag.fail_embedding = True
    result = await read_paper(
        library, rag, PaperReadingRequest(paper_id="paper-a", question="method")
    )
    assert result.retrieval == "bm25"
    assert result.warnings
    assert "private" not in result.model_dump_json()


@pytest.mark.asyncio
async def test_failed_reindex_keeps_previous_complete_catalogue(library):
    rag = PaperRAG()
    await add_paper(library, rag)
    rag.fail_embedding = True
    with pytest.raises(RuntimeError):
        await add_paper(library, rag, text="replacement")
    assert library.blocks("paper-a")[1]["text"] == "Our method uses retrieval."


@pytest.mark.asyncio
async def test_invalid_citations_are_not_returned_as_real_sources(library):
    rag = PaperRAG()
    await add_paper(library, rag)
    rag.response = "A claim [E999] and [E0]."
    result = await read_paper(
        library, rag, PaperReadingRequest(paper_id="paper-a", question="method")
    )
    assert result.status == "uncited"
    assert result.citations == []
    assert "[E999]" not in result.answer
    assert "引用无效" in result.answer


@pytest.mark.asyncio
async def test_model_failure_uses_safe_application_error(library):
    rag = PaperRAG()
    await add_paper(library, rag)
    rag.query_error = RuntimeError("secret key")
    with pytest.raises(ModelUnavailableError) as error:
        await read_paper(
            library, rag, PaperReadingRequest(paper_id="paper-a", question="method")
        )
    assert "secret" not in str(error.value)


@pytest.mark.asyncio
async def test_overview_covers_different_sections_and_discloses_gaps(library):
    rag = PaperRAG()
    original = library.upload_dir / "test.pdf"
    original.write_bytes(b"%PDF-1.4\n")
    content = []
    for section in (
        "Abstract",
        "Methods",
        "Experiments",
        "Results",
        "Limitations",
        "Conclusion",
    ):
        content.extend(
            [
                {"type": "text", "text": section, "text_level": 1},
                {"type": "text", "text": section + " detailed paragraph."},
            ]
        )
    await index_paper(library, rag, "test", "test.pdf", original, content)
    rows, method, warnings = await retrieve(
        library, rag, PaperReadingRequest(paper_id="test", task="overview")
    )
    assert method == "section_coverage"
    assert {row["section"] for row in rows} == {
        "Abstract",
        "Methods",
        "Experiments",
        "Results",
        "Limitations",
        "Conclusion",
    }
    assert len(rows) <= 12
    assert any("片段" in warning for warning in warnings)
    assert all(row["page"] is None for row in rows)


def test_reference_numbers_support_chinese_without_matching_prefixes():
    assert reference_keys("解释图2中的流程，结合 Table 20 和 Eq. 3.1") == {
        ("figure", "2"),
        ("table", "20"),
        ("equation", "3.1"),
    }
    assert ("figure", "2") not in reference_keys("Figure 20")


def test_assets_cannot_escape_roots_or_be_replaced_by_external_symlinks(
    library, tmp_path
):
    outside = tmp_path / "private.png"
    Image.new("RGB", (10, 10)).save(outside)
    inside = library.output_dir / "plot.png"
    Image.new("RGB", (10, 10)).save(inside)
    rows = library.prepare(
        "paper-a",
        [
            {"type": "image", "image_caption": ["Figure 1"], "img_path": str(inside)},
            {"type": "image", "image_caption": ["Figure 2"], "img_path": str(outside)},
        ],
    )
    library.save("paper-a", "demo.pdf", library.upload_dir / "demo.pdf", rows, "")
    assert rows[1]["asset"] is None
    with pytest.raises(AppError):
        library.asset("paper-b", rows[0]["id"])
    inside.unlink()
    inside.symlink_to(outside)
    with pytest.raises(AppError):
        library.asset("paper-a", rows[0]["id"])


def test_paper_upload_read_browse_and_restart_api(app_settings):
    rag = PaperRAG()
    with TestClient(create_app(app_settings, rag_factory=lambda _: rag)) as client:
        response = client.post(
            "/papers", files={"file": ("demo.pdf", b"%PDF-1.4\n", "application/pdf")}
        )
        assert response.status_code == 202
        task_id = response.json()["task_id"]
        # Waiting via the service avoids unbounded HTTP polling in tests.
        client.portal.call(client.app.state.service.wait_for_document, task_id)
        paper_id = client.get(f"/documents/{task_id}").json()["document_id"]
        paper = client.get("/papers").json()[0]
        assert paper["paper_id"] == paper_id
        assert paper["vector_indexed"]
        assert rag.process_calls == []
        blocks = client.get(f"/papers/{paper_id}/content").json()
        image = next(item for item in blocks if item["modality"] == "image")
        asset = client.get(image["image_url"])
        assert asset.status_code == 200
        assert asset.headers["content-type"] == "image/jpeg"
        answer = client.post(
            "/papers/read", json={"paper_id": paper_id, "question": "Figure 2"}
        )
        assert answer.status_code == 200
        assert answer.json()["vision_used"]
        assert answer.json()["evidence"][0]["page"] == 2
        history_id = answer.json()["history_id"]
        assert history_id
        history_url = f"/papers/{paper_id}/readings"
        assert client.get(history_url).json()[0]["history_id"] == history_id
        assert client.get(history_url + "?offset=-1").status_code == 400
        assert client.get(history_url + "?limit=101").status_code == 400
        assert client.get(history_url + "/missing").status_code == 404
        assert str(app_settings.output_dir) not in answer.text
        assert client.get(f"/papers/{paper_id}/original").content == b"%PDF-1.4\n"
        assert (
            client.get(f"/papers/other/assets/{image['evidence_id']}").status_code
            == 404
        )
        assert client.get(f"/papers/{paper_id}/content?offset=-1").status_code == 400
        assert (
            client.post(
                "/papers/read", json={"paper_id": paper_id, "question": " "}
            ).status_code
            == 400
        )
        assert (
            client.post(
                "/papers/read", json={"paper_id": "missing", "question": "method"}
            ).status_code
            == 404
        )
        assert (
            client.post(
                "/papers", files={"file": ("image.png", b"payload")}
            ).status_code
            == 415
        )
    with TestClient(
        create_app(app_settings, rag_factory=lambda _: PaperRAG())
    ) as restarted:
        assert restarted.get("/papers").json()[0]["paper_id"] == paper_id
        assert restarted.get(f"/papers/{paper_id}/content").json()
        record = restarted.get(f"/papers/{paper_id}/readings/{history_id}").json()
        assert record["request"]["question"] == "Figure 2"
        assert record["response"] == answer.json()


@pytest.mark.asyncio
async def test_history_is_scoped_paginated_and_keeps_snapshot_after_reindex(library):
    rag = PaperRAG()
    await add_paper(library, rag)
    await add_paper(library, rag, "paper-b")
    request = PaperReadingRequest(paper_id="paper-a", question="method")
    response = await read_paper(library, rag, request)
    ids = [library.save_reading(request, response) for _ in range(3)]
    assert [item.history_id for item in library.readings("paper-a", 1, 1)] == [ids[1]]
    assert library.readings("paper-b") == []
    with pytest.raises(AppError) as error:
        library.reading("paper-b", ids[0])
    assert error.value.code == "READING_NOT_FOUND"
    snapshot = library.reading("paper-a", ids[0])
    await add_paper(library, rag, text="Changed method")
    assert library.reading("paper-a", ids[0]) == snapshot
    assert "Changed" not in snapshot.response.evidence[-1].text
    response.evidence[0].paper_id = "paper-b"
    with pytest.raises(ValueError, match="another paper"):
        library.save_reading(request, response)


@pytest.mark.asyncio
async def test_failed_generation_does_not_create_history_and_save_failure_keeps_answer(
    app_settings, monkeypatch
):
    import sqlite3
    from app.services.knowledge_base import KnowledgeBaseService

    rag = PaperRAG()
    service = KnowledgeBaseService(app_settings, rag_factory=lambda _: rag)
    await service.initialize()
    try:
        await add_paper(service.papers, rag)
        request = PaperReadingRequest(paper_id="paper-a", question="method")
        rag.query_error = RuntimeError("private model error")
        with pytest.raises(ModelUnavailableError):
            await service.read_paper(request)
        assert service.papers.readings("paper-a") == []
        rag.query_error = None

        def fail_save(*args):
            raise sqlite3.OperationalError("private filesystem path")

        monkeypatch.setattr(service.papers, "save_reading", fail_save)
        answer = await service.read_paper(request)
        assert answer.answer and answer.history_id is None
        assert any("未能保存" in warning for warning in answer.warnings)
        assert "private" not in answer.model_dump_json()
    finally:
        await service.close()


def test_failed_paper_processing_is_not_published(app_settings):
    rag = PaperRAG()
    rag.parse_error = RuntimeError("private failure")
    with TestClient(create_app(app_settings, rag_factory=lambda _: rag)) as client:
        created = client.post(
            "/papers", files={"file": ("demo.pdf", b"%PDF-1.4\n")}
        ).json()
        client.portal.call(
            client.app.state.service.wait_for_document, created["task_id"]
        )
        assert client.get("/papers").json() == []
        task = client.get(f"/documents/{created['task_id']}").json()
        assert task["status"] == "failed"
        assert "private" not in task["error"]


@pytest.mark.asyncio
async def test_chart_panels_keep_shared_caption_and_all_originals(library):
    rag = PaperRAG()
    original = library.upload_dir / "chart.pdf"
    original.write_bytes(b"%PDF-1.4\n")
    content = []
    for index in range(3):
        image = library.output_dir / f"panel-{index}.png"
        Image.new("RGB", (10, 10), "white").save(image)
        content.append(
            {
                "type": "chart",
                "img_path": str(image),
                "page_idx": 3,
                "chart_caption": ["Figure 3: Evaluation"] if index == 2 else [],
            }
        )
    content.extend(
        [
            {"type": "page_number", "text": "877", "page_idx": 3},
            {"type": "list", "sub_type": "ref_text", "list_items": ["Reference noise"]},
        ]
    )
    await index_paper(library, rag, "chart", "chart.pdf", original, content)
    assert len(library.blocks("chart")) == 3
    result = await read_paper(
        library, rag, PaperReadingRequest(paper_id="chart", question="Figure 3")
    )
    assert result.retrieval == "figure_lookup"
    assert result.vision_used
    assert len(result.evidence) == 3
    assert [item.evidence_id for item in result.evidence] == [
        "chart-0-0",
        "chart-1-0",
        "chart-2-0",
    ]
    messages = rag.generations[-1][1]["messages"]
    assert sum(part["type"] == "image_url" for part in messages[1]["content"]) == 3


@pytest.mark.asyncio
async def test_no_retrieval_baseline_uses_model_knowledge_without_sources(library):
    rag = PaperRAG()
    await add_paper(library, rag)
    rag.fail_embedding = True
    result = await read_paper(
        library,
        rag,
        PaperReadingRequest(
            paper_id="paper-a",
            question="What is this paper about?",
            retrieval_mode="none",
        ),
    )
    assert result.retrieval == "none"
    assert result.evidence == []
    assert result.citations == []
    assert not result.vision_used
    prompt, options = rag.generations[-1]
    assert "Our method uses retrieval" not in prompt
    assert "已有知识" in options["system_prompt"]
    assert options["temperature"] == 0
    assert "[E1]" not in result.answer
