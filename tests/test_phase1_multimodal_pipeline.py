from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from examples.simple_multimodal_rag import (
    DOCUMENT_ID,
    FIXED_QUESTIONS,
    build_rag,
    load_content_list,
    normalize_completion_kwargs,
    run_pipeline,
    validate_content_list,
)


class FakeRAG:
    def __init__(self, *, insert_error=None, query_error=None):
        self.insert_error = insert_error
        self.query_error = query_error
        self.insert_calls = []
        self.query_calls = []
        self.finalize_calls = 0
        self.answers = {
            "text": "项目代号是 Aurora。",
            "image": "架构图包含知识图谱构建。",
            "table": "Q2 的文档数量是 150。",
            "equation": "F1 综合精确率与召回率。",
        }

    async def insert_content_list(self, **kwargs):
        self.insert_calls.append(kwargs)
        if self.insert_error:
            raise self.insert_error

    async def aquery(self, question, **kwargs):
        self.query_calls.append((question, kwargs))
        if self.query_error:
            raise self.query_error
        case = next(case for case in FIXED_QUESTIONS if case["question"] == question)
        return self.answers[case["id"]]

    async def finalize_storages(self):
        self.finalize_calls += 1


def test_phase1_fixture_contains_four_modalities_and_existing_image():
    content = load_content_list()

    assert {item["type"] for item in content} == {
        "text",
        "image",
        "table",
        "equation",
    }
    image = next(item for item in content if item["type"] == "image")
    assert Path(image["img_path"]).is_absolute()
    assert Path(image["img_path"]).is_file()


def test_fixture_validation_rejects_a_missing_modality():
    content = [item for item in load_content_list() if item["type"] != "equation"]

    with pytest.raises(ValueError, match="missing modalities.*equation"):
        validate_content_list(content)


def test_fixture_validation_rejects_a_missing_required_field():
    content = deepcopy(load_content_list())
    next(item for item in content if item["type"] == "table").pop("table_body")

    with pytest.raises(ValueError, match="missing required field: table_body"):
        validate_content_list(content)


def test_rag_builder_rejects_missing_credentials(monkeypatch, tmp_path):
    monkeypatch.delenv("LLM_BINDING_API_KEY", raising=False)
    monkeypatch.delenv("EMBEDDING_BINDING_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="LLM_BINDING_API_KEY is not configured"):
        build_rag(tmp_path)

    monkeypatch.setenv("LLM_BINDING_API_KEY", "configured-for-test")
    with pytest.raises(
        RuntimeError, match="EMBEDDING_BINDING_API_KEY is not configured"
    ):
        build_rag(tmp_path)


def test_deepseek_keyword_extraction_uses_supported_json_mode():
    normalized = normalize_completion_kwargs(
        {"keyword_extraction": True}, "https://api.deepseek.com"
    )

    assert "keyword_extraction" not in normalized
    assert normalized["response_format"] == {"type": "json_object"}


def test_other_providers_keep_structured_output_options():
    options = {"keyword_extraction": True, "temperature": 0}

    assert normalize_completion_kwargs(options, "https://api.openai.com/v1") == options


@pytest.mark.asyncio
async def test_pipeline_inserts_and_queries_all_modalities_in_mix_mode():
    rag = FakeRAG()

    result = await run_pipeline(rag, load_content_list())

    assert result["passed"] is True
    assert result["document_id"] == DOCUMENT_ID
    assert len(result["answers"]) == 4
    assert all(answer["passed"] for answer in result["answers"])
    assert len(rag.insert_calls) == 1
    assert rag.insert_calls[0]["doc_id"] == DOCUMENT_ID
    assert {item["type"] for item in rag.insert_calls[0]["content_list"]} == {
        "text",
        "image",
        "table",
        "equation",
    }
    assert len(rag.query_calls) == 4
    assert all(options["mode"] == "mix" for _, options in rag.query_calls)
    assert all(options["vlm_enhanced"] is False for _, options in rag.query_calls)
    assert all(options["enable_rerank"] is False for _, options in rag.query_calls)
    assert rag.finalize_calls == 1


@pytest.mark.asyncio
async def test_pipeline_finalizes_storage_when_insertion_fails():
    rag = FakeRAG(insert_error=RuntimeError("insert failed"))

    with pytest.raises(RuntimeError, match="insert failed"):
        await run_pipeline(rag, load_content_list())

    assert rag.finalize_calls == 1


@pytest.mark.asyncio
async def test_pipeline_finalizes_storage_when_query_fails():
    rag = FakeRAG(query_error=RuntimeError("query failed"))

    with pytest.raises(RuntimeError, match="query failed"):
        await run_pipeline(rag, load_content_list())

    assert rag.finalize_calls == 1


@pytest.mark.asyncio
async def test_pipeline_can_repeat_with_stable_document_id():
    first_run = FakeRAG()
    second_run = FakeRAG()

    first_result = await run_pipeline(first_run, load_content_list())
    second_result = await run_pipeline(second_run, load_content_list())

    assert first_result["document_id"] == second_result["document_id"] == DOCUMENT_ID
    assert first_result["passed"] is second_result["passed"] is True
    assert first_run.finalize_calls == second_run.finalize_calls == 1
