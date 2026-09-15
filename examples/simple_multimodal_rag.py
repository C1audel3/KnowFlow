#!/usr/bin/env python
"""Minimal text/image/table/equation RAG loop for the resume project."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from functools import partial
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

from raganything import RAGAnything, RAGAnythingConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTENT_FILE = PROJECT_ROOT / "data" / "samples" / "phase1_content.json"
REQUIRED_MODALITIES = {"text", "image", "table", "equation"}
PLACEHOLDER_VALUES = {"", "replace-with-your-api-key", "your_api_key"}
DOCUMENT_ID = "phase1-multimodal-demo-v1"

FIXED_QUESTIONS = (
    {
        "id": "text",
        "question": "这个轻量级多模态知识库的项目代号是什么？",
        "expected_keywords": ("Aurora",),
    },
    {
        "id": "image",
        "question": "架构图展示的流程中包含哪一种图结构构建？",
        "expected_keywords": ("知识图谱",),
    },
    {
        "id": "table",
        "question": "试运行统计表中 Q2 的文档数量是多少？",
        "expected_keywords": ("150",),
    },
    {
        "id": "equation",
        "question": "F1 公式综合了哪两个指标？",
        "expected_keywords": ("精确率", "召回率"),
    },
)


def load_content_list(
    content_file: Path = DEFAULT_CONTENT_FILE,
) -> list[dict[str, Any]]:
    """Load the sample and resolve media paths relative to its JSON file."""
    content_file = content_file.resolve()
    content = json.loads(content_file.read_text(encoding="utf-8"))
    if not isinstance(content, list):
        raise ValueError("content_list JSON root must be a list")

    for item in content:
        if isinstance(item, dict) and item.get("type") == "image":
            image_path = Path(str(item.get("img_path", "")))
            if not image_path.is_absolute():
                item["img_path"] = str((content_file.parent / image_path).resolve())

    validate_content_list(content)
    return content


def validate_content_list(content: list[dict[str, Any]]) -> None:
    """Fail early when the deterministic phase-one fixture is malformed."""
    if not content:
        raise ValueError("content_list cannot be empty")

    observed = {item.get("type") for item in content if isinstance(item, dict)}
    missing = REQUIRED_MODALITIES - observed
    if missing:
        raise ValueError(f"content_list is missing modalities: {sorted(missing)}")

    required_fields = {
        "text": ("text",),
        "image": ("img_path", "image_caption"),
        "table": ("table_body", "table_caption"),
        "equation": ("latex", "text"),
    }
    for index, item in enumerate(content):
        if not isinstance(item, dict):
            raise ValueError(f"content_list[{index}] must be an object")
        content_type = item.get("type")
        if content_type not in REQUIRED_MODALITIES:
            raise ValueError(
                f"content_list[{index}] has unsupported type: {content_type}"
            )
        if not isinstance(item.get("page_idx"), int):
            raise ValueError(f"content_list[{index}].page_idx must be an integer")
        for field_name in required_fields[content_type]:
            if not item.get(field_name):
                raise ValueError(
                    f"content_list[{index}] is missing required field: {field_name}"
                )

    for item in content:
        if item["type"] == "image" and not Path(item["img_path"]).is_file():
            raise ValueError(f"sample image does not exist: {item['img_path']}")


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if value.lower() in PLACEHOLDER_VALUES:
        raise RuntimeError(
            f"{name} is not configured. Copy env.example to .env and set real credentials."
        )
    return value


def normalize_completion_kwargs(
    kwargs: dict[str, Any], base_url: str
) -> dict[str, Any]:
    """Translate LightRAG structured output hints for DeepSeek Chat Completions."""
    normalized = dict(kwargs)
    if "api.deepseek.com" not in base_url.lower():
        return normalized

    keyword_extraction = normalized.pop("keyword_extraction", False)
    response_format = normalized.get("response_format")
    if keyword_extraction or (
        response_format is not None and not isinstance(response_format, dict)
    ):
        # DeepSeek Chat Completions supports json_object, but not the Pydantic
        # model class accepted by OpenAI's beta parse helper.
        normalized["response_format"] = {"type": "json_object"}
    return normalized


def build_rag(working_dir: Path) -> RAGAnything:
    """Create RAGAnything using one OpenAI-compatible endpoint configuration."""
    llm_api_key = _required_env("LLM_BINDING_API_KEY")
    embedding_api_key = _required_env("EMBEDDING_BINDING_API_KEY")
    llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    vision_model = os.getenv("VISION_MODEL", llm_model)
    llm_base_url = os.getenv("LLM_BINDING_HOST", "https://api.openai.com/v1")
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    embedding_base_url = os.getenv("EMBEDDING_BINDING_HOST", llm_base_url)
    embedding_dim = int(os.getenv("EMBEDDING_DIM", "1536"))

    async def llm_model_func(
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> str:
        kwargs = normalize_completion_kwargs(kwargs, llm_base_url)
        return await openai_complete_if_cache(
            model=llm_model,
            prompt=prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            api_key=llm_api_key,
            base_url=llm_base_url,
            **kwargs,
        )

    async def vision_model_func(
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list[dict[str, Any]] | None = None,
        image_data: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> str:
        kwargs = normalize_completion_kwargs(kwargs, llm_base_url)
        if messages:
            return await openai_complete_if_cache(
                model=vision_model,
                prompt="",
                messages=messages,
                api_key=llm_api_key,
                base_url=llm_base_url,
                **kwargs,
            )
        if image_data:
            content = [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_data}"},
                },
            ]
            image_messages: list[dict[str, Any]] = []
            if system_prompt:
                image_messages.append({"role": "system", "content": system_prompt})
            image_messages.append({"role": "user", "content": content})
            return await openai_complete_if_cache(
                model=vision_model,
                prompt="",
                messages=image_messages,
                api_key=llm_api_key,
                base_url=llm_base_url,
                **kwargs,
            )
        return await llm_model_func(
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            **kwargs,
        )

    embedding_func = EmbeddingFunc(
        embedding_dim=embedding_dim,
        max_token_size=8192,
        func=partial(
            openai_embed.func,
            model=embedding_model,
            api_key=embedding_api_key,
            base_url=embedding_base_url,
            embedding_dim=embedding_dim,
        ),
    )
    config = RAGAnythingConfig(
        working_dir=str(working_dir),
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
        enable_audio_processing=False,
        enable_video_processing=False,
        display_content_stats=True,
    )
    return RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        vision_model_func=vision_model_func,
        embedding_func=embedding_func,
    )


async def run_pipeline(
    rag: Any,
    content_list: list[dict[str, Any]],
    *,
    force_reprocess: bool = False,
) -> dict[str, Any]:
    """Insert, query, verify fixed facts, and always release storage resources."""
    started_at = time.perf_counter()
    answers: list[dict[str, Any]] = []
    try:
        await rag.insert_content_list(
            content_list=content_list,
            file_path="phase1_multimodal_fixture.json",
            doc_id=DOCUMENT_ID,
            display_stats=True,
            force_multimodal_reprocess=force_reprocess,
        )
        for case in FIXED_QUESTIONS:
            query_started_at = time.perf_counter()
            answer = await rag.aquery(
                case["question"],
                mode="mix",
                vlm_enhanced=False,
                enable_rerank=False,
            )
            if not isinstance(answer, str) or not answer.strip():
                raise RuntimeError(f"query {case['id']} returned an empty answer")
            missing_keywords = [
                keyword
                for keyword in case["expected_keywords"]
                if keyword.lower() not in answer.lower()
            ]
            answers.append(
                {
                    "id": case["id"],
                    "question": case["question"],
                    "answer": answer.strip(),
                    "expected_keywords": list(case["expected_keywords"]),
                    "passed": not missing_keywords,
                    "latency_seconds": round(time.perf_counter() - query_started_at, 3),
                }
            )

        failed = [answer["id"] for answer in answers if not answer["passed"]]
        if failed:
            raise RuntimeError(f"fixed-answer checks failed: {', '.join(failed)}")
        return {
            "document_id": DOCUMENT_ID,
            "modalities": sorted(REQUIRED_MODALITIES),
            "answers": answers,
            "passed": True,
            "elapsed_seconds": round(time.perf_counter() - started_at, 3),
        }
    finally:
        await rag.finalize_storages()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--content-file", type=Path, default=DEFAULT_CONTENT_FILE)
    parser.add_argument(
        "--working-dir",
        type=Path,
        default=Path(os.getenv("WORKING_DIR", "./rag_storage_app")),
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate local fixtures without making model API calls.",
    )
    parser.add_argument(
        "--force-reprocess",
        action="store_true",
        help="Reprocess multimodal items even when the document ID already exists.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Run the same insertion/query flow repeatedly to check idempotency.",
    )
    return parser.parse_args()


async def async_main(args: argparse.Namespace) -> None:
    content_list = load_content_list(args.content_file)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "valid": True,
                    "content_file": str(args.content_file.resolve()),
                    "modalities": sorted(REQUIRED_MODALITIES),
                    "blocks": len(content_list),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if args.repeat < 1:
        raise ValueError("--repeat must be at least 1")

    for run_number in range(1, args.repeat + 1):
        rag = build_rag(args.working_dir)
        result = await run_pipeline(
            rag,
            content_list,
            force_reprocess=args.force_reprocess,
        )
        result["run"] = run_number
        print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    asyncio.run(async_main(parse_args()))


if __name__ == "__main__":
    main()
