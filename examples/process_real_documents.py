#!/usr/bin/env python
"""Parse, index, and query the deterministic phase-two document fixtures."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from raganything.callbacks import MetricsCallback

from examples.simple_multimodal_rag import PROJECT_ROOT, build_rag

DEFAULT_DOCUMENTS = (
    PROJECT_ROOT / "data" / "samples" / "phase2_multimodal.pdf",
    PROJECT_ROOT / "data" / "samples" / "phase2_document.md",
    PROJECT_ROOT / "data" / "samples" / "phase2_image.png",
)
SUPPORTED_EXTENSIONS = {".pdf", ".md", ".png", ".jpg", ".jpeg"}

FIXED_QUESTIONS = (
    {
        "id": "pdf_text",
        "question": "Atlas 项目计划在哪个季度发布？",
        "expected_keywords": ("2027", "Q3"),
    },
    {
        "id": "pdf_table",
        "question": "试验结果中 Sensor-B 的延迟是多少？",
        "expected_keywords": ("18", "ms"),
    },
    {
        "id": "pdf_image",
        "question": "架构图中的视觉网关叫什么？",
        "expected_keywords": ("ORION",),
    },
    {
        "id": "pdf_equation",
        "question": "报告中的 F1 公式综合哪两个指标？",
        "expected_keywords": ("精确率", "召回率"),
    },
    {
        "id": "markdown",
        "question": "Atlas 知识库试点的负责人是谁？",
        "expected_keywords": ("Lin Qiao",),
    },
    {
        "id": "image",
        "question": "独立图片中的 IMAGE CODE 是什么？",
        "expected_keywords": ("PIXEL-42",),
    },
)


def validate_document_paths(documents: list[Path]) -> list[Path]:
    """Validate real sample inputs before any model or storage work begins."""
    if not documents:
        raise ValueError("at least one document is required")

    validated = []
    for document in documents:
        path = document.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"document not found: {path}")
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"unsupported document type: {path.suffix.lower()}")
        if path.stat().st_size == 0:
            raise ValueError(f"document is empty: {path}")
        validated.append(path)
    return validated


def count_content_types(content_list: list[dict[str, Any]]) -> dict[str, int]:
    """Return stable content type counts for reports and API responses."""
    counts = Counter(
        str(item.get("type", "unknown"))
        for item in content_list
        if isinstance(item, dict)
    )
    return dict(sorted(counts.items()))


def parser_options(path: Path, backend: str, timeout: int) -> dict[str, Any]:
    """Use MinerU only for binary documents; Markdown parsing is direct."""
    if path.suffix.lower() == ".md":
        return {}
    return {"backend": backend, "timeout": timeout}


def _metric_delta(after: dict[str, Any], before: dict[str, Any], key: str) -> float:
    return round(float(after[key]) - float(before[key]), 3)


async def run_document_pipeline(
    rag: Any,
    documents: list[Path],
    output_dir: Path,
    *,
    backend: str = "pipeline",
    timeout: int = 900,
) -> dict[str, Any]:
    """Run real-document ingestion and fixed queries with guaranteed cleanup."""
    documents = validate_document_paths(documents)
    metrics = MetricsCallback()
    rag.callback_manager.register(metrics)
    started_at = time.perf_counter()
    document_results: list[dict[str, Any]] = []
    answers: list[dict[str, Any]] = []

    try:
        for path in documents:
            options = parser_options(path, backend, timeout)
            before = dict(metrics.metrics)
            document_started_at = time.perf_counter()
            await rag.process_document_complete(
                file_path=str(path),
                output_dir=str(output_dir),
                parse_method="auto",
                display_stats=True,
                **options,
            )
            after = dict(metrics.metrics)

            # This second call is served by RAGAnything's parse cache and gives
            # the application a public, structured result for stats/reporting.
            content_list, doc_id = await rag.parse_document(
                file_path=str(path),
                output_dir=str(output_dir),
                parse_method="auto",
                display_stats=False,
                **options,
            )
            document_results.append(
                {
                    "file": path.name,
                    "size_bytes": path.stat().st_size,
                    "document_id": doc_id,
                    "content_blocks": len(content_list),
                    "content_types": count_content_types(content_list),
                    "parse_seconds": _metric_delta(after, before, "total_parse_time"),
                    "text_insert_seconds": _metric_delta(
                        after, before, "total_insert_time"
                    ),
                    "multimodal_seconds": _metric_delta(
                        after, before, "total_multimodal_time"
                    ),
                    "total_seconds": round(
                        time.perf_counter() - document_started_at, 3
                    ),
                    "passed": True,
                }
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
            "documents": document_results,
            "answers": answers,
            "documents_passed": len(document_results),
            "questions_passed": len(answers),
            "passed": True,
            "elapsed_seconds": round(time.perf_counter() - started_at, 3),
        }
    finally:
        await rag.finalize_storages()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("documents", nargs="*", type=Path, default=DEFAULT_DOCUMENTS)
    parser.add_argument(
        "--working-dir", type=Path, default=Path("./rag_storage_phase2")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("./output_phase2"))
    parser.add_argument("--backend", default="pipeline")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate fixture files without parsing or calling model APIs.",
    )
    return parser.parse_args()


async def async_main(args: argparse.Namespace) -> None:
    documents = validate_document_paths(list(args.documents))
    if args.validate_only:
        print(
            json.dumps(
                {
                    "valid": True,
                    "documents": [
                        {
                            "file": path.name,
                            "extension": path.suffix.lower(),
                            "size_bytes": path.stat().st_size,
                        }
                        for path in documents
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    rag = build_rag(args.working_dir)
    result = await run_document_pipeline(
        rag,
        documents,
        args.output_dir,
        backend=args.backend,
        timeout=args.timeout,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    asyncio.run(async_main(parse_args()))


if __name__ == "__main__":
    main()
