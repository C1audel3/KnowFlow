"""Export blinded offline review packets and aggregate explicit reviewer judgments."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
from html import escape
import json
from pathlib import Path
import random

from app.paper_evaluation import load_cases
from scripts.benchmark_paper_reading import save_atomic, sha256

FIELDS = [
    "review_id",
    "record_sha256",
    "reviewer",
    "source_checked",
    "semantic_correct",
    "citation_supported",
    "error_stage",
    "notes",
]
SEMANTIC = {"correct", "partial", "incorrect", "uncertain"}
CITATION = {"supported", "partial", "unsupported", "not_applicable", "uncertain"}
STAGES = {
    "none",
    "parsing",
    "retrieval",
    "generation",
    "citation",
    "service",
    "annotation",
    "uncertain",
}


def records(raw: dict, cases: dict) -> list[dict]:
    """Bind ratings to the complete saved response, not just its answer string."""
    result, seen = [], set()
    for row in raw["observations"]:
        key = (row["case_id"], row["mode"])
        if key in seen or key[0] not in cases:
            raise ValueError("duplicate observation or unknown case")
        if key[1] not in {"no_rag", "text_rag", "multimodal_rag"}:
            raise ValueError("unknown experiment mode")
        seen.add(key)
        digest = hashlib.sha256(
            json.dumps(row, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()
        result.append(
            {"review_id": "R-" + digest[:16], "record_sha256": digest, "row": row}
        )
    if not result:
        raise ValueError("empty experiment")
    return result


def export_packet(
    raw_path: Path, dataset: Path, manifest_path: Path, output: Path
) -> None:
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    expected = raw.get("rescoring_metadata", {}).get(
        "corrected_dataset_sha256", raw["metadata"]["dataset_sha256"]
    )
    if expected != sha256(dataset):
        raise ValueError("dataset differs from the saved experiment annotations")
    if raw["metadata"]["manifest_sha256"] != sha256(manifest_path):
        raise ValueError("source manifest differs from the experiment")
    cases = {case.id: case for case in load_cases(dataset)}
    papers = {
        paper["key"]: paper for paper in json.loads(manifest_path.read_text())["papers"]
    }
    items = records(raw, cases)
    random.Random(20261112).shuffle(items)
    # A fresh directory prevents accidental loss of an existing reviewer's work.
    output.mkdir(parents=True, exist_ok=False)
    sections = []
    for item in items:
        row, review_id = item["row"], item["review_id"]
        case = cases[row["case_id"]]
        paper = papers[case.paper_key]
        url = paper["source_url"]
        if not url.startswith("https://"):
            raise ValueError("source links must use HTTPS")
        response = row.get("response") or {}
        evidence = []
        for index, block in enumerate(response.get("evidence", []), 1):
            page = block.get("page")
            end = block.get("page_end")
            location = f"PDF 第 {page} 页" if page else "页码未知"
            if end:
                location += f" 至 {end} 页"
            evidence.append(
                f"<details><summary>E{index} · {escape(location)} · {escape(block['modality'])}</summary>"
                f"<pre>{escape(block['text'])}</pre></details>"
            )
        locators = ", ".join(
            f"PDF p.{locator.page} {locator.kind}" for locator in case.gold_evidence
        )
        sections.append(
            f'<article id="{review_id}"><h2>{review_id}</h2>'
            f"<p>{escape(paper['title'])}</p><h3>问题</h3><pre>{escape(case.question)}</pre>"
            f"<h3>待审回答</h3><pre>{escape(response.get('answer', '请求失败，无回答'))}</pre>"
            f"<h3>返回的证据片段</h3>{''.join(evidence) or '<p>无返回证据。</p>'}"
            f"<details><summary>核对参考与原文（参考答案也可能有误）</summary>"
            f"<pre>{escape(case.reference_answer)}</pre><p>{escape(locators)}</p>"
            f'<a href="{escape(url, quote=True)}" target="_blank" rel="noreferrer">打开官方原文，按 PDF 页序核对图文</a>'
            "</details></article>"
        )
    html = (
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>论文阅读人工核验</title>'
        "<meta http-equiv=\"Content-Security-Policy\" content=\"default-src 'none'; style-src 'unsafe-inline'\">"
        "<style>body{max-width:960px;margin:32px auto;padding:20px;font:16px/1.7 sans-serif}"
        "article{border-top:2px solid #bbb;margin-top:40px}pre{white-space:pre-wrap;overflow-wrap:anywhere}"
        "summary{cursor:pointer}details{margin:12px 0}</style>"
        "<h1>论文阅读人工核验</h1><p>按编号在 ratings.csv 填写。隐藏配置名、时延与自动分数；"
        "证据数量等仍可能透露配置，因此仅为辅助盲审。此页不执行脚本，不加载外部资源。</p>"
        "<p>先独立判断回答，再展开参考；必须对照原文核对引用。片段不能替代原始图表。"
        "参考错误请标 annotation 并写备注；不确定填 uncertain。标准见开发文档。</p>"
        + "".join(sections)
        + "</html>"
    )
    (output / "review.html").write_text(html, encoding="utf-8")
    with (output / "ratings.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for item in items:
            writer.writerow({key: item[key] for key in FIELDS[:2]})
    save_atomic(
        output / "mapping.private.json",
        {
            "raw_sha256": sha256(raw_path),
            "dataset_sha256": sha256(dataset),
            "items": items,
        },
    )


def aggregate(mapping: dict, ratings: list[dict]) -> dict:
    expected = {item["review_id"]: item for item in mapping["items"]}
    seen, reviewed = set(), []
    for rating in ratings:
        key = rating.get("review_id")
        if key not in expected or key in seen:
            raise ValueError("unknown or duplicate review ID")
        seen.add(key)
        item = expected[key]
        if rating.get("record_sha256") != item["record_sha256"]:
            raise ValueError("rating does not match its saved response")
        if not any(rating.get(field, "").strip() for field in FIELDS[2:]):
            continue
        if not rating.get("reviewer", "").strip():
            raise ValueError("reviewer identity is required for filled ratings")
        semantic, citation = (
            rating.get("semantic_correct"),
            rating.get("citation_supported"),
        )
        if (
            semantic not in SEMANTIC
            or citation not in CITATION
            or rating.get("source_checked") not in {"yes", "no"}
            or rating.get("error_stage") not in STAGES
        ):
            raise ValueError("invalid or incomplete rating; see review protocol")
        row = item["row"]
        if (
            row.get("response")
            and rating["source_checked"] != "yes"
            and (semantic != "uncertain" or citation != "uncertain")
        ):
            raise ValueError(
                "definitive judgments require checking the original source"
            )
        if not row.get("response"):
            if (
                semantic != "uncertain"
                or citation != "not_applicable"
                or rating["error_stage"] != "service"
            ):
                raise ValueError(
                    "failed requests need uncertain/not_applicable/service ratings"
                )
        elif row["mode"] == "no_rag":
            if citation not in {"not_applicable", "uncertain"}:
                raise ValueError(
                    "no-retrieval answers have no evidence citations to score"
                )
        elif row["category"] != "refusal" and citation == "not_applicable":
            raise ValueError("answerable RAG cases require a citation judgment")
        reviewed.append((item, rating))
    if seen != set(expected):
        raise ValueError(
            "rating sheet is missing review IDs; leave unreviewed rows blank"
        )
    groups = defaultdict(list)
    for item in mapping["items"]:
        groups[item["row"]["mode"]].append(item)
    summary = {}
    for mode, items in sorted(groups.items()):
        filled = [
            (item, rating) for item, rating in reviewed if item["row"]["mode"] == mode
        ]
        # Uncertain, failed and unreviewed samples never enter semantic denominators.
        semantic = [
            rating
            for item, rating in filled
            if item["row"].get("response") and rating["semantic_correct"] != "uncertain"
        ]
        citations = [
            rating
            for _, rating in filled
            if rating["citation_supported"] in {"supported", "partial", "unsupported"}
        ]
        summary[mode] = {
            "requests": len(items),
            "reviewed": len(filled),
            "coverage": len(filled) / len(items),
            "semantic_judged": len(semantic),
            "semantic_correct_rate_reviewed": sum(
                rating["semantic_correct"] == "correct" for rating in semantic
            )
            / len(semantic)
            if semantic
            else None,
            "citation_judged": len(citations),
            "citation_supported_rate_reviewed": sum(
                rating["citation_supported"] == "supported" for rating in citations
            )
            / len(citations)
            if citations
            else None,
        }
    return {
        "raw_sha256": mapping.get("raw_sha256"),
        "dataset_sha256": mapping.get("dataset_sha256"),
        "status": "pending"
        if not reviewed
        else "complete"
        if len(reviewed) == len(expected)
        else "partial",
        "total": len(expected),
        "reviewed": len(reviewed),
        "by_mode": summary,
        "scope": "Reviewer-entered judgments only; rates use judged subsets. Coverage and denominators must accompany rates. Reviewer independence is not automatically verified.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export")
    export.add_argument("--raw", type=Path, required=True)
    export.add_argument(
        "--dataset", type=Path, default=Path("data/evaluation/paper_reading_real.jsonl")
    )
    export.add_argument(
        "--manifest", type=Path, default=Path("data/benchmarks/papers/manifest.json")
    )
    export.add_argument("--output", type=Path, required=True)
    score = sub.add_parser("summarize")
    score.add_argument("--packet", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "export":
        export_packet(args.raw, args.dataset, args.manifest, args.output)
    else:
        mapping = json.loads(
            (args.packet / "mapping.private.json").read_text(encoding="utf-8")
        )
        with (args.packet / "ratings.csv").open(
            encoding="utf-8-sig", newline=""
        ) as stream:
            ratings = list(csv.DictReader(stream))
        save_atomic(args.output, aggregate(mapping, ratings))


if __name__ == "__main__":
    main()
