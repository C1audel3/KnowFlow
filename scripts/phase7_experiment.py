"""Run the reproducible phase-7 retrieval and grounded-generation experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.benchmark import (  # noqa: E402
    BM25Retriever,
    EvidenceBlock,
    aggregate_metrics,
    citation_metrics,
    reciprocal_rank_fusion,
    retrieval_metrics,
)

DEFAULT_DATASET = PROJECT_ROOT / "data/evaluation/phase7_benchmark.jsonl"
DEFAULT_CORPUS = PROJECT_ROOT / "data/benchmarks/phase7/documents"
DEFAULT_OUTPUT = PROJECT_ROOT / "report/artifacts/phase7_experiment.json"
EVIDENCE_HEADER = "[EVIDENCE:"
SYSTEM_PROMPT = """你是封闭知识库问答助手。只能依据给定证据回答。
若证据未包含答案，明确回答“档案未提供，无法确定”，不得猜测。
回答必须简洁，并在末尾引用所用证据，格式严格为 [EVIDENCE:证据ID]。"""
NO_RAG_PROMPT = """你在进行闭卷事实问答，没有外部文档。请简洁回答；
若问题依赖未提供的项目档案，必须回答“档案未提供，无法确定”，不得猜测。"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--embedding-model", default="bge-m3")
    parser.add_argument("--model", default=os.getenv("LLM_MODEL", "deepseek-chat"))
    parser.add_argument("--base-url", default=os.getenv("LLM_BINDING_HOST", "https://api.deepseek.com"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-generation", action="store_true")
    return parser.parse_args()


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [case["id"] for case in cases]
    if len(cases) != 120 or len(ids) != len(set(ids)):
        raise ValueError("phase-7 dataset must contain 120 unique cases")
    if Counter(case["difficulty"] for case in cases) != Counter({"easy": 40, "medium": 40, "hard": 40}):
        raise ValueError("difficulty split must be 40/40/40")
    return cases


def load_blocks(corpus: Path) -> list[EvidenceBlock]:
    blocks: list[EvidenceBlock] = []
    for path in sorted(corpus.glob("*.md")):
        current_id: str | None = None
        lines: list[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith(EVIDENCE_HEADER) and line.endswith("]"):
                if current_id:
                    blocks.append(EvidenceBlock(current_id, path.stem, current_id.rsplit("-", 1)[-1], "\n".join(lines).strip()))
                current_id = line[len(EVIDENCE_HEADER):-1].lower()
                lines = []
            elif current_id:
                lines.append(line)
        if current_id:
            blocks.append(EvidenceBlock(current_id, path.stem, current_id.rsplit("-", 1)[-1], "\n".join(lines).strip()))
    if len(blocks) != 48 or len({block.id for block in blocks}) != 48:
        raise ValueError("corpus must contain 48 unique evidence blocks")
    return blocks


def embed(texts: list[str], base_url: str, model: str) -> list[list[float]]:
    response = httpx.post(
        base_url.rstrip("/") + "/api/embed",
        json={"model": model, "input": texts, "truncate": True},
        timeout=300,
    )
    response.raise_for_status()
    vectors = response.json()["embeddings"]
    if len(vectors) != len(texts):
        raise RuntimeError("Ollama returned an unexpected embedding count")
    return vectors


def cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    denominator = math.sqrt(sum(a * a for a in left) * sum(b * b for b in right))
    return numerator / denominator if denominator else 0.0


def group_summary(rows: list[dict[str, Any]], field: str, metric: str) -> dict[str, float]:
    grouped: defaultdict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row[field]].append(float(row[metric]))
    return {key: round(statistics.fmean(values), 4) for key, values in sorted(grouped.items())}


def run_retrieval(cases: list[dict[str, Any]], blocks: list[EvidenceBlock], args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, list[str]]]:
    started = time.perf_counter()
    bm25 = BM25Retriever(blocks)
    block_vectors = embed([block.content for block in blocks], args.ollama_url, args.embedding_model)
    query_vectors = embed([case["question"] for case in cases], args.ollama_url, args.embedding_model)
    rankings: dict[str, dict[str, list[str]]] = {name: {} for name in ("bm25", "dense", "hybrid_rrf")}
    rows: list[dict[str, Any]] = []
    for case, query_vector in zip(cases, query_vectors):
        sparse = [identifier for identifier, _ in bm25.search(case["question"], top_k=10)]
        dense = [blocks[index].id for index in sorted(range(len(blocks)), key=lambda index: (-cosine(query_vector, block_vectors[index]), blocks[index].id))[:10]]
        hybrid = [identifier for identifier, _ in reciprocal_rank_fusion([sparse, dense], top_k=10)]
        for strategy, ranking in (("bm25", sparse), ("dense", dense), ("hybrid_rrf", hybrid)):
            rankings[strategy][case["id"]] = ranking
            metric_row: dict[str, Any] = {"case_id": case["id"], "type": case["type"], "difficulty": case["difficulty"], "strategy": strategy}
            for k in (1, 3, 5):
                metric_row.update(retrieval_metrics(ranking, case["evidence_ids"], k))
            rows.append(metric_row)
    summaries: dict[str, Any] = {}
    for strategy in rankings:
        selected = [row for row in rows if row["strategy"] == strategy]
        summaries[strategy] = {
            "overall": aggregate_metrics([{key: row[key] for key in row if key not in {"case_id", "type", "difficulty", "strategy"}} for row in selected]),
            "recall@5_by_type": group_summary(selected, "type", "recall@5"),
            "recall@5_by_difficulty": group_summary(selected, "difficulty", "recall@5"),
        }
    return {"duration_ms": round((time.perf_counter() - started) * 1000), "summaries": summaries, "observations": rows}, rankings["hybrid_rrf"]


def answer_correct(case: dict[str, Any], answer: str) -> bool:
    normalized = "".join(answer.lower().split())
    return all("".join(keyword.lower().split()) in normalized for keyword in case["expected_keywords"])


def wilson(correct: int, total: int, z: float = 1.96) -> list[float]:
    proportion = correct / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return [round(center - margin, 4), round(center + margin, 4)]


def paired_comparison(rag_rows: list[dict[str, Any]], baseline_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rag = {row["case_id"]: bool(row["correct"]) for row in rag_rows}
    baseline = {row["case_id"]: bool(row["correct"]) for row in baseline_rows}
    rag_only = sum(rag[key] and not baseline[key] for key in rag)
    baseline_only = sum(baseline[key] and not rag[key] for key in rag)
    discordant = rag_only + baseline_only
    tail = sum(math.comb(discordant, index) for index in range(min(rag_only, baseline_only) + 1)) / 2**discordant if discordant else 0.5
    rag_accuracy, baseline_accuracy = sum(rag.values()) / len(rag), sum(baseline.values()) / len(baseline)
    return {
        "absolute_accuracy_gain_pp": round((rag_accuracy - baseline_accuracy) * 100, 2),
        "relative_accuracy_improvement": round((rag_accuracy / baseline_accuracy - 1) * 100, 2) if baseline_accuracy else None,
        "rag_wilson_95_ci": wilson(sum(rag.values()), len(rag)),
        "baseline_wilson_95_ci": wilson(sum(baseline.values()), len(baseline)),
        "mcnemar": {"rag_correct_baseline_wrong": rag_only, "rag_wrong_baseline_correct": baseline_only, "exact_two_sided_p_value": min(1.0, 2 * tail)},
    }


def generation_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    correct = sum(row["correct"] for row in rows)
    answerable = [row for row in rows if row["type"] != "refusal"]
    refusal = [row for row in rows if row["type"] == "refusal"]
    latencies = [row["latency_ms"] for row in rows]
    return {
        "correct": correct,
        "total": total,
        "accuracy": round(correct / total, 4),
        "accuracy_by_type": group_summary(rows, "type", "correct"),
        "accuracy_by_difficulty": group_summary(rows, "difficulty", "correct"),
        "answerable_accuracy": round(sum(row["correct"] for row in answerable) / len(answerable), 4),
        "refusal_accuracy": round(sum(row["correct"] for row in refusal) / len(refusal), 4),
        "hallucination_rate": round(sum(not row["correct"] for row in refusal) / len(refusal), 4),
        "citation_precision": round(statistics.fmean(row["citation_precision"] for row in rows), 4),
        "citation_recall": round(statistics.fmean(row["citation_recall"] for row in rows), 4),
        "latency_ms": {"mean": round(statistics.fmean(latencies), 2), "p50": round(statistics.median(latencies), 2), "p95": sorted(latencies)[math.ceil(total * 0.95) - 1]},
        "total_tokens": sum(row.get("total_tokens", 0) for row in rows),
    }


def run_generation(cases: list[dict[str, Any]], blocks: list[EvidenceBlock], hybrid: dict[str, list[str]], args: argparse.Namespace) -> dict[str, Any]:
    key = os.getenv("LLM_BINDING_API_KEY", "").strip()
    if not key or "replace" in key.lower():
        raise RuntimeError("LLM_BINDING_API_KEY is not configured")
    block_by_id = {block.id: block for block in blocks}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output.with_suffix(".checkpoint.json")
    saved: dict[str, dict[str, Any]] = json.loads(checkpoint.read_text(encoding="utf-8")) if checkpoint.exists() else {}

    def one(case: dict[str, Any], system: str) -> dict[str, Any]:
        system_name = "hybrid_rag" if system == SYSTEM_PROMPT else "no_rag"
        key_name = f"{system_name}:{case['id']}"
        if key_name in saved:
            return saved[key_name]
        context = "\n\n".join(f"[EVIDENCE:{identifier}]\n{block_by_id[identifier].content}" for identifier in hybrid[case["id"]][:5]) if system_name == "hybrid_rag" else ""
        user = f"证据：\n{context}\n\n问题：{case['question']}" if context else case["question"]
        client = OpenAI(api_key=key, base_url=args.base_url, timeout=180)
        started = time.perf_counter()
        response = client.chat.completions.create(model=args.model, temperature=0, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        latency = round((time.perf_counter() - started) * 1000)
        answer = (response.choices[0].message.content or "").strip()
        citations = citation_metrics(answer, case["evidence_ids"]) if system_name == "hybrid_rag" else {"citation_precision": 0.0, "citation_recall": 0.0}
        return {"key": key_name, "system": system_name, "case_id": case["id"], "type": case["type"], "difficulty": case["difficulty"], "answer": answer, "correct": answer_correct(case, answer), "latency_ms": latency, "total_tokens": getattr(response.usage, "total_tokens", 0) or 0, **citations}

    pending = [(case, system) for case in cases for system in (SYSTEM_PROMPT, NO_RAG_PROMPT) if f"{'hybrid_rag' if system == SYSTEM_PROMPT else 'no_rag'}:{case['id']}" not in saved]
    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(one, case, system): (case, system) for case, system in pending}
        for future in as_completed(futures):
            row = future.result()
            saved[row["key"]] = row
            completed += 1
            checkpoint.write_text(json.dumps(saved, ensure_ascii=False, indent=2), encoding="utf-8")
            if completed % 10 == 0 or completed == len(pending):
                print(f"generation {completed}/{len(pending)}")
    case_by_id = {case["id"]: case for case in cases}
    rows = list(saved.values())
    if len(rows) != len(cases) * 2:
        raise RuntimeError("generation checkpoint is incomplete")
    for row in rows:
        row["correct"] = answer_correct(case_by_id[row["case_id"]], row["answer"])
        if row["system"] == "hybrid_rag":
            row.update(citation_metrics(row["answer"], case_by_id[row["case_id"]]["evidence_ids"]))
    system_rows = {name: [row for row in rows if row["system"] == name] for name in ("hybrid_rag", "no_rag")}
    return {
        "systems": {name: generation_summary(items) for name, items in system_rows.items()},
        "comparison": paired_comparison(system_rows["hybrid_rag"], system_rows["no_rag"]),
        "observations": sorted(rows, key=lambda row: row["key"]),
    }


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    args = parse_args()
    cases, blocks = load_cases(args.dataset), load_blocks(args.corpus)
    retrieval, hybrid = run_retrieval(cases, blocks, args)
    generation = None if args.skip_generation else run_generation(cases, blocks, hybrid, args)
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {"cases": len(cases), "documents": len(list(args.corpus.glob('*.md'))), "evidence_blocks": len(blocks), "sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(), "type_counts": dict(Counter(case["type"] for case in cases)), "difficulty_counts": dict(Counter(case["difficulty"] for case in cases))},
        "configuration": {"embedding_model": args.embedding_model, "generation_model": args.model, "temperature": 0, "top_k": 5, "rrf_rank_constant": 60},
        "retrieval": retrieval,
        "generation": generation,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    retrieval_csv = args.output.with_name("phase7_retrieval_observations.csv")
    with retrieval_csv.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(retrieval["observations"][0]))
        writer.writeheader()
        writer.writerows(retrieval["observations"])
    if generation:
        generation_csv = args.output.with_name("phase7_generation_observations.csv")
        with generation_csv.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(generation["observations"][0]))
            writer.writeheader()
            writer.writerows(generation["observations"])
        args.output.with_suffix(".checkpoint.json").unlink(missing_ok=True)
    print(json.dumps({"dataset": payload["dataset"], "retrieval": retrieval["summaries"], "generation": generation and generation["systems"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
