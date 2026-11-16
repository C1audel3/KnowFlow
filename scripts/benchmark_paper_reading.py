"""Run paired real-paper ablations against the application's reading HTTP API."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import threading
import time

from dotenv import load_dotenv

from app.paper_evaluation import automatic_checks, load_cases, summarize
from app.schemas.papers import PaperReadingRequest
from ui.client import APIClient

ROOT = Path(__file__).resolve().parents[1]
MODES = ("no_rag", "text_rag", "multimodal_rag")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def write_review(path: Path, observations: list[dict], cases: dict) -> None:
    """Preserve human review files rather than overwriting existing annotations."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        fields = [
            "case_id",
            "split",
            "mode",
            "question",
            "reference_answer",
            "answer",
            "semantic_correct",
            "citation_supported",
            "error_stage",
            "notes",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in observations:
            case = cases[row["case_id"]]
            writer.writerow(
                {
                    "case_id": case.id,
                    "split": case.split,
                    "mode": row["mode"],
                    "question": case.question,
                    "reference_answer": case.reference_answer,
                    "answer": (row.get("response") or {}).get("answer", ""),
                }
            )


def main() -> int:
    load_dotenv(ROOT / ".env", override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "data/evaluation/paper_reading_real.jsonl",
    )
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "data/benchmarks/papers/manifest.json"
    )
    parser.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=2)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    cases = [
        case
        for case in load_cases(args.dataset)
        if args.split == "all" or case.split == args.split
    ]
    by_id = {case.id: case for case in cases}
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    expected = {paper["key"] for paper in manifest["papers"]}
    if not {case.paper_key for case in cases} <= expected:
        raise ValueError("dataset refers to a paper absent from the manifest")
    llm = os.getenv("LLM_MODEL", "")
    vision = os.getenv("VISION_MODEL", llm)
    if "multimodal_rag" in args.modes and llm != vision:
        raise ValueError(
            "paired image ablation requires identical text and vision model names"
        )
    metadata = {
        "dataset_sha256": sha256(args.dataset),
        "manifest_sha256": sha256(args.manifest),
        "source_sha256": {
            paper["key"]: paper["sha256"] for paper in manifest["papers"]
        },
        "model": llm,
        "vision_model": vision,
        "embedding_model": os.getenv("EMBEDDING_MODEL", ""),
        "temperature": 0,
        "max_output_tokens": 1800,
        "top_k": 5,
        "reading_code_sha256": sha256(ROOT / "app/services/paper_reading.py"),
        "scoring_code_sha256": sha256(ROOT / "app/paper_evaluation.py"),
        "library_code_sha256": sha256(ROOT / "app/services/paper_library.py"),
        "source_adapter_code_sha256": sha256(ROOT / "app/services/paper_sources.py"),
        "runner_code_sha256": sha256(Path(__file__)),
    }
    fingerprint = hashlib.sha256(
        json.dumps(metadata, sort_keys=True).encode()
    ).hexdigest()
    output = (
        args.output or ROOT / f"data/experiments/paper_benchmark_raw/{args.split}.json"
    )
    summary_path = (
        args.summary
        or ROOT / f"data/experiments/paper_benchmark_{args.split}_summary.json"
    )
    saved = {"experiment_id": fingerprint, "metadata": metadata, "observations": []}
    if output.exists():
        saved = json.loads(output.read_text(encoding="utf-8"))
        if saved["experiment_id"] != fingerprint:
            raise ValueError(
                "checkpoint differs from current code/data/models; use a new output path"
            )
    existing = {
        (row["case_id"], row["mode"])
        for row in saved["observations"]
        if row.get("response")
    }
    lock = threading.Lock()
    client = APIClient(args.api_url, timeout=240)
    catalogue = {paper.paper_id for paper in client.papers()}
    if not {case.paper_key for case in cases} <= catalogue:
        client.close()
        raise ValueError(
            "import the pinned papers into this API's catalogue before running"
        )

    def run(case, mode):
        started = time.perf_counter()
        record = {
            "case_id": case.id,
            "paper_key": case.paper_key,
            "split": case.split,
            "mode": mode,
            "category": case.category,
            "visual_probe": case.visual_probe,
        }
        try:
            response = client.paper_read(
                PaperReadingRequest(
                    paper_id=case.paper_key,
                    question=case.question,
                    use_images=mode == "multimodal_rag",
                    retrieval_mode="none" if mode == "no_rag" else "retrieved",
                )
            )
            record.update(
                response=response.model_dump(), checks=automatic_checks(case, response)
            )
        except Exception as error:
            record.update(
                response=None, error_code=getattr(error, "code", type(error).__name__)
            )
        record["end_to_end_ms"] = round((time.perf_counter() - started) * 1000)
        with lock:
            saved["observations"] = [
                row
                for row in saved["observations"]
                if (row["case_id"], row["mode"]) != (case.id, mode)
            ]
            saved["observations"].append(record)
            save_atomic(output, saved)
        return f"{case.id}:{mode}:" + ("ok" if record["response"] else "failed")

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = []
            for index, case in enumerate(cases):
                modes = (
                    args.modes[index % len(args.modes) :]
                    + args.modes[: index % len(args.modes)]
                )
                for mode in modes:
                    if (case.id, mode) not in existing:
                        futures.append(pool.submit(run, case, mode))
            for future in as_completed(futures):
                print(future.result(), flush=True)
    finally:
        client.close()
    observations = [
        row
        for row in saved["observations"]
        if row["case_id"] in by_id and row["mode"] in args.modes
    ]
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": fingerprint,
        "metadata": metadata,
        "case_count": len(cases),
        "split": args.split,
        "summary": summarize(observations),
        "failed_requests": [
            {
                "case_id": row["case_id"],
                "mode": row["mode"],
                "error_code": row["error_code"],
            }
            for row in observations
            if not row.get("response")
        ],
    }
    save_atomic(summary_path, summary)
    write_review(output.with_suffix(".review.csv"), observations, by_id)
    print(json.dumps(summary["summary"], ensure_ascii=False, indent=2))
    return 0 if all(row.get("response") for row in observations) else 1


if __name__ == "__main__":
    raise SystemExit(main())
