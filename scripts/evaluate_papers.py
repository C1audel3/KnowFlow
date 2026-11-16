"""Functional paper-reading regression against the same API used by the UI."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from app.evaluation import load_dataset, run_evaluation, summarize_results
from app.schemas.papers import PaperReadingRequest
from ui.client import APIClient

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-id", required=True)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--dataset", type=Path, default=ROOT / "data/evaluation/paper_demo.jsonl"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/experiments/paper_reading_results.json",
    )
    parser.add_argument("--text-only", action="store_true")
    args = parser.parse_args()
    cases = load_dataset(args.dataset)
    client = APIClient(args.api_url, timeout=180)
    observations = []

    def query(question):
        result = client.paper_read(
            PaperReadingRequest(
                paper_id=args.paper_id,
                question=question,
                use_images=not args.text_only,
            )
        )
        observations.append({"question": question, **result.model_dump()})
        return SimpleNamespace(
            answer=result.answer,
            duration_ms=result.duration_ms,
            sources=[
                item.evidence_id
                for index, item in enumerate(result.evidence, 1)
                if f"E{index}" in result.citations
            ],
            mode=result.retrieval,
        )

    try:
        results = run_evaluation(cases, query)
    finally:
        client.close()
    output = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "paper_id": args.paper_id,
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "text_only": args.text_only,
        "scope": "Synthetic-demo keyword regression; not semantic accuracy or a real-paper benchmark.",
        "summary": summarize_results(cases, results, 1),
        "results": [result.model_dump() for result in results],
        "reading_observations": observations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2))
    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
