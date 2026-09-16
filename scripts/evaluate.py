"""Run the fixed multimodal evaluation set against the KnowFlow HTTP API."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from app.core.config import PROJECT_ROOT
from app.evaluation import load_dataset, run_evaluation, summarize_results
from ui.client import APIClient

DEFAULT_DATASET = PROJECT_ROOT / "data" / "evaluation" / "phase5_questions.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "report" / "artifacts" / "phase5_results.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--api-url",
        default=os.getenv("API_BASE_URL", "http://127.0.0.1:8000"),
    )
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--fail-on-check", action="store_true")
    return parser.parse_args()


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    cases = load_dataset(args.dataset)
    dataset_sha256 = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    distribution = Counter(case.type for case in cases)
    client = APIClient(args.api_url, timeout=args.timeout)
    try:
        health = client.health()
        if health.status != "ok" or not health.rag_initialized:
            raise RuntimeError("API reports that RAG is not ready")
        results = run_evaluation(cases, client.query, rounds=args.rounds)
    finally:
        client.close()

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset.name,
        "dataset_sha256": dataset_sha256,
        "dataset_distribution": dict(sorted(distribution.items())),
        "run_config": {"rounds": args.rounds, "timeout_seconds": args.timeout},
        "models": {
            "llm": os.getenv("LLM_MODEL", "not-recorded"),
            "embedding": os.getenv("EMBEDDING_MODEL", "not-recorded"),
            "embedding_dim": int(os.getenv("EMBEDDING_DIM", "0")),
        },
        "summary": summarize_results(cases, results, args.rounds),
        "results": [item.model_dump(mode="json") for item in results],
    }


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    args = parse_args()
    if args.rounds <= 0:
        raise SystemExit("--rounds must be positive")
    report = build_report(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"result_file={args.output}")
    if args.fail_on_check and report["summary"]["rule_pass_rate"] < 1:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
