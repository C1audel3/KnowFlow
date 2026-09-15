"""Verify the phase-three HTTP query against an existing real knowledge base."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from dotenv import load_dotenv
from fastapi.testclient import TestClient

from app.core.config import PROJECT_ROOT, AppSettings
from app.main import create_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--working-dir",
        type=Path,
        default=Path("./rag_storage_phase2"),
        help="Existing LightRAG directory populated during phase two.",
    )
    parser.add_argument("--question", default="Atlas 项目计划在哪个季度发布？")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    working_dir = args.working_dir
    if not working_dir.is_absolute():
        working_dir = PROJECT_ROOT / working_dir
    settings = replace(AppSettings.from_env(), working_dir=working_dir.resolve())

    with TestClient(create_app(settings)) as client:
        health = client.get("/health")
        health.raise_for_status()
        response = client.post("/query", json={"question": args.question})
        response.raise_for_status()

    result = response.json()
    print(
        json.dumps(
            {
                "health": health.json(),
                "question": args.question,
                "answer": result["answer"],
                "sources": result["sources"],
                "duration_ms": result["duration_ms"],
                "mode": result["mode"],
                "passed": bool(result["answer"].strip()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
