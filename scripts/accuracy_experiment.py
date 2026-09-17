"""Run a reproducible KnowFlow RAG versus no-RAG accuracy experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from app.accuracy import (
    AccuracyCase,
    load_accuracy_dataset,
    mcnemar_exact,
    score_answers,
    summarize_accuracy,
)
from app.core.config import PROJECT_ROOT

DEFAULT_DATASET = PROJECT_ROOT / "data" / "evaluation" / "accuracy_experiment.jsonl"
DEFAULT_RAG_RESULTS = PROJECT_ROOT / "report" / "artifacts" / "phase5_results.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "report" / "artifacts" / "accuracy_experiment.json"
PLACEHOLDER_KEYS = {"", "replace-with-your-api-key", "your_api_key"}
BASELINE_SYSTEM_PROMPT = (
    "你正在参加一个闭卷事实问答实验。你没有外部知识库或用户文档。"
    "请直接、简洁地回答问题；如果问题依赖未提供的私有或合成资料，"
    "必须明确说明不知道或资料未提供，不得猜测。"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--rag-results", type=Path, default=DEFAULT_RAG_RESULTS)
    parser.add_argument("--rag-round", type=int, default=1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=os.getenv("LLM_MODEL", "deepseek-flash"))
    parser.add_argument(
        "--base-url",
        default=os.getenv("LLM_BINDING_HOST", "https://api.deepseek.com"),
    )
    parser.add_argument("--timeout", type=float, default=180)
    return parser.parse_args()


def load_rag_answers(path: Path, round_number: int) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    answers: dict[str, str] = {}
    for result in payload.get("results", []):
        if result.get("round") != round_number:
            continue
        case_id = result.get("case_id")
        answer = result.get("answer")
        if not isinstance(case_id, str) or not isinstance(answer, str):
            raise ValueError("RAG result contains an invalid answer")
        if case_id in answers:
            raise ValueError(f"duplicate RAG result id: {case_id}")
        answers[case_id] = answer
    if not answers:
        raise ValueError(f"RAG result has no answers for round {round_number}")
    return answers


def run_no_rag_baseline(
    cases: list[AccuracyCase],
    client: OpenAI,
    *,
    model: str,
) -> tuple[dict[str, str], dict[str, int]]:
    answers: dict[str, str] = {}
    durations: dict[str, int] = {}
    for case in cases:
        started_at = time.perf_counter()
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": BASELINE_SYSTEM_PROMPT},
                {"role": "user", "content": case.question},
            ],
        )
        duration_ms = round((time.perf_counter() - started_at) * 1000)
        answer = response.choices[0].message.content
        if not isinstance(answer, str) or not answer.strip():
            raise RuntimeError(f"baseline returned an empty answer for {case.id}")
        answers[case.id] = answer.strip()
        durations[case.id] = duration_ms
        print(f"baseline {len(answers)}/{len(cases)}: {case.id} ({duration_ms} ms)")
    return answers, durations


def build_report(
    *,
    cases: list[AccuracyCase],
    dataset_path: Path,
    rag_results_path: Path,
    rag_round: int,
    rag_answers: dict[str, str],
    baseline_answers: dict[str, str],
    baseline_durations: dict[str, int],
    model: str,
) -> dict[str, Any]:
    rag_observations = score_answers(cases, rag_answers)
    baseline_observations = score_answers(cases, baseline_answers)
    for item in baseline_observations:
        item["duration_ms"] = baseline_durations[item["case_id"]]

    rag_summary = summarize_accuracy(rag_observations)
    baseline_summary = summarize_accuracy(baseline_observations)
    gain = round((rag_summary["accuracy"] - baseline_summary["accuracy"]) * 100, 2)
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": "paired_rag_vs_no_rag_closed_set_accuracy",
        "dataset": dataset_path.name,
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "case_count": len(cases),
        "model": model,
        "temperature": 0,
        "baseline_system_prompt": BASELINE_SYSTEM_PROMPT,
        "scoring": {
            "unit": "one binary correctness decision per question",
            "rule": "pre-registered all/any/forbidden keyword rubric",
            "confidence_interval": "Wilson score interval, 95%",
            "paired_test": "exact two-sided McNemar test",
        },
        "rag_source": {
            "artifact": rag_results_path.name,
            "round": rag_round,
        },
        "comparison": {
            "accuracy_gain_percentage_points": gain,
            "mcnemar": mcnemar_exact(rag_observations, baseline_observations),
        },
        "systems": {
            "knowflow_rag": {
                "summary": rag_summary,
                "observations": rag_observations,
            },
            "no_rag_baseline": {
                "summary": baseline_summary,
                "latency_mean_ms": round(
                    sum(baseline_durations.values()) / len(baseline_durations), 2
                ),
                "observations": baseline_observations,
            },
        },
    }


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    args = parse_args()
    if args.rag_round <= 0:
        raise SystemExit("--rag-round must be positive")
    key = os.getenv("LLM_BINDING_API_KEY", "").strip()
    if key.lower() in PLACEHOLDER_KEYS:
        raise SystemExit("LLM_BINDING_API_KEY is not configured")

    cases = load_accuracy_dataset(args.dataset)
    rag_answers = load_rag_answers(args.rag_results, args.rag_round)
    client = OpenAI(api_key=key, base_url=args.base_url, timeout=args.timeout)
    try:
        baseline_answers, durations = run_no_rag_baseline(
            cases, client, model=args.model
        )
    finally:
        client.close()

    report = build_report(
        cases=cases,
        dataset_path=args.dataset,
        rag_results_path=args.rag_results,
        rag_round=args.rag_round,
        rag_answers=rag_answers,
        baseline_answers=baseline_answers,
        baseline_durations=durations,
        model=args.model,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "knowflow_rag": report["systems"]["knowflow_rag"]["summary"],
                "no_rag_baseline": report["systems"]["no_rag_baseline"]["summary"],
                "comparison": report["comparison"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"result_file={args.output}")


if __name__ == "__main__":
    main()
