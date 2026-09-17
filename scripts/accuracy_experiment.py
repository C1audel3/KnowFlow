"""Run a reproducible, repeated KnowFlow RAG versus no-RAG experiment."""

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
    answerability_summary,
    effect_size,
    efficiency_summary,
    load_accuracy_dataset,
    mcnemar_exact,
    repeatability_summary,
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

RunRecords = dict[str, dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--rag-results", type=Path, default=DEFAULT_RAG_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=os.getenv("LLM_MODEL", "deepseek-flash"))
    parser.add_argument(
        "--base-url",
        default=os.getenv("LLM_BINDING_HOST", "https://api.deepseek.com"),
    )
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=180)
    return parser.parse_args()


def load_rag_runs(path: Path, rounds: int) -> list[RunRecords]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    runs: list[RunRecords] = [dict() for _ in range(rounds)]
    for result in payload.get("results", []):
        round_number = result.get("round")
        if not isinstance(round_number, int) or not 1 <= round_number <= rounds:
            continue
        case_id = result.get("case_id")
        answer = result.get("answer")
        duration = result.get("end_to_end_duration_ms")
        if (
            not isinstance(case_id, str)
            or not isinstance(answer, str)
            or not isinstance(duration, int)
        ):
            raise ValueError("RAG result contains an invalid observation")
        current = runs[round_number - 1]
        if case_id in current:
            raise ValueError(f"duplicate RAG result id: {case_id}@{round_number}")
        current[case_id] = {"answer": answer, "duration_ms": duration}
    if any(not run for run in runs):
        raise ValueError(f"RAG result does not contain {rounds} complete rounds")
    return runs


def load_rag_answers(path: Path, round_number: int) -> dict[str, str]:
    """Compatibility helper for selecting one saved RAG round."""
    if round_number <= 0:
        raise ValueError("round number must be positive")
    run = load_rag_runs(path, round_number)[round_number - 1]
    return {case_id: record["answer"] for case_id, record in run.items()}


def run_no_rag_baseline(
    cases: list[AccuracyCase],
    client: OpenAI,
    *,
    model: str,
    round_number: int = 1,
) -> RunRecords:
    records: RunRecords = {}
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
        usage = response.usage
        records[case.id] = {
            "answer": answer.strip(),
            "duration_ms": duration_ms,
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        }
        print(
            f"baseline round {round_number}: {len(records)}/{len(cases)} "
            f"{case.id} ({duration_ms} ms)"
        )
    return records


def score_runs(
    cases: list[AccuracyCase], runs: list[RunRecords]
) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for round_number, run in enumerate(runs, start=1):
        answers = {case_id: record["answer"] for case_id, record in run.items()}
        scored = score_answers(cases, answers)
        for item in scored:
            record = run[item["case_id"]]
            item["round"] = round_number
            item["duration_ms"] = record.get("duration_ms")
            for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
                if record.get(field) is not None:
                    item[field] = record[field]
        observations.extend(scored)
    return observations


def _per_round_summary(
    observations: list[dict[str, Any]], rounds: int
) -> dict[str, Any]:
    return {
        str(round_number): summarize_accuracy(
            [item for item in observations if item["round"] == round_number]
        )
        for round_number in range(1, rounds + 1)
    }


def _token_summary(observations: list[dict[str, Any]]) -> dict[str, Any] | None:
    with_usage = [item for item in observations if "total_tokens" in item]
    if not with_usage:
        return None
    total_prompt = sum(item["prompt_tokens"] for item in with_usage)
    total_completion = sum(item["completion_tokens"] for item in with_usage)
    total = sum(item["total_tokens"] for item in with_usage)
    return {
        "requests_with_usage": len(with_usage),
        "prompt_tokens": total_prompt,
        "completion_tokens": total_completion,
        "total_tokens": total,
        "mean_tokens_per_request": round(total / len(with_usage), 2),
    }


def _system_report(
    observations: list[dict[str, Any]], rounds: int, *, include_tokens: bool
) -> dict[str, Any]:
    result = {
        "summary": summarize_accuracy(observations),
        "answerability": answerability_summary(observations),
        "repeatability": repeatability_summary(observations, rounds),
        "efficiency": efficiency_summary(observations),
        "per_round": _per_round_summary(observations, rounds),
        "observations": observations,
    }
    if include_tokens:
        result["token_usage"] = _token_summary(observations)
    return result


def build_report(
    *,
    cases: list[AccuracyCase],
    dataset_path: Path,
    rag_results_path: Path,
    rag_runs: list[RunRecords],
    baseline_runs: list[RunRecords],
    model: str,
) -> dict[str, Any]:
    if len(rag_runs) != len(baseline_runs) or len(rag_runs) < 2:
        raise ValueError("both systems require the same number of repeated rounds")
    rounds = len(rag_runs)
    rag_observations = score_runs(cases, rag_runs)
    baseline_observations = score_runs(cases, baseline_runs)
    rag_summary = summarize_accuracy(rag_observations)
    baseline_summary = summarize_accuracy(baseline_observations)
    first_rag = [item for item in rag_observations if item["round"] == 1]
    first_baseline = [item for item in baseline_observations if item["round"] == 1]

    return {
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": "repeated_paired_rag_vs_no_rag_closed_set_accuracy",
        "dataset": dataset_path.name,
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "case_count": len(cases),
        "rounds": rounds,
        "request_count": len(cases) * rounds * 2,
        "model": model,
        "temperature": 0,
        "baseline_system_prompt": BASELINE_SYSTEM_PROMPT,
        "scoring": {
            "unit": "one binary correctness decision per question and round",
            "rule": "pre-registered all/any/forbidden keyword rubric",
            "confidence_interval": "Wilson score interval, 95%",
            "paired_test": "exact two-sided McNemar test on round one",
        },
        "rag_source": {
            "artifact": rag_results_path.name,
            "rounds": list(range(1, rounds + 1)),
            "latency_scope": "cached repeated queries; not comparable to baseline",
        },
        "comparison": {
            **effect_size(rag_summary["accuracy"], baseline_summary["accuracy"]),
            "mcnemar_round_one": mcnemar_exact(first_rag, first_baseline),
        },
        "systems": {
            "knowflow_rag": _system_report(
                rag_observations, rounds, include_tokens=False
            ),
            "no_rag_baseline": _system_report(
                baseline_observations, rounds, include_tokens=True
            ),
        },
    }


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    args = parse_args()
    if args.rounds < 2:
        raise SystemExit("--rounds must be at least 2")
    key = os.getenv("LLM_BINDING_API_KEY", "").strip()
    if key.lower() in PLACEHOLDER_KEYS:
        raise SystemExit("LLM_BINDING_API_KEY is not configured")

    cases = load_accuracy_dataset(args.dataset)
    rag_runs = load_rag_runs(args.rag_results, args.rounds)
    client = OpenAI(api_key=key, base_url=args.base_url, timeout=args.timeout)
    try:
        baseline_runs = [
            run_no_rag_baseline(
                cases,
                client,
                model=args.model,
                round_number=round_number,
            )
            for round_number in range(1, args.rounds + 1)
        ]
    finally:
        client.close()

    report = build_report(
        cases=cases,
        dataset_path=args.dataset,
        rag_results_path=args.rag_results,
        rag_runs=rag_runs,
        baseline_runs=baseline_runs,
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
