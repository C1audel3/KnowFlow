"""Deterministic evaluation models, runner, and aggregate metrics."""

from __future__ import annotations

import json
import math
import statistics
import time
from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

EvaluationType = Literal[
    "text", "table", "image", "equation", "cross_document", "refusal"
]


class EvaluationCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    question: str = Field(min_length=1, max_length=2000)
    type: EvaluationType
    expected_keywords: list[str] = Field(default_factory=list)
    expected_any: list[str] = Field(default_factory=list)
    forbidden_keywords: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_positive_expectation(self) -> EvaluationCase:
        if not self.expected_keywords and not self.expected_any:
            raise ValueError("at least one expected keyword rule is required")
        return self


class EvaluationQueryResponse(Protocol):
    answer: str
    duration_ms: int
    sources: list[str]
    mode: str


class EvaluationResult(BaseModel):
    case_id: str
    round: int
    type: EvaluationType
    question: str
    answer: str | None
    sources: list[str]
    request_success: bool
    nonempty_answer: bool
    expected_all_passed: bool
    expected_any_passed: bool
    forbidden_passed: bool
    passed: bool
    missing_keywords: list[str]
    matched_any_keyword: str | None
    found_forbidden_keywords: list[str]
    backend_duration_ms: int | None
    end_to_end_duration_ms: int
    api_overhead_ms: int | None
    error_code: str | None = None
    error_message: str | None = None


def load_dataset(path: Path) -> list[EvaluationCase]:
    cases: list[EvaluationCase] = []
    identifiers: set[str] = set()
    with path.open(encoding="utf-8") as dataset:
        for line_number, raw_line in enumerate(dataset, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                payload = json.loads(line)
                case = EvaluationCase.model_validate(payload)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"invalid dataset line {line_number}") from exc
            if case.id in identifiers:
                raise ValueError(f"duplicate evaluation id: {case.id}")
            identifiers.add(case.id)
            cases.append(case)
    if not cases:
        raise ValueError("evaluation dataset is empty")
    return cases


def _contains(answer: str, keyword: str) -> bool:
    return keyword.casefold() in answer.casefold()


def evaluate_answer(case: EvaluationCase, answer: str) -> dict[str, Any]:
    stripped = answer.strip()
    missing = [
        keyword
        for keyword in case.expected_keywords
        if not _contains(stripped, keyword)
    ]
    matched_any = next(
        (keyword for keyword in case.expected_any if _contains(stripped, keyword)), None
    )
    forbidden = [
        keyword for keyword in case.forbidden_keywords if _contains(stripped, keyword)
    ]
    expected_any_passed = not case.expected_any or matched_any is not None
    return {
        "nonempty_answer": bool(stripped),
        "expected_all_passed": not missing,
        "expected_any_passed": expected_any_passed,
        "forbidden_passed": not forbidden,
        "passed": bool(stripped)
        and not missing
        and expected_any_passed
        and not forbidden,
        "missing_keywords": missing,
        "matched_any_keyword": matched_any,
        "found_forbidden_keywords": forbidden,
    }


def run_evaluation(
    cases: Iterable[EvaluationCase],
    query: Callable[[str], EvaluationQueryResponse],
    *,
    rounds: int = 1,
    clock: Callable[[], float] = time.perf_counter,
) -> list[EvaluationResult]:
    if rounds <= 0:
        raise ValueError("rounds must be positive")
    case_list = list(cases)
    results: list[EvaluationResult] = []
    for round_number in range(1, rounds + 1):
        for case in case_list:
            started_at = clock()
            try:
                response = query(case.question)
                elapsed_ms = round((clock() - started_at) * 1000)
                checks = evaluate_answer(case, response.answer)
                overhead = max(0, elapsed_ms - response.duration_ms)
                results.append(
                    EvaluationResult(
                        case_id=case.id,
                        round=round_number,
                        type=case.type,
                        question=case.question,
                        answer=response.answer.strip(),
                        sources=response.sources,
                        request_success=True,
                        backend_duration_ms=response.duration_ms,
                        end_to_end_duration_ms=elapsed_ms,
                        api_overhead_ms=overhead,
                        **checks,
                    )
                )
            except Exception as exc:
                elapsed_ms = round((clock() - started_at) * 1000)
                code = getattr(exc, "code", type(exc).__name__)
                message = getattr(exc, "message", "评测请求失败")
                results.append(
                    EvaluationResult(
                        case_id=case.id,
                        round=round_number,
                        type=case.type,
                        question=case.question,
                        answer=None,
                        sources=[],
                        request_success=False,
                        nonempty_answer=False,
                        expected_all_passed=False,
                        expected_any_passed=False,
                        forbidden_passed=False,
                        passed=False,
                        missing_keywords=list(case.expected_keywords),
                        matched_any_keyword=None,
                        found_forbidden_keywords=[],
                        backend_duration_ms=None,
                        end_to_end_duration_ms=elapsed_ms,
                        api_overhead_ms=None,
                        error_code=str(code)[:80],
                        error_message=str(message)[:200],
                    )
                )
    return results


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _percentile(values: list[int], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction, 2)


def _latency_metrics(results: list[EvaluationResult]) -> dict[str, Any]:
    end_to_end = [
        item.end_to_end_duration_ms for item in results if item.request_success
    ]
    backend = [
        item.backend_duration_ms
        for item in results
        if item.request_success and item.backend_duration_ms is not None
    ]
    overhead = [
        item.api_overhead_ms
        for item in results
        if item.request_success and item.api_overhead_ms is not None
    ]
    slowest = max(
        (item for item in results if item.request_success),
        key=lambda item: item.end_to_end_duration_ms,
        default=None,
    )
    return {
        "end_to_end_mean_ms": round(statistics.fmean(end_to_end), 2)
        if end_to_end
        else None,
        "end_to_end_p50_ms": _percentile(end_to_end, 0.5),
        "end_to_end_p95_ms": _percentile(end_to_end, 0.95),
        "backend_mean_ms": round(statistics.fmean(backend), 2) if backend else None,
        "api_overhead_mean_ms": round(statistics.fmean(overhead), 2)
        if overhead
        else None,
        "slowest_case_id": slowest.case_id if slowest else None,
        "slowest_end_to_end_ms": slowest.end_to_end_duration_ms if slowest else None,
    }


def summarize_results(
    cases: list[EvaluationCase], results: list[EvaluationResult], rounds: int
) -> dict[str, Any]:
    total = len(results)
    grouped: dict[str, list[EvaluationResult]] = defaultdict(list)
    by_round: dict[int, list[EvaluationResult]] = defaultdict(list)
    for result in results:
        grouped[result.type].append(result)
        by_round[result.round].append(result)

    def result_rates(items: list[EvaluationResult]) -> dict[str, Any]:
        count = len(items)
        return {
            "requests": count,
            "request_success_rate": _rate(
                sum(item.request_success for item in items), count
            ),
            "nonempty_answer_rate": _rate(
                sum(item.nonempty_answer for item in items), count
            ),
            "rule_pass_rate": _rate(sum(item.passed for item in items), count),
            "latency": _latency_metrics(items),
        }

    refusal = grouped.get("refusal", [])
    return {
        "case_count": len(cases),
        "rounds": rounds,
        "request_count": total,
        **result_rates(results),
        "refusal_pass_rate": _rate(sum(item.passed for item in refusal), len(refusal)),
        "by_type": {
            evaluation_type: result_rates(items)
            for evaluation_type, items in sorted(grouped.items())
        },
        "by_round": {
            str(round_number): result_rates(items)
            for round_number, items in sorted(by_round.items())
        },
        "failed_case_ids": [
            f"{item.case_id}@{item.round}" for item in results if not item.passed
        ],
    }
