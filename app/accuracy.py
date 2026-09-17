"""Closed-set factual accuracy scoring for the KnowFlow ablation experiment."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import ConfigDict, Field

from app.evaluation import EvaluationCase, EvaluationType, evaluate_answer


class AccuracyCase(EvaluationCase):
    model_config = ConfigDict(extra="forbid")

    reference_answer: str = Field(min_length=1)

    def evaluation_case(self) -> EvaluationCase:
        return EvaluationCase.model_validate(
            self.model_dump(exclude={"reference_answer"})
        )


def load_accuracy_dataset(path: Path) -> list[AccuracyCase]:
    cases: list[AccuracyCase] = []
    identifiers: set[str] = set()
    with path.open(encoding="utf-8") as dataset:
        for line_number, raw_line in enumerate(dataset, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                case = AccuracyCase.model_validate_json(line)
            except ValueError as exc:
                raise ValueError(
                    f"invalid accuracy dataset line {line_number}"
                ) from exc
            if case.id in identifiers:
                raise ValueError(f"duplicate accuracy id: {case.id}")
            identifiers.add(case.id)
            cases.append(case)
    if not cases:
        raise ValueError("accuracy dataset is empty")
    return cases


def score_answers(
    cases: list[AccuracyCase], answers: dict[str, str]
) -> list[dict[str, Any]]:
    expected_ids = {case.id for case in cases}
    missing = expected_ids - answers.keys()
    extra = answers.keys() - expected_ids
    if missing or extra:
        raise ValueError(
            f"answer ids do not match dataset (missing={sorted(missing)}, "
            f"extra={sorted(extra)})"
        )

    observations: list[dict[str, Any]] = []
    for case in cases:
        answer = answers[case.id].strip()
        checks = evaluate_answer(case.evaluation_case(), answer)
        observations.append(
            {
                "case_id": case.id,
                "type": case.type,
                "question": case.question,
                "reference_answer": case.reference_answer,
                "answer": answer,
                "correct": checks["passed"],
                "missing_keywords": checks["missing_keywords"],
                "matched_any_keyword": checks["matched_any_keyword"],
                "found_forbidden_keywords": checks["found_forbidden_keywords"],
            }
        )
    return observations


def wilson_interval(correct: int, total: int, z: float = 1.96) -> list[float]:
    if total <= 0 or not 0 <= correct <= total:
        raise ValueError("correct and total must define a non-empty binomial sample")
    proportion = correct / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2))
        / denominator
    )
    return [round(max(0, center - margin), 4), round(min(1, center + margin), 4)]


def summarize_accuracy(observations: list[dict[str, Any]]) -> dict[str, Any]:
    if not observations:
        raise ValueError("accuracy observations are empty")
    grouped: dict[EvaluationType, list[dict[str, Any]]] = defaultdict(list)
    for item in observations:
        grouped[item["type"]].append(item)

    def metrics(items: list[dict[str, Any]]) -> dict[str, Any]:
        correct = sum(bool(item["correct"]) for item in items)
        total = len(items)
        return {
            "correct": correct,
            "total": total,
            "accuracy": round(correct / total, 4),
            "wilson_95_ci": wilson_interval(correct, total),
        }

    category_metrics = {
        category: metrics(items) for category, items in sorted(grouped.items())
    }
    overall = metrics(observations)
    overall["macro_accuracy"] = round(
        sum(item["accuracy"] for item in category_metrics.values())
        / len(category_metrics),
        4,
    )
    overall["by_type"] = category_metrics
    overall["incorrect_case_ids"] = [
        item["case_id"] for item in observations if not item["correct"]
    ]
    return overall


def mcnemar_exact(
    first: list[dict[str, Any]], second: list[dict[str, Any]]
) -> dict[str, Any]:
    first_by_id = {item["case_id"]: bool(item["correct"]) for item in first}
    second_by_id = {item["case_id"]: bool(item["correct"]) for item in second}
    if first_by_id.keys() != second_by_id.keys():
        raise ValueError("paired systems must contain identical case ids")

    first_only = sum(first_by_id[key] and not second_by_id[key] for key in first_by_id)
    second_only = sum(second_by_id[key] and not first_by_id[key] for key in first_by_id)
    discordant = first_only + second_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(discordant, index)
            for index in range(min(first_only, second_only) + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2 * tail)
    return {
        "first_correct_second_wrong": first_only,
        "first_wrong_second_correct": second_only,
        "discordant_pairs": discordant,
        "exact_two_sided_p_value": round(p_value, 6),
    }


def percentile(values: list[int], probability: float) -> float:
    if not values or not 0 <= probability <= 1:
        raise ValueError("percentile requires values and a probability in [0, 1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction, 2)


def efficiency_summary(observations: list[dict[str, Any]]) -> dict[str, Any]:
    if not observations:
        raise ValueError("efficiency observations are empty")
    lengths = [len(item["answer"]) for item in observations]
    durations = [
        int(item["duration_ms"])
        for item in observations
        if item.get("duration_ms") is not None
    ]
    result: dict[str, Any] = {
        "answer_length_mean_chars": round(statistics.fmean(lengths), 2),
        "answer_length_p50_chars": percentile(lengths, 0.5),
    }
    if durations:
        result["latency"] = {
            "samples": len(durations),
            "mean_ms": round(statistics.fmean(durations), 2),
            "p50_ms": percentile(durations, 0.5),
            "p95_ms": percentile(durations, 0.95),
            "min_ms": min(durations),
            "max_ms": max(durations),
        }
    return result


def answerability_summary(observations: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [item for item in observations if item["type"] != "refusal"]
    refusal = [item for item in observations if item["type"] == "refusal"]

    def accuracy(items: list[dict[str, Any]]) -> dict[str, Any]:
        correct = sum(bool(item["correct"]) for item in items)
        return {
            "correct": correct,
            "total": len(items),
            "accuracy": round(correct / len(items), 4) if items else None,
        }

    return {
        "answerable_questions": accuracy(answerable),
        "unanswerable_refusal_questions": accuracy(refusal),
    }


def repeatability_summary(
    observations: list[dict[str, Any]], rounds: int
) -> dict[str, Any]:
    if rounds < 2:
        raise ValueError("repeatability requires at least two rounds")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_round: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in observations:
        grouped[item["case_id"]].append(item)
        by_round[int(item["round"])].append(item)
    if set(by_round) != set(range(1, rounds + 1)):
        raise ValueError("observations do not contain every requested round")
    if any(len(items) != rounds for items in grouped.values()):
        raise ValueError("each case must have exactly one observation per round")

    correctness_agreements = 0
    exact_answer_matches = 0
    for items in grouped.values():
        ordered = sorted(items, key=lambda item: item["round"])
        correctness_agreements += len({item["correct"] for item in ordered}) == 1
        normalized_answers = {" ".join(item["answer"].split()) for item in ordered}
        exact_answer_matches += len(normalized_answers) == 1

    round_accuracies = {
        str(round_number): round(
            sum(bool(item["correct"]) for item in items) / len(items), 4
        )
        for round_number, items in sorted(by_round.items())
    }
    return {
        "rounds": rounds,
        "case_count": len(grouped),
        "correctness_agreement_rate": round(correctness_agreements / len(grouped), 4),
        "exact_answer_match_rate": round(exact_answer_matches / len(grouped), 4),
        "accuracy_by_round": round_accuracies,
        "accuracy_range": round(
            max(round_accuracies.values()) - min(round_accuracies.values()), 4
        ),
    }


def effect_size(first_accuracy: float, second_accuracy: float) -> dict[str, Any]:
    absolute_gain = first_accuracy - second_accuracy
    second_error = 1 - second_accuracy
    first_error = 1 - first_accuracy
    return {
        "accuracy_gain_percentage_points": round(absolute_gain * 100, 2),
        "accuracy_ratio": round(first_accuracy / second_accuracy, 4)
        if second_accuracy
        else None,
        "relative_accuracy_improvement": round(absolute_gain / second_accuracy, 4)
        if second_accuracy
        else None,
        "error_reduction_rate": round((second_error - first_error) / second_error, 4)
        if second_error
        else None,
    }
