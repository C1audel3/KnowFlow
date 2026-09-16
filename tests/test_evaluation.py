from __future__ import annotations

import json
from collections import Counter
from types import SimpleNamespace

import pytest

from app.evaluation import (
    EvaluationCase,
    evaluate_answer,
    load_dataset,
    run_evaluation,
    summarize_results,
)
from scripts import evaluate as evaluate_script
from ui.client import UIError


def response(answer: str, duration_ms: int = 100):
    return SimpleNamespace(
        answer=answer,
        duration_ms=duration_ms,
        sources=[],
        mode="mix",
    )


def test_real_dataset_has_balanced_fixed_cases():
    cases = load_dataset(evaluate_script.DEFAULT_DATASET)
    distribution = Counter(case.type for case in cases)

    assert len(cases) == 16
    assert distribution == {
        "text": 4,
        "table": 3,
        "image": 3,
        "equation": 2,
        "cross_document": 2,
        "refusal": 2,
    }
    assert len({case.id for case in cases}) == len(cases)


def test_dataset_loader_rejects_invalid_and_duplicate_lines(tmp_path):
    invalid = tmp_path / "invalid.jsonl"
    invalid.write_text('{"id":"bad"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        load_dataset(invalid)

    duplicate = tmp_path / "duplicate.jsonl"
    record = {
        "id": "same",
        "question": "question",
        "type": "text",
        "expected_keywords": ["answer"],
    }
    duplicate.write_text(
        json.dumps(record) + "\n" + json.dumps(record) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="duplicate evaluation id"):
        load_dataset(duplicate)


def test_dataset_loader_skips_comments_and_rejects_empty(tmp_path):
    dataset = tmp_path / "empty.jsonl"
    dataset.write_text("# comment\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_dataset(dataset)


def test_answer_rules_cover_all_any_and_forbidden_keywords():
    case = EvaluationCase(
        id="rules",
        question="q",
        type="refusal",
        expected_keywords=["Atlas"],
        expected_any=["未提供", "不知道"],
        forbidden_keywords=["100 万元"],
    )

    passed = evaluate_answer(case, "ATLAS 的预算未提供。")
    failed = evaluate_answer(case, "Atlas 的预算为 100 万元。")

    assert passed["passed"] is True
    assert passed["matched_any_keyword"] == "未提供"
    assert failed["passed"] is False
    assert failed["found_forbidden_keywords"] == ["100 万元"]


def test_case_requires_at_least_one_positive_rule():
    with pytest.raises(ValueError, match="expected keyword"):
        EvaluationCase(id="empty", question="q", type="text")


def test_runner_continues_after_safe_request_error():
    cases = [
        EvaluationCase(
            id="first", question="one", type="text", expected_keywords=["ok"]
        ),
        EvaluationCase(
            id="second", question="two", type="text", expected_keywords=["ok"]
        ),
    ]

    def query(question):
        if question == "one":
            raise UIError("模型服务暂时不可用", "MODEL_UNAVAILABLE")
        return response("ok")

    ticks = iter([0.0, 0.01, 1.0, 1.2])
    results = run_evaluation(cases, query, clock=lambda: next(ticks))

    assert len(results) == 2
    assert results[0].request_success is False
    assert results[0].error_code == "MODEL_UNAVAILABLE"
    assert results[0].error_message == "模型服务暂时不可用"
    assert results[1].passed is True
    assert results[1].end_to_end_duration_ms == 200


def test_runner_records_rounds_backend_and_api_overhead():
    case = EvaluationCase(
        id="latency", question="q", type="table", expected_keywords=["18"]
    )
    ticks = iter([0.0, 0.15, 1.0, 1.05])

    results = run_evaluation(
        [case],
        lambda _: response("18 ms", duration_ms=100),
        rounds=2,
        clock=lambda: next(ticks),
    )

    assert [item.round for item in results] == [1, 2]
    assert results[0].end_to_end_duration_ms == 150
    assert results[0].api_overhead_ms == 50
    assert results[1].api_overhead_ms == 0


def test_summary_reports_rates_categories_percentiles_and_failures():
    cases = [
        EvaluationCase(id="ok", question="q1", type="text", expected_keywords=["yes"]),
        EvaluationCase(id="no", question="q2", type="refusal", expected_any=["未提供"]),
    ]
    ticks = iter([0.0, 0.1, 1.0, 1.3])
    results = run_evaluation(
        cases,
        lambda question: response("yes" if question == "q1" else "invented", 50),
        clock=lambda: next(ticks),
    )

    summary = summarize_results(cases, results, rounds=1)

    assert summary["case_count"] == 2
    assert summary["request_success_rate"] == 1
    assert summary["rule_pass_rate"] == 0.5
    assert summary["refusal_pass_rate"] == 0
    assert summary["by_type"]["text"]["rule_pass_rate"] == 1
    assert summary["latency"]["end_to_end_p50_ms"] == 200
    assert summary["latency"]["end_to_end_p95_ms"] == 290
    assert summary["failed_case_ids"] == ["no@1"]


def test_build_report_contains_no_credentials(monkeypatch, tmp_path):
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "one",
                "question": "q",
                "type": "text",
                "expected_keywords": ["answer"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class FakeClient:
        def __init__(self, base_url, timeout):
            self.base_url = base_url

        def health(self):
            return SimpleNamespace(status="ok", rag_initialized=True)

        def query(self, question):
            return response("answer", 10)

        def close(self):
            return None

    monkeypatch.setattr(evaluate_script, "APIClient", FakeClient)
    monkeypatch.setenv("LLM_BINDING_API_KEY", "top-secret")
    args = SimpleNamespace(
        dataset=dataset,
        output=tmp_path / "result.json",
        api_url="http://test",
        timeout=1,
        rounds=1,
    )

    report = evaluate_script.build_report(args)

    assert report["summary"]["rule_pass_rate"] == 1
    assert len(report["dataset_sha256"]) == 64
    assert report["dataset_distribution"] == {"text": 1}
    assert report["run_config"] == {"rounds": 1, "timeout_seconds": 1}
    assert "top-secret" not in json.dumps(report)
