from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.accuracy import (
    AccuracyCase,
    load_accuracy_dataset,
    mcnemar_exact,
    score_answers,
    summarize_accuracy,
    wilson_interval,
)
from scripts import accuracy_experiment


def case(case_id: str, category: str = "text") -> AccuracyCase:
    return AccuracyCase(
        id=case_id,
        question=f"question {case_id}",
        type=category,
        reference_answer="yes",
        expected_keywords=["yes"],
    )


def test_accuracy_dataset_is_fixed_and_balanced():
    cases = load_accuracy_dataset(accuracy_experiment.DEFAULT_DATASET)

    assert len(cases) == 16
    assert len({item.id for item in cases}) == 16
    assert {item.type for item in cases} == {
        "text",
        "table",
        "image",
        "equation",
        "cross_document",
        "refusal",
    }
    assert all(item.reference_answer for item in cases)


def test_accuracy_loader_rejects_empty_invalid_and_duplicate(tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("# empty\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_accuracy_dataset(empty)

    invalid = tmp_path / "invalid.jsonl"
    invalid.write_text('{"id":"bad"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        load_accuracy_dataset(invalid)

    duplicate = tmp_path / "duplicate.jsonl"
    payload = case("same").model_dump_json()
    duplicate.write_text(payload + "\n" + payload + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_accuracy_dataset(duplicate)


def test_score_answers_requires_exact_ids_and_applies_rubric():
    cases = [case("one"), case("two")]

    observations = score_answers(cases, {"one": "YES", "two": "no"})

    assert [item["correct"] for item in observations] == [True, False]
    with pytest.raises(ValueError, match="missing"):
        score_answers(cases, {"one": "yes"})


def test_wilson_interval_and_accuracy_summary():
    cases = [case("one", "text"), case("two", "table")]
    observations = score_answers(cases, {"one": "yes", "two": "no"})

    summary = summarize_accuracy(observations)

    assert summary["accuracy"] == 0.5
    assert summary["macro_accuracy"] == 0.5
    assert summary["by_type"]["text"]["accuracy"] == 1
    assert summary["incorrect_case_ids"] == ["two"]
    assert wilson_interval(16, 16) == [0.8064, 1.0]


def test_wilson_interval_rejects_invalid_samples():
    with pytest.raises(ValueError, match="non-empty"):
        wilson_interval(0, 0)
    with pytest.raises(ValueError, match="non-empty"):
        wilson_interval(2, 1)


def test_mcnemar_exact_reports_paired_disagreements():
    cases = [case(str(index)) for index in range(4)]
    first = score_answers(cases, {str(index): "yes" for index in range(4)})
    second = score_answers(cases, {"0": "yes", "1": "no", "2": "no", "3": "no"})

    result = mcnemar_exact(first, second)

    assert result == {
        "first_correct_second_wrong": 3,
        "first_wrong_second_correct": 0,
        "discordant_pairs": 3,
        "exact_two_sided_p_value": 0.25,
    }


def test_load_rag_answers_selects_one_round_and_rejects_duplicates(tmp_path):
    artifact = tmp_path / "rag.json"
    artifact.write_text(
        json.dumps(
            {
                "results": [
                    {"case_id": "one", "round": 1, "answer": "first"},
                    {"case_id": "one", "round": 2, "answer": "second"},
                ]
            }
        ),
        encoding="utf-8",
    )

    assert accuracy_experiment.load_rag_answers(artifact, 1) == {"one": "first"}
    with pytest.raises(ValueError, match="no answers"):
        accuracy_experiment.load_rag_answers(artifact, 3)


def test_no_rag_baseline_records_answers_and_durations(monkeypatch):
    class Completions:
        def create(self, **kwargs):
            assert kwargs["temperature"] == 0
            assert kwargs["messages"][0]["content"] == (
                accuracy_experiment.BASELINE_SYSTEM_PROMPT
            )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="yes"))]
            )

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=Completions()),
    )
    ticks = iter([1.0, 1.125])
    monkeypatch.setattr(accuracy_experiment.time, "perf_counter", lambda: next(ticks))

    answers, durations = accuracy_experiment.run_no_rag_baseline(
        [case("one")], client, model="model"
    )

    assert answers == {"one": "yes"}
    assert durations == {"one": 125}


def test_build_report_contains_statistics_and_no_credentials(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(case("one").model_dump_json() + "\n", encoding="utf-8")
    rag = tmp_path / "rag.json"
    rag.write_text("{}", encoding="utf-8")

    report = accuracy_experiment.build_report(
        cases=[case("one")],
        dataset_path=dataset,
        rag_results_path=rag,
        rag_round=1,
        rag_answers={"one": "yes"},
        baseline_answers={"one": "no"},
        baseline_durations={"one": 50},
        model="deepseek-flash",
    )

    assert report["systems"]["knowflow_rag"]["summary"]["accuracy"] == 1
    assert report["systems"]["no_rag_baseline"]["summary"]["accuracy"] == 0
    assert report["comparison"]["accuracy_gain_percentage_points"] == 100
    assert "api_key" not in json.dumps(report).lower()
