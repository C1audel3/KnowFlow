import copy
import csv
import json
from pathlib import Path

import pytest

from app.paper_evaluation import load_cases
from scripts.benchmark_paper_reading import sha256
from scripts.review_paper_reading import aggregate, export_packet, records

ROOT = Path(__file__).resolve().parents[1]


def observations():
    return {
        "observations": [
            {
                "case_id": "case-a",
                "mode": mode,
                "category": "figure",
                "response": {"answer": "<script>alert(1)</script>", "evidence": []},
            }
            for mode in ("no_rag", "text_rag", "multimodal_rag")
        ]
    }


def rating_sheet(items):
    return [
        {"review_id": item["review_id"], "record_sha256": item["record_sha256"]}
        for item in items
    ]


def test_unreviewed_and_partial_metrics_keep_human_denominators_explicit():
    items = records(observations(), {"case-a": True})
    sheet = rating_sheet(items)
    pending = aggregate({"items": items}, sheet)
    assert pending["status"] == "pending"
    assert all(
        row["semantic_correct_rate_reviewed"] is None
        for row in pending["by_mode"].values()
    )
    sheet[1].update(
        reviewer="tester",
        source_checked="yes",
        semantic_correct="partial",
        citation_supported="unsupported",
        error_stage="generation",
        notes="wrong direction",
    )
    summary = aggregate({"items": items}, sheet)
    assert summary["status"] == "partial"
    assert summary["reviewed"] == 1
    assert summary["by_mode"]["text_rag"]["semantic_correct_rate_reviewed"] == 0
    assert summary["by_mode"]["multimodal_rag"]["semantic_judged"] == 0


@pytest.mark.parametrize(
    "mutation",
    ["duplicate", "missing", "hash", "unchecked", "baseline_citation", "no_reviewer"],
)
def test_reviews_reject_misattributed_or_invalid_judgments(mutation):
    items = records(observations(), {"case-a": True})
    sheet = rating_sheet(items)
    sheet[0].update(
        reviewer="tester",
        source_checked="yes",
        semantic_correct="correct",
        citation_supported="not_applicable",
        error_stage="none",
    )
    if mutation == "duplicate":
        sheet.append(sheet[0])
    elif mutation == "missing":
        sheet.pop()
    elif mutation == "hash":
        sheet[0]["record_sha256"] = "wrong"
    elif mutation == "unchecked":
        sheet[0]["source_checked"] = "no"
    elif mutation == "baseline_citation":
        sheet[0]["citation_supported"] = "supported"
    else:
        sheet[0]["reviewer"] = ""
    with pytest.raises(ValueError):
        aggregate({"items": items}, sheet)


def test_unknown_and_duplicate_observations_rejected_and_hash_binds_evidence():
    raw = observations()
    before = records(raw, {"case-a": True})
    modified = copy.deepcopy(raw)
    modified["observations"][0]["response"]["evidence"].append(
        {"text": "different evidence"}
    )
    after = records(modified, {"case-a": True})
    assert before[0]["record_sha256"] != after[0]["record_sha256"]
    raw["observations"].append(raw["observations"][0])
    with pytest.raises(ValueError, match="duplicate"):
        records(raw, {"case-a": True})


def test_failed_requests_cannot_be_counted_as_semantically_correct():
    raw = observations()
    raw["observations"][1]["response"] = None
    items = records(raw, {"case-a": True})
    sheet = rating_sheet(items)
    sheet[1].update(
        reviewer="tester",
        source_checked="no",
        semantic_correct="uncertain",
        citation_supported="not_applicable",
        error_stage="service",
    )
    result = aggregate({"items": items}, sheet)
    assert result["by_mode"]["text_rag"]["semantic_judged"] == 0
    sheet[1]["semantic_correct"] = "correct"
    with pytest.raises(ValueError, match="failed requests"):
        aggregate({"items": items}, sheet)


def test_export_is_blinded_escaped_bound_to_annotations_and_preserves_review(tmp_path):
    dataset = ROOT / "data/evaluation/paper_reading_real.jsonl"
    manifest = ROOT / "data/benchmarks/papers/manifest.json"
    raw = observations()
    for row in raw["observations"]:
        row["case_id"] = "rag2020-architecture"
    raw["metadata"] = {
        "dataset_sha256": sha256(dataset),
        "manifest_sha256": sha256(manifest),
    }
    path = tmp_path / "raw.json"
    path.write_text(json.dumps(raw))
    packet = tmp_path / "packet"
    export_packet(path, dataset, manifest, packet)
    html = (packet / "review.html").read_text()
    assert "&lt;script&gt;" in html and "<script>" not in html
    assert not any(mode in html for mode in ("no_rag", "text_rag", "multimodal_rag"))
    with (packet / "ratings.csv").open(encoding="utf-8-sig") as stream:
        sheet = list(csv.DictReader(stream))
    assert len(sheet) == 3 and all(not row["semantic_correct"] for row in sheet)
    with pytest.raises(FileExistsError):
        export_packet(path, dataset, manifest, packet)
    raw["metadata"]["dataset_sha256"] = "wrong"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="dataset differs"):
        export_packet(path, dataset, manifest, tmp_path / "wrong")


def test_visual_cases_are_separate_development_diagnostics():
    cases = load_cases(ROOT / "data/evaluation/paper_reading_visual_dev.jsonl")
    old = load_cases(ROOT / "data/evaluation/paper_reading_real.jsonl")
    assert len(cases) == 6
    assert {case.id for case in cases}.isdisjoint(case.id for case in old)
    assert all(case.split == "dev" and case.visual_probe for case in cases)
    assert all("pending_human_review" in case.review_status for case in cases)
