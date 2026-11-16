from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.paper_evaluation import (
    EvidenceLocator,
    automatic_checks,
    context_hash,
    load_cases,
    locator_matches,
    summarize,
)
from app.schemas.papers import PaperEvidence, PaperReadingResponse
from scripts.benchmark_paper_reading import save_atomic, write_review

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/evaluation/paper_reading_real.jsonl"


def response(text="44.1 and 44.5 [E1]", evidence=None):
    return PaperReadingResponse(
        answer=text,
        evidence=evidence or [],
        citations=["E1"],
        status="answered",
        duration_ms=10,
        vision_used=False,
        retrieval="figure_lookup",
    )


def test_real_dataset_has_pinned_sources_and_separated_evidence_groups():
    cases = load_cases(DATASET)
    manifest = json.loads((ROOT / "data/benchmarks/papers/manifest.json").read_text())
    pages = {paper["key"]: paper["pages"] for paper in manifest["papers"]}
    assert len(cases) == 30
    assert sum(case.split == "dev" for case in cases) == 9
    assert sum(case.split == "test" for case in cases) == 21
    assert {case.paper_key for case in cases} == set(pages)
    for case in cases:
        assert all(
            locator.page <= pages[case.paper_key] for locator in case.gold_evidence
        )
        assert "pending_human_review" in case.review_status


def test_loader_rejects_group_leakage_between_dev_and_test(tmp_path):
    cases = load_cases(DATASET)
    first, second = cases[0].model_dump(), cases[1].model_dump()
    second.update(group_id=first["group_id"], split="test")
    path = tmp_path / "cases.jsonl"
    path.write_text(json.dumps(first) + "\n" + json.dumps(second))
    with pytest.raises(ValueError, match="cross"):
        load_cases(path)


def test_source_locator_requires_page_type_and_exact_figure_number():
    item = PaperEvidence(
        evidence_id="x",
        paper_id="a",
        page=4,
        section="Results",
        modality="chart",
        text="Figure 3: Evaluation",
    )
    assert locator_matches(
        EvidenceLocator(page=4, kind="figure", reference="figure:3"), item
    )
    assert not locator_matches(EvidenceLocator(page=3, kind="figure"), item)
    assert not locator_matches(EvidenceLocator(page=4, kind="table"), item)
    assert not locator_matches(
        EvidenceLocator(page=4, kind="figure", reference="figure:30"), item
    )


def test_keyword_and_locator_checks_do_not_claim_semantic_correctness():
    case = next(case for case in load_cases(DATASET) if case.id == "rag2020-nq-table")
    item = PaperEvidence(
        evidence_id="x",
        paper_id="rag2020",
        page=6,
        section="Results",
        modality="table",
        text="Table 1: scores",
    )
    answer = response(evidence=[item])
    checks = automatic_checks(case, answer)
    assert checks["keyword_rule_pass"]
    assert checks["locator_recall"] == 1
    assert checks["cited_locator_recall"] == 1
    summary = summarize(
        [
            {
                "case_id": case.id,
                "split": "test",
                "mode": "text_rag",
                "response": answer.model_dump(),
                "checks": checks,
            }
        ]
    )
    metrics = summary["by_split_and_mode"]["test:text_rag"]
    assert metrics["semantic_accuracy"] is None
    assert metrics["citation_support_rate"] is None


def test_image_ablation_detects_changed_context():
    first = response()
    second = response(
        evidence=[
            PaperEvidence(
                evidence_id="x",
                paper_id="a",
                page=1,
                section="",
                modality="text",
                text="different",
            )
        ]
    )
    assert context_hash(first) != context_hash(second)
    rows = [
        {
            "case_id": "a",
            "split": "test",
            "mode": mode,
            "response": answer.model_dump(),
            "checks": {
                "keyword_rule_pass": False,
                "locator_recall": None,
                "context_hash": context_hash(answer),
            },
        }
        for mode, answer in [("text_rag", first), ("multimodal_rag", second)]
    ]
    assert summarize(rows)["context_pair_checks"] == [
        {"case_id": "a", "same_context": False}
    ]


def test_checkpoint_is_atomic_and_human_review_is_preserved(tmp_path):
    path = tmp_path / "checkpoint.json"
    save_atomic(path, {"complete": True})
    assert json.loads(path.read_text())["complete"]
    assert not path.with_suffix(".json.tmp").exists()
    review = tmp_path / "review.csv"
    review.write_text("my manual review")
    write_review(review, [], {})
    assert review.read_text() == "my manual review"


def test_cross_page_span_extends_source_only_with_matching_metadata(tmp_path):
    from app.services.paper_sources import enrich_page_ranges

    text = "This is a sufficiently long paragraph continuing onto the next PDF page."
    middle = tmp_path / "paper_middle.json"
    middle.write_text(
        json.dumps(
            {
                "pdf_info": [
                    {
                        "para_blocks": [
                            {
                                "type": "text",
                                "lines": [
                                    {"spans": [{"content": text[:45]}]},
                                    {
                                        "spans": [
                                            {"content": text[45:], "cross_page": True}
                                        ]
                                    },
                                ],
                            }
                        ]
                    },
                    {"para_blocks": []},
                ]
            }
        )
    )
    content = [
        {"type": "text", "text": text, "page_idx": 0},
        {"type": "text", "text": "A different paragraph", "page_idx": 0},
    ]
    enriched = enrich_page_ranges(content, middle)
    assert enriched[0]["_page_end_idx"] == 1
    assert "_page_end_idx" not in enriched[1]
    assert "_page_end_idx" not in content[0]
    evidence = PaperEvidence(
        evidence_id="x",
        paper_id="p",
        page=1,
        page_end=2,
        section="",
        modality="text",
        text=text,
    )
    assert locator_matches(EvidenceLocator(page=2, kind="text"), evidence)
    assert not locator_matches(EvidenceLocator(page=3, kind="text"), evidence)


def test_numbered_equation_supports_parentheses_and_latex_tags():
    from app.services.paper_reading import reference_keys

    assert reference_keys("公式（1）和 Eq. (2)") == {
        ("equation", "1"),
        ("equation", "2"),
    }
    assert reference_keys(r"L=x+y \tag{2}") == {("equation", "2")}


def test_annotation_correction_rescores_without_changing_saved_answers():
    import copy
    from scripts.rescore_paper_benchmark import rescore

    original = load_cases(DATASET)
    corrected = [case.model_copy(deep=True) for case in original]
    corrected[0].answer_groups = [["corrected keyword"]]
    answer = response(text="corrected keyword [E1]")
    raw = {
        "metadata": {},
        "observations": [
            {
                "case_id": original[0].id,
                "split": "dev",
                "mode": "text_rag",
                "response": answer.model_dump(),
                "checks": {"keyword_rule_pass": False},
            }
        ],
    }
    before = copy.deepcopy(raw)
    result = rescore(raw, original, corrected)
    assert result["observations"][0]["checks"]["keyword_rule_pass"]
    assert (
        result["observations"][0]["response"] == before["observations"][0]["response"]
    )
    assert raw == before
    corrected[0].question = "A new question"
    with pytest.raises(ValueError, match="changed questions"):
        rescore(raw, original, corrected)
