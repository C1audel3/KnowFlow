import copy

import pytest

from scripts.audit_paper_visual import audit


def paired():
    response = {
        "answer": "Answer",
        "evidence": [],
        "citations": [],
        "status": "uncited",
        "duration_ms": 1,
        "vision_used": False,
        "retrieval": "figure_lookup",
    }
    return {
        "observations": [
            {
                "case_id": "a",
                "mode": mode,
                "response": {**response, "vision_used": image},
            }
            for mode, image in (("text_rag", False), ("multimodal_rag", True))
        ]
    }


def test_pair_audit_recomputes_context_instead_of_trusting_saved_checks():
    raw = paired()
    assert audit(raw, {"a"})["all_pairs_valid"]
    raw["observations"][1]["response"]["evidence"] = [
        {
            "evidence_id": "b",
            "paper_id": "paper",
            "page": 1,
            "section": "section",
            "modality": "text",
            "text": "different",
        }
    ]
    assert audit(raw, {"a"})["pairs"][0]["same_context"] is False


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "extra"])
def test_pair_audit_rejects_incomplete_or_duplicate_experiments(mutation):
    raw = paired()
    if mutation == "missing":
        raw["observations"].pop()
    elif mutation == "duplicate":
        raw["observations"].append(copy.deepcopy(raw["observations"][0]))
    else:
        raw["observations"].append(
            {"case_id": "b", "mode": "text_rag", "response": None}
        )
    with pytest.raises(ValueError):
        audit(raw, {"a"})


def test_pair_audit_keeps_failed_and_missing_image_pairs_visible():
    raw = paired()
    raw["observations"][1]["response"]["vision_used"] = False
    assert audit(raw, {"a"})["pairs"][0]["image_control_valid"] is False
    raw["observations"][1]["response"] = None
    result = audit(raw, {"a"})
    assert result["successful_requests"] == 1
    assert result["pairs"][0]["same_context"] is None
    assert result["all_pairs_valid"] is False
