"""Check paired visual-development runs and bind assistant diagnostics to saved answers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.paper_evaluation import context_hash, load_cases
from app.schemas.papers import PaperReadingResponse
from scripts.benchmark_paper_reading import save_atomic, sha256

MODES = ("text_rag", "multimodal_rag")


def audit(raw: dict, case_ids: set[str]) -> dict:
    rows = {}
    for row in raw["observations"]:
        key = row["case_id"], row["mode"]
        if key in rows:
            raise ValueError("duplicate case/mode")
        rows[key] = row
    if set(rows) != {(case_id, mode) for case_id in case_ids for mode in MODES}:
        raise ValueError("expected exactly one response per case and mode")
    pairs = []
    for case_id in sorted(case_ids):
        responses = [rows[case_id, mode].get("response") for mode in MODES]
        successful = all(responses)
        parsed = [
            PaperReadingResponse.model_validate(response)
            for response in responses
            if response
        ]
        same_context = (
            context_hash(parsed[0]) == context_hash(parsed[1]) if successful else None
        )
        image_control = (
            not parsed[0].vision_used and parsed[1].vision_used if successful else None
        )
        pairs.append(
            {
                "case_id": case_id,
                "both_successful": successful,
                "same_context": same_context,
                "image_control_valid": image_control,
            }
        )
    return {
        "expected_requests": len(case_ids) * 2,
        "successful_requests": sum(bool(row.get("response")) for row in rows.values()),
        "all_pairs_valid": all(
            pair["both_successful"]
            and pair["same_context"]
            and pair["image_control_valid"]
            for pair in pairs
        ),
        "pairs": pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/evaluation/paper_reading_visual_dev.jsonl"),
    )
    parser.add_argument("--notes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = json.loads(args.raw.read_text(encoding="utf-8"))
    notes = json.loads(args.notes.read_text(encoding="utf-8"))
    if raw["metadata"]["dataset_sha256"] != sha256(args.dataset):
        raise ValueError("dataset differs from generation")
    if notes["raw_sha256"] != sha256(args.raw):
        raise ValueError("diagnostic notes refer to another run")
    ids = {case.id for case in load_cases(args.dataset)}
    if (
        len(notes["cases"]) != len(ids)
        or {row["case_id"] for row in notes["cases"]} != ids
    ):
        raise ValueError("diagnostics must cover every case exactly once")
    result = audit(raw, ids)
    result.update(
        raw_sha256=sha256(args.raw),
        dataset_sha256=sha256(args.dataset),
        notes_sha256=sha256(args.notes),
        audit_code_sha256=sha256(Path(__file__)),
        assessment_type="unblinded_assistant_development_diagnostic",
        independent_human_review="pending",
        semantic_accuracy=None,
        citation_support_rate=None,
        cases=notes["cases"],
        scope="Diagnostic notes are assistant judgments, not independent human metrics or held-out generalization results.",
    )
    save_atomic(args.output, result)


if __name__ == "__main__":
    main()
