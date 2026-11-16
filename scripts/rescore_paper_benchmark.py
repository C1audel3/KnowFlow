"""Re-score saved answers after source-verified annotation corrections, without inference."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from app.paper_evaluation import PaperCase, automatic_checks, load_cases, summarize
from app.schemas.papers import PaperReadingResponse
from scripts.benchmark_paper_reading import save_atomic, write_review

ROOT = Path(__file__).resolve().parents[1]
STABLE_FIELDS = (
    "id",
    "paper_key",
    "split",
    "group_id",
    "category",
    "question",
    "visual_probe",
)


def rescore(raw: dict, original: list[PaperCase], corrected: list[PaperCase]) -> dict:
    before = {case.id: case for case in original}
    after = {case.id: case for case in corrected}
    if set(before) != set(after):
        raise ValueError("correction must preserve the same case IDs")
    changes = []
    for key in before:
        old, new = before[key].model_dump(), after[key].model_dump()
        if any(old[field] != new[field] for field in STABLE_FIELDS):
            raise ValueError(
                "cannot reuse generated answers for changed questions or splits"
            )
        fields = [field for field in old if old[field] != new[field]]
        if fields:
            changes.append({"case_id": key, "changed_annotation_fields": fields})
    result = copy.deepcopy(raw)
    for row in result["observations"]:
        if row["case_id"] not in after:
            raise ValueError("saved response refers to an unknown case")
        if row.get("response"):
            row["checks"] = automatic_checks(
                after[row["case_id"]],
                PaperReadingResponse.model_validate(row["response"]),
            )
    result["annotation_corrections"] = changes
    result["summary"] = summarize(result["observations"])
    result["failed_requests"] = [
        {
            "case_id": row["case_id"],
            "mode": row["mode"],
            "error_code": row.get("error_code"),
        }
        for row in result["observations"]
        if not row.get("response")
    ]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument(
        "--original-dataset",
        type=Path,
        default=ROOT / "data/evaluation/paper_reading_real_frozen_v1.jsonl",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "data/evaluation/paper_reading_real.jsonl",
    )
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    raw = json.loads(args.raw.read_text(encoding="utf-8"))
    original_hash = hashlib.sha256(args.original_dataset.read_bytes()).hexdigest()
    if raw["metadata"]["dataset_sha256"] != original_hash:
        raise ValueError(
            "the original dataset does not match the generation checkpoint"
        )
    corrected = load_cases(args.dataset)
    result = rescore(raw, load_cases(args.original_dataset), corrected)
    result["rescoring_metadata"] = {
        "original_dataset_sha256": original_hash,
        "corrected_dataset_sha256": hashlib.sha256(
            args.dataset.read_bytes()
        ).hexdigest(),
        "scoring_code_sha256": hashlib.sha256(
            (ROOT / "app/paper_evaluation.py").read_bytes()
        ).hexdigest(),
        "rescore_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "generation_repeated": False,
    }
    output = args.raw.with_name(args.raw.stem + "_reviewed.json")
    save_atomic(output, result)
    write_review(
        output.with_suffix(".review.csv"),
        result["observations"],
        {case.id: case for case in corrected},
    )
    save_atomic(
        args.summary,
        {key: value for key, value in result.items() if key != "observations"},
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
