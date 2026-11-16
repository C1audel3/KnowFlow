"""Source locators and conservative automatic metrics for real-paper experiments."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.papers import PaperEvidence, PaperReadingResponse
from app.services.paper_reading import reference_keys


class EvidenceLocator(BaseModel):
    page: int = Field(ge=1)
    kind: Literal["text", "figure", "table", "equation"]
    terms_any: list[str] = Field(default_factory=list)
    reference: str | None = None


class PaperCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    paper_key: str
    split: Literal["dev", "test"]
    group_id: str
    category: Literal["method", "figure", "table", "equation", "limitation", "refusal"]
    question: str
    reference_answer: str
    answer_groups: list[list[str]] = Field(min_length=1)
    gold_evidence: list[EvidenceLocator]
    visual_probe: bool
    review_status: str

    @model_validator(mode="after")
    def evidence_required(self) -> "PaperCase":
        if self.category != "refusal" and not self.gold_evidence:
            raise ValueError("answerable cases need source locators")
        if any(
            not group or any(not word.strip() for word in group)
            for group in self.answer_groups
        ):
            raise ValueError("answer groups cannot be empty")
        return self


def load_cases(path: Path) -> list[PaperCase]:
    cases = [
        PaperCase.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not cases or len({case.id for case in cases}) != len(cases):
        raise ValueError("dataset is empty or contains duplicate IDs")
    splits: dict[str, str] = {}
    for case in cases:
        if case.group_id in splits and splits[case.group_id] != case.split:
            raise ValueError(
                "the same evidence group cannot cross development/test splits"
            )
        splits[case.group_id] = case.split
    return cases


def normalized(text: str) -> str:
    return "".join(text.casefold().split())


def locator_matches(locator: EvidenceLocator, evidence: PaperEvidence) -> bool:
    kinds = {
        "text": {"text", "list"},
        "figure": {"image", "chart"},
        "table": {"table"},
        "equation": {"equation"},
    }
    if (
        evidence.page is None
        or not evidence.page <= locator.page <= (evidence.page_end or evidence.page)
        or evidence.modality not in kinds[locator.kind]
    ):
        return False
    if locator.reference:
        kind, number = locator.reference.split(":", 1)
        if (kind, number) not in reference_keys(evidence.text):
            return False
    return not locator.terms_any or any(
        normalized(term) in normalized(evidence.text) for term in locator.terms_any
    )


def context_hash(response: PaperReadingResponse) -> str:
    content = [item.model_dump() for item in response.evidence]
    return hashlib.sha256(
        json.dumps(content, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def automatic_checks(case: PaperCase, response: PaperReadingResponse) -> dict:
    text = normalized(response.answer)
    group_hits = [
        any(normalized(word) in text for word in group) for group in case.answer_groups
    ]
    located = [
        any(locator_matches(locator, item) for item in response.evidence)
        for locator in case.gold_evidence
    ]
    cited = [
        item
        for index, item in enumerate(response.evidence, 1)
        if f"E{index}" in response.citations
    ]
    cited_hits = [
        any(locator_matches(locator, item) for item in cited)
        for locator in case.gold_evidence
    ]
    return {
        "keyword_rule_pass": all(group_hits),
        "keyword_group_hits": group_hits,
        "locator_recall": sum(located) / len(located) if located else None,
        "cited_locator_recall": sum(cited_hits) / len(cited_hits)
        if cited_hits
        else None,
        "has_valid_citation": bool(response.citations),
        "context_hash": context_hash(response),
    }


def summarize(observations: list[dict]) -> dict:
    grouped = defaultdict(list)
    for row in observations:
        grouped[(row["split"], row["mode"])].append(row)
    summary = {}
    for (split, mode), rows in sorted(grouped.items()):
        success = [row for row in rows if row.get("response")]
        locators = [
            row["checks"]["locator_recall"]
            for row in success
            if row["checks"]["locator_recall"] is not None
        ]
        summary[f"{split}:{mode}"] = {
            "requests": len(rows),
            "successful": len(success),
            "keyword_rule_pass_rate": sum(
                row["checks"]["keyword_rule_pass"] for row in success
            )
            / len(rows),
            "locator_recall_mean": sum(locators) / len(locators) if locators else None,
            "vision_used_requests": sum(
                row["response"]["vision_used"] for row in success
            ),
            "mean_duration_ms": sum(row["response"]["duration_ms"] for row in success)
            / len(success)
            if success
            else None,
            "semantic_accuracy": None,
            "citation_support_rate": None,
            "manual_review": "pending",
        }
    pairs = defaultdict(dict)
    for row in observations:
        if row.get("response"):
            pairs[row["case_id"]][row["mode"]] = row
    comparable = []
    for case_id, pair in pairs.items():
        if "text_rag" in pair and "multimodal_rag" in pair:
            comparable.append(
                {
                    "case_id": case_id,
                    "same_context": pair["text_rag"]["checks"]["context_hash"]
                    == pair["multimodal_rag"]["checks"]["context_hash"],
                }
            )
    return {
        "by_split_and_mode": summary,
        "context_pair_checks": comparable,
        "scope": "Keyword rules and page/type source locators only; human semantic and citation-support review required.",
    }
