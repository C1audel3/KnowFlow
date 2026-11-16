"""Recover page ranges explicitly marked by MinerU's cross-page spans."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any


def prefix(text: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]", "", text.casefold())[:80]


def enrich_page_ranges(
    content: list[dict[str, Any]], middle_path: Path
) -> list[dict[str, Any]]:
    """Only extend a range when both a matching prefix and cross_page flag exist.

    The flat content list assigns merged paragraphs to their starting page. The
    middle JSON preserves explicit flags on spans carried over from the next page.
    No range is inferred from semantic similarity or paragraph length.
    """
    try:
        payload = json.loads(middle_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return content
    pages = payload.get("pdf_info", []) if isinstance(payload, dict) else []
    if not isinstance(pages, list):
        return content
    cross_page_prefixes: dict[int, set[str]] = {}
    for page_idx, page in enumerate(pages[:-1]):
        if not isinstance(page, dict):
            continue
        for block in page.get("para_blocks", []):
            if not isinstance(block, dict) or block.get("type") != "text":
                continue
            spans = [
                span
                for line in block.get("lines", [])
                for span in line.get("spans", [])
            ]
            if any(span.get("cross_page") is True for span in spans):
                key = prefix(" ".join(str(span.get("content", "")) for span in spans))
                if len(key) >= 30:
                    cross_page_prefixes.setdefault(page_idx, set()).add(key)
    result = []
    for item in content:
        copy = dict(item)
        if item.get("type") == "text" and type(item.get("page_idx")) is int:
            key = prefix(str(item.get("text", "")))
            if key in cross_page_prefixes.get(item["page_idx"], set()):
                copy["_page_end_idx"] = item["page_idx"] + 1
        result.append(copy)
    return result
