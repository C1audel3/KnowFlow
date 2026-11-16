"""Fetch pinned official PDFs, audit parser output, and index real papers locally."""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import tempfile

import httpx
from dotenv import load_dotenv
import pypdfium2 as pdfium

from app.paper_evaluation import load_cases, locator_matches
from app.services.paper_library import PaperLibrary
from app.services.paper_reading import index_paper
from app.services.paper_sources import enrich_page_ranges
from examples.simple_multimodal_rag import build_rag

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data/benchmarks/papers/manifest.json"
DEFAULT_DATASET = ROOT / "data/evaluation/paper_reading_real.jsonl"


def verified_pdf(paper: dict, downloads: Path) -> Path:
    path = downloads / f"{paper['key']}.pdf"
    data = path.read_bytes()
    if (
        not data.startswith(b"%PDF-")
        or hashlib.sha256(data).hexdigest() != paper["sha256"]
    ):
        raise ValueError(f"PDF hash mismatch: {paper['key']}")
    return path


def fetch(papers: list[dict], downloads: Path) -> None:
    downloads.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=90, follow_redirects=True) as client:
        for paper in papers:
            path = downloads / f"{paper['key']}.pdf"
            if not path.exists():
                response = client.get(paper["source_url"])
                response.raise_for_status()
                data = response.content
                if (
                    not data.startswith(b"%PDF-")
                    or hashlib.sha256(data).hexdigest() != paper["sha256"]
                ):
                    raise ValueError(
                        f"Downloaded PDF does not match pinned source: {paper['key']}"
                    )
                path.write_bytes(data)
            verified_pdf(paper, downloads)
            print(f"Verified {paper['key']}")


def parsed_content(root: Path, paper_key: str) -> list[dict]:
    path = root / paper_key / "auto" / f"{paper_key}_content_list.json"
    content = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(content, list) or any(
        not isinstance(item, dict) for item in content
    ):
        raise ValueError("expected a flat MinerU content list")
    for item in content:
        value = item.get("img_path")
        if value:
            asset = Path(value)
            asset = (asset if asset.is_absolute() else path.parent / asset).resolve()
            if not asset.is_relative_to(root.resolve()):
                raise ValueError("parsed image escapes the configured output directory")
            item["img_path"] = str(asset)
    return enrich_page_ranges(content, path.with_name(f"{paper_key}_middle.json"))


def audit(
    papers: list[dict], downloads: Path, parsed_root: Path, dataset: Path
) -> dict:
    cases = load_cases(dataset)
    report = {
        "scope": "Parser output and source locator checks, not answer accuracy.",
        "parser": "mineru",
        "parser_version": version("mineru"),
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "papers": [],
    }
    with tempfile.TemporaryDirectory() as temporary:
        library = PaperLibrary(Path(temporary), downloads, parsed_root)
        for paper in papers:
            source = verified_pdf(paper, downloads)
            with pdfium.PdfDocument(source) as document:
                pages = len(document)
            if pages != paper["pages"]:
                raise ValueError("PDF page count changed")
            content = parsed_content(parsed_root, paper["key"])
            rows = library.prepare(paper["key"], content)
            evidence = [library.public(row) for row in rows]
            relevant = [case for case in cases if case.paper_key == paper["key"]]
            matches = {
                case.id: [
                    any(locator_matches(locator, item) for item in evidence)
                    for locator in case.gold_evidence
                ]
                for case in relevant
            }
            observed_pages = {
                item.get("page_idx") + 1
                for item in content
                if type(item.get("page_idx")) is int
            }
            report["papers"].append(
                {
                    "paper_key": paper["key"],
                    "source_sha256": paper["sha256"],
                    "pages": pages,
                    "parsed_pages": sorted(observed_pages),
                    "page_coverage_complete": observed_pages
                    == set(range(1, pages + 1)),
                    "raw_modalities": dict(
                        Counter(item.get("type", "unknown") for item in content)
                    ),
                    "indexed_modalities": dict(
                        Counter(row["modality"] for row in rows)
                    ),
                    "evidence_blocks": len(rows),
                    "registered_assets": sum(bool(row["asset"]) for row in rows),
                    "cross_page_blocks": sum(
                        row.get("page_end") is not None for row in rows
                    ),
                    "unreadable_assets": sum(
                        bool(item.get("img_path"))
                        and not Path(item["img_path"]).is_file()
                        for item in content
                    ),
                    "gold_locator_matches": matches,
                }
            )
    return report


async def ingest(papers: list[dict], args: argparse.Namespace) -> None:
    load_dotenv(ROOT / ".env", override=False)
    args.upload_dir.mkdir(parents=True, exist_ok=True)
    library = PaperLibrary(args.working_dir, args.upload_dir, args.parsed_root)
    rag = build_rag(args.working_dir)
    for paper in papers:
        source = verified_pdf(paper, args.downloads)
        target = args.upload_dir / source.name
        shutil.copyfile(source, target)
        await index_paper(
            library,
            rag,
            paper["key"],
            source.name,
            target,
            parsed_content(args.parsed_root, paper["key"]),
        )
        print(f"Indexed {paper['key']}: {library.get(paper['key']).block_count} blocks")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["fetch", "audit", "ingest"])
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--downloads", type=Path, default=ROOT / "data/benchmarks/papers/downloads"
    )
    parser.add_argument(
        "--parsed-root", type=Path, default=ROOT / "output_papers_benchmark"
    )
    parser.add_argument(
        "--working-dir", type=Path, default=ROOT / "rag_storage_papers_benchmark"
    )
    parser.add_argument(
        "--upload-dir", type=Path, default=ROOT / "data/uploads/papers_benchmark"
    )
    parser.add_argument(
        "--report", type=Path, default=ROOT / "data/experiments/paper_parse_audit.json"
    )
    args = parser.parse_args()
    papers = json.loads(args.manifest.read_text(encoding="utf-8"))["papers"]
    if args.action == "fetch":
        fetch(papers, args.downloads)
    elif args.action == "ingest":
        asyncio.run(ingest(papers, args))
    else:
        report = audit(papers, args.downloads, args.parsed_root, args.dataset)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
