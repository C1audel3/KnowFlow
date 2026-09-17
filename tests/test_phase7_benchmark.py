from __future__ import annotations

from app.benchmark import (
    BM25Retriever,
    EvidenceBlock,
    citation_metrics,
    reciprocal_rank_fusion,
    retrieval_metrics,
)


def test_bm25_prefers_matching_evidence():
    blocks = [
        EvidenceBlock("a", "doc-a", "text", "Atlas owner Lin Qiao"),
        EvidenceBlock("b", "doc-b", "table", "Sensor latency 18 ms"),
    ]
    assert BM25Retriever(blocks).search("Atlas owner", 1)[0][0] == "a"


def test_retrieval_metrics_support_multiple_relevant_blocks():
    metrics = retrieval_metrics(["a", "x", "b"], {"a", "b"}, 3)
    assert metrics["recall@3"] == 1
    assert metrics["precision@3"] == 0.666667
    assert metrics["mrr"] == 1
    assert metrics["ndcg@3"] > 0.9


def test_rrf_and_citation_metrics():
    assert reciprocal_rank_fusion([["a", "b"], ["b", "a"]], top_k=2) == [
        ("a", 1 / 61 + 1 / 62),
        ("b", 1 / 61 + 1 / 62),
    ]
    assert citation_metrics("答案 [EVIDENCE:a] [EVIDENCE:x]", {"a", "b"}) == {
        "citation_precision": 0.5,
        "citation_recall": 0.5,
    }
