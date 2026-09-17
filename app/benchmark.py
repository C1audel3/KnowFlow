"""Retrieval and grounded-generation metrics for the phase-7 benchmark."""

from __future__ import annotations

import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.%-]*|[\u4e00-\u9fff]")
EVIDENCE_PATTERN = re.compile(r"\[EVIDENCE:([a-z0-9-]+)\]", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


@dataclass(frozen=True, slots=True)
class EvidenceBlock:
    id: str
    document_id: str
    modality: str
    content: str


class BM25Retriever:
    """Small dependency-free BM25 implementation for a reproducible baseline."""

    def __init__(self, blocks: Sequence[EvidenceBlock], k1: float = 1.5, b: float = 0.75):
        if not blocks:
            raise ValueError("blocks cannot be empty")
        self.blocks = list(blocks)
        self.k1 = k1
        self.b = b
        self.tokens = [tokenize(block.content) for block in blocks]
        self.lengths = [len(tokens) for tokens in self.tokens]
        self.average_length = statistics.fmean(self.lengths)
        document_frequency: Counter[str] = Counter()
        for tokens in self.tokens:
            document_frequency.update(set(tokens))
        count = len(blocks)
        self.idf = {
            token: math.log(1 + (count - frequency + 0.5) / (frequency + 0.5))
            for token, frequency in document_frequency.items()
        }

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        query_tokens = tokenize(query)
        ranked: list[tuple[str, float]] = []
        for block, tokens, length in zip(self.blocks, self.tokens, self.lengths):
            frequencies = Counter(tokens)
            score = 0.0
            for token in query_tokens:
                frequency = frequencies[token]
                if not frequency:
                    continue
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * length / self.average_length
                )
                score += self.idf.get(token, 0.0) * frequency * (self.k1 + 1) / denominator
            ranked.append((block.id, score))
        ranked.sort(key=lambda item: (-item[1], item[0]))
        return ranked[:top_k]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]], *, top_k: int = 5, rank_constant: int = 60
) -> list[tuple[str, float]]:
    scores: defaultdict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, identifier in enumerate(ranking, start=1):
            scores[identifier] += 1 / (rank_constant + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]


def retrieval_metrics(retrieved: Sequence[str], relevant: Iterable[str], k: int = 5) -> dict[str, float]:
    relevant_set = set(relevant)
    if not relevant_set:
        raise ValueError("relevant evidence cannot be empty")
    selected = list(retrieved[:k])
    hits = [1 if identifier in relevant_set else 0 for identifier in selected]
    recall = len(set(selected) & relevant_set) / len(relevant_set)
    precision = sum(hits) / k
    reciprocal_rank = next((1 / rank for rank, hit in enumerate(hits, 1) if hit), 0.0)
    dcg = sum(hit / math.log2(rank + 1) for rank, hit in enumerate(hits, 1))
    ideal_hits = min(len(relevant_set), k)
    ideal_dcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return {
        f"recall@{k}": round(recall, 6),
        f"precision@{k}": round(precision, 6),
        "mrr": round(reciprocal_rank, 6),
        f"ndcg@{k}": round(dcg / ideal_dcg, 6),
    }


def aggregate_metrics(rows: Sequence[dict[str, float]]) -> dict[str, float]:
    if not rows:
        raise ValueError("metric rows cannot be empty")
    return {
        key: round(statistics.fmean(row[key] for row in rows), 6)
        for key in rows[0]
    }


def extract_evidence_ids(text: str) -> list[str]:
    return list(dict.fromkeys(match.lower() for match in EVIDENCE_PATTERN.findall(text)))


def citation_metrics(answer: str, relevant: Iterable[str]) -> dict[str, float]:
    cited = extract_evidence_ids(answer)
    relevant_set = set(relevant)
    correct = set(cited) & relevant_set
    return {
        "citation_precision": round(len(correct) / len(cited), 6) if cited else 0.0,
        "citation_recall": round(len(correct) / len(relevant_set), 6),
    }
