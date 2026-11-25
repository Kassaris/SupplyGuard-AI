"""
Retrieval metrics (pure functions - unit-tested in tests/test_benchmark_metrics.py).

All metrics work on DOCUMENTS, not chunks: a contract split into 2 chunks is
one answer.  `ranked` = unique source_ids in result order.

  recall@k  share of the relevant documents found in the top k.
            "Did the user get what they need on the first screen?"
  MRR       1 / rank of the first relevant document (0 if none in the list).
            "How far down is the first useful answer?"
  leaks     results the asking user is NOT allowed to see.  Must be 0 -
            a security property, reported for every query.
"""

from __future__ import annotations

from statistics import mean


def unique_sources(hits) -> list[str]:
    """Chunk hits -> document ranking (first occurrence wins)."""
    seen, out = set(), []
    for h in hits:
        if h.source_id not in seen:
            seen.add(h.source_id)
            out.append(h.source_id)
    return out


def recall_at(ranked: list[str], relevant: list[str], k: int) -> float | None:
    """None when there is nothing to find (pure scope queries): such queries are
    excluded from recall averages instead of counting as perfect or as zero."""
    if not relevant:
        return None
    return len(set(ranked[:k]) & set(relevant)) / len(relevant)


def reciprocal_rank(ranked: list[str], relevant: list[str]) -> float | None:
    if not relevant:
        return None
    for i, sid in enumerate(ranked, start=1):
        if sid in relevant:
            return 1.0 / i
    return 0.0


def leaks(hits, allowed_scopes: set[str]) -> list[str]:
    return [h.source_id for h in hits if h.access_scope not in allowed_scopes]


def avg(values) -> float | None:
    vals = [v for v in values if v is not None]
    return round(mean(vals), 3) if vals else None
