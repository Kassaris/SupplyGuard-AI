"""Retrieval benchmark: pgvector vs ChromaDB vs keyword vs hybrid.

Runs every question in benchmarks/queries.jsonl through each search mode and
reports recall@5, MRR, scope leaks and search latency.

Requires the Docker services with the documents indexed, and the embedding
settings in .env (one embedding call per question).

    uv run python benchmarks/run_benchmark.py
"""

import json
import os
import time
from pathlib import Path

from supplyguard_ai.config import get_settings
from supplyguard_ai.domain.models import Role, User
from supplyguard_ai.persistence.db import make_pool
from supplyguard_ai.retrieval.metrics import avg, leaks, recall_at, reciprocal_rank, unique_sources
from supplyguard_ai.retrieval.service import build_postgres_retrieval

ROOT = Path(__file__).resolve().parent.parent
QUERIES = ROOT / "benchmarks" / "queries.jsonl"
REPORT = ROOT / "reports" / "retrieval_benchmark.md"

MODES = ["pgvector", "chroma", "keyword", "hybrid"]
K = 5

# Same scopes as u-pm-01 in support-data/config/users.csv.
USER = User(
    user_id="u-pm-01",
    name="Procurement manager",
    role=Role.PROCUREMENT_MANAGER,
    document_scopes=frozenset({"public", "internal", "procurement"}),
    factory_access=frozenset({"F1", "F2", "F3"}),
)


def load_queries() -> list[dict]:
    lines = QUERIES.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def run_mode(retrieval, mode: str, queries: list[dict]) -> list[dict]:
    results = []
    for q in queries:
        start = time.perf_counter()
        # Fetch more than K so MRR can also credit a hit just below the top K.
        hits = retrieval.search(USER, q["query"], k=K * 2, mode=mode)
        latency_ms = (time.perf_counter() - start) * 1000

        ranked = unique_sources(hits)
        results.append({
            "id": q["id"],
            "type": q["type"],
            "recall": recall_at(ranked, q["relevant"], K),
            "mrr": reciprocal_rank(ranked, q["relevant"]),
            "leaks": len(leaks(hits, USER.document_scopes)),
            "latency_ms": latency_ms,
            "top": ranked[:K],
        })
    return results


def summary_row(mode: str, results: list[dict]) -> str:
    return (
        f"| {mode} | {avg(r['recall'] for r in results)} | {avg(r['mrr'] for r in results)} "
        f"| {sum(r['leaks'] for r in results)} | {avg(r['latency_ms'] for r in results):.0f} |"
    )


def write_report(all_results: dict[str, list[dict]], queries: list[dict]) -> None:
    types = sorted({q["type"] for q in queries if q["relevant"]})
    lines = [
        "# Retrieval benchmark",
        "",
        f"{len(queries)} questions, user `{USER.user_id}` (scopes: {', '.join(sorted(USER.document_scopes))}), "
        f"metrics on the top {K} documents.",
        "",
        f"| Mode | Recall@{K} | MRR | Scope leaks | Latency (ms) |",
        "|---|---|---|---|---|",
        *[summary_row(mode, results) for mode, results in all_results.items()],
        "",
        f"## Recall@{K} by question type",
        "",
        "| Mode | " + " | ".join(types) + " |",
        "|---|" + "---|" * len(types),
    ]
    for mode, results in all_results.items():
        cells = [str(avg(r["recall"] for r in results if r["type"] == t)) for t in types]
        lines.append(f"| {mode} | " + " | ".join(cells) + " |")

    lines += ["", "## Misses (relevant document not in the top 5)", ""]
    relevant = {q["id"]: q["relevant"] for q in queries}
    for mode, results in all_results.items():
        misses = [r["id"] for r in results if r["recall"] is not None and r["recall"] < 1]
        lines.append(f"- **{mode}**: {', '.join(misses) or 'none'}")

    lines += ["", "## Notes", "",
              "- Latency is search time only: query embeddings are computed once before timing,",
              "  so every mode is measured on the same footing.",
              "- `keyword` is Postgres full-text search; `hybrid` fuses ChromaDB and keyword results (RRF).",
              "- Scope leaks must be 0 for every mode."]
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Per-question detail, useful to explain individual misses.
    detail = {mode: [{**r, "relevant": relevant[r["id"]]} for r in results] for mode, results in all_results.items()}
    REPORT.with_suffix(".json").write_text(json.dumps(detail, indent=2), encoding="utf-8")


def main() -> None:
    # Some antivirus tools on Windows set SSLKEYLOGFILE to a path that crashes
    # the database and HTTPS clients. Drop it before any connection is opened.
    os.environ.pop("SSLKEYLOGFILE", None)

    queries = load_queries()
    pool = make_pool(get_settings(), max_size=4)
    try:
        retrieval = build_postgres_retrieval(pool)

        # Embed every question once up front, so latency compares the stores, not the embedding API.
        for q in queries:
            retrieval.embeddings.embed_query(q["query"])

        all_results = {mode: run_mode(retrieval, mode, queries) for mode in MODES}
    finally:
        pool.close()

    write_report(all_results, queries)
    print(REPORT.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
