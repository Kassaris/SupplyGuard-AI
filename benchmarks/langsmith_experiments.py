"""Upload the retrieval benchmark to LangSmith and run one experiment per mode.

Creates the dataset `supplyguard-retrieval-benchmark` from queries.jsonl (once),
then runs pgvector, chroma, keyword and hybrid as separate experiments so they
can be compared side by side in the LangSmith UI.

Requires the same services as run_benchmark.py plus LANGSMITH_* in .env.

    uv run python benchmarks/langsmith_experiments.py
"""

import os

import truststore
from dotenv import load_dotenv
from langsmith import Client, evaluate

from run_benchmark import K, MODES, USER, load_queries
from supplyguard_ai.config import get_settings
from supplyguard_ai.persistence.db import make_pool
from supplyguard_ai.retrieval.metrics import leaks, recall_at, reciprocal_rank, unique_sources
from supplyguard_ai.retrieval.service import build_postgres_retrieval

DATASET = "supplyguard-retrieval-benchmark"


def ensure_dataset(client: Client) -> None:
    """Create the dataset on the first run. If queries.jsonl changes, rename DATASET."""
    if client.has_dataset(dataset_name=DATASET):
        return
    queries = load_queries()
    dataset = client.create_dataset(DATASET, description="Questions with the documents that answer them.")
    client.create_examples(
        dataset_id=dataset.id,
        inputs=[{"query": q["query"]} for q in queries],
        outputs=[{"relevant": q["relevant"]} for q in queries],
        metadata=[{"id": q["id"], "type": q["type"]} for q in queries],
    )


# --- evaluators: (run, example) -> score, as in the LangSmith guide ---

def recall_at_k(run, example) -> dict:
    return {"key": f"recall@{K}", "score": recall_at(run.outputs["sources"], example.outputs["relevant"], K)}


def mrr(run, example) -> dict:
    return {"key": "mrr", "score": reciprocal_rank(run.outputs["sources"], example.outputs["relevant"])}


def no_leaks(run, example) -> dict:
    return {"key": "no_scope_leaks", "score": 1.0 if run.outputs["leaks"] == 0 else 0.0}


def main() -> None:
    os.environ.pop("SSLKEYLOGFILE", None)  # see run_benchmark.py
    # Trust the OS certificate store, as in scripts/check_langsmith.py (HTTPS inspection by antivirus/proxies).
    truststore.inject_into_ssl()
    load_dotenv()  # LangSmith reads LANGSMITH_* from the environment

    client = Client()
    ensure_dataset(client)

    pool = make_pool(get_settings(), max_size=4)
    try:
        retrieval = build_postgres_retrieval(pool)
        for mode in MODES:
            def search(inputs: dict, mode=mode) -> dict:
                hits = retrieval.search(USER, inputs["query"], k=K * 2, mode=mode)
                return {"sources": unique_sources(hits), "leaks": len(leaks(hits, USER.document_scopes))}

            evaluate(
                search,
                data=DATASET,
                evaluators=[recall_at_k, mrr, no_leaks],
                experiment_prefix=f"retrieval-{mode}",
                metadata={"mode": mode, "k": K, "user": USER.user_id},
                client=client,
            )
    finally:
        pool.close()


if __name__ == "__main__":
    main()
