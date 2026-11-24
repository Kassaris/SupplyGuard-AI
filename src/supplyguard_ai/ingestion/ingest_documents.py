"""
Index documents into pgvector + ChromaDB, and approved memories into pgvector.

Run by the `doc-ingest` service in docker-compose.yml:
    python -m supplyguard_ai.ingestion.ingest_documents [--rebuild]

STEPS
-----
1. Load the 212 documents (DocumentService) and the 40 approved memories.
2. Chunk the documents (ingestion/chunking.py) -> 272 chunks with provenance.
3. EMBED ONCE: both indexes use the same embedding model, so each chunk is
   embedded a single time and the vectors are written to pgvector AND Chroma
   (`add_embeddings` / `collection.upsert`).  Half the API cost and time,
   and the two indexes are guaranteed to hold identical vectors - which makes
   the pgvector vs. Chroma benchmark a fair comparison of the *stores*.
4. Upsert by deterministic id -> re-running never duplicates anything.

`--rebuild` drops and recreates the tables/collection (e.g. after changing
the chunk size).  Without it, ingestion is an idempotent upsert.

Retry: the compose job retries the whole script up to 10 times (e.g. Chroma
still starting).  Embedding calls retry inside the client (max_retries=3).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from langchain_core.embeddings import Embeddings

from supplyguard_ai.config import Settings, get_settings
from supplyguard_ai.ingestion.chunking import chunk_all, memory_document
from supplyguard_ai.persistence.db import ensure_app_schema, make_pool
from supplyguard_ai.retrieval import stores
from supplyguard_ai.services.documents import DocumentService
from supplyguard_ai.services.repository import CsvRepository

log = logging.getLogger("ingest_documents")
BATCH = 64


def _batches(items: list, size: int = BATCH):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def ingest(settings: Settings, embeddings: Embeddings, pool, engine, chroma, rebuild: bool = False,
           names: dict[str, str] | None = None) -> dict[str, int]:
    """Testable core (main() only wires real dependencies).

    `names` overrides table / collection names - integration tests use
    throw-away names so they never touch the real, Azure-embedded index.
    """
    n = {"documents": stores.DOCUMENTS_TABLE, "memories": stores.MEMORIES_TABLE,
         "collection": stores.CHROMA_COLLECTION} | (names or {})
    stores.ensure_vector_tables(engine, pool, overwrite=rebuild, documents_table=n["documents"], memories_table=n["memories"])
    if rebuild:
        try:
            chroma.delete_collection(n["collection"])
        except Exception:  # noqa: BLE001 - collection may not exist yet
            pass

    pg_docs = stores.documents_pg_store(engine, embeddings, n["documents"])
    pg_mem = stores.memories_pg_store(engine, embeddings, n["memories"])
    collection = chroma.get_or_create_collection(n["collection"], metadata={"hnsw:space": "cosine"})

    # ---- documents: embed once, write twice
    chunks = chunk_all(DocumentService(settings.support_data_dir).all_documents())
    done = 0
    for batch in _batches(chunks):
        texts = [c.page_content for c in batch]
        ids = [c.id for c in batch]
        metas = [c.metadata for c in batch]
        vectors = embeddings.embed_documents(texts)
        pg_docs.add_embeddings(texts, vectors, metadatas=metas, ids=ids)
        collection.upsert(ids=ids, embeddings=vectors, metadatas=metas, documents=texts)
        done += len(batch)
        log.info("indexed %d/%d chunks", done, len(chunks))

    # ---- approved incident memories: pgvector only (separate table)
    mems = [memory_document(m) for m in CsvRepository(settings.support_data_dir).approved_memories()]
    for batch in _batches(mems):
        pg_mem.add_documents(batch, ids=[m.id for m in batch])

    return {"documents": len({c.metadata['source_id'] for c in chunks}), "chunks": len(chunks),
            "chroma_count": collection.count(), "memories": len(mems)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--rebuild", action="store_true", help="drop and recreate the indexes first")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    from supplyguard_ai.llm import get_embeddings

    settings = get_settings()
    pool = make_pool(settings, max_size=2)
    t = time.time()
    try:
        ensure_app_schema(pool)
        counts = ingest(settings, get_embeddings(), pool, stores.get_engine(), stores.chroma_client(settings), args.rebuild)
    finally:
        pool.close()
    print(f"ingest_documents: {counts} in {time.time() - t:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
