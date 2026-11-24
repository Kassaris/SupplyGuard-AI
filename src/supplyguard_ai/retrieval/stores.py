"""
Vector store factories: where the embeddings live.

ROLES (assignment §05)
----------------------
  pgvector  vec.documents          baseline document retrieval
            vec.incident_memories  semantic search over approved memories (separate table)
  ChromaDB  collection "documents" alternative index: retrieval experiments + hybrid path

WHY langchain_postgres.PGVectorStore
------------------------------------
* LangChain `VectorStore` interface: similarity search, scores, MMR,
  `as_retriever()` - same API as Chroma, so the benchmark can swap them.
* METADATA COLUMNS: access_scope, doc_type, supplier_id ... become real SQL
  columns (not only JSON), so the scope filter is a plain indexed WHERE.
* Table per collection in our own schema `vec` ("separate collections or
  tables" for documents vs. memories).

WHY THE asyncpg DRIVER
----------------------
PGVectorStore runs on SQLAlchemy's async engine.  psycopg's async mode
cannot run on Windows' default ProactorEventLoop; asyncpg can.  Our own
sync code keeps using the psycopg pool (persistence/db.py).

WHY chromadb.HttpClient
-----------------------
Chroma runs as its own server in docker compose (port 8001 on the host).
The client package version (1.5.9) matches the server image exactly.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_postgres import Column, PGEngine, PGVectorStore

from supplyguard_ai.config import Settings, get_settings
from supplyguard_ai.ingestion.chunking import MEMORY_FIELDS, METADATA_FIELDS
from supplyguard_ai.llm import EMBEDDING_DIM

SCHEMA = "vec"
DOCUMENTS_TABLE = "documents"
MEMORIES_TABLE = "incident_memories"
CHROMA_COLLECTION = "documents"

_TYPES = {"version": "INTEGER", "chunk_no": "INTEGER", "contains_instructions": "BOOLEAN"}


# --------------------------------------------------------------------------- postgres
def pg_url(s: Settings) -> str:
    return (f"postgresql+asyncpg://{s.postgres_user}:{s.postgres_password.get_secret_value()}"
            f"@{s.postgres_host}:{s.postgres_port}/{s.postgres_db}")


@lru_cache(maxsize=1)
def get_engine() -> PGEngine:
    """One engine per process (it owns a background event loop + connection pool)."""
    return PGEngine.from_connection_string(pg_url(get_settings()))


def _table_exists(pool, table: str) -> bool:
    with pool.connection() as conn:
        row = conn.execute("SELECT to_regclass(%s) AS t", (f"{SCHEMA}.{table}",)).fetchone()
    return row["t"] is not None


def ensure_vector_tables(engine: PGEngine, pool, overwrite: bool = False,
                         documents_table: str = DOCUMENTS_TABLE, memories_table: str = MEMORIES_TABLE) -> None:
    """Create vec.documents and vec.incident_memories if missing.

    id column is TEXT (default would be UUID) so chunk ids stay readable and
    deterministic: 'DOC-CON-S004-v1#0'.
    """
    specs = {
        documents_table: [Column(f, _TYPES.get(f, "TEXT")) for f in METADATA_FIELDS],
        memories_table: [Column(f, "TEXT") for f in MEMORY_FIELDS],
    }
    for table, cols in specs.items():
        if overwrite or not _table_exists(pool, table):
            engine.init_vectorstore_table(
                table, EMBEDDING_DIM, schema_name=SCHEMA, id_column=Column("id", "TEXT"),
                metadata_columns=cols, overwrite_existing=overwrite,
            )
    with pool.connection() as conn:
        # Full-text index for the KEYWORD backend (hybrid search): exact terms
        # like "PO-2026-9002" or "ISO 9001" that embeddings blur.
        # table names are code constants / test names, never user input
        conn.execute(f"CREATE INDEX IF NOT EXISTS {documents_table}_fts_idx ON {SCHEMA}.{documents_table} "
                     "USING gin (to_tsvector('english', content))")
        conn.execute(f"CREATE INDEX IF NOT EXISTS {documents_table}_scope_idx ON {SCHEMA}.{documents_table} (access_scope)")


def documents_pg_store(engine: PGEngine, embeddings: Embeddings, table: str = DOCUMENTS_TABLE) -> PGVectorStore:
    return PGVectorStore.create_sync(engine, embeddings, table, schema_name=SCHEMA,
                                     id_column="id", metadata_columns=list(METADATA_FIELDS))


def memories_pg_store(engine: PGEngine, embeddings: Embeddings, table: str = MEMORIES_TABLE) -> PGVectorStore:
    return PGVectorStore.create_sync(engine, embeddings, table, schema_name=SCHEMA,
                                     id_column="id", metadata_columns=list(MEMORY_FIELDS))


# --------------------------------------------------------------------------- chroma
def chroma_client(s: Settings | None = None):
    import chromadb

    s = s or get_settings()
    return chromadb.HttpClient(host=s.chroma_host, port=s.chroma_port)


def documents_chroma_store(client, embeddings: Embeddings, collection: str = CHROMA_COLLECTION):
    from langchain_chroma import Chroma

    # cosine distance, same as pgvector, so the two indexes are comparable in the benchmark
    return Chroma(collection_name=collection, embedding_function=embeddings, client=client,
                  collection_metadata={"hnsw:space": "cosine"})
