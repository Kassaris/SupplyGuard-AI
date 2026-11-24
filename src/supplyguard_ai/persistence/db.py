"""
Database connection pool + application schema.

WHY A CONNECTION POOL (psycopg_pool)
------------------------------------
Opening a Postgres connection costs ~10-50 ms.  Chainlit serves several users,
LangGraph writes a checkpoint after every node, and the services read
operational data constantly.  A pool keeps a few connections open and lends
them out (`with pool.connection() as conn:`), returning them afterwards.

WHY psycopg 3
-------------
It is what `langgraph-checkpoint-postgres` uses, so the checkpointer and our
own stores can share ONE pool.  `dict_row` returns rows as dicts, which is
exactly the `Row` shape our CSV repository already produced - so business
services don't notice which backend they run on.

SCHEMA OWNERSHIP
----------------
docker/postgres/001_extensions.sql  -> extension + schemas (needs superuser, runs once)
APP_SCHEMA_SQL below                -> app tables, idempotent, run at start-up
ingestion/load_structured.py        -> ops.* tables (truncate + reload)
PostgresSaver.setup()               -> LangGraph checkpoint tables
"""

from __future__ import annotations

from functools import lru_cache

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from supplyguard_ai.config import Settings, get_settings

APP_SCHEMA_SQL = """
CREATE SCHEMA IF NOT EXISTS ops;
CREATE SCHEMA IF NOT EXISTS app;
CREATE SCHEMA IF NOT EXISTS vec;

-- Users and roles (seeded from support-data/config/users.csv).
CREATE TABLE IF NOT EXISTS app.users (
    user_id          text PRIMARY KEY,
    name             text NOT NULL,
    role             text NOT NULL,
    document_scopes  text[] NOT NULL,
    factory_access   text[] NOT NULL,
    password_hash    text            -- set by the UI login setup; NULL = cannot log in
);

-- Payload-bound approvals. The full ApprovalRecord is kept as JSONB (it is
-- read and written as a whole); hot fields are columns for querying.
CREATE TABLE IF NOT EXISTS app.approvals (
    approval_id   text PRIMARY KEY,
    case_id       text NOT NULL,
    status        text NOT NULL,
    payload_hash  text NOT NULL,
    record        jsonb NOT NULL,
    updated_at    timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS approvals_case_idx ON app.approvals (case_id);

-- Append-only audit trail. No UPDATE/DELETE in the code base.
CREATE TABLE IF NOT EXISTS app.audit_events (
    id          bigserial PRIMARY KEY,
    event_type  text NOT NULL,
    case_id     text NOT NULL,
    actor       text NOT NULL,
    details     jsonb NOT NULL,
    at          timestamptz NOT NULL
);
CREATE INDEX IF NOT EXISTS audit_case_idx ON app.audit_events (case_id, id);

-- Procurement ledger = idempotency table. One row per executed write.
-- The PRIMARY KEY on idempotency_key is what makes a replay a no-op: a second
-- INSERT with the same key cannot succeed, even under a race.
CREATE TABLE IF NOT EXISTS app.procurement_executions (
    idempotency_key  text PRIMARY KEY,
    case_id          text NOT NULL,
    approval_id      text NOT NULL,
    operation        text NOT NULL,
    payload          jsonb NOT NULL,
    reference        text NOT NULL,
    cost_eur         double precision NOT NULL,
    result           jsonb NOT NULL,
    executed_at      timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS proc_case_idx ON app.procurement_executions (case_id);

-- Case registry: who opened which case, its phase - for "resume the case I
-- was reviewing yesterday" and for approvers' to-do lists. The case STATE
-- itself lives in the LangGraph checkpoints (thread_id = case_id).
CREATE TABLE IF NOT EXISTS app.cases (
    case_id     text PRIMARY KEY,
    created_by  text NOT NULL,
    title       text NOT NULL,
    phase       text NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now(),
    updated_at  timestamptz NOT NULL DEFAULT now()
);

-- users who acted in a case (clarified, signed, rejected); added later -> idempotent migration
ALTER TABLE app.cases ADD COLUMN IF NOT EXISTS participants text[] NOT NULL DEFAULT '{}';

-- New PO numbers for the simulated procurement system.
CREATE SEQUENCE IF NOT EXISTS app.po_number_seq START 9100;
"""


def make_pool(settings: Settings | None = None, min_size: int = 1, max_size: int = 10) -> ConnectionPool:
    """Create a pool.  `autocommit=True` is required by PostgresSaver and keeps
    our single-statement writes simple; multi-statement writes use explicit
    `with conn.transaction():` blocks."""
    s = settings or get_settings()
    pool = ConnectionPool(
        conninfo=s.postgres_dsn,
        min_size=min_size,
        max_size=max_size,
        kwargs={"autocommit": True, "row_factory": dict_row, "prepare_threshold": 0},
        open=True,
    )
    pool.wait(timeout=30)  # fail fast with a clear error if the DB is not reachable
    return pool


@lru_cache(maxsize=1)
def get_pool() -> ConnectionPool:
    """Process-wide pool (one per app process)."""
    return make_pool()


def ensure_app_schema(pool: ConnectionPool) -> None:
    """Create app tables if missing.  Safe to call on every start-up."""
    with pool.connection() as conn:
        # prepare=False: the pool uses prepare_threshold=0 (as LangGraph's
        # PostgresSaver recommends), and a multi-statement script cannot be a
        # prepared statement.
        conn.execute(APP_SCHEMA_SQL, prepare=False)
