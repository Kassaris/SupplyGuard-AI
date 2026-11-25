"""
Durable Postgres implementations of the service-layer storage protocols.

  PostgresAuditLog       -> AuditStore         (app.audit_events, append-only)
  PostgresApprovalStore  -> ApprovalStore      (app.approvals)
  PostgresLedger         -> ProcurementLedger  (app.procurement_executions)

Each class mirrors its in-memory twin method by method, so the same unit
tests describe both; tests/test_postgres.py runs the Postgres versions
against the real database.

SQL SAFETY: every value is passed as a parameter (%s / %(name)s), never
formatted into the SQL string -> no SQL injection, even from LLM-produced
payloads.  `Jsonb(...)` adapts Python dicts to JSONB.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from supplyguard_ai.domain.models import ApprovalRecord, ExecutionResult, ProposedAction
from supplyguard_ai.services.audit import AuditEvent


# --------------------------------------------------------------------------- audit
class PostgresAuditLog:
    """Append-only: this class has no update or delete method on purpose."""

    def __init__(self, pool: ConnectionPool) -> None:
        self.pool = pool

    def record(self, event_type: str, case_id: str, actor: str, details: dict[str, Any] | None = None) -> AuditEvent:
        ev = AuditEvent(event_type, case_id, actor, dict(details or {}), datetime.now(timezone.utc))
        with self.pool.connection() as conn:
            conn.execute(
                "INSERT INTO app.audit_events (event_type, case_id, actor, details, at) VALUES (%s, %s, %s, %s, %s)",
                # default=str: details may contain dates/enums; store them as text
                (ev.event_type, ev.case_id, ev.actor, Jsonb(ev.details, dumps=_dumps), ev.at),
            )
        return ev

    def events(self, case_id: str | None = None) -> list[AuditEvent]:
        q = "SELECT event_type, case_id, actor, details, at FROM app.audit_events"
        params: tuple = ()
        if case_id is not None:
            q, params = q + " WHERE case_id = %s", (case_id,)
        with self.pool.connection() as conn:
            rows = conn.execute(q + " ORDER BY id", params).fetchall()
        return [AuditEvent(r["event_type"], r["case_id"], r["actor"], r["details"], r["at"]) for r in rows]


def _dumps(obj: Any) -> str:
    import json

    return json.dumps(obj, default=str)


# --------------------------------------------------------------------------- approvals
class PostgresApprovalStore:
    def __init__(self, pool: ConnectionPool) -> None:
        self.pool = pool

    def get(self, approval_id: str) -> ApprovalRecord | None:
        with self.pool.connection() as conn:
            row = conn.execute("SELECT record FROM app.approvals WHERE approval_id = %s", (approval_id,)).fetchone()
        return ApprovalRecord.model_validate(row["record"]) if row else None

    def save(self, rec: ApprovalRecord) -> None:
        """Upsert the whole record (it is always read and written as a unit)."""
        with self.pool.connection() as conn:
            conn.execute(
                """INSERT INTO app.approvals (approval_id, case_id, status, payload_hash, record, updated_at)
                   VALUES (%s, %s, %s, %s, %s, now())
                   ON CONFLICT (approval_id) DO UPDATE
                      SET status = EXCLUDED.status, record = EXCLUDED.record, updated_at = now()""",
                (rec.approval_id, rec.case_id, rec.status.value, rec.payload_hash, Jsonb(rec.model_dump(mode="json"))),
            )

    def for_case(self, case_id: str) -> list[ApprovalRecord]:
        with self.pool.connection() as conn:
            rows = conn.execute("SELECT record FROM app.approvals WHERE case_id = %s ORDER BY updated_at", (case_id,)).fetchall()
        return [ApprovalRecord.model_validate(r["record"]) for r in rows]


# --------------------------------------------------------------------------- procurement ledger
class PostgresLedger:
    def __init__(self, pool: ConnectionPool) -> None:
        self.pool = pool

    def get(self, key: str) -> ExecutionResult | None:
        with self.pool.connection() as conn:
            row = conn.execute("SELECT result FROM app.procurement_executions WHERE idempotency_key = %s", (key,)).fetchone()
        return ExecutionResult.model_validate(row["result"]) if row else None

    def record(self, key: str, case_id: str, approval_id: str, action: ProposedAction, result: ExecutionResult) -> bool:
        """Atomic 'execute once': ON CONFLICT DO NOTHING + RETURNING tells us
        whether WE inserted the row.  If another process got there first, no
        row comes back and we report a duplicate."""
        with self.pool.connection() as conn:
            row = conn.execute(
                """INSERT INTO app.procurement_executions
                       (idempotency_key, case_id, approval_id, operation, payload, reference, cost_eur, result)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (idempotency_key) DO NOTHING
                   RETURNING idempotency_key""",
                (key, case_id, approval_id, action.kind, Jsonb(action.payload), result.reference,
                 action.cost_eur, Jsonb(result.model_dump(mode="json"))),
            ).fetchone()
        return row is not None

    def spent(self, case_id: str) -> float:
        with self.pool.connection() as conn:
            row = conn.execute("SELECT COALESCE(SUM(cost_eur), 0) AS s FROM app.procurement_executions WHERE case_id = %s",
                               (case_id,)).fetchone()
        return float(row["s"])

    def created_purchase_orders(self, component_id: str | None = None) -> list[dict]:
        q = "SELECT payload, reference FROM app.procurement_executions WHERE operation = 'create_purchase_order'"
        params: tuple = ()
        if component_id:
            q, params = q + " AND payload->>'component_id' = %s", (component_id,)
        with self.pool.connection() as conn:
            rows = conn.execute(q + " ORDER BY executed_at", params).fetchall()
        return [{**r["payload"], "po_id": r["reference"], "status": "open"} for r in rows]

    def next_po_number(self) -> int:
        """A Postgres SEQUENCE never hands out the same number twice, even across processes."""
        with self.pool.connection() as conn:
            return int(conn.execute("SELECT nextval('app.po_number_seq') AS n").fetchone()["n"])
