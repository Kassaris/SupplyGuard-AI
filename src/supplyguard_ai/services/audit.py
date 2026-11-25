"""
Audit log: "every action can be audited" (assignment §04, USER CONTROL).

Append-only list of events.  Services call `record()` for every state change
(approval requested/signed/rejected, PO created, duplicate write blocked ...).

Why a separate tiny class?  Because it is a cross-cutting concern: approvals,
procurement and later the LangGraph nodes all write to it, and the Postgres
version (`audit_events` table) only has to replace this one class.
Events are never updated or deleted - that is what makes them an audit trail.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol


@dataclass(frozen=True)
class AuditEvent:
    event_type: str
    case_id: str
    actor: str  # user_id, or "system" / "agent:<name>"
    details: dict[str, Any]
    at: datetime


class AuditStore(Protocol):
    """Implemented by AuditLog (memory) and PostgresAuditLog (app.audit_events)."""

    def record(self, event_type: str, case_id: str, actor: str, details: dict[str, Any] | None = None) -> AuditEvent: ...
    def events(self, case_id: str | None = None) -> list[AuditEvent]: ...


@dataclass
class AuditLog:
    _events: list[AuditEvent] = field(default_factory=list)

    def record(self, event_type: str, case_id: str, actor: str, details: dict[str, Any] | None = None) -> AuditEvent:
        ev = AuditEvent(event_type, case_id, actor, dict(details or {}), datetime.now(timezone.utc))
        self._events.append(ev)
        return ev

    def events(self, case_id: str | None = None) -> list[AuditEvent]:
        return [e for e in self._events if case_id is None or e.case_id == case_id]
