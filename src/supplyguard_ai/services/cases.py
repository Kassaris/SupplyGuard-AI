"""
Case registry: the list of incident cases (not their content).

The full state of a case lives in the LangGraph checkpoints (thread_id =
case_id).  Checkpoints can't be queried by "who" or "phase", so this small
registry answers the UI's list questions:
  * "Resume the case I was reviewing yesterday"  -> my cases, newest first
  * approvers' inbox                              -> cases awaiting approval
Two implementations, as everywhere: in memory (tests) and Postgres (app.cases).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol

from supplyguard_ai.domain.models import Role, User

APPROVER_ROLES = {Role.PROCUREMENT_MANAGER, Role.FINANCE_CONTROLLER, Role.FINANCE_DIRECTOR}


class CaseRegistry(Protocol):
    def create(self, case_id: str, user_id: str, title: str, phase: str) -> None: ...
    def update(self, case_id: str, phase: str) -> None: ...
    def add_participant(self, case_id: str, user_id: str) -> None: ...
    def visible_to(self, user: User, limit: int = 10) -> list[dict]: ...
    def can_access(self, user: User, case_id: str) -> bool: ...


def _is_visible(r: dict, user: User) -> bool:
    """THE access rule for cases (list AND open):
      * the creator,
      * anyone who acted in it (answered a clarification, signed / rejected) -
        so an approver can re-read a case after it closed,
      * approver roles, while it awaits approval (their inbox)."""
    return (r["created_by"] == user.user_id
            or user.user_id in (r.get("participants") or [])
            or (r["phase"] == "awaiting_approval" and user.role in APPROVER_ROLES))


def _visible(rows: list[dict], user: User, limit: int) -> list[dict]:
    return sorted((r for r in rows if _is_visible(r, user)), key=lambda r: r["updated_at"], reverse=True)[:limit]


@dataclass
class InMemoryCaseRegistry:
    _rows: dict[str, dict] = field(default_factory=dict)

    def create(self, case_id, user_id, title, phase):
        now = datetime.now(timezone.utc)
        self._rows.setdefault(case_id, {"case_id": case_id, "created_by": user_id, "title": title,
                                        "phase": phase, "participants": [], "created_at": now, "updated_at": now})

    def update(self, case_id, phase):
        if case_id in self._rows:
            self._rows[case_id] |= {"phase": phase, "updated_at": datetime.now(timezone.utc)}

    def add_participant(self, case_id, user_id):
        row = self._rows.get(case_id)
        if row is not None and user_id not in row["participants"]:
            row["participants"].append(user_id)

    def visible_to(self, user, limit=10):
        return _visible(list(self._rows.values()), user, limit)

    def can_access(self, user, case_id):
        row = self._rows.get(case_id)
        return row is not None and _is_visible(row, user)


class PostgresCaseRegistry:
    def __init__(self, pool) -> None:
        self.pool = pool

    def create(self, case_id, user_id, title, phase):
        with self.pool.connection() as conn:
            conn.execute("INSERT INTO app.cases (case_id, created_by, title, phase) VALUES (%s, %s, %s, %s) "
                         "ON CONFLICT (case_id) DO NOTHING", (case_id, user_id, title[:200], phase))

    def update(self, case_id, phase):
        with self.pool.connection() as conn:
            conn.execute("UPDATE app.cases SET phase = %s, updated_at = now() WHERE case_id = %s", (phase, case_id))

    def add_participant(self, case_id, user_id):
        with self.pool.connection() as conn:  # array_append only if not already present
            conn.execute("UPDATE app.cases SET participants = array_append(participants, %s) "
                         "WHERE case_id = %s AND NOT (%s = ANY(participants))", (user_id, case_id, user_id))

    def visible_to(self, user, limit=10):
        with self.pool.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM app.cases WHERE created_by = %s OR %s = ANY(participants) OR phase = 'awaiting_approval' "
                "ORDER BY updated_at DESC LIMIT 200", (user.user_id, user.user_id)).fetchall()
        return _visible(rows, user, limit)

    def can_access(self, user, case_id):
        with self.pool.connection() as conn:
            row = conn.execute("SELECT * FROM app.cases WHERE case_id = %s", (case_id,)).fetchone()
        return row is not None and _is_visible(row, user)
