"""
Identity & authorization service.

ASSIGNMENT (§05): the application service layer owns "User identity, case
access ...".  (§04 step 1): "The user's role decides which documents can be
retrieved."

KEY SECURITY DECISION: authorization is enforced in CODE, never in the prompt.
-----------------------------------------------------------------------------
It is tempting to write in the system prompt "do not show finance documents
to viewers".  A prompt-injected document or a clever user can talk the LLM
out of that.  Instead:
  * the retrieval tools receive the `User` object from the runtime context
    (not from the LLM's tool arguments!),
  * `allowed_document_scopes()` becomes a hard metadata filter on pgvector /
    ChromaDB queries,
  * so a forbidden document never even reaches the model's context window.

Chainlit will call `authenticate()` from its password/header auth callback.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from supplyguard_ai.domain.models import Role, User


class AuthorizationError(PermissionError):
    """Raised when a user tries something their role does not allow."""


# Which roles may approve / execute / read finance data.  Kept as data so it
# is easy to audit and to test.
_CAN_REQUEST_APPROVAL = {Role.PROCUREMENT_MANAGER, Role.INCIDENT_COORDINATOR, Role.OPERATIONS_ANALYST}
_READ_ONLY = {Role.VIEWER}


@dataclass
class IdentityService:
    """Holds the known users.  Built from users.csv (unit tests) or from
    app.users in Postgres (production, seeded by ingestion/seed_users.py)."""

    _users: dict[str, User]

    @classmethod
    def from_csv(cls, users_csv: Path) -> "IdentityService":
        users = {}
        with users_csv.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                users[row["user_id"]] = User(
                    user_id=row["user_id"],
                    name=row["name"],
                    role=Role(row["role"]),
                    # "public;internal;procurement" -> frozenset({...}); empty -> empty set
                    document_scopes=frozenset(s for s in row["document_scopes"].split(";") if s),
                    factory_access=frozenset(f for f in row["factory_access"].split(";") if f),
                )
        return cls(users)

    @classmethod
    def from_rows(cls, rows: list[dict]) -> "IdentityService":
        """From app.users rows (text[] columns arrive as Python lists)."""
        return cls({
            r["user_id"]: User(user_id=r["user_id"], name=r["name"], role=Role(r["role"]),
                               document_scopes=frozenset(r["document_scopes"]),
                               factory_access=frozenset(r["factory_access"]))
            for r in rows
        })

    # ------------------------------------------------------------- identity
    def authenticate(self, user_id: str) -> User:
        """Look up a user.  (Demo-grade: real deployments would verify a
        password hash or an SSO token here before returning the user.)"""
        try:
            return self._users[user_id]
        except KeyError:
            raise AuthorizationError(f"Unknown user {user_id!r}") from None

    def all_users(self) -> list[User]:
        return list(self._users.values())

    # -------------------------------------------------------- authorization
    @staticmethod
    def allowed_document_scopes(user: User) -> set[str]:
        """Used as a metadata filter in every retrieval query."""
        return set(user.document_scopes)

    @staticmethod
    def can_read_document(user: User, access_scope: str) -> bool:
        return access_scope in user.document_scopes

    @staticmethod
    def require_can_open_case(user: User) -> None:
        if user.role in _READ_ONLY:
            raise AuthorizationError(f"{user.name} ({user.role}) has read-only access")

    @staticmethod
    def require_can_request_approval(user: User) -> None:
        if user.role not in _CAN_REQUEST_APPROVAL:
            raise AuthorizationError(f"{user.role} may not submit recovery plans for approval")

    @staticmethod
    def require_can_approve_as(user: User, role: str) -> None:
        """A user can only sign an approval slot that matches their own role."""
        if user.role.value != role:
            raise AuthorizationError(f"{user.name} is {user.role}, cannot approve as {role}")
