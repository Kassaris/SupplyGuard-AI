"""
Approval service: human-in-the-loop that is binding, not decorative.

ASSIGNMENT RULES
  §07  "Human approval must bind to the specific payload. If supplier,
        quantity, price, or relevant conditions change, the system must
        revalidate and obtain approval again."
  §02  "Do not create a purchase order without approval."
  policies.json  approval tiers by amount:
        <= 25k  procurement_manager
        <= 75k  procurement_manager + finance_controller
        <= 90k  finance_director

HOW BINDING WORKS
-----------------
  request()  -> stores the exact actions + SHA-256 hash of them
  approve()  -> each required role signs; status APPROVED when all signed
  verify()   -> procurement calls this before EVERY write: the action it is
                about to execute must be one of the approved actions, byte for
                byte.  A changed qty / price / supplier => different JSON =>
                no match => refused.

A new request for the same case automatically SUPERSEDES the previous one,
so an old approval can never be reused for a modified plan.

STORAGE
-------
`ApprovalStore` protocol: in memory for unit tests, `PostgresApprovalStore`
(persistence/stores.py) in production.  The LangGraph checkpoint keeps only
the `approval_id`; the record itself is in app.approvals, so an app restart
while approval is pending loses nothing (assignment §02, last complication).
Records are read, changed and explicitly `save()`d - never mutated in place
inside the store - so both backends behave identically.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol

from supplyguard_ai.domain.models import ApprovalRecord, ApprovalStatus, ProposedAction, RecoveryPlan, User
from supplyguard_ai.services.audit import AuditStore
from supplyguard_ai.services.identity import AuthorizationError, IdentityService


class ApprovalError(RuntimeError):
    pass


def required_roles_for(amount_eur: float, tiers: list[dict]) -> list[str]:
    """Map an amount to approver roles using policies.json tiers.

    'procurement_manager + finance_controller' -> ['procurement_manager', 'finance_controller']
    """
    for tier in sorted(tiers, key=lambda t: t["up_to"]):
        if amount_eur <= tier["up_to"]:
            return [r.strip() for r in tier["approver"].split("+")]
    raise ApprovalError(f"EUR {amount_eur:,.2f} exceeds the highest approval tier; cannot be approved")


class ApprovalStore(Protocol):
    """Where approval records live.  In memory for unit tests, Postgres
    (persistence/stores.py) in production - so an approval survives a restart."""

    def get(self, approval_id: str) -> ApprovalRecord | None: ...
    def save(self, rec: ApprovalRecord) -> None: ...
    def for_case(self, case_id: str) -> list[ApprovalRecord]: ...


@dataclass
class InMemoryApprovalStore:
    _records: dict[str, ApprovalRecord] = field(default_factory=dict)

    def get(self, approval_id: str) -> ApprovalRecord | None:
        # copy: callers mutate the record and must call save() - same contract as Postgres
        rec = self._records.get(approval_id)
        return rec.model_copy(deep=True) if rec else None

    def save(self, rec: ApprovalRecord) -> None:
        self._records[rec.approval_id] = rec.model_copy(deep=True)

    def for_case(self, case_id: str) -> list[ApprovalRecord]:
        return [r.model_copy(deep=True) for r in self._records.values() if r.case_id == case_id]


@dataclass
class ApprovalService:
    tiers: list[dict]
    audit: AuditStore
    store: ApprovalStore = field(default_factory=InMemoryApprovalStore)

    # ----------------------------------------------------------- request
    def request(self, case_id: str, plan: RecoveryPlan, requester: User) -> ApprovalRecord:
        IdentityService.require_can_request_approval(requester)
        payload_hash = ApprovalRecord.hash_actions(plan.actions)

        # Supersede any older open/approved request for this case with a different payload.
        for rec in self.store.for_case(case_id):
            if rec.status in (ApprovalStatus.PENDING, ApprovalStatus.APPROVED):
                if rec.payload_hash == payload_hash and rec.status == ApprovalStatus.PENDING:
                    return rec  # idempotent: same plan already waiting -> reuse (survives retries/restarts)
                rec.status = ApprovalStatus.SUPERSEDED
                self.store.save(rec)
                self.audit.record("approval.superseded", case_id, requester.user_id, {"approval_id": rec.approval_id})

        rec = ApprovalRecord(
            approval_id=f"APR-{uuid.uuid4().hex[:10]}",
            case_id=case_id,
            payload_hash=payload_hash,
            actions=plan.actions,
            total_cost_eur=plan.total_cost_eur,
            required_roles=required_roles_for(plan.total_cost_eur, self.tiers),
            requested_by=requester.user_id,
        )
        self.store.save(rec)
        self.audit.record("approval.requested", case_id, requester.user_id,
                          {"approval_id": rec.approval_id, "total_cost_eur": rec.total_cost_eur,
                           "required_roles": rec.required_roles, "payload_hash": payload_hash})
        return rec

    # ---------------------------------------------------------- decisions
    def get(self, approval_id: str) -> ApprovalRecord:
        rec = self.store.get(approval_id)
        if rec is None:
            raise ApprovalError(f"Unknown approval {approval_id}")
        return rec

    def approve(self, approval_id: str, approver: User) -> ApprovalRecord:
        rec = self.get(approval_id)
        if rec.status != ApprovalStatus.PENDING:
            raise ApprovalError(f"Approval {approval_id} is {rec.status}, not pending")
        role = approver.role.value
        if role not in rec.required_roles:
            raise AuthorizationError(f"{role} is not a required approver for EUR {rec.total_cost_eur:,.0f}")
        # Each user has exactly ONE role, and each slot is keyed by role, so a
        # two-role tier automatically needs two different people (four-eyes).
        rec.approvals[role] = approver.user_id
        if set(rec.required_roles) <= set(rec.approvals):
            rec.status = ApprovalStatus.APPROVED
            rec.decided_by = approver.user_id
            rec.decided_at = datetime.now(timezone.utc)
        self.store.save(rec)
        self.audit.record("approval.signed", rec.case_id, approver.user_id,
                          {"approval_id": approval_id, "role": role, "status": rec.status})
        return rec

    def reject(self, approval_id: str, approver: User, reason: str) -> ApprovalRecord:
        rec = self.get(approval_id)
        if approver.role.value not in rec.required_roles:
            raise AuthorizationError(f"{approver.role} cannot reject this approval")
        rec.status = ApprovalStatus.REJECTED
        rec.decided_by = approver.user_id
        rec.rejection_reason = reason
        rec.decided_at = datetime.now(timezone.utc)
        self.store.save(rec)
        self.audit.record("approval.rejected", rec.case_id, approver.user_id, {"approval_id": approval_id, "reason": reason})
        return rec

    # ------------------------------------------------------------- verify
    def verify(self, approval_id: str, action: ProposedAction) -> ApprovalRecord:
        """Gatekeeper used by the procurement system before every write."""
        rec = self.get(approval_id)
        if rec.status != ApprovalStatus.APPROVED:
            raise ApprovalError(f"Approval {approval_id} is {rec.status}; execution refused")
        if ApprovalRecord.hash_actions(rec.actions) != rec.payload_hash:
            raise ApprovalError("Stored approval was tampered with")
        if action not in rec.actions:  # pydantic models compare field-by-field
            raise ApprovalError("Action payload differs from the approved payload; re-approval required")
        return rec
