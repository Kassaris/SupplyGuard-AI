"""
Simulated procurement system (assignment §03 step 9, §05 "MCP integrations").

Implements the contract in support-data/config/procurement_system_operations.json:
  * read operations are free,
  * every write needs an `approval_id` bound to the exact payload,
  * every write carries an `idempotency_key` (case_id + step); repeating a
    key returns the FIRST result and creates nothing new.

WHY IDEMPOTENCY MATTERS HERE
----------------------------
Complication §02: "The application restarts while approval is pending. The
investigation must resume without duplicating actions."  LangGraph resumes
from its last checkpoint; if the crash happened *after* the PO was created
but *before* the checkpoint was written, the execute node runs again.  With
idempotency keys the second call is a harmless no-op returning the same PO id.

WHY A PYTHON CLASS FIRST, MCP LATER
-----------------------------------
This class is the system's business logic.  In the MCP step we wrap these
exact methods in a FastMCP server (separate process).

STORAGE
-------
`ProcurementLedger` protocol: InMemoryLedger for unit tests,
PostgresLedger (app.procurement_executions, idempotency_key PRIMARY KEY)
in production.  Created POs, spent amount per case and replays are all
derived from that one table, so there is a single source of truth.  Agents then reach it
only through MCP tools, which proves the integration boundary - but the rules
(approval, caps, certification) live here and are unit-tested without MCP.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Protocol

from supplyguard_ai.domain.models import ExecutionResult, ProposedAction
from supplyguard_ai.services.approvals import ApprovalError, ApprovalService
from supplyguard_ai.services.audit import AuditStore
from supplyguard_ai.services.repository import OperationalRepository


# --------------------------------------------------------------------------- ledger
class ProcurementLedger(Protocol):
    """Durable record of executed writes = the idempotency table.

    `record()` must be ATOMIC: it returns False if the key already exists
    (Postgres: PRIMARY KEY + INSERT ... ON CONFLICT DO NOTHING), so even two
    processes racing on the same key cannot both execute.
    """

    def get(self, key: str) -> ExecutionResult | None: ...
    def record(self, key: str, case_id: str, approval_id: str, action: ProposedAction, result: ExecutionResult) -> bool: ...
    def spent(self, case_id: str) -> float: ...
    def created_purchase_orders(self, component_id: str | None = None) -> list[dict]: ...
    def next_po_number(self) -> int: ...


@dataclass
class InMemoryLedger:
    _rows: dict[str, dict] = field(default_factory=dict)
    _seq: int = 9100

    def get(self, key: str) -> ExecutionResult | None:
        row = self._rows.get(key)
        return ExecutionResult.model_validate(row["result"]) if row else None

    def record(self, key, case_id, approval_id, action, result) -> bool:
        if key in self._rows:
            return False
        self._rows[key] = {"case_id": case_id, "approval_id": approval_id, "operation": action.kind,
                           "payload": action.payload, "cost_eur": action.cost_eur,
                           "result": result.model_dump(mode="json")}
        return True

    def spent(self, case_id: str) -> float:
        return sum(r["cost_eur"] for r in self._rows.values() if r["case_id"] == case_id)

    def created_purchase_orders(self, component_id: str | None = None) -> list[dict]:
        return [
            {**r["payload"], "po_id": r["result"]["reference"], "status": "open"}
            for r in self._rows.values()
            if r["operation"] == "create_purchase_order" and component_id in (None, r["payload"]["component_id"])
        ]

    def next_po_number(self) -> int:
        n, self._seq = self._seq, self._seq + 1
        return n


# --------------------------------------------------------------------------- system
@dataclass
class SimulatedProcurementSystem:
    repo: OperationalRepository
    approvals: ApprovalService
    audit: AuditStore
    spending_cap_eur: float
    ledger: ProcurementLedger = field(default_factory=InMemoryLedger)

    # ------------------------------------------------------------- reads
    def get_inventory(self, component_id: str) -> list[dict]:
        return self.repo.inventory(component_id)

    def get_open_purchase_orders(self, component_id: str) -> list[dict]:
        return self.repo.open_purchase_orders(component_id) + self.ledger.created_purchase_orders(component_id)

    def get_supplier_catalog(self, component_id: str) -> list[dict]:
        return self.repo.supplier_catalog(component_id)

    def created_purchase_orders(self) -> list[dict]:
        """POs created by approved executions (all components)."""
        return self.ledger.created_purchase_orders()

    # ------------------------------------------------------------- writes
    def execute(self, case_id: str, step: str, approval_id: str, action: ProposedAction, actor: str) -> ExecutionResult:
        """Single entry point for all writes -> one place enforces all rules."""
        key = f"{case_id}:{step}"
        if (first := self.ledger.get(key)) is not None:
            return self._duplicate(first, case_id, actor, key)

        try:
            self.approvals.verify(approval_id, action)  # 1. approval bound to THIS payload
            self._check_business_rules(case_id, action)  # 2. operation-specific requirements
        except (ApprovalError, ValueError) as exc:
            # Rejections are NOT stored under the key: after re-approval the same step may succeed.
            self.audit.record("procurement.rejected", case_id, actor, {"step": step, "reason": str(exc)})
            return ExecutionResult(idempotency_key=key, operation=action.kind, status="rejected", message=str(exc))

        reference = self._reference(action)
        result = ExecutionResult(
            idempotency_key=key, operation=action.kind, status="created", reference=reference,
            message=f"{action.kind} executed ({reference}), EUR {action.cost_eur:,.2f}",
        )
        if not self.ledger.record(key, case_id, approval_id, action, result):
            # Lost a race with another process executing the same key: theirs counts.
            return self._duplicate(self.ledger.get(key), case_id, actor, key)
        self.audit.record("procurement.executed", case_id, actor,
                          {"step": step, "approval_id": approval_id, "reference": reference, "cost_eur": action.cost_eur})
        return result

    def execute_approved(self, case_id: str, step: str, approval_id: str, kind: str, payload: dict,
                         actor: str) -> ExecutionResult:
        """Execute by PAYLOAD - the form an MCP client can send.

        A remote caller only knows the operation and its payload, not our
        internal ProposedAction (cost, rationale ...).  We look the payload up
        in the APPROVED record: it must match an approved action exactly
        (kind + every payload field).  Same qty but another supplier, or one
        unit more -> no match -> rejected.  That is "approval binds to the
        payload" enforced at the system boundary.
        """
        key = f"{case_id}:{step}"
        if (first := self.ledger.get(key)) is not None:
            return self._duplicate(first, case_id, actor, key)
        try:
            rec = self.approvals.get(approval_id)
        except ApprovalError as exc:
            return ExecutionResult(idempotency_key=key, operation=kind, status="rejected", message=str(exc))
        match = next((a for a in rec.actions if a.kind == kind and a.payload == payload), None)
        if match is None:
            self.audit.record("procurement.rejected", case_id, actor,
                              {"step": step, "reason": "payload not in approved actions", "payload": payload})
            return ExecutionResult(idempotency_key=key, operation=kind, status="rejected",
                                   message="payload does not match any approved action; re-approval required")
        return self.execute(case_id, step, approval_id, match, actor)

    # ---------------------------------------------------------- internals
    def _duplicate(self, first: ExecutionResult, case_id: str, actor: str, key: str) -> ExecutionResult:
        self.audit.record("procurement.duplicate_blocked", case_id, actor, {"idempotency_key": key})
        return first.model_copy(update={"status": "duplicate", "message": f"already executed: {first.message}"})

    def _check_business_rules(self, case_id: str, action: ProposedAction) -> None:
        p = action.payload
        if self.ledger.spent(case_id) + action.cost_eur > self.spending_cap_eur:
            raise ValueError(f"would exceed spending cap EUR {self.spending_cap_eur:,.0f}")

        if action.kind == "create_purchase_order":
            sup = self.repo.supplier(p["supplier_id"])
            expiry = sup and sup.get("certification_expiry")
            if not (isinstance(expiry, date) and expiry > self.repo.scenario_date()):
                raise ValueError(f"supplier {p['supplier_id']} has no valid certification")

        elif action.kind == "expedite_purchase_order":
            po = self.repo.purchase_order(p["po_id"])
            if not po or po["status"] != "open" or po.get("expedite_allowed") != "yes":
                raise ValueError(f"{p['po_id']} cannot be expedited (not open or expedite not allowed by contract)")

        elif action.kind == "request_stock_transfer":
            src = next((r for r in self.repo.inventory(p["component_id"]) if r["location_id"] == p["from_location"]), None)
            usable = (src["on_hand"] - src["reserved"] - src["quarantined"]) if src else 0
            if usable < p["qty"]:
                raise ValueError(f"only {usable} usable units at {p['from_location']}, {p['qty']} requested")

    def _reference(self, action: ProposedAction) -> str:
        """The business reference of the write (new PO number, transfer id ...)."""
        if action.kind == "create_purchase_order":
            return f"PO-2026-{self.ledger.next_po_number()}"
        if action.kind == "expedite_purchase_order":
            return f"{action.payload['po_id']}-EXP"
        if action.kind == "request_stock_transfer":
            return f"TR-{action.payload['from_location']}-{action.payload['to_location']}-{action.payload['qty']}"
        return f"SCHED-{action.payload.get('product_id')}"
