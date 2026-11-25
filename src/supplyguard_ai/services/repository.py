"""
Data repository: the ONLY module that knows where operational data lives.

PATTERN: Repository (a.k.a. data-access gateway)
------------------------------------------------
Business services (ImpactService, RecoveryPlanner, ...) ask the repository
questions like "inventory for MC-17" and get plain Python records back.
They never open files or write SQL themselves.

WHY START WITH CSV, NOT POSTGRES?
---------------------------------
The final architecture stores operational records in PostgreSQL (§05).
But:
  1. The business logic is the riskiest part (25% of the grade is numerical
     correctness).  An in-memory CSV repository lets us unit-test it in
     milliseconds with zero Docker.
  2. Because services depend on the `OperationalRepository` *Protocol*, a
     `PostgresRepository` with the same method names can be dropped in later
     without touching a single business rule (Dependency Inversion).

The CSV loader is also what the Postgres ingestion script will reuse.

DATA-QUALITY NOTE
-----------------
The data has planted problems (data_quality_manifest.csv).  Two are handled
here because they are about *identity* of records:
  * ambiguous supplier names ("Atlas" -> Atlas Components OR Atlas Logistik),
  * the ERP `erp_status` / `certification_status` columns can be stale, so we
    expose the raw `certification_expiry` date and let services decide.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import date
from functools import cached_property
from pathlib import Path
from typing import Any, Protocol

Row = dict[str, Any]


class OperationalRepository(Protocol):
    """What business services are allowed to ask.  A Postgres implementation
    must provide the same methods (structural typing, no inheritance needed)."""

    def scenario_date(self) -> date: ...
    def policies(self) -> dict[str, Any]: ...
    def supplier(self, supplier_id: str) -> Row | None: ...
    def suppliers(self) -> list[Row]: ...
    def inventory(self, component_id: str) -> list[Row]: ...
    def locations(self) -> list[Row]: ...
    def bom_for_component(self, component_id: str) -> list[Row]: ...
    def production_schedule(self, product_ids: set[str]) -> list[Row]: ...
    def customer_orders(self, product_ids: set[str]) -> list[Row]: ...
    def open_purchase_orders(self, component_id: str) -> list[Row]: ...
    def purchase_order(self, po_id: str) -> Row | None: ...
    def supplier_catalog(self, component_id: str) -> list[Row]: ...
    def customers(self) -> list[Row]: ...
    def shipment_updates(self, po_id: str) -> list[Row]: ...
    def data_source_ids(self) -> set[str]: ...
    def approved_memories(self) -> list[Row]: ...


# Column -> converter.  CSV gives strings; we convert once, at load time,
# so the rest of the code works with real ints / floats / dates.
_INT_COLS = {
    "qty", "on_hand", "reserved", "quarantined", "usable", "planned_units", "qty_per_unit",
    "lead_time_days", "monthly_capacity_units", "moq_units", "reserved_qty", "priority_tier",
    "customer_priority_tier", "expedite_days_gain", "delay_days",
}
_FLOAT_COLS = {
    "unit_price_eur", "revenue_eur", "standard_cost_eur", "list_price_eur", "expedite_fee_pct",
    "late_penalty_pct_per_week_capped5", "recovery_cost_eur",
}
_DATE_COLS = {
    "due_date", "week_start", "order_date", "promised_date", "actual_delivery_date",
    "last_counted_date", "certification_expiry", "event_date", "eta", "effective_date",
}


def _convert(row: Row) -> Row:
    out: Row = {}
    for k, v in row.items():
        v = v.strip() if isinstance(v, str) else v
        if v in ("", None):
            out[k] = None  # blank -> None (e.g. missing supplier_id on some POs)
        elif k in _INT_COLS:
            out[k] = int(float(v))
        elif k in _FLOAT_COLS:
            out[k] = float(v)
        elif k in _DATE_COLS:
            try:
                out[k] = date.fromisoformat(v)
            except ValueError:  # planted bad dates: keep raw string, don't crash ingestion
                out[k] = v
        else:
            out[k] = v
    return out


@dataclass
class CsvRepository:
    """Reads support-data/structured/*.csv into memory (read-only).

    `cached_property` -> each CSV is parsed the first time it is needed and
    then kept.  The whole data set is ~1 MB, so memory is not a concern.
    """

    data_dir: Path
    _cache: dict[str, list[Row]] = field(default_factory=dict, repr=False)

    # ---------------------------------------------------------------- helpers
    def _table(self, name: str) -> list[Row]:
        if name not in self._cache:
            path = self.data_dir / "structured" / f"{name}.csv"
            with path.open(encoding="utf-8", newline="") as fh:
                self._cache[name] = [_convert(r) for r in csv.DictReader(fh)]
        return self._cache[name]

    @cached_property
    def _policies(self) -> dict[str, Any]:
        return json.loads((self.data_dir / "config" / "policies.json").read_text(encoding="utf-8"))

    # ------------------------------------------------------------- protocol
    def scenario_date(self) -> date:
        """'Today' for the simulation (2026-10-19).  Never use date.today() in
        business logic: results must be reproducible for evaluation."""
        return date.fromisoformat(self._policies["scenario_date"])

    def policies(self) -> dict[str, Any]:
        return self._policies

    def supplier(self, supplier_id: str) -> Row | None:
        return next((s for s in self._table("suppliers") if s["supplier_id"] == supplier_id), None)

    def suppliers(self) -> list[Row]:
        return self._table("suppliers")

    def inventory(self, component_id: str) -> list[Row]:
        return [r for r in self._table("inventory") if r["component_id"] == component_id]

    def locations(self) -> list[Row]:
        return self._table("locations")

    def bom_for_component(self, component_id: str) -> list[Row]:
        return [r for r in self._table("bill_of_materials") if r["component_id"] == component_id]

    def production_schedule(self, product_ids: set[str]) -> list[Row]:
        return [r for r in self._table("production_schedule") if r["product_id"] in product_ids]

    def customer_orders(self, product_ids: set[str]) -> list[Row]:
        return [r for r in self._table("customer_orders") if r["product_id"] in product_ids]

    def open_purchase_orders(self, component_id: str) -> list[Row]:
        return [
            r for r in self._table("purchase_orders")
            if r["component_id"] == component_id and r["status"] == "open"
        ]

    def purchase_order(self, po_id: str) -> Row | None:
        return next((r for r in self._table("purchase_orders") if r["po_id"] == po_id), None)

    def supplier_catalog(self, component_id: str) -> list[Row]:
        """Who can supply this component, at what price / lead time / capacity."""
        return [r for r in self._table("supplier_components") if r["component_id"] == component_id]

    def customers(self) -> list[Row]:
        return self._table("customers")

    def shipment_updates(self, po_id: str) -> list[Row]:
        """Carrier / supplier events for one PO, oldest first (may contain duplicates - planted)."""
        rows = [r for r in self._table("shipment_updates") if r["po_id"] == po_id]
        return sorted(rows, key=lambda r: (str(r["event_date"]), r["shipment_event_id"]))

    @cached_property
    def _memories(self) -> list[Row]:
        path = self.data_dir / "structured" / "approved_incident_memories.jsonl"
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def approved_memories(self) -> list[Row]:
        """The 40 approved incident memories (lessons from past cases).
        Baseline exact-match source; pgvector semantic search replaces it later."""
        return self._memories

    def data_source_ids(self) -> set[str]:
        """Ids in the provenance registry (SRC-WMS-INVENTORY, ...).  Used to
        check that an agent cites only sources that really exist."""
        # + the memory store, which has no row in data_sources.csv
        return {r["source_id"] for r in self._table("data_sources")} | {"SRC-APPROVED-INCIDENT-MEMORIES"}

    # ------------------------------------------------------- entity resolution
    def resolve_supplier(self, name_or_id: str) -> list[Row]:
        """Return ALL suppliers matching a user-typed name or id.

        Returning a list (not a single row) is the point: if the user typed
        "Atlas" we get two candidates, and the LangGraph intake node must ASK
        the user which one is meant instead of guessing (assignment §04 step 2).
        """
        needle = name_or_id.strip().lower()
        exact_id = [s for s in self.suppliers() if s["supplier_id"].lower() == needle]
        if exact_id:
            return exact_id
        exact_name = [s for s in self.suppliers() if s["name"].lower() == needle]
        if exact_name:
            return exact_name
        return [s for s in self.suppliers() if needle in s["name"].lower()]
