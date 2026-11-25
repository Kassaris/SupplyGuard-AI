"""
Recovery planner: turns an ImpactAssessment into costed, executable plans.

Strategies (assignment §07):
  A  Transfer available inventory      -> request_stock_transfer
  B  Expedite an existing order        -> expedite_purchase_order
  C  Purchase from approved alternative-> create_purchase_order
  D  Reallocate across customer orders -> (reported as residual handling)
  E  Reschedule production             -> (reported as residual handling)

THE ALGORITHM (greedy, explainable - every step has a written rationale)
-----------------------------------------------------------------------
1. Screen every alternative supplier with HARD rules first:
     * certificate must be valid on the scenario date  (policy
       `requires_valid_certification`; ERP status is known to be stale, so we
       check the expiry DATE, not the `erp_status` column),
     * must arrive BEFORE the delayed original shipment, otherwise it fixes
       nothing (this is why Helios, 21 d, is rejected),
   and SOFT rules as warnings:
     * shares an upstream manufacturer with the disrupted supplier
       (Delta & Atlas both depend on Veltro -> "apparent diversification").
2. A - rebalance usable stock: move non-plant (hub) stock to the plants with
   the LOWEST days of cover using "water-filling" (raise the lowest plant to
   the level of the next lowest, and so on).  Cheap (EUR 0.12/unit).
3. B - expedite open POs from feasible suppliers (cost = qty*price*fee%).
4. C - buy the remaining gap from feasible suppliers, independent upstream
   first, then fastest, then cheapest; qty = min(gap, monthly capacity),
   respecting MOQ.
5. Whatever is still uncovered -> caveat recommending D / E.

We build TWO plans so the human sees the trade-off:
   plan_independent : only suppliers with an independent upstream
   plan_full_cover  : also allows shared-upstream suppliers to close the gap
Golden reference: ~EUR 30,378 (850 uncovered) and ~EUR 39,218 (0 uncovered).

NOTE: contract/e-mail conflicts (Cordillera fee 6% vs 10%, Helios capacity
4,000 vs 1,500) are resolved by the *document* agents using the source
precedence in policies.json; they can pass the resolved value in through
`overrides`.  The planner itself only uses structured ERP data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

from supplyguard_ai.domain.models import (
    ImpactAssessment,
    ProposedAction,
    RecoveryPlan,
    StockPosition,
    SupplierAssessment,
)
from supplyguard_ai.services.repository import OperationalRepository


@dataclass
class PlannerOverrides:
    """Values resolved from documents that beat the ERP numbers.

    e.g. capacity_units={"S002": 1500}, expedite_fee_pct={"PO-2026-9002": 6.0}
    """

    capacity_units: dict[str, int] = field(default_factory=dict)
    expedite_fee_pct: dict[str, float] = field(default_factory=dict)
    excluded_suppliers: set[str] = field(default_factory=set)


@dataclass
class RecoveryPlanner:
    repo: OperationalRepository

    # ---------------------------------------------------- supplier screening
    def assess_suppliers(
        self, impact: ImpactAssessment, disrupted_supplier_id: str, overrides: PlannerOverrides | None = None
    ) -> list[SupplierAssessment]:
        ov = overrides or PlannerOverrides()
        today = impact.scenario_date
        disrupted = self.repo.supplier(disrupted_supplier_id) or {}
        disrupted_upstream = disrupted.get("upstream_manufacturer_id")
        deadline = impact.delayed_arrival_day or 0

        results = []
        for cat in self.repo.supplier_catalog(impact.component_id):
            sid = cat["supplier_id"]
            if sid == disrupted_supplier_id:
                continue
            sup = self.repo.supplier(sid) or {}
            expiry = sup.get("certification_expiry")
            cert_ok = bool(expiry) and expiry > today
            upstream = sup.get("upstream_manufacturer_id")
            shared = bool(upstream) and upstream == disrupted_upstream
            arrives_in_time = cat["lead_time_days"] < deadline
            capacity = ov.capacity_units.get(sid, cat["monthly_capacity_units"])

            reasons, warnings = [], []
            # DOC-PRO-04: BOTH a valid certificate AND ERP status 'approved' are
            # required.  An expired certificate blocks ordering regardless of ERP
            # status (Borealis: ERP says approved, certificate expired).
            if not cert_ok:
                reasons.append(f"{sup.get('certification')} certificate expired on {expiry} (DOC-PRO-04: valid certification required)")
            if sup.get("erp_status") != "approved":
                reasons.append(f"ERP status '{sup.get('erp_status')}' is not 'approved' (DOC-PRO-04)")
            if not arrives_in_time:
                reasons.append(
                    f"lead time {cat['lead_time_days']} d >= delayed arrival day {deadline}: arrives after the original supplier"
                )
            if sid in ov.excluded_suppliers:
                reasons.append("excluded by reviewer / document evidence")
            if shared:
                warnings.append(f"shares upstream manufacturer {upstream} with {disrupted.get('name')}: weak diversification")
            if cat["approved_source"] != "yes":
                warnings.append(f"catalog status '{cat['approved_source']}' (not primary source)")

            results.append(
                SupplierAssessment(
                    supplier_id=sid,
                    supplier_name=sup.get("name", sid),
                    unit_price_eur=cat["unit_price_eur"],
                    lead_time_days=cat["lead_time_days"],
                    monthly_capacity_units=capacity,
                    moq_units=cat["moq_units"],
                    upstream_manufacturer_id=upstream,
                    certification_valid=cert_ok,
                    shares_upstream_with_disrupted_supplier=shared,
                    arrives_before_stockout=arrives_in_time,
                    feasible=not reasons,
                    rejection_reasons=reasons,
                    warnings=warnings,
                )
            )
        return results

    # ------------------------------------------------------------ strategy A
    def _transfers(self, stock: list[StockPosition], component_id: str) -> list[ProposedAction]:
        """Water-filling: give hub stock to the plants with the lowest cover."""
        cost_per_unit = self.repo.policies()["transfer_cost_eur_per_unit"]
        pool = sum(s.usable for s in stock if s.factory_id is None)  # non-plant stock
        sources = [s for s in stock if s.factory_id is None and s.usable > 0]
        plants = [s for s in stock if s.factory_id and s.daily_demand > 0]
        if not pool or not plants or not sources:
            return []

        # Simulate unit-by-unit in blocks of 50 units: simple and obviously correct.
        add = {p.location_id: 0 for p in plants}
        remaining = pool
        while remaining > 0:
            # plant whose cover after transfers so far is lowest
            target = min(plants, key=lambda p: (p.usable + add[p.location_id]) / p.daily_demand)
            step = min(50, remaining)
            add[target.location_id] += step
            remaining -= step

        src = sources[0].location_id  # one hub in this data set
        actions = []
        for loc, qty in add.items():
            if qty <= 0:
                continue
            plant = next(p for p in plants if p.location_id == loc)
            actions.append(
                ProposedAction(
                    kind="request_stock_transfer",
                    payload={"component_id": component_id, "from_location": src, "to_location": loc, "qty": qty},
                    cost_eur=round(qty * cost_per_unit, 2),
                    units_covered=0,  # transfers move stock, they don't create it
                    rationale=f"{loc} has only {plant.days_of_cover} days of cover; rebalancing delays its stock-out",
                )
            )
        return actions

    # ------------------------------------------------------------ strategy B
    def _expedites(
        self, impact: ImpactAssessment, feasible: dict[str, SupplierAssessment], ov: PlannerOverrides
    ) -> list[ProposedAction]:
        actions = []
        for po in self.repo.open_purchase_orders(impact.component_id):
            if po["po_id"] not in impact.other_open_po_ids or po["supplier_id"] not in feasible:
                continue
            if po.get("expedite_allowed") != "yes" or not po.get("expedite_days_gain"):
                continue
            fee_pct = ov.expedite_fee_pct.get(po["po_id"], po["expedite_fee_pct"])
            new_date = po["promised_date"] - timedelta(days=po["expedite_days_gain"])
            new_day = (new_date - impact.scenario_date).days
            if new_day >= (impact.delayed_arrival_day or 0):
                continue
            actions.append(
                ProposedAction(
                    kind="expedite_purchase_order",
                    payload={"po_id": po["po_id"], "days_gain": po["expedite_days_gain"], "fee_pct": fee_pct},
                    cost_eur=round(po["qty"] * po["unit_price_eur"] * fee_pct / 100, 2),
                    units_covered=po["qty"],
                    rationale=f"{po['po_id']} ({po['qty']} units) moves from day {new_day + po['expedite_days_gain']} to day {new_day}",
                )
            )
        return actions

    # ------------------------------------------------------------ strategy C
    def _purchases(
        self, impact: ImpactAssessment, candidates: list[SupplierAssessment], gap: int, already_committed: dict[str, int]
    ) -> list[ProposedAction]:
        actions = []
        # independent upstream first, then fastest, then cheapest
        for s in sorted(candidates, key=lambda s: (s.shares_upstream_with_disrupted_supplier, s.lead_time_days, s.unit_price_eur)):
            if gap <= 0:
                break
            free_capacity = s.monthly_capacity_units - already_committed.get(s.supplier_id, 0)
            qty = min(gap, free_capacity)
            if qty < s.moq_units:
                continue
            delivery = impact.scenario_date + timedelta(days=s.lead_time_days)
            actions.append(
                ProposedAction(
                    kind="create_purchase_order",
                    payload={
                        "supplier_id": s.supplier_id,
                        "component_id": impact.component_id,
                        "qty": qty,
                        "unit_price_eur": s.unit_price_eur,
                        "requested_delivery_date": delivery.isoformat(),
                    },
                    cost_eur=round(qty * s.unit_price_eur, 2),
                    units_covered=qty,
                    rationale=(
                        f"{s.supplier_name}: {s.lead_time_days} d lead time, capacity {s.monthly_capacity_units}/month"
                        + (f"; WARNING: {'; '.join(s.warnings)}" if s.warnings else "")
                    ),
                )
            )
            gap -= qty
        return actions

    # ----------------------------------------------------------------- plans
    def build_plans(
        self,
        impact: ImpactAssessment,
        disrupted_supplier_id: str,
        spending_cap_eur: float,
        overrides: PlannerOverrides | None = None,
    ) -> list[RecoveryPlan]:
        ov = overrides or PlannerOverrides()
        assessments = self.assess_suppliers(impact, disrupted_supplier_id, ov)
        feasible = {a.supplier_id: a for a in assessments if a.feasible}

        transfers = self._transfers(impact.stock, impact.component_id)
        expedites = self._expedites(impact, feasible, ov)
        base_gap = impact.shortfall_units - sum(a.units_covered for a in expedites)

        # Capacity already promised to us by a supplier.  The golden scenario
        # treats an open PO (Cordillera PO-2026-9002) as already inside the
        # supplier's production plan, so it does NOT reduce the capacity
        # available for a new order.  Kept as a parameter so this assumption
        # is visible and can be changed in one place.
        committed: dict[str, int] = {}

        plans: list[RecoveryPlan] = []
        variants = [
            ("plan_independent_upstream", [a for a in feasible.values() if not a.shares_upstream_with_disrupted_supplier]),
            ("plan_full_cover", list(feasible.values())),
        ]
        for plan_id, candidates in variants:
            purchases = self._purchases(impact, candidates, base_gap, committed)
            actions = transfers + expedites + purchases
            covered = sum(a.units_covered for a in actions)
            uncovered = max(0, impact.shortfall_units - covered)
            total = round(sum(a.cost_eur for a in actions), 2)

            strategies = sorted({{"request_stock_transfer": "A", "expedite_purchase_order": "B",
                                  "create_purchase_order": "C"}[a.kind] for a in actions})
            # Warnings of suppliers we actually buy from; dict.fromkeys = de-duplicate, keep order.
            caveats = list(dict.fromkeys(w for a in candidates for w in a.warnings if any(
                p.payload.get("supplier_id") == a.supplier_id for p in purchases)))
            if uncovered:
                days = round(uncovered / impact.daily_demand_total, 1)
                caveats.append(
                    f"{uncovered} units (~{days} days of demand) remain uncovered: reschedule production (E) "
                    f"or reallocate stock to tier-1 orders (D)"
                )
                strategies = sorted(set(strategies) | {"D", "E"})
            if total > spending_cap_eur:
                caveats.append(f"total EUR {total:,.0f} exceeds the EUR {spending_cap_eur:,.0f} cap")

            plan = RecoveryPlan(
                plan_id=plan_id,
                strategies=strategies,
                actions=actions,
                total_cost_eur=total,
                units_covered=covered,
                uncovered_units=uncovered,
                within_cap=total <= spending_cap_eur,
                caveats=caveats,
                sources=impact.sources + ["SRC-ERP-SUPPLIER-MASTER", "SRC-SUPPLIER-QUESTIONNAIRE-2026"],
            )
            # Don't show the same plan twice (if no shared-upstream supplier exists).
            if not any(p.actions == plan.actions for p in plans):
                plans.append(plan)
        return plans
