"""
Impact service: deterministic answer to "what does this delay break?"

WHY DETERMINISTIC CODE AND NOT THE LLM?
---------------------------------------
Assignment §03 step 6: "Calculate trade-offs, using deterministic business
tools."  LLMs are unreliable at multi-step arithmetic and can't be audited.
So the split of responsibilities is:

    Python (this file)  -> computes every number, with sources
    LLM (agents)        -> decides WHICH calculation to run, interprets
                           documents, explains the result in plain language

These functions become LangChain tools later (`@tool`), so an agent calls
`assess_impact("S001", "MC-17", 12)` and receives a validated
`ImpactAssessment` it can quote, but cannot alter.

THE MODEL (matches support-data/evaluation/golden_scenario.json)
---------------------------------------------------------------
  day 0                = scenario date (2026-10-19)
  daily demand         = sum over products using the component of
                         planned_units/week * qty_per_unit / 7     (per factory)
  usable stock         = on_hand - reserved - quarantined          (per location)
  original arrival day = PO promised_date - day 0                  (7)
  delayed arrival day  = original + delay                          (19)
  shortfall            = demand * delayed_day - total usable stock (5,350)
                         i.e. demand not covered until Atlas arrives,
                         assuming stock can be pooled (transfers).
  at-risk orders       = open customer orders on affected products due
                         within 60 days (eval_cases.jsonl definition).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta

from supplyguard_ai.domain.models import AffectedOrder, ImpactAssessment, StockPosition
from supplyguard_ai.services.repository import OperationalRepository

AT_RISK_HORIZON_DAYS = 60


class ImpactError(ValueError):
    """Bad input (unknown component, no matching PO, ...)."""


@dataclass
class ImpactService:
    repo: OperationalRepository

    # --------------------------------------------------------------- demand
    def daily_demand_by_factory(self, component_id: str, on: date | None = None) -> dict[str, float]:
        """Component consumption per factory per day, from the production plan.

        We use the plan week that contains `on` (default: scenario date) -
        the plan is flat for MC-17, but taking the *current* week is the
        correct general rule.
        """
        on = on or self.repo.scenario_date()
        bom = self.repo.bom_for_component(component_id)
        qty_per_unit = {b["product_id"]: b["qty_per_unit"] for b in bom}  # P058 uses 3 x MC-17!
        demand: dict[str, float] = defaultdict(float)
        for row in self.repo.production_schedule(set(qty_per_unit)):
            week_start: date = row["week_start"]
            if week_start <= on < week_start + timedelta(days=7):
                demand[row["factory_id"]] += row["planned_units"] * qty_per_unit[row["product_id"]] / 7
        return dict(demand)

    # ---------------------------------------------------------------- stock
    def stock_positions(self, component_id: str) -> list[StockPosition]:
        """Per-location stock, with each plant's own days of cover.

        `usable` is recomputed rather than trusted from the CSV column, so a
        stale `usable` value can't hide reserved / quarantined stock.
        """
        loc_to_factory = {l["location_id"]: l["factory_id"] for l in self.repo.locations()}
        demand = self.daily_demand_by_factory(component_id)
        positions = []
        for r in self.repo.inventory(component_id):
            factory = loc_to_factory.get(r["location_id"])  # hub -> None
            usable = max(0, r["on_hand"] - r["reserved"] - r["quarantined"])
            positions.append(
                StockPosition(
                    location_id=r["location_id"],
                    factory_id=factory,
                    on_hand=r["on_hand"],
                    reserved=r["reserved"],
                    quarantined=r["quarantined"],
                    usable=usable,
                    daily_demand=demand.get(factory, 0.0) if factory else 0.0,
                )
            )
        # Deterministic order, independent of the storage backend (CSV file
        # order vs. SQL row order differed - found by the Postgres parity
        # test): plants by factory id first, then non-plant stock (hub).
        positions.sort(key=lambda p: (p.factory_id is None, p.factory_id or "", p.location_id))
        return positions

    # ---------------------------------------------------------------- main
    def assess(self, supplier_id: str, component_id: str, delay_days: int) -> ImpactAssessment:
        today = self.repo.scenario_date()
        if not self.repo.bom_for_component(component_id):
            raise ImpactError(f"Component {component_id} is not used in any product BOM")

        # 1. Which open PO from this supplier is delayed?  If several, the
        #    earliest one is the one that hurts first.
        open_pos = self.repo.open_purchase_orders(component_id)
        supplier_pos = sorted(
            (p for p in open_pos if p["supplier_id"] == supplier_id and isinstance(p["promised_date"], date)),
            key=lambda p: p["promised_date"],
        )
        affected_po = supplier_pos[0] if supplier_pos else None
        other_pos = [p["po_id"] for p in open_pos if p["supplier_id"] != supplier_id]

        # 2. Stock & demand.
        stock = self.stock_positions(component_id)
        total_usable = sum(s.usable for s in stock)
        daily_demand = sum(self.daily_demand_by_factory(component_id).values())

        # 3. Timeline.  Without a PO we can't know the original arrival; we
        #    then assume the delay starts today (conservative).
        original_day = (affected_po["promised_date"] - today).days if affected_po else 0
        delayed_day = original_day + delay_days
        stock_at_arrival = total_usable - daily_demand * original_day
        first_stockout = round(total_usable / daily_demand, 1) if daily_demand else None
        shortfall = max(0, round(daily_demand * delayed_day - total_usable))

        # 4. Business impact: open orders on affected products due soon.
        affected_products = sorted({b["product_id"] for b in self.repo.bom_for_component(component_id)})
        horizon = today + timedelta(days=AT_RISK_HORIZON_DAYS)
        at_risk = [
            AffectedOrder(
                order_id=o["order_id"],
                customer_id=o["customer_id"],
                product_id=o["product_id"],
                qty=o["qty"],
                due_date=o["due_date"],
                revenue_eur=o["revenue_eur"],
                priority_tier=o["customer_priority_tier"],
                late_penalty_pct_per_week=o["late_penalty_pct_per_week_capped5"] or 0.0,
            )
            for o in self.repo.customer_orders(set(affected_products))
            if o["status"] == "open" and isinstance(o["due_date"], date) and o["due_date"] <= horizon
        ]
        at_risk.sort(key=lambda o: (o.priority_tier, o.due_date))  # tier-1 customers first

        return ImpactAssessment(
            component_id=component_id,
            scenario_date=today,
            affected_po_id=affected_po["po_id"] if affected_po else None,
            original_arrival_day=original_day,
            delayed_arrival_day=delayed_day,
            stock=stock,
            total_on_hand=sum(s.on_hand for s in stock),
            total_reserved=sum(s.reserved for s in stock),
            total_quarantined=sum(s.quarantined for s in stock),
            total_usable=total_usable,
            daily_demand_total=round(daily_demand, 2),
            stock_at_original_arrival=round(stock_at_arrival, 1),
            first_stockout_day=first_stockout,
            shortfall_units=shortfall,
            affected_products=affected_products,
            other_open_po_ids=other_pos,
            at_risk_orders=at_risk,
            # Revenue EXPOSURE, not expected loss (assignment §07).
            revenue_at_risk_eur=round(sum(o.revenue_eur for o in at_risk), 2),
            # Real source ids from structured/data_sources.csv (provenance registry).
            sources=[
                "SRC-WMS-INVENTORY", "SRC-ERP-BOM", "SRC-MES-SCHEDULE",
                "SRC-PROCUREMENT-PO", "SRC-ERP-ORDERS",
            ],
        )
