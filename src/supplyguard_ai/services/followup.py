"""
Deterministic answers to the follow-up questions of assignment §03:

  "Why did you reject Supplier Helios?"                      -> explain_supplier()
  "What changes if we increase the spending limit to €90,000?" -> what_if_cap()
  "Can we protect our highest-priority customers using existing stock?" -> protect_priority_customers()

Each function reads the SAVED case (LangGraph state) plus live data and
returns numbers with sources.  The follow-up agent (ui/followup_agent.py)
only chooses which function to call and phrases the answer.  A what-if never
changes the case: it computes, it doesn't approve.
"""

from __future__ import annotations

from supplyguard_ai.domain.models import ImpactAssessment
from supplyguard_ai.services.approvals import ApprovalError, required_roles_for


def explain_supplier(svc, case: dict, supplier: str) -> dict:
    """Every reason behind a supplier verdict: rules, graph, documents, external feed.

    `supplier` may be an id (S002) or a name ("Helios"): users ask by name.
    Found in the UI test: the agent passed "Helios" as an id, got "not a
    catalog supplier" and told the user Helios was never considered.
    """
    candidates = svc.repo.resolve_supplier(supplier)
    if not candidates:
        return {"error": f"no supplier matches '{supplier}'"}
    if len(candidates) > 1:
        return {"ambiguous": True, "candidates": [{"supplier_id": c["supplier_id"], "name": c["name"]} for c in candidates]}
    supplier_id = candidates[0]["supplier_id"]
    facts = case["intake"]["facts"]
    impact = ImpactAssessment.model_validate(case["impact"])
    rules = next((a for a in svc.planner.assess_suppliers(impact, facts["supplier_id"]) if a.supplier_id == supplier_id), None)
    if rules is None:
        return {"error": f"{supplier_id} is not a catalog supplier of {facts['component_id']}"}
    inv = case.get("investigation", {})
    return {
        "supplier": f"{rules.supplier_name} ({supplier_id})",
        "verdict": "feasible" if rules.feasible else "rejected",
        "hard_rule_failures": rules.rejection_reasons,          # DOC-PRO-04 + timing
        "risks": rules.warnings,                                 # DOC-PRO-05 shared upstream ...
        "numbers": {"unit_price_eur": rules.unit_price_eur, "lead_time_days": rules.lead_time_days,
                    "monthly_capacity_units": rules.monthly_capacity_units,
                    "delayed_arrival_day": impact.delayed_arrival_day},
        "investigator_verdict": next((v for v in inv.get("supplier_verdicts", []) if v["supplier_id"] == supplier_id), None),
        "document_conflicts": [c for c in inv.get("conflicts", []) if c["supplier_id"] == supplier_id],
        "external_alerts": [a for a in inv.get("external_alerts", []) if a["supplier_id"] == supplier_id],
        "sources": ["SRC-ERP-SUPPLIER-MASTER", "DOC-PRO-04-v1", "DOC-PRO-05-v1"],
    }


def what_if_cap(svc, case: dict, cap_eur: float) -> dict:
    """Re-run the deterministic planner with another cap.  Nothing is approved or saved."""
    facts = case["intake"]["facts"]
    impact = ImpactAssessment.model_validate(case["impact"])
    policies = svc.repo.policies()
    plans = svc.planner.build_plans(impact, facts["supplier_id"], cap_eur)
    out = []
    for p in plans:
        try:
            roles = required_roles_for(p.total_cost_eur, policies["approval_tiers_eur"])
        except ApprovalError:
            roles = ["not approvable (above highest tier)"]
        out.append({"plan_id": p.plan_id, "total_cost_eur": p.total_cost_eur, "uncovered_units": p.uncovered_units,
                    "within_new_cap": p.within_cap, "approval_needed": " + ".join(roles), "caveats": p.caveats})
    current = case.get("spending_cap_eur")
    notes = []
    if cap_eur > policies["incident_spending_cap_eur"]:
        notes.append(f"A cap above EUR {policies['incident_spending_cap_eur']:,.0f} needs finance director approval "
                     f"(max EUR {policies['raised_cap_eur_with_finance_director_approval']:,.0f}, DOC-PRO-03).")
    if all(p["within_new_cap"] for p in out) and current and all(p.total_cost_eur <= current for p in plans):
        notes.append(f"All plans already fit the current cap of EUR {current:,.0f}: raising it changes nothing "
                     "unless a more expensive option (e.g. a larger alternative purchase) is wanted.")
    return {"cap_eur": cap_eur, "current_cap_eur": current, "plans": out, "notes": notes,
            "sources": ["config/policies.json", "SRC-ERP-SUPPLIER-MASTER", "SRC-PROCUREMENT-PO"]}


def protect_priority_customers(svc, case: dict) -> dict:
    """Allocate EXISTING stock to at-risk orders by priority (DOC-PRO-07: tier 1 first, then earliest due).

    Static allocation: reserved stock stays with the order it is reserved for;
    usable stock goes to the remaining orders in priority order.  It ignores
    production timing - an upper bound on what existing stock can protect.
    """
    facts = case["intake"]["facts"]
    impact = ImpactAssessment.model_validate(case["impact"])
    qpu = {b["product_id"]: b["qty_per_unit"] for b in svc.repo.bom_for_component(facts["component_id"])}
    reserved = {r["customer_order_id"]: r["reserved_qty"] for r in svc.repo._table("reservations")
                if r["component_id"] == facts["component_id"]}

    free = impact.total_usable
    rows = []
    for o in sorted(impact.at_risk_orders, key=lambda o: (o.priority_tier, o.due_date)):
        need = o.qty * qpu[o.product_id]
        from_reserved = min(need, reserved.get(o.order_id, 0))
        from_usable = min(need - from_reserved, free)
        free -= from_usable
        rows.append({"order_id": o.order_id, "customer_id": o.customer_id, "tier": o.priority_tier,
                     "due_date": o.due_date.isoformat(), "need_units": need, "from_reserved": from_reserved,
                     "from_usable": from_usable, "covered": from_reserved + from_usable >= need})

    tier1 = [r for r in rows if r["tier"] == 1]
    covered1 = [r for r in tier1 if r["covered"]]
    first_gap = next((r for r in tier1 if not r["covered"]), None)
    return {
        "usable_stock": impact.total_usable, "reserved_stock": impact.total_reserved,
        "tier1_orders": len(tier1), "tier1_fully_covered": len(covered1),
        "tier1_units_needed": sum(r["need_units"] for r in tier1),
        "tier1_units_covered": sum(r["from_reserved"] + r["from_usable"] for r in tier1),
        "first_uncovered_tier1_order": first_gap,
        "orders_below_tier1_covered": sum(r["covered"] for r in rows if r["tier"] > 1),
        "covered_tier1_orders": [r["order_id"] for r in covered1],
        "assumption": "static allocation of current stock, production timing ignored (upper bound)",
        "sources": ["SRC-WMS-INVENTORY", "SRC-ERP-ORDERS", "SRC-ERP-BOM", "DOC-PRO-06-v1", "DOC-PRO-07-v1"],
    }
