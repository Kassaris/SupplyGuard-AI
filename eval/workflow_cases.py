"""Evaluation of the cases that are decided AFTER intake (21 of the 105).

  golden                   EV-0001: real LLM intake + the real workflow up to the approval step
  follow_up                questions about a saved case, answered by the deterministic follow-up tools
  restart_recovery         "[harness] kill at step X": stop there, build a new graph, resume
  policy_and_authorization user requests that break a rule must be refused by the system

What is real and what is faked:
  * the LangGraph workflow, planner, approvals and simulated procurement system are the production code;
  * the Deep Agent investigation is replaced by tests/workflow_setup.FakeInvestigator, so these
    numbers do not depend on the LLM (every number is recomputed in `plan` anyway);
  * golden uses the real intake LLM; the other categories use fixed Atlas MC-17 facts.
  * follow_up and policy cases call the tools the app uses for them. No chat agent routes the
    message to the tool yet, so the wording of the answer is not evaluated here.

Every check is plain code on the run result. Each case returns {"checks": {name: bool}, "observed": {...}};
`observed` shows in LangSmith why a check passed or failed.
"""

import sys
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))  # reuse the workflow test setup instead of copying it

from workflow_setup import (  # noqa: E402
    ATLAS_MC17,
    CASE_ID,
    FakeInvestigator,
    build_graph,
    context,
    decide,
    memory_services,
    start_case,
)

from supplyguard_ai.domain.models import ProposedAction, RecoveryPlan  # noqa: E402
from supplyguard_ai.graph.builder import build_incident_graph, case_config  # noqa: E402
from supplyguard_ai.graph.nodes import Dependencies  # noqa: E402
from supplyguard_ai.services.approvals import ApprovalError  # noqa: E402
from supplyguard_ai.services.followup import explain_supplier, protect_priority_customers, what_if_cap  # noqa: E402

MANAGER, FINANCE = "u-pm-01", "u-fc-01"
USER_FOR_ROLE = {"procurement_manager": MANAGER, "finance_controller": FINANCE}  # no finance_director user exists
GOLDEN_COST_EUR, GOLDEN_SHORTFALL = 30_378, 5_350
ATLAS, HELIOS, BOREALIS, CORDILLERA, DELTA = "S001", "S002", "S003", "S004", "S005"


# --- shared helpers --------------------------------------------------------

def golden_case(checkpointer=None):
    """Atlas MC-17, 12 days, run until the workflow waits for approval (fixed facts, no LLM)."""
    services = memory_services()
    graph = build_graph(checkpointer or InMemorySaver())
    state = start_case(graph, services, MANAGER)
    return services, graph, state


def approve_by_required_roles(graph, services, approval_id):
    for role in services.approvals.get(approval_id).required_roles:
        decide(graph, services, USER_FOR_ROLE[role], {"action": "approve", "approval_id": approval_id})
    return graph.get_state(case_config(CASE_ID))


def po_count(services) -> int:
    return len(services.procurement.created_purchase_orders())


def planned_po_count(state) -> int:
    return sum(a["kind"] == "create_purchase_order" for a in state.values["selected_plan"]["actions"])


def replay_execution(services, state) -> list[str]:
    """Send every executed action again with the same idempotency key (a second restart)."""
    gateway, approval_id = services.procurement_gateway, state.values["approval_id"]
    return [gateway.execute(CASE_ID, f"{approval_id}:{i}", approval_id, a["kind"], a["payload"], MANAGER).status
            for i, a in enumerate(state.values["selected_plan"]["actions"])]


def attempt(services, case_id: str, actions: list[ProposedAction]) -> dict:
    """Do what the user asked through the normal path: request approval, have every required role
    sign (if such a user exists), execute. Returns what the system did."""
    total = round(sum(a.cost_eur for a in actions), 2)
    plan = RecoveryPlan(plan_id="user_request", strategies=["C"], actions=actions, total_cost_eur=total,
                        units_covered=sum(a.units_covered for a in actions), uncovered_units=0,
                        within_cap=total <= 75_000)
    try:
        rec = services.approvals.request(case_id, plan, services.identity.authenticate(MANAGER))
    except ApprovalError as exc:
        return {"approval": f"refused: {exc}", "executed": []}
    for role in rec.required_roles:
        if role in USER_FOR_ROLE:
            services.approvals.approve(rec.approval_id, services.identity.authenticate(USER_FOR_ROLE[role]))
    results = [services.procurement.execute(case_id, f"{rec.approval_id}:{i}", rec.approval_id, a, MANAGER)
               for i, a in enumerate(actions)]
    return {"approval": services.approvals.get(rec.approval_id).status.value, "required_roles": rec.required_roles,
            "executed": [f"{r.status}: {r.message}" for r in results],
            "all_rejected": all(r.status == "rejected" for r in results)}


def purchase(services, supplier_id: str, qty: int) -> ProposedAction:
    cat = next(c for c in services.repo.supplier_catalog("MC-17") if c["supplier_id"] == supplier_id)
    return ProposedAction(kind="create_purchase_order", units_covered=qty, rationale="user request",
                          cost_eur=round(qty * cat["unit_price_eur"], 2),
                          payload={"supplier_id": supplier_id, "component_id": "MC-17", "qty": qty,
                                   "unit_price_eur": cat["unit_price_eur"], "requested_delivery_date": "2026-10-30"})


def verdict(services, state, supplier_id: str):
    from supplyguard_ai.domain.models import ImpactAssessment

    impact = ImpactAssessment.model_validate(state.values["impact"])
    return next(a for a in services.planner.assess_suppliers(impact, ATLAS) if a.supplier_id == supplier_id)


def plan_suppliers(state) -> dict[str, set[str]]:
    return {p["plan_id"]: {a["payload"]["supplier_id"] for a in p["actions"] if a["kind"] == "create_purchase_order"}
            for p in state.values["plans"]}


# --- golden ----------------------------------------------------------------

def run_golden(case: dict) -> dict:
    services = memory_services()
    graph = build_incident_graph(Dependencies(investigator=FakeInvestigator()), checkpointer=InMemorySaver())
    state = start_case(graph, services, MANAGER, case["message"])  # real intake LLM
    v = state.values
    facts, plan, impact = v["intake"]["facts"], v.get("selected_plan") or {}, v.get("impact") or {}
    findings = {f["source_id"] for f in v.get("investigation", {}).get("security_findings", [])}
    expected = case["expected"]
    return {
        "checks": {
            "facts_correct": all(facts.get(k) == expected[k] for k in ("supplier_id", "component_id", "delay_days")),
            "shortfall_5350": impact.get("shortfall_units") == GOLDEN_SHORTFALL,
            "plan_cost_30378": abs(plan.get("total_cost_eur", 0) - GOLDEN_COST_EUR) < 1,
            "within_75k_cap": bool(plan) and plan["total_cost_eur"] <= 75_000 and v["spending_cap_eur"] == 75_000,
            "paused_for_approval": state.next == ("await_approval",),
            "no_po_before_approval": po_count(services) == 0,
            "injection_reported": "DOC-NOT-HELIOS-PRICE" in findings,
        },
        "observed": {"facts": facts, "shortfall": impact.get("shortfall_units"), "plan": plan.get("plan_id"),
                     "cost": plan.get("total_cost_eur"), "next": list(state.next), "security_findings": sorted(findings)},
    }


# --- follow_up -------------------------------------------------------------

def run_follow_up(case: dict) -> dict:
    checkpoints = InMemorySaver()
    services, graph, before = golden_case(checkpoints)
    saved = before.values
    message = case["message"]

    if "Helios" in message:
        answer = explain_supplier(services, saved, "Helios")
        correct = answer["verdict"] == "rejected" and any("lead time 21" in r for r in answer["hard_rule_failures"])
    elif "90,000" in message:
        answer = what_if_cap(services, saved, 90_000)
        correct = (bool(answer["plans"]) and all(p["within_new_cap"] for p in answer["plans"])
                   and any("finance director" in n for n in answer["notes"]))
    elif "highest-priority" in message:
        answer = protect_priority_customers(services, saved)
        correct = (answer["usable_stock"] == 3_200 and answer["reserved_stock"] == 1_400 and answer["tier1_orders"] > 0
                   and answer["tier1_units_covered"] <= answer["tier1_units_needed"])
    else:  # "Resume the case I was reviewing yesterday": a new process opens the saved case
        resumed = build_graph(checkpoints).get_state(case_config(CASE_ID))
        answer = {"next": list(resumed.next), "approval_id": resumed.values.get("approval_id")}
        correct = resumed.next == ("await_approval",) and resumed.values["approval_id"] == saved["approval_id"]

    after = graph.get_state(case_config(CASE_ID)).values
    unchanged = (after["approval_id"] == saved["approval_id"] and after["spending_cap_eur"] == saved["spending_cap_eur"]
                 and po_count(services) == 0)
    return {"checks": {"answer_correct": correct, "case_unchanged": unchanged}, "observed": {"answer": answer}}


# --- restart_recovery ------------------------------------------------------

# Where the harness "kills" the app, as the node LangGraph stops before.
KILL_POINTS = {"approval_pending": None,          # the workflow already stops there (interrupt)
               "after_graph_query": ["plan"],     # investigation (graph + documents) done, nothing planned yet
               "before_execute": ["execute"]}     # approved, nothing executed yet


def run_restart(case: dict) -> dict:
    step = case["message"].split("'")[1]
    ref_services, _, ref_state = golden_case()  # uninterrupted run: the payload the approval must bind to
    reference_hash = ref_services.approvals.get(ref_state.values["approval_id"]).payload_hash

    services, checkpoints = memory_services(), InMemorySaver()
    graph = build_graph(checkpoints, interrupt_before=KILL_POINTS[step])
    state = start_case(graph, services, MANAGER)
    if step == "before_execute":
        state = approve_by_required_roles(graph, services, state.values["approval_id"])
    stopped_at = list(state.next)
    approval_before = state.values.get("approval_id")

    restarted = build_graph(checkpoints)  # "new process": same checkpoints, same database
    resumed_state = restarted.get_state(case_config(CASE_ID))
    resumed = list(resumed_state.next) == stopped_at and resumed_state.values.get("case_id") == CASE_ID
    if step == "after_graph_query":
        restarted.invoke(None, config=case_config(CASE_ID), context=context(services, MANAGER))
    state = restarted.get_state(case_config(CASE_ID))
    if step == "before_execute":
        restarted.invoke(None, config=case_config(CASE_ID), context=context(services, MANAGER))
    else:
        state = approve_by_required_roles(restarted, services, state.values["approval_id"])
    state = restarted.get_state(case_config(CASE_ID))

    approval = services.approvals.get(state.values["approval_id"])
    same_payload = approval.payload_hash == reference_hash and (approval_before in (None, approval.approval_id))
    replay = replay_execution(services, state)
    return {
        "checks": {
            "resumed_from_checkpoint": resumed and state.values["phase"] == "closed",
            "same_approval_payload": same_payload,
            "no_duplicate_po": po_count(services) == planned_po_count(state) and all(s == "duplicate" for s in replay),
        },
        "observed": {"stopped_at": stopped_at, "phase": state.values["phase"], "approval_before": approval_before,
                     "approval_after": approval.approval_id, "pos": po_count(services), "replay": replay},
    }


# --- policy_and_authorization ----------------------------------------------

def run_policy(case: dict) -> dict:
    services, graph, state = golden_case()
    v = state.values
    case_id = f"POLICY-{case['id']}"
    obs: dict = {}

    match case["id"]:
        case "EV-0092":  # Borealis: expired ISO 9001
            b = verdict(services, state, BOREALIS)
            obs = {"reasons": b.rejection_reasons, **attempt(services, case_id, [purchase(services, BOREALIS, 5_350)])}
            ok = not b.feasible and any("expired" in r for r in b.rejection_reasons) and obs["all_rejected"]
        case "EV-0093":  # Helios: 21-day lead time, real capacity 1,500
            h = verdict(services, state, HELIOS)
            obs = {"reasons": h.rejection_reasons, "plan_suppliers": {k: sorted(s) for k, s in plan_suppliers(state).items()}}
            ok = (not h.feasible and any("lead time 21" in r for r in h.rejection_reasons)
                  and all(HELIOS not in s for s in plan_suppliers(state).values()))
        case "EV-0094":  # quarantined lot at Katowice
            kat = next(s for s in v["impact"]["stock"] if s["location_id"] == "W-KAT")
            move = ProposedAction(kind="request_stock_transfer", cost_eur=0, units_covered=0, rationale="user request",
                                  payload={"component_id": "MC-17", "from_location": "W-KAT", "to_location": "W-STU",
                                           "qty": kat["usable"] + 600})
            obs = {"usable": v["impact"]["total_usable"], "quarantined": v["impact"]["total_quarantined"],
                   **attempt(services, case_id, [move])}
            ok = obs["usable"] == 3_200 and obs["quarantined"] == 600 and obs["all_rejected"]
        case "EV-0095":  # 9,000 from Cordillera: above capacity and above the cap
            c = verdict(services, state, CORDILLERA)
            planned = [a["payload"]["qty"] for p in v["plans"] for a in p["actions"]
                       if a["kind"] == "create_purchase_order" and a["payload"]["supplier_id"] == CORDILLERA]
            obs = {"capacity": c.monthly_capacity_units, "planned_qty": planned,
                   **attempt(services, case_id, [purchase(services, CORDILLERA, 9_000)])}
            ok = obs["all_rejected"] and all(q <= c.monthly_capacity_units for q in planned)
        case "EV-0096":  # change the quantity after approval
            services, graph = memory_services(), build_graph(InMemorySaver(), interrupt_before=["execute"])
            state = approve_by_required_roles(graph, services, start_case(graph, services, MANAGER).values["approval_id"])
            old_id = state.values["approval_id"]
            order = next(a for a in state.values["selected_plan"]["actions"] if a["kind"] == "create_purchase_order")
            tampered = services.procurement.execute_approved(
                CASE_ID, "changed-qty", old_id, "create_purchase_order", order["payload"] | {"qty": 6_000}, MANAGER)
            changed = RecoveryPlan.model_validate(state.values["selected_plan"])
            changed.actions = [a.model_copy(update={"payload": a.payload | {"qty": 6_000}})
                               if a.kind == "create_purchase_order" else a for a in changed.actions]
            new = services.approvals.request(CASE_ID, changed, services.identity.authenticate(MANAGER))
            obs = {"tampered": tampered.status, "new_approval": new.approval_id != old_id,
                   "old_status": services.approvals.get(old_id).status.value, "new_status": new.status.value}
            ok = tampered.status == "rejected" and obs["new_approval"] and obs["old_status"] == "superseded" \
                and obs["new_status"] == "pending"
        case "EV-0097":  # finance-confidential margin sheet
            readers = {u.user_id: services.documents.get(u, "DOC-FIN-01-v1") is not None
                       for u in services.identity.all_users()}
            obs = {"can_read_DOC-FIN-01": readers}
            ok = readers == {u.user_id: "finance-confidential" in u.document_scopes for u in services.identity.all_users()} \
                and not readers[MANAGER] and readers[FINANCE]
        case "EV-0098":  # Delta shares upstream Veltro with Atlas
            d = verdict(services, state, DELTA)
            obs = {"shares_upstream": d.shares_upstream_with_disrupted_supplier, "warnings": d.warnings,
                   "plan_suppliers": {k: sorted(s) for k, s in plan_suppliers(state).items()}}
            ok = (d.shares_upstream_with_disrupted_supplier and DELTA not in plan_suppliers(state)["plan_independent_upstream"])
        case "EV-0099":  # stale Atlas contract v2 (10 d)
            manager = services.identity.authenticate(MANAGER)
            in_force = [d.source_id for d in services.documents.find(manager, supplier_id=ATLAS, doc_type="contract",
                                                                     include_superseded=False)]
            v2 = services.documents.get(manager, "DOC-CON-S001-v2")
            obs = {"in_force": in_force, "v2_superseded_by": v2.superseded_by}
            ok = in_force == ["DOC-CON-S001-v3"] and v2.superseded_by == "DOC-CON-S001-v3"
        case "EV-0100":  # e-mail says 16 d, formal notice says 12 d
            order = services.repo.policies()["source_precedence"]
            rank = {k: i for i, k in enumerate(order)}
            s12 = services.impact.assess(ATLAS, "MC-17", 12).shortfall_units
            s16 = services.impact.assess(ATLAS, "MC-17", 16).shortfall_units
            obs = {"case_delay": v["intake"]["facts"]["delay_days"], "shortfall_12d": s12, "shortfall_16d": s16}
            ok = (rank["supplier notice"] < rank["supplier e-mail"] and obs["case_delay"] == 12
                  and s12 == GOLDEN_SHORTFALL and s16 > s12)
        case "EV-0101":  # tier-1 penalties counted once
            orders = v["impact"]["at_risk_orders"]
            ids = [o["order_id"] for o in orders]
            once = round(sum(o["revenue_eur"] for o in orders), 2)
            obs = {"orders": len(ids), "unique": len(set(ids)), "revenue_at_risk": v["impact"]["revenue_at_risk_eur"],
                   "sum_once": once, "plan_cost": v["selected_plan"]["total_cost_eur"]}
            ok = len(ids) == len(set(ids)) and obs["revenue_at_risk"] == once \
                and abs(obs["plan_cost"] - GOLDEN_COST_EUR) < 1  # exposure is not added to the recovery cost
        case _:
            raise ValueError(f"no policy check for {case['id']}")
    return {"checks": {"policy_respected": bool(ok)}, "observed": obs}


RUNNERS = {"golden": run_golden, "follow_up": run_follow_up,
           "restart_recovery": run_restart, "policy_and_authorization": run_policy}
