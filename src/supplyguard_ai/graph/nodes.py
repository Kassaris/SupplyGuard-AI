"""
Nodes of the incident workflow.  Each node = one step of assignment §03.

    intake ──► (clarify ⟲) ──► investigate ──► plan ──► challenge ──► request_approval ──► await_approval ⟲
                                                  ▲                                         │ approve / reject / modify
                                                  └────────────── modify ───────────────────┘
                                                                           approve ──► execute ──► report
                                                                           reject  ─────────────► report

NODE CONTRACT (LangGraph)
-------------------------
    def node(state: IncidentState, runtime: Runtime[SupplyGuardContext]) -> dict
* `state`   - the current case (read-only by convention),
* `runtime` - `.context` = who is acting + services; `.stream_writer` = push
              progress events to the UI (stream_mode="custom"),
* return    - ONLY the keys that changed.

WHERE THE LLM IS USED - AND WHERE IT IS NOT
-------------------------------------------
  LLM:  intake (extraction chain), investigate (Deep Agent + subagents).
  CODE: plan, challenge, approval, execute, report.
Every number that reaches the approval screen or the procurement system is
recomputed deterministically in `plan`.  The Deep Agent's report only
contributes *choices* (which plan, which document-resolved values), and those
are validated (`validate_overrides`, plan id must exist, plan must fit the cap).

HUMAN-IN-THE-LOOP: `interrupt()`
--------------------------------
`interrupt(payload)` pauses the graph and saves a checkpoint.  The app shows
`payload` to the user and later resumes with
    graph.invoke(Command(resume=<answer>), config={"configurable": {"thread_id": case_id}}, context=<user ctx>)
On resume LangGraph RE-RUNS THE NODE FROM THE TOP and `interrupt()` returns
the answer.  Therefore: no side effects BEFORE an interrupt in the same node.
That's why approval is split into `request_approval` (side effect: create the
approval record) and `await_approval` (interrupt first, then apply decision).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable
from langgraph.runtime import Runtime
from langgraph.types import interrupt

from supplyguard_ai.agents.contract_agent import validate_overrides
from supplyguard_ai.agents.investigator import investigate
from supplyguard_ai.agents.schemas import Conflict, InvestigationReport, PlannerOverride
from supplyguard_ai.core.chains import IntakeResult, build_intake_chain
from supplyguard_ai.core.context import SupplyGuardContext
from supplyguard_ai.domain.models import ApprovalStatus, ImpactAssessment, ProposedAction, RecoveryPlan
from supplyguard_ai.graph.state import IncidentState
from supplyguard_ai.services.approvals import ApprovalError, required_roles_for
from supplyguard_ai.services.identity import AuthorizationError, IdentityService
from supplyguard_ai.services.recovery import PlannerOverrides

Ctx = Runtime[SupplyGuardContext]


@dataclass
class Dependencies:
    """The LLM-backed parts, injectable so tests can replace them with fakes.

    intake_chain_factory: repo -> Runnable returning IntakeResult
    investigator:         a Deep Agent graph (None = build the real one lazily)
    """

    intake_chain_factory: Callable[[Any], Runnable] = field(default=lambda repo: build_intake_chain(repo))
    investigator: Any = None


def _progress(runtime: Ctx, phase: str, detail: str = "") -> None:
    """Push a progress event to the UI.  Visible with stream_mode="custom";
    a no-op for plain .invoke()."""
    if runtime.stream_writer:
        runtime.stream_writer({"phase": phase, "detail": detail})


class IncidentNodes:
    def __init__(self, deps: Dependencies | None = None) -> None:
        self.deps = deps or Dependencies()

    # ===================================================================== 1. intake
    def intake(self, state: IncidentState, runtime: Ctx) -> dict:
        """§03 step 1: extract structured facts; decide whether to ask the user."""
        ctx = runtime.context
        IdentityService.require_can_open_case(ctx.user)  # viewers can't open cases
        _progress(runtime, "intake", "extracting facts")

        result: IntakeResult = self.deps.intake_chain_factory(ctx.services.repo).invoke({"message": state["request"]})
        policies = ctx.services.repo.policies()
        # The user's cap wins, but can never exceed the policy cap (raising it is a separate, approved change).
        cap = min(result.facts.spending_cap_eur or policies["incident_spending_cap_eur"], policies["incident_spending_cap_eur"])
        ctx.services.audit.record("case.intake", state["case_id"], ctx.user.user_id, result.facts.model_dump(mode="json"))
        for f in result.security_findings:  # embedded instructions in the user's own message
            ctx.services.audit.record("security.prompt_injection_detected", state["case_id"], "intake",
                                      {"source_id": f["source_id"], "description": f["description"]})
        return {
            "intake": result.model_dump(mode="json"),
            "spending_cap_eur": state.get("spending_cap_eur", cap),
            "phase": "awaiting_clarification" if result.needs_clarification else "investigating",
        }

    # ===================================================================== 2. clarify
    def clarify(self, state: IncidentState, runtime: Ctx) -> dict:
        """Ask instead of guessing (§04 step 2).  NO side effects before interrupt()."""
        intake = state["intake"]
        answer = interrupt({
            "type": "clarification",
            "question": intake["clarification_question"],
            "candidates": intake.get("supplier_candidates", []),
        })
        # Re-run intake on the original message + the answer: the extraction
        # chain sees the full context, and the resolution logic re-checks it.
        return {
            "request": f"{state['request']}\nClarification from user: {answer}",
            "messages": [AIMessage(intake["clarification_question"]), HumanMessage(str(answer))],
            "phase": "intake",
        }

    # ===================================================================== 3. investigate
    def investigate(self, state: IncidentState, runtime: Ctx) -> dict:
        """§03 steps 2-5, 7: the Deep Agent investigation (impact, evidence, dependencies, options)."""
        ctx = runtime.context
        _progress(runtime, "investigating", "deep agent started")
        # Sandbox session for agent-written analysis scripts (SANDBOX_URL in compose).
        # Optional: if the service is down the investigation runs without `execute`.
        sandbox = None
        try:
            from supplyguard_ai.agents.sandbox import SandboxUnavailable, make_sandbox

            sandbox = make_sandbox(ctx.services.settings)
        except SandboxUnavailable as exc:
            _progress(runtime, "investigating", f"sandbox unavailable, continuing without it ({exc})")
        try:
            result = investigate(
                ctx, IntakeResult.model_validate(state["intake"]), state["request"], state["spending_cap_eur"],
                agent=self.deps.investigator, sandbox=sandbox,
            )
        finally:
            if sandbox is not None:
                sandbox.close()  # deletes the session's workdir on the sandbox service
        _progress(runtime, "investigating", "deep agent finished")
        return {
            "investigation": result.report.model_dump(mode="json"),
            "investigation_files": result.files,
            "provenance_issues": result.provenance_issues,
            "messages": [AIMessage(f"Investigation finished.\n\n{result.report.summary}")],
            "phase": "planning",
        }

    # ===================================================================== 4. plan
    def plan(self, state: IncidentState, runtime: Ctx) -> dict:
        """§03 step 6: deterministic trade-offs.  Recomputes EVERYTHING from data;
        the investigation only contributes validated choices."""
        ctx = runtime.context
        svc = ctx.services
        facts = state["intake"]["facts"]
        inv = InvestigationReport.model_validate(state["investigation"])

        impact = svc.impact.assess(facts["supplier_id"], facts["component_id"], facts["delay_days"])
        overrides, rejected = validate_overrides(
            [PlannerOverride.model_validate(o) for o in inv.planner_overrides],
            [Conflict.model_validate(c) for c in inv.conflicts],
            state.get("provenance_issues", []),
            svc.repo,
        )
        overrides = PlannerOverrides(
            capacity_units=overrides.capacity_units,
            expedite_fee_pct=overrides.expedite_fee_pct,
            excluded_suppliers=set(state.get("excluded_suppliers", [])),
        )
        cap = state["spending_cap_eur"]
        plans = svc.planner.build_plans(impact, facts["supplier_id"], cap, overrides)

        # Selection order: user's explicit choice > agent's recommendation > first plan within cap.
        by_id = {p.plan_id: p for p in plans}
        selected = None
        for candidate in (state.get("preferred_plan_id"), inv.recommended_plan_id):
            if candidate in by_id and by_id[candidate].within_cap:
                selected = by_id[candidate]
                break
        if selected is None:
            selected = next((p for p in plans if p.within_cap), None)

        return {
            "impact": impact.model_dump(mode="json"),
            "plans": [p.model_dump(mode="json") for p in plans],
            "selected_plan": selected.model_dump(mode="json") if selected else None,
            "override_rejections": rejected,
            "phase": "challenging",
        }

    # ===================================================================== 5. challenge
    def challenge(self, state: IncidentState, runtime: Ctx) -> dict:
        """§03 step 7: deterministic policy checks on the SELECTED plan, plus the
        agent's critic findings.  A blocker stops the case before approval."""
        ctx = runtime.context
        svc = ctx.services
        blockers, warnings, notes = [], [], []
        plan_d = state.get("selected_plan")
        if not plan_d:
            blockers.append(f"No recovery plan fits the spending cap of EUR {state['spending_cap_eur']:,.0f}.")
            return {"challenge": {"blockers": blockers, "warnings": warnings, "notes": notes}, "phase": "challenging"}

        plan = RecoveryPlan.model_validate(plan_d)
        impact = ImpactAssessment.model_validate(state["impact"])
        facts = state["intake"]["facts"]
        verdicts = {a.supplier_id: a for a in svc.planner.assess_suppliers(impact, facts["supplier_id"])}

        if plan.total_cost_eur > state["spending_cap_eur"]:
            blockers.append(f"Cost EUR {plan.total_cost_eur:,.2f} exceeds the cap EUR {state['spending_cap_eur']:,.0f} (DOC-PRO-03).")
        for a in plan.actions:
            p = a.payload
            if a.kind == "create_purchase_order":
                v = verdicts.get(p["supplier_id"])
                if v is None or not v.feasible:
                    blockers.append(f"PO to {p['supplier_id']} violates qualification rules: {v.rejection_reasons if v else 'unknown supplier'} (DOC-PRO-04).")
                elif v.shares_upstream_with_disrupted_supplier:
                    warnings.append(f"{v.supplier_name} shares upstream {v.upstream_manufacturer_id} with the disrupted supplier: not real diversification (DOC-PRO-05).")
            elif a.kind == "expedite_purchase_order":
                po = svc.repo.purchase_order(p["po_id"])
                if not po or po["status"] != "open" or po.get("expedite_allowed") != "yes":
                    blockers.append(f"{p['po_id']} cannot be expedited (DOC-PRO-08).")
            elif a.kind == "request_stock_transfer":
                src = next((s for s in impact.stock if s.location_id == p["from_location"]), None)
                if not src or src.usable < p["qty"]:
                    blockers.append(f"Transfer of {p['qty']} from {p['from_location']} exceeds usable stock (DOC-PRO-06).")
        # External risk alerts (risk_sweep) on suppliers this plan actually buys from / expedites with.
        used = {a.payload.get("supplier_id") for a in plan.actions if a.kind == "create_purchase_order"}
        used |= {(svc.repo.purchase_order(a.payload["po_id"]) or {}).get("supplier_id")
                 for a in plan.actions if a.kind == "expedite_purchase_order"}
        for al in state.get("investigation", {}).get("external_alerts", []):
            if al["supplier_id"] in used:
                (warnings if al["severity"] == "high" else notes).append(
                    f"External alert ({al['severity']}) for {al['supplier_name']} via {al['via']}, region "
                    f"{al['region']}: {al['text']} [EXTERNAL-RISK-FEED, untrusted]")
        if plan.uncovered_units:
            warnings.append(f"{plan.uncovered_units} units remain uncovered; handle via rescheduling (E) or tier-1 reallocation (D).")

        inv = state.get("investigation", {})
        if state.get("provenance_issues"):
            warnings.append(f"Unverified citations in the investigation: {state['provenance_issues']}")
        notes += [f"Override not applied: {r}" for r in state.get("override_rejections", [])]
        notes += [f"Critic: {c}" for c in inv.get("critic_findings", [])]
        notes += [f"Security: {s['source_id']} contained injected instructions - ignored." for s in inv.get("security_findings", [])]
        roles = required_roles_for(plan.total_cost_eur, svc.repo.policies()["approval_tiers_eur"])
        notes.append(f"Approval tier: {' + '.join(roles)} (DOC-PRO-03).")
        notes.append(f"Revenue at risk EUR {impact.revenue_at_risk_eur:,.0f} is exposure, not a loss, and is not part of the recovery cost (DOC-PRO-12).")

        return {"challenge": {"blockers": blockers, "warnings": warnings, "notes": notes}, "phase": "challenging"}

    # ===================================================================== 6. request approval
    def request_approval(self, state: IncidentState, runtime: Ctx) -> dict:
        """§03 step 8: create the payload-bound approval record (idempotent:
        same payload -> same pending record, so a re-run never duplicates it)."""
        ctx = runtime.context
        plan = RecoveryPlan.model_validate(state["selected_plan"])
        rec = ctx.services.approvals.request(state["case_id"], plan, ctx.user)
        lines = [f"Approval needed ({' + '.join(rec.required_roles)}) for plan {plan.plan_id}: "
                 f"EUR {plan.total_cost_eur:,.2f} total, {plan.uncovered_units} units uncovered."]
        lines += [f"- {a.kind}: {a.payload} (EUR {a.cost_eur:,.2f})" for a in plan.actions]
        return {
            "approval_id": rec.approval_id,
            "approval_status": rec.status.value,
            "messages": [AIMessage("\n".join(lines))],
            "phase": "awaiting_approval",
        }

    # ===================================================================== 7. await approval
    def await_approval(self, state: IncidentState, runtime: Ctx) -> dict:
        """Pause for a human decision.  The DECIDER is whoever resumes the graph
        (runtime.context.user, set by the app after login) - never a user id
        inside the resume payload, which a client could forge.

        Resume value: {"action": "approve" | "reject" | "modify", "approval_id": str (required for
                       approve/reject), "reason": str,
                       "changes": {"spending_cap_eur"?, "excluded_suppliers"?, "preferred_plan_id"?}}
        """
        ctx = runtime.context
        svc = ctx.services
        rec = svc.approvals.get(state["approval_id"])
        decision = interrupt({
            "type": "approval",
            "approval_id": rec.approval_id,
            "payload_hash": rec.payload_hash,
            "plan": state["selected_plan"],
            "required_roles": rec.required_roles,
            "signed_by": rec.approvals,
            "challenge": state.get("challenge", {}),
            "evidence": {k: state.get("investigation", {}).get(k) for k in ("key_numbers", "supplier_verdicts", "conflicts")},
        })
        # ---- everything below runs exactly once per resume
        action = (decision or {}).get("action")
        user = ctx.user
        update: dict[str, Any] = {"last_decision": {**(decision or {}), "by": user.user_id, "role": user.role.value}}

        # A decision must name the approval the human actually looked at.
        # Found in the UI test: after "switch plan", the OLD card's Approve
        # button was still clickable and would have approved the NEW payload.
        if action in ("approve", "reject") and (decision or {}).get("approval_id") != rec.approval_id:
            return update | {"messages": [AIMessage(
                f"Decision refused: it refers to approval {(decision or {}).get('approval_id') or '(none)'}, "
                f"but the pending request is {rec.approval_id}. Please review the current approval card.")]}

        if action == "approve":
            try:
                rec = svc.approvals.approve(rec.approval_id, user)
                msg = f"{user.name} ({user.role.value}) approved. Status: {rec.status.value}."
            except (AuthorizationError, ApprovalError) as exc:
                msg = f"Approval by {user.name} not accepted: {exc}"
            update |= {"approval_status": rec.status.value, "messages": [AIMessage(msg)]}

        elif action == "reject":
            try:
                rec = svc.approvals.reject(rec.approval_id, user, decision.get("reason", ""))
                update |= {"approval_status": rec.status.value, "phase": "rejected",
                           "messages": [AIMessage(f"Plan rejected by {user.name}: {decision.get('reason', '')}")]}
            except AuthorizationError as exc:
                update |= {"messages": [AIMessage(f"Rejection not accepted: {exc}")]}

        elif action == "modify":
            # Any change -> re-plan -> new payload -> new approval (the old one gets SUPERSEDED).
            changes = decision.get("changes", {})
            raised = svc.repo.policies()["raised_cap_eur_with_finance_director_approval"]
            if "spending_cap_eur" in changes:
                update["spending_cap_eur"] = min(float(changes["spending_cap_eur"]), raised)
            if "excluded_suppliers" in changes:
                update["excluded_suppliers"] = list(changes["excluded_suppliers"])
            if "preferred_plan_id" in changes:
                update["preferred_plan_id"] = changes["preferred_plan_id"]
            update |= {"approval_status": "modified", "phase": "planning",
                       "messages": [AIMessage(f"Changes received {changes}; re-planning and requesting a new approval.")]}
        else:
            update |= {"messages": [AIMessage("Unrecognised decision; still awaiting approval.")]}
        return update

    # ===================================================================== 8. execute
    def execute(self, state: IncidentState, runtime: Ctx) -> dict:
        """§03 step 9: run the approved actions in the procurement system.

        Sends only kind + payload (what an external system can know); the
        procurement side re-checks them against the approved record.

        idempotency key = case + approval + action index: if the process dies
        mid-way and this node re-runs, already-executed actions return
        "duplicate" with the original reference instead of running twice.
        """
        ctx = runtime.context
        _progress(runtime, "executing")
        results = []
        gateway = ctx.services.procurement_gateway  # MCP server in docker compose, in-process otherwise
        for i, action_d in enumerate(state["selected_plan"]["actions"]):
            action = ProposedAction.model_validate(action_d)
            r = gateway.execute(state["case_id"], f"{state['approval_id']}:{i}", state["approval_id"],
                                action.kind, action.payload, ctx.user.user_id)
            results.append(r.model_dump(mode="json"))
        ok = sum(r["status"] in ("created", "duplicate") for r in results)
        return {
            "execution_results": results,
            "messages": [AIMessage(f"Executed {ok}/{len(results)} approved actions: "
                                   + ", ".join(r["reference"] or r["message"] for r in results))],
            "phase": "reporting",
        }

    # ===================================================================== 9. report
    def report(self, state: IncidentState, runtime: Ctx) -> dict:
        """§03 step 10: incident report (DOC-PRO-09 template) + approved summary for future cases."""
        md, memory = render_report(state)
        if state.get("challenge", {}).get("blockers"):
            phase = "no_feasible_plan"
        elif state.get("approval_status") == ApprovalStatus.REJECTED.value:
            phase = "rejected"
        else:
            phase = "closed"
        runtime.context.services.audit.record("case.closed", state["case_id"], runtime.context.user.user_id, {"phase": phase})
        return {"report_md": md, "resolution_memory": memory, "phase": phase,
                "messages": [AIMessage(f"Case {state['case_id']} {phase}. Incident report ready.")]}


# =========================================================================== report rendering
def render_report(state: IncidentState) -> tuple[str, str | None]:
    """Deterministic markdown so every number in the report is the number that
    was computed and approved.  Sections follow skills/incident-report/SKILL.md."""
    inv, imp, plan = state.get("investigation", {}), state.get("impact", {}), state.get("selected_plan") or {}
    facts = state.get("intake", {}).get("facts", {})
    ch = state.get("challenge", {})
    results = state.get("execution_results", [])
    out = [f"# Incident report - {state['case_id']}", ""]

    out += ["## 1. Summary", "_Investigator summary (AI-generated, written before the human decision):_",
            inv.get("summary", "-"), ""]
    out += ["## 2. Facts and sources",
            f"- Supplier {facts.get('supplier_id')} ({facts.get('supplier_name')}), component {facts.get('component_id')}, "
            f"stated delay {facts.get('delay_days')} d, affected PO {imp.get('affected_po_id')}, scenario date {imp.get('scenario_date')}",
            f"- Sources: {', '.join(imp.get('sources', []))}", ""]
    if imp:
        out += ["## 3. Impact",
                f"- Stock: on hand {imp['total_on_hand']}, reserved {imp['total_reserved']}, quarantined {imp['total_quarantined']}, "
                f"**usable {imp['total_usable']}**",
                "- Days of cover: " + ", ".join(f"{s['location_id']} {s['days_of_cover']}" for s in imp["stock"] if s.get("days_of_cover")),
                f"- Demand {imp['daily_demand_total']}/day; first stock-out day {imp['first_stockout_day']}; **shortfall {imp['shortfall_units']} units**",
                f"- Affected products: {', '.join(imp['affected_products'])}; at-risk orders: {len(imp['at_risk_orders'])}",
                f"- Revenue at risk (exposure, not a loss): EUR {imp['revenue_at_risk_eur']:,.0f}", ""]
    if state.get("plans"):
        out += ["## 4. Options compared", "| Plan | Strategies | Cost EUR | Uncovered | Within cap |", "|---|---|---|---|---|"]
        out += [f"| {p['plan_id']} | {''.join(p['strategies'])} | {p['total_cost_eur']:,.2f} | {p['uncovered_units']} | {p['within_cap']} |"
                for p in state["plans"]]
        out += [f"- {v['supplier_name']} ({v['supplier_id']}): {v['verdict']} - {v['reason']}" for v in inv.get("supplier_verdicts", [])]
        out.append("")
    if not plan:
        rec_lines = ["No feasible plan."]
    elif plan.get("plan_id") == inv.get("recommended_plan_id"):
        rec_lines = [f"**{plan['plan_id']}** (recommended by the investigation): {inv.get('recommendation_rationale', '-')}"]
    else:  # the human chose another plan than the agent: say so, don't show the agent's rationale as if it fit
        rec_lines = [f"**{plan['plan_id']}** - selected by the approver.",
                     f"The investigation had recommended {inv.get('recommended_plan_id')}: {inv.get('recommendation_rationale', '-')}"]
    out += ["## 5. Recommendation", *rec_lines, ""]
    out += ["## 6. Approval record",
            f"- Approval {state.get('approval_id')} - status {state.get('approval_status')} - last decision {state.get('last_decision')}", ""]
    out += ["## 7. Execution outcome"] + ([f"- {r['operation']}: {r['status']} {r['reference'] or r['message']}" for r in results] or ["- nothing executed"]) + [""]
    out += ["## 8. Residual risks"] + [f"- {w}" for w in ch.get("warnings", []) + ch.get("blockers", [])] + \
           [f"- Conflict {c['supplier_id']} {c['topic']}: used {c['resolved_value']} ({c.get('planning_risk') or 'no residual risk'})"
            for c in inv.get("conflicts", [])] + [""]
    out += ["## 9. Lessons learned"] + [f"- {m}" for m in inv.get("similar_past_incidents", [])] + ["- (to be completed by the incident coordinator)", ""]
    out += ["## 10. Security findings"] + ([f"- {s['source_id']}: {s['description']} - {s['action_taken']}" for s in inv.get("security_findings", [])] or ["- none"])

    memory = None
    executed = [r for r in results if r["status"] in ("created", "duplicate")]
    if plan and executed and len(executed) == len(results):  # only APPROVED + fully EXECUTED outcomes become memory
        memory = (f"supplier delay ({facts.get('delay_days')} d) on {facts.get('component_id')} from {facts.get('supplier_id')} "
                  f"resolved with strategies {''.join(plan['strategies'])} at EUR {plan['total_cost_eur']:,.0f}; "
                  f"{plan['uncovered_units']} units handled by D/E.")
    return "\n".join(out), memory
