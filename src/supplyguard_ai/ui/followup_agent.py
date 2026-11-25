"""
Follow-up Q&A over a saved case (assignment §03: "The user must also be able
to ask follow-up questions").

A small `create_agent` - bounded like the contract analyst:
  * tools bound to THIS case's saved state (closures): case_summary,
    explain_supplier, what_if_spending_cap, protect_priority_customers,
  * plus the shared read-only tools (search_documents, trace_dependencies),
  * no write tools: a what-if never changes the case; changing the plan goes
    through the approval card's "Modify" button -> re-plan -> new approval.

Why closures instead of a `case_id` argument?  The model can't ask about a
case it wasn't opened on: the case is fixed by our code, like the user.
Built per question (compiling a small agent takes milliseconds).
"""

from __future__ import annotations

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from supplyguard_ai.agents.middleware import audit_tool_calls
from supplyguard_ai.core.context import SupplyGuardContext
from supplyguard_ai.core.prompts import GROUNDING_RULES
from supplyguard_ai.core.tools import search_documents, trace_dependencies
from supplyguard_ai.services import followup as F

PROMPT = (
    "You answer follow-up questions about ONE saved supply-chain incident case for a procurement team.\n"
    "Call case_summary first if you need the case facts. Use explain_supplier for 'why was supplier X rejected/chosen', "
    "what_if_spending_cap for 'what if the limit is EUR N', protect_priority_customers for 'can existing stock protect "
    "our top customers'. Answer in short paragraphs or a small table, quote the tool numbers exactly, cite sources in "
    "[brackets]. A what-if is an estimate only - to change the plan the user must use Modify on the approval card.\n"
    "You CANNOT execute, order, release, transfer, approve or change anything. If the user asks you to, say clearly "
    "that you can't, explain what the rules allow, and that a new plan needs human approval. Never write that "
    "something 'has been' executed/ordered/released unless that exact action is listed in execution_results.\n\n"
    + GROUNDING_RULES
)


def case_tools(svc, case: dict) -> list:
    component = (case.get("intake", {}).get("facts") or {}).get("component_id")
    quarantine_lots = [{k: str(v) for k, v in r.items()} for r in svc.repo._table("quarantine_lots")
                       if r["component_id"] == component]

    @tool
    def case_summary() -> dict:
        """The saved case: facts, impact totals, selected plan, approval status, execution results, report excerpt."""
        imp = case.get("impact", {})
        return {
            "case_id": case.get("case_id"), "phase": case.get("phase"), "facts": case.get("intake", {}).get("facts"),
            "impact": {k: imp.get(k) for k in ("total_usable", "total_reserved", "total_quarantined", "shortfall_units",
                                               "daily_demand_total", "first_stockout_day", "revenue_at_risk_eur",
                                               "affected_products")},
            "at_risk_orders": len(imp.get("at_risk_orders", [])),
            # stock per location incl. what is NOT usable (found by the evaluation: without this the agent
            # improvised about a quarantined lot and claimed it had been "released")
            "stock_by_location": [{k: s.get(k) for k in ("location_id", "on_hand", "reserved", "quarantined", "usable")}
                                  for s in imp.get("stock", [])],
            "quarantine_lots": quarantine_lots,
            "stock_rules": "usable = on hand - reserved - quarantined; quarantined lots cannot be used or transferred; "
                           "reserved stock only via customer priority (DOC-PRO-06, DOC-PRO-07)",
            "plans": [{k: p[k] for k in ("plan_id", "total_cost_eur", "uncovered_units", "within_cap")} for p in case.get("plans", [])],
            "selected_plan": (case.get("selected_plan") or {}).get("plan_id"),
            "spending_cap_eur": case.get("spending_cap_eur"),
            "approval": {"id": case.get("approval_id"), "status": case.get("approval_status")},
            "challenge": case.get("challenge"),
            "execution_results": case.get("execution_results", []),
            "supplier_verdicts": case.get("investigation", {}).get("supplier_verdicts", []),
        }

    @tool
    def explain_supplier(supplier: str) -> dict:
        """Why a supplier was rejected or accepted: hard rules, risks, document conflicts, external alerts.
        `supplier` = supplier name as the user said it (e.g. "Helios") or id (e.g. "S002")."""
        return F.explain_supplier(svc, case, supplier)

    @tool
    def what_if_spending_cap(cap_eur: float) -> dict:
        """Re-plan with another spending cap (estimate only, nothing is approved or changed)."""
        return F.what_if_cap(svc, case, cap_eur)

    @tool
    def protect_priority_customers() -> dict:
        """How many tier-1 customer orders existing stock (usable + reserved) can protect, tier 1 first then by due date."""
        return F.protect_priority_customers(svc, case)

    return [case_summary, explain_supplier, what_if_spending_cap, protect_priority_customers]


def answer(ctx: SupplyGuardContext, case: dict, question: str, history: list | None = None, model=None) -> str:
    if model is None:
        from supplyguard_ai.llm import get_chat_model

        model = get_chat_model()
    agent = create_agent(
        model,
        tools=[*case_tools(ctx.services, case), search_documents, trace_dependencies],
        system_prompt=PROMPT,
        middleware=[audit_tool_calls, ToolCallLimitMiddleware(run_limit=8, exit_behavior="continue"),
                    ModelCallLimitMiddleware(run_limit=8, exit_behavior="end")],
        context_schema=SupplyGuardContext,
        name="case_followup",
    )
    state = agent.invoke({"messages": [*(history or [])[-6:], HumanMessage(question)]}, context=ctx,
                         config={"run_name": "case_followup", "metadata": {"case_id": ctx.case_id}})
    return state["messages"][-1].content
