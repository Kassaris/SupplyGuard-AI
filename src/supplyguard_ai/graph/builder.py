"""
Builds and compiles the incident workflow graph.

WHY LANGGRAPH FOR THE MAIN LIFECYCLE (assignment §05)
----------------------------------------------------
"Main incident lifecycle, routing, persistent execution, interrupts, and
recovery."  The lifecycle is a FIXED business process (intake -> ... ->
approval -> execution) with a few decision points.  Encoding it as an
explicit graph - instead of letting one big agent decide the order - means:
  * the approval gate can't be skipped: there is simply no edge from
    `challenge` to `execute`,
  * every step is checkpointed -> resume after restart, time-travel debug,
  * each node is testable in isolation,
  * the LLM-heavy parts (intake, investigate) are just two nodes among nine.

ROUTING FUNCTIONS
-----------------
A conditional edge calls a small pure function on the state and returns the
name of the next node.  Keeping them pure (no I/O) makes the control flow
easy to read and to unit-test.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from supplyguard_ai.core.context import SupplyGuardContext
from supplyguard_ai.graph.nodes import Dependencies, IncidentNodes
from supplyguard_ai.graph.state import IncidentState


# --------------------------------------------------------------------------- routers
def route_after_intake(state: IncidentState) -> str:
    return "clarify" if state["intake"]["needs_clarification"] else "investigate"


def route_after_challenge(state: IncidentState) -> str:
    return "report" if state["challenge"]["blockers"] else "request_approval"


def route_after_decision(state: IncidentState) -> str:
    status = state.get("approval_status")
    if status == "approved":
        return "execute"
    if status == "rejected":
        return "report"
    if status == "modified":
        return "plan"  # re-plan -> challenge -> NEW approval (old one superseded)
    return "await_approval"  # still pending (e.g. 1 of 2 signatures) -> ask the next approver


# --------------------------------------------------------------------------- builder
def build_incident_graph(deps: Dependencies | None = None, checkpointer=None, store=None, interrupt_before=None):
    """Assemble nodes + edges and compile.

    checkpointer: InMemorySaver for tests / PostgresSaver in production.
                  REQUIRED for interrupts: a paused graph must be saved somewhere.
    store:        long-term memory (approved resolutions) - added with pgvector.
    """
    n = IncidentNodes(deps)
    g = StateGraph(IncidentState, context_schema=SupplyGuardContext)

    for name in ("intake", "clarify", "investigate", "plan", "challenge",
                 "request_approval", "await_approval", "execute", "report"):
        g.add_node(name, getattr(n, name))

    g.add_edge(START, "intake")
    g.add_conditional_edges("intake", route_after_intake, ["clarify", "investigate"])
    g.add_edge("clarify", "intake")
    g.add_edge("investigate", "plan")
    g.add_edge("plan", "challenge")
    g.add_conditional_edges("challenge", route_after_challenge, ["request_approval", "report"])
    g.add_edge("request_approval", "await_approval")
    g.add_conditional_edges("await_approval", route_after_decision, ["execute", "report", "plan", "await_approval"])
    g.add_edge("execute", "report")
    g.add_edge("report", END)

    # interrupt_before: used by the evaluation harness to simulate "the app died right
    # before node X" (LangGraph checkpoints and stops exactly there).
    return g.compile(checkpointer=checkpointer, store=store, interrupt_before=interrupt_before,
                     name="supplyguard_incident")


# --------------------------------------------------------------------------- helpers for the app
def new_case_input(case_id: str, message: str) -> dict:
    """Initial state for a new case."""
    from langchain_core.messages import HumanMessage

    return {"case_id": case_id, "request": message, "messages": [HumanMessage(message)], "phase": "intake"}


def case_config(case_id: str) -> dict:
    """thread_id = case_id: each case is one persistent LangGraph thread, so
    'resume the case I was reviewing yesterday' = invoke with the same id."""
    return {"configurable": {"thread_id": case_id}, "recursion_limit": 100,
            "run_name": "incident_workflow", "metadata": {"case_id": case_id}}
