"""
State of one incident case - the "memory" of the LangGraph workflow.

LANGGRAPH MENTAL MODEL (Week 2, day 5): State, Nodes, Edges
-----------------------------------------------------------
* State  = this TypedDict.  Every node receives it and returns a PARTIAL
           update (only the keys it changed).
* Nodes  = Python functions (graph/nodes.py).
* Edges  = which node runs next; conditional edges decide from the state
           (graph/builder.py).
With a checkpointer, the whole state is saved after every node.  That is
what makes "resume the case I was reviewing yesterday" and "survive a restart
while approval is pending" possible: the state is reloaded from the
checkpoint by `thread_id` (= case id).

DESIGN DECISIONS
----------------
1. JSON-friendly values only (dicts, lists, str, numbers).  Pydantic objects
   are stored as `model_dump(mode="json")`.  Checkpoints are then plain JSON
   in Postgres: readable, portable, and safe across code upgrades.
2. `messages` uses the `add_messages` REDUCER: a node returning
   {"messages": [msg]} APPENDS to the conversation instead of replacing it.
   All other keys are simply overwritten by the latest update.
3. Secrets, services and the user object are NOT in state.  They travel in
   the runtime context (core/context.py), so they never land in checkpoints.
4. `phase` is a human-readable status for the UI and for routing on resume.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, NotRequired, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

Phase = Literal[
    "intake", "awaiting_clarification", "investigating", "planning", "challenging",
    "awaiting_approval", "executing", "reporting", "closed", "rejected", "no_feasible_plan",
]


class IncidentState(TypedDict):
    # --- conversation (appended by the add_messages reducer)
    messages: Annotated[list[AnyMessage], add_messages]

    # --- case identity
    case_id: str
    phase: NotRequired[Phase]
    request: NotRequired[str]  # the user's incident message (+ clarifications)

    # --- intake (core/chains.py IntakeResult as dict)
    intake: NotRequired[dict[str, Any]]

    # --- investigation (Deep Agent InvestigationReport as dict)
    investigation: NotRequired[dict[str, Any]]
    investigation_files: NotRequired[dict[str, str]]
    provenance_issues: NotRequired[list[str]]

    # --- deterministic analysis
    impact: NotRequired[dict[str, Any]]  # ImpactAssessment
    plans: NotRequired[list[dict[str, Any]]]  # RecoveryPlan list
    selected_plan: NotRequired[dict[str, Any]]
    override_rejections: NotRequired[list[str]]
    challenge: NotRequired[dict[str, Any]]  # {"blockers": [...], "warnings": [...], "notes": [...]}

    # --- decision inputs the user may change ("what if the cap is EUR 90,000?")
    spending_cap_eur: NotRequired[float]
    excluded_suppliers: NotRequired[list[str]]
    preferred_plan_id: NotRequired[str | None]

    # --- approval & execution
    approval_id: NotRequired[str | None]
    approval_status: NotRequired[str | None]
    last_decision: NotRequired[dict[str, Any] | None]
    execution_results: NotRequired[list[dict[str, Any]]]

    # --- outputs
    report_md: NotRequired[str]
    resolution_memory: NotRequired[str | None]  # approved summary for future cases
