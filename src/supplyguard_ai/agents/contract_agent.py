"""
Contract analyst: a BOUNDED specialist agent built with `create_agent`.

ASSIGNMENT §05: "create_agent - A bounded specialist agent, such as contract analysis."

WHY AN AGENT HERE (and not a chain)
-----------------------------------
Which documents matter is not known in advance: the model lists documents,
sees "SUPERSEDED", reads the newer version, notices an e-mail contradicting
the contract, reads it too ...  That "decide what to read next" loop is exactly
the ReAct pattern that `create_agent` implements.

WHY `create_agent` (and not a Deep Agent)
-----------------------------------------
The task is narrow: ~10 documents, one output schema.  It needs no planning
todo-list, no virtual filesystem, no subagents.  `create_agent` is the
smallest construct that does the job, and smaller = cheaper, faster, easier
to evaluate and to secure.  The Deep Agent (next file) handles the open-ended
investigation and can delegate to THIS agent.

WHAT MAKES IT "BOUNDED"
-----------------------
1. Tools: only document reading + policy lookup.  No impact/plan tools, no
   writes at all.                                     (least privilege)
2. Limits: max 30 tool calls and 20 model calls per run.  (cost / loops)
3. Output: must be a valid `ContractFindings` object.     (shape)
4. Provenance guard: every cited source is verified after the run.   (truth)

THE PIECES OF `create_agent(...)` USED BELOW
--------------------------------------------
  model           -> our configurable LLM from llm.py
  tools           -> LangChain tools from core/tools.py
  system_prompt   -> role + grounding rules + two skills' text
  middleware      -> audit, limits, provenance (agents/middleware.py)
  response_format -> ToolStrategy(ContractFindings): the final answer is a
                     "tool call" whose arguments are validated by Pydantic;
                     on a validation error the model is told the error and
                     retries.  Works with any tool-calling model (Azure,
                     OpenAI, Ollama), unlike provider-native JSON modes.
  context_schema  -> SupplyGuardContext: who is asking (scopes!)
"""

from __future__ import annotations

from functools import lru_cache

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware
from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from supplyguard_ai.agents.middleware import audit_tool_calls, check_provenance, provenance_guard
from supplyguard_ai.agents.schemas import Conflict, ContractFindings, PlannerOverride
from supplyguard_ai.config import PROJECT_ROOT
from supplyguard_ai.core.context import SupplyGuardContext
from supplyguard_ai.core.prompts import CONTRACT_ANALYST_PROMPT
from supplyguard_ai.core.tools import DOCUMENT_TOOLS, get_policy
from supplyguard_ai.services.recovery import PlannerOverrides
from supplyguard_ai.services.repository import OperationalRepository

CONTRACT_TOOLS = [*DOCUMENT_TOOLS, get_policy]
MAX_TOOL_CALLS = 30
MAX_MODEL_CALLS = 20


class ContractAgentError(RuntimeError):
    """The agent stopped without a valid structured answer (e.g. hit a limit)."""


class ContractReview(BaseModel):
    """What callers (LangGraph node, Deep Agent tool) receive."""

    findings: ContractFindings
    provenance_issues: list[str]
    tool_calls: int


# --------------------------------------------------------------------------- prompt
def load_skill_body(name: str) -> str:
    """Read skills/<name>/SKILL.md without its YAML front matter.

    The Deep Agent loads skills on demand through SkillsMiddleware.  This
    create_agent has no skills middleware, so we inline the two skills it
    always needs.  One source of truth: edit the SKILL.md, both agents change.
    """
    raw = (PROJECT_ROOT / "skills" / name / "SKILL.md").read_text(encoding="utf-8")
    return raw.split("---", 2)[2].strip() if raw.startswith("---") else raw


def build_system_prompt() -> str:
    return "\n\n".join([
        CONTRACT_ANALYST_PROMPT,
        load_skill_body("source-precedence"),
        load_skill_body("untrusted-content"),
        "Finish by returning ContractFindings.\n"
        "planner_overrides are the values the cost planner will USE:\n"
        "- expedite_fee_pct: the fee from the WINNING source of your conflict resolution (strict on cost). "
        "The higher e-mail fee belongs in planning_risk as a sensitivity, NOT in planner_overrides.\n"
        "- capacity_units: the LOWER credible capacity (conservative on feasibility), citing its source.",
    ])


# --------------------------------------------------------------------------- agent
def build_contract_agent(model: BaseChatModel | None = None):
    """Create the compiled agent graph.

    `model` is injectable so tests can pass a fake chat model.
    The result is a LangGraph `CompiledStateGraph`: it supports .invoke,
    .stream, checkpointers and can be embedded as a node/subgraph later.
    """
    if model is None:
        from supplyguard_ai.llm import get_chat_model

        model = get_chat_model()

    return create_agent(
        model,
        tools=CONTRACT_TOOLS,
        system_prompt=build_system_prompt(),
        middleware=[
            audit_tool_calls,
            # exit_behavior="continue": further tool calls are refused with an
            # error message, but the model may still write its final answer.
            ToolCallLimitMiddleware(run_limit=MAX_TOOL_CALLS, exit_behavior="continue"),
            # Hard stop for runaway loops; the runner then raises ContractAgentError.
            ModelCallLimitMiddleware(run_limit=MAX_MODEL_CALLS, exit_behavior="end"),
            provenance_guard,
        ],
        response_format=ToolStrategy(ContractFindings),
        context_schema=SupplyGuardContext,
        name="contract_analyst",  # shows up in LangSmith traces
    )


@lru_cache(maxsize=1)
def _default_agent():
    """Building the agent compiles a graph; do it once per process."""
    return build_contract_agent()


# --------------------------------------------------------------------------- runner
def build_task(component_id: str, supplier_ids: list[str], focus: str | None = None) -> str:
    """The user-message we send to the agent.  Built by code, not by the LLM,
    so the scope of the analysis is fixed and testable."""
    lines = [
        f"Component: {component_id}",
        f"Disrupted supplier: {supplier_ids[0]}",
        f"Alternative suppliers to review: {', '.join(supplier_ids[1:]) or 'none'}",
        "For each supplier: read the contract in force, any supplier notice and e-mail.",
        "Extract delay, lead time, capacity, price, expedite fee/days and penalties.",
        "Report every conflict between sources and resolve it with source precedence.",
    ]
    if focus:
        lines.append(f"Focus question from the user: {focus}")
    return "\n".join(lines)


def analyze_contracts(
    ctx: SupplyGuardContext,
    component_id: str,
    supplier_ids: list[str],
    focus: str | None = None,
    agent=None,
) -> ContractReview:
    """Run the agent once and return validated findings.

    `context=ctx` is how the logged-in user reaches the tools (scopes).
    """
    agent = agent or _default_agent()
    state = agent.invoke(
        {"messages": [HumanMessage(build_task(component_id, supplier_ids, focus))]},
        context=ctx,
        config={"run_name": "contract_analyst", "tags": ["contract-agent"],
                "metadata": {"case_id": ctx.case_id, "user_role": ctx.user.role.value}},
    )
    findings = state.get("structured_response")
    if not isinstance(findings, ContractFindings):
        raise ContractAgentError("contract agent stopped without a structured answer (limit reached?)")

    tool_calls = sum(len(getattr(m, "tool_calls", []) or []) for m in state["messages"])
    return ContractReview(
        findings=findings,
        provenance_issues=state.get("provenance_issues", check_provenance(findings, state["messages"], ctx)),
        tool_calls=tool_calls,
    )


# --------------------------------------------------------------------------- hand-off to planner
# Plausibility bounds: an LLM-extracted number is never trusted blindly.
_BOUNDS = {"expedite_fee_pct": (0.0, 50.0), "capacity_units": (1.0, 1_000_000.0)}
_TOPIC_OF = {"expedite_fee_pct": "expedite_fee", "capacity_units": "capacity"}


def _contradicts_resolution(o, conflicts: list[Conflict]) -> str | None:
    """Is this override inconsistent with the agent's OWN conflict resolution?

    Found in a live run: the agent resolved Cordillera's fee to the contract's
    6%, then put the e-mail's 10% into planner_overrides.  Rule (skill
    `source-precedence`): fees follow the winning source; capacity may use a
    LOWER-authority source only if it is the more conservative (lower) value.
    """
    for c in conflicts:
        if c.topic != _TOPIC_OF[o.kind] or o.source_id not in {x.source_id for x in c.claims}:
            continue
        if o.source_id == c.resolved_source_id:
            return None
        if o.kind == "capacity_units":
            values = [float(m) for x in c.claims if (m := _first_number(x.value)) is not None]
            if values and o.value <= min(values):
                return None  # conservative capacity: allowed
        return f"uses {o.source_id} but the conflict was resolved in favour of {c.resolved_source_id}"
    return None


def _first_number(text: str) -> str | None:
    import re

    m = re.search(r"\d[\d,]*(?:\.\d+)?", text)
    return m.group(0).replace(",", "") if m else None


def _key_is_valid(o, repo: OperationalRepository) -> bool:
    """Fee overrides must name an existing OPEN PO; capacity overrides an existing supplier.
    (Found in a live run: the model keyed a fee by supplier id 'S001'.)"""
    if o.kind == "expedite_fee_pct":
        po = repo.purchase_order(o.key)
        return bool(po) and po["status"] == "open"
    return repo.supplier(o.key) is not None


def validate_overrides(
    overrides: list[PlannerOverride],
    conflicts: list[Conflict],
    provenance_issues: list[str],
    repo: OperationalRepository,
) -> tuple[PlannerOverrides, list[str]]:
    """Convert LLM-proposed overrides into deterministic planner inputs.

    Shared by the contract agent AND the Deep Agent report (LangGraph `plan`
    node).  An override is accepted only if (a) the key is a real open PO /
    supplier, (b) the value is within plausible bounds, (c) its source passed
    the provenance check and (d) it is consistent with the conflict
    resolution.  Rejected ones are returned as messages for the UI.
    """
    bad_sources = {issue.split(":", 1)[0] for issue in provenance_issues}
    ov, rejected = PlannerOverrides(), []
    for o in overrides:
        lo, hi = _BOUNDS[o.kind]
        if not _key_is_valid(o, repo):
            rejected.append(f"{o.kind} {o.key}={o.value}: key is not an open PO / known supplier")
        elif o.source_id in bad_sources:
            rejected.append(f"{o.kind} {o.key}={o.value}: unverified source {o.source_id}")
        elif not lo <= o.value <= hi:
            rejected.append(f"{o.kind} {o.key}={o.value}: outside plausible range {lo}-{hi}")
        elif reason := _contradicts_resolution(o, conflicts):
            rejected.append(f"{o.kind} {o.key}={o.value}: {reason}")
        elif o.kind == "expedite_fee_pct":
            ov.expedite_fee_pct[o.key] = o.value
        else:
            ov.capacity_units[o.key] = int(o.value)
    return ov, rejected


def to_planner_overrides(review: ContractReview, repo: OperationalRepository) -> tuple[PlannerOverrides, list[str]]:
    """Convenience wrapper for a ContractReview."""
    f = review.findings
    return validate_overrides(f.planner_overrides, f.conflicts, review.provenance_issues, repo)
