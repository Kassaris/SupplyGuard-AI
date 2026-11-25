"""
Agent middleware: deterministic code that runs AROUND the agent loop.

WHAT MIDDLEWARE IS (Week 2, day 8)
----------------------------------
`create_agent` builds a small LangGraph:  model -> tools -> model -> ... -> end.
Middleware plugs hooks into that loop:

    before_agent ─► [ before_model ─► wrap_model_call(MODEL) ─► after_model
                       ─► wrap_tool_call(TOOL) ]*  ─► after_agent

We use hooks for things that must happen EVERY time and must not depend on
the model "remembering" an instruction:

  audit_tool_calls   (wrap_tool_call) every tool call -> audit log
  provenance_guard   (after_agent)    every cited source_id must exist, be
                                      readable by this user, and have
                                      actually been read in this run
  limits             (prebuilt)       ToolCallLimit / ModelCallLimit keep the
                                      agent BOUNDED (cost, latency, runaway loops)

Why decorators (`@wrap_tool_call`, `@after_agent`) and not subclasses of
AgentMiddleware?  A hook that needs no configuration is one function; the
decorator generates the class.  That's the style taught in the course.
"""

from __future__ import annotations

import re
from typing import Any, NotRequired

from langchain.agents.middleware import AgentState, after_agent, wrap_model_call, wrap_tool_call
from langchain_core.messages import ToolMessage
from pydantic import BaseModel

from supplyguard_ai.core.context import SupplyGuardContext


# --------------------------------------------------------------------------- audit
@wrap_tool_call
def audit_tool_calls(request, handler):
    """Record which tool the agent called, with which arguments, then run it.

    `handler(request)` executes the real tool; we call it unchanged, so this
    hook observes and never alters behaviour.  The audit trail answers the
    reviewer question "what exactly did the AI look at before recommending?".
    """
    ctx = request.runtime.context
    result = handler(request)
    if isinstance(ctx, SupplyGuardContext):
        ctx.services.audit.record(
            "agent.tool_call",
            ctx.case_id or "-",
            f"agent:{ctx.user.user_id}",
            {"tool": request.tool_call["name"], "args": request.tool_call.get("args", {})},
        )
    return result


# --------------------------------------------------------------------------- provenance
class ProvenanceState(AgentState):
    """Agent state + one extra key written by `provenance_guard`.

    Custom state keys must be declared, otherwise LangGraph drops them.
    """

    provenance_issues: NotRequired[list[str]]


# read_document output starts with "[<source_id> | <type> | ..." (see core/tools.py)
_READ_HEADER = re.compile(r"^\[([A-Za-z0-9_.-]+) \|")


def _cited_source_ids(obj: Any) -> set[str]:
    """Walk a dumped Pydantic object and collect every source id it cites."""
    found: set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("source_id", "resolved_source_id") and isinstance(v, str):
                found.add(v)
            else:
                found |= _cited_source_ids(v)
    elif isinstance(obj, list):
        for v in obj:
            found |= _cited_source_ids(v)
    return found


def _documents_read(messages: list) -> set[str]:
    """source_ids returned by successful read_document calls in this run."""
    read = set()
    for m in messages:
        if isinstance(m, ToolMessage) and m.name == "read_document" and isinstance(m.content, str):
            if match := _READ_HEADER.match(m.content):
                read.add(match.group(1))
    return read


def check_provenance(response: BaseModel, messages: list, ctx: SupplyGuardContext) -> list[str]:
    """Pure function (easy to unit-test) behind the provenance guard.

    Returns human-readable issues; empty list = every citation is verified.
    """
    cited = _cited_source_ids(response.model_dump(mode="json"))
    # Reads visible in THIS agent's messages, plus reads done by its subagents
    # in the same case.  Subagent tool messages never appear in the parent's
    # message list (only their final answer does), but `audit_tool_calls`
    # recorded them.  Denied reads are harmless here: the access check below
    # still rejects documents this user can't see.
    events = ctx.services.audit.events(ctx.case_id or "-")
    read = _documents_read(messages) | {
        e.details["args"].get("source_id")
        for e in events
        if e.event_type == "agent.tool_call" and e.details.get("tool") == "read_document"
    } | {  # search_documents returns chunk text too: its hits count as seen
        sid for e in events if e.event_type == "agent.search_results" for sid in e.details.get("source_ids", [])
    }
    structured_sources = ctx.services.repo.data_source_ids()
    issues = []
    for sid in sorted(cited):
        if sid in structured_sources or sid.startswith("config/"):
            continue  # ERP / config sources come from tools, not documents
        if ctx.services.documents.get(ctx.user, sid) is None:
            issues.append(f"{sid}: cited but does not exist or is not accessible to {ctx.user.role}")
        elif sid not in read:
            issues.append(f"{sid}: cited but never read in this run")
    return issues


@after_agent(state_schema=ProvenanceState)
def provenance_guard(state: ProvenanceState, runtime) -> dict | None:
    """After the agent finishes, verify its citations deterministically.

    We do NOT silently fix the answer: issues are written to state so the
    LangGraph workflow can show them, lower confidence, or re-run the agent.
    """
    response = state.get("structured_response")
    ctx = runtime.context
    if not isinstance(response, BaseModel) or not isinstance(ctx, SupplyGuardContext):
        return None
    issues = check_provenance(response, state["messages"], ctx)
    if issues:
        ctx.services.audit.record("agent.provenance_issues", ctx.case_id or "-", "middleware", {"issues": issues})
    return {"provenance_issues": issues}


# --------------------------------------------------------------------------- tool hiding
def hide_tools(*names: str):
    """Build a middleware that removes the named tools from every model call.

    Found in two live runs: agents ran `grep` over their virtual file system
    (which only holds skills and notes), found no "MC-17" and concluded the
    component had no BOM - a false blocker that leaked into the summary.
    No agent needs file *search*: skills are read by exact path.  Removing
    the tools is deterministic; asking the model not to use them is not.

    `wrap_model_call` sees the request just before the LLM call;
    `request.override(tools=...)` returns a modified copy (requests are immutable).
    """
    hidden = set(names)

    @wrap_model_call(name=f"HideTools_{'_'.join(sorted(hidden))}")
    def _hide(request, handler):
        visible = [t for t in request.tools if getattr(t, "name", None) not in hidden
                   and not (isinstance(t, dict) and t.get("name") in hidden)]
        return handler(request.override(tools=visible))

    return _hide


HIDE_FILE_SEARCH = ("grep", "glob")
