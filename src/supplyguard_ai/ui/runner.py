"""
CaseRunner: drives the LangGraph incident workflow for a UI - without
depending on any UI framework.

WHY A SEPARATE RUNNER
---------------------
Chainlit callbacks are hard to unit-test.  Everything that is not "draw it
on screen" lives here: starting a case, streaming progress, recognising an
interrupt (clarification / approval), resuming with the CURRENT user as
the actor, and keeping the case registry in sync.  The Chainlit file only
translates `Event`s into messages and buttons.

STREAMING
---------
`graph.stream(..., stream_mode=["updates", "custom"])` yields
  ("custom",  {"phase": ..., "detail": ...})   <- runtime.stream_writer in nodes
  ("updates", {"<node>": {...partial state...}})  after each node
  ("updates", {"__interrupt__": (Interrupt(value=...),)})  when the graph pauses
We map them to `Event`s.  New AIMessages a node returns become chat messages.

SYNC ON PURPOSE
---------------
The runner is synchronous: the checkpointer (PostgresSaver), the services and
the agents are sync, and psycopg's async mode doesn't run on Windows' default
event loop.  Chainlit runs it in a worker thread (see chainlit_app.py).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterator, Literal

from langchain_core.messages import AIMessage
from langgraph.types import Command

from supplyguard_ai.core.context import SupplyGuardContext
from supplyguard_ai.domain.models import Role, User
from supplyguard_ai.graph.builder import case_config, new_case_input
from supplyguard_ai.services.identity import IdentityService

EventKind = Literal["progress", "node", "message", "clarification", "approval", "done", "error"]

NODE_LABELS = {
    "intake": "Extracting incident facts",
    "clarify": "Clarification received",
    "investigate": "Investigation (deep agent, subagents, documents, graph)",
    "plan": "Deterministic recovery planning",
    "challenge": "Policy and evidence checks",
    "request_approval": "Approval request created",
    "await_approval": "Approval decision recorded",
    "execute": "Executing approved actions",
    "report": "Incident report",
}


@dataclass
class Event:
    kind: EventKind
    data: dict[str, Any] = field(default_factory=dict)


def new_case_id() -> str:
    return f"CASE-{datetime.now(timezone.utc):%Y%m%d}-{uuid.uuid4().hex[:6].upper()}"


class CaseRunner:
    def __init__(self, services, graph) -> None:
        self.svc = services
        self.graph = graph

    # ------------------------------------------------------------ public API
    def start(self, user: User, message: str, case_id: str | None = None) -> tuple[str, Iterator[Event]]:
        # Check the role BEFORE registering the case: otherwise a viewer's refused
        # request still left an empty case row behind (found in the UI test).
        IdentityService.require_can_open_case(user)  # raises AuthorizationError (a PermissionError)
        case_id = case_id or new_case_id()
        self.svc.cases.create(case_id, user.user_id, message.strip().splitlines()[0][:120], "intake")
        return case_id, self._run(new_case_input(case_id, message), user, case_id)

    def resume(self, user: User, case_id: str, value: Any) -> Iterator[Event]:
        """Continue a paused case.  `user` = whoever clicked - the graph uses
        THEM as approver/decider (runtime context), never an id in `value`."""
        if user.role != Role.VIEWER:  # viewers can't act, so they never become participants
            self.svc.cases.add_participant(case_id, user.user_id)
        return self._run(Command(resume=value), user, case_id)

    def state(self, case_id: str) -> dict:
        return self.graph.get_state(case_config(case_id)).values or {}

    def pending(self, case_id: str) -> dict | None:
        """The interrupt a case is waiting on (after a restart, from the checkpoint)."""
        snap = self.graph.get_state(case_config(case_id))
        for task in snap.tasks or ():
            for itr in getattr(task, "interrupts", ()) or ():
                return itr.value
        return None

    # ------------------------------------------------------------ internals
    def _run(self, inp, user: User, case_id: str) -> Iterator[Event]:
        ctx = SupplyGuardContext(user=user, services=self.svc, case_id=case_id)
        phase = None
        try:
            for mode, chunk in self.graph.stream(inp, config=case_config(case_id), context=ctx,
                                                 stream_mode=["updates", "custom"]):
                if mode == "custom":
                    yield Event("progress", chunk)
                    continue
                for node, update in chunk.items():
                    if node == "__interrupt__":
                        value = update[0].value
                        self.svc.cases.update(case_id, "awaiting_approval" if value["type"] == "approval"
                                              else "awaiting_clarification")
                        yield Event(value["type"], {"case_id": case_id, **value})
                        return
                    update = update or {}
                    phase = update.get("phase", phase)
                    yield Event("node", {"node": node, "label": NODE_LABELS.get(node, node), "phase": phase})
                    for m in update.get("messages", []):
                        if isinstance(m, AIMessage) and m.content:
                            yield Event("message", {"text": m.content, "node": node})
        except PermissionError as exc:
            yield Event("error", {"message": str(exc)})
            return
        final = self.state(case_id)
        self.svc.cases.update(case_id, final.get("phase", phase or "unknown"))
        yield Event("done", {"case_id": case_id, "phase": final.get("phase"), "report_md": final.get("report_md"),
                             "execution_results": final.get("execution_results", [])})
