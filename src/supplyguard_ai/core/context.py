"""
Runtime context: WHO is asking and WHICH services to use, for one run.

THE PROBLEM
-----------
Tools need the current user (to filter documents by access scope) and the
services (to compute impact).  Three bad ways to give them that:
  1. Global variables          -> two Chainlit users at once would clash.
  2. Tool arguments            -> the LLM would fill in `user_id`.  A prompt
                                  injection could then say "use user u-fc-01"
                                  and read finance-confidential data.
  3. Putting it in the prompt  -> same problem: the model controls it.

THE SOLUTION: LangGraph / LangChain "runtime context"
-----------------------------------------------------
`create_agent(..., context_schema=SupplyGuardContext)` and
`graph.invoke(..., context=SupplyGuardContext(...))`.  The context is set by
OUR code (Chainlit after login), travels with the run, and is injected into
tools as `runtime.context`.  It is never shown to, or chosen by, the model.
It is also NOT stored in checkpoints (only state is), so services and
secrets don't end up serialized in Postgres.
"""

from __future__ import annotations

from dataclasses import dataclass

from supplyguard_ai.domain.models import User
from supplyguard_ai.services.container import AppServices


@dataclass(frozen=True)  # frozen: a tool can't swap the user mid-run
class SupplyGuardContext:
    user: User  # authenticated by Chainlit; decides document scopes
    services: AppServices  # shared service instances (audit log, approvals ...)
    case_id: str | None = None  # incident case this run belongs to (for audit)
