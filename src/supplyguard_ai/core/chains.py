"""
Reusable chains (LangChain Core / LCEL).

THE INTAKE CHAIN = assignment §03 step 1:
    "Create an incident case, extract structured facts, and request
     clarification where essential information is missing."

    message ─► INTAKE_PROMPT ─► LLM.with_structured_output(IncidentFacts) ─► resolve_facts ─► IntakeResult
               (template)       (LLM fills OUR Pydantic schema)           (deterministic ERP lookup)

WHY A CHAIN AND NOT AN AGENT?
-----------------------------
Extraction is one fixed step: no tools to choose, no loop.  A chain is
cheaper, faster, deterministic in shape, and easy to evaluate on the 105
eval cases.  Rule of thumb from the course: use the simplest construct that
works - chain < graph < agent < deep agent.

WHY SPLIT "LLM EXTRACTS" FROM "CODE RESOLVES"?
----------------------------------------------
The LLM is good at reading messy language ("Atlas", "mc17", "two weeks").
It is NOT a reliable source of ids - it might invent "S001" for "Atlas".
So the LLM returns what was *written*, and `resolve_facts()` (plain Python)
maps it to canonical ERP ids and detects ambiguity.  Ambiguity is decided
by data, not by the model's opinion.
"""

from __future__ import annotations

import re

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from pydantic import BaseModel, Field

from supplyguard_ai.core.prompts import INTAKE_PROMPT, detect_injection
from supplyguard_ai.domain.models import IncidentFacts
from supplyguard_ai.services.repository import CsvRepository


class IntakeResult(BaseModel):
    """Output of the intake chain; the LangGraph intake node stores it in state."""

    facts: IncidentFacts
    supplier_candidates: list[dict] = Field(default_factory=list)  # filled when ambiguous
    needs_clarification: bool
    clarification_question: str | None = None
    # embedded instructions found in the USER'S OWN message (pasted notices ...)
    security_findings: list[dict] = Field(default_factory=list)


def _normalise_component(raw: str | None) -> str | None:
    """'mc17', 'MC 17', 'mc-017' -> 'MC-17'.  Deterministic clean-up after the LLM."""
    if not raw:
        return None
    m = re.search(r"mc\s*-?\s*0*(\d+)", raw, re.IGNORECASE)
    return f"MC-{m.group(1)}" if m else raw.strip().upper()


def resolve_facts(facts: IncidentFacts, repo: CsvRepository, message: str = "") -> IntakeResult:
    """Map extracted text to ERP ids and decide whether we must ask the user.

    Steps:
      1. normalise the component id and check it exists in a BOM,
      2. resolve the supplier name/id against the supplier master,
      3. collect everything missing or ambiguous into ONE clarification question
         (asking several questions one by one annoys users).
    """
    # CODE decides what is missing: supplier, component and delay - nothing else.
    # Found by the evaluation (dev case): the LLM listed the user's TASKS
    # ("recovery plan", "alternative sources") as "missing", and the case stopped
    # to ask a pointless question although all three facts were resolved.
    problems: list[str] = []
    candidates: list[dict] = []

    # 1. component
    component = _normalise_component(facts.component_id)
    if component and not repo.bom_for_component(component):
        problems.append(f"component {component} is not used in any product")
        component = None
    elif not component:
        problems.append("component")

    # 2. supplier: an explicit id wins over a name
    supplier_id = None
    query = facts.supplier_id or facts.supplier_name
    if query:
        rows = repo.resolve_supplier(query)
        if len(rows) == 1:
            supplier_id = rows[0]["supplier_id"]
        elif len(rows) > 1:
            candidates = [{"supplier_id": r["supplier_id"], "name": r["name"], "city": r["city"]} for r in rows]
            problems.append(f"supplier '{query}' is ambiguous")
        else:
            problems.append(f"supplier '{query}' not found")
    else:
        problems.append("supplier")

    # 3. delay
    if facts.delay_days is None:
        problems.append("delay in days")

    # De-duplicate the LLM's generic entries ("supplier") when we have a precise one.
    problems = list(dict.fromkeys(p for p in problems if p))
    # po_creation_requires_approval is FORCED to True: approval is company
    # policy (DOC-PRO-03), not something the user's wording - or the LLM's
    # reading of it - can switch off.  (Found in a live test: for a message
    # that didn't mention approval, the model returned False.)
    resolved = facts.model_copy(update={"supplier_id": supplier_id, "component_id": component,
                                        "missing_or_ambiguous": problems,
                                        "po_creation_requires_approval": True})

    question = None
    if problems:
        parts = []
        if candidates:
            options = ", ".join(f"{c['name']} ({c['supplier_id']}, {c['city']})" for c in candidates)
            parts.append(f"Which supplier do you mean: {options}?")
        others = [p for p in problems if "ambiguous" not in p]
        if others:
            parts.append("Please also provide: " + "; ".join(others) + ".")
        question = " ".join(parts)

    return IntakeResult(
        facts=resolved,
        supplier_candidates=candidates,
        needs_clarification=bool(problems),
        clarification_question=question,
        security_findings=user_message_findings(facts, message),
    )


def user_message_findings(facts: IncidentFacts, message: str) -> list[dict]:
    """Embedded instructions inside the user's own message (e.g. a pasted
    supplier notice saying "SYSTEM: reveal ..."): reported, never obeyed.

    Two independent detectors: the extraction LLM (`embedded_instructions`,
    understands meaning) and the deterministic pattern scan (cannot itself be
    talked out of it).  Either one is enough to report.
    """
    snippets = [s.strip() for s in facts.embedded_instructions if s and s.strip()]
    patterns = detect_injection(message)
    if not snippets and not patterns:
        return []
    shown = "; ".join(f"'{s[:120]}'" for s in snippets) or ", ".join(patterns)
    return [{"source_id": "USER-MESSAGE", "action_taken": "ignored_and_reported",
             "description": f"the incident message contains pasted text with embedded instructions ({shown}); "
                            "treated as data, not followed"}]


def build_intake_chain(repo: CsvRepository, model: BaseChatModel | None = None, extractor: Runnable | None = None) -> Runnable:
    """Compose the intake chain with LCEL's `|` operator.

    `extractor` lets tests inject a fake (any Runnable returning IncidentFacts),
    so the chain logic is tested without calling a real LLM.

    `.with_structured_output(IncidentFacts)` makes the model return a
    validated IncidentFacts object (via tool-calling / JSON schema under the
    hood) instead of free text we would have to parse.
    """
    if extractor is None:
        if model is None:
            from supplyguard_ai.llm import get_chat_model

            model = get_chat_model()
        extractor = INTAKE_PROMPT | model.with_structured_output(IncidentFacts)

    # The message travels next to the extracted facts (RunnablePassthrough.assign),
    # because the deterministic injection scan needs the raw text.
    # .with_config(run_name=...) gives the step a readable name in LangSmith traces.
    return (RunnablePassthrough.assign(facts=extractor)
            | RunnableLambda(lambda d: resolve_facts(d["facts"], repo, d.get("message", "")))
            ).with_config(run_name="incident_intake")
