"""Shared setup for the end-to-end workflow tests.

Runs the real LangGraph workflow, approval service and simulated procurement
system in memory. Only the two LLM-backed parts are replaced, as the
workflow allows (graph/nodes.py, Dependencies):
  * intake: the real chain with a fake extractor returning fixed facts,
  * investigation: a fake Deep Agent returning a fixed report.
Everything after that - planning, policy checks, approval, execution - is
the production code.
"""

from langchain_core.runnables import RunnableLambda
from langgraph.types import Command

from supplyguard_ai.agents.schemas import InvestigationReport
from supplyguard_ai.config import get_settings
from supplyguard_ai.core.chains import build_intake_chain
from supplyguard_ai.core.context import SupplyGuardContext
from supplyguard_ai.domain.models import IncidentFacts
from supplyguard_ai.graph.builder import build_incident_graph, case_config, new_case_input
from supplyguard_ai.graph.nodes import Dependencies
from supplyguard_ai.services.container import AppServices

CASE_ID = "CASE-TEST-001"
ATLAS_MC17 = {"supplier_name": "Atlas Components", "component_id": "MC-17", "delay_days": 12}


def memory_services() -> AppServices:
    """App services with in-memory stores and no external services."""
    settings = get_settings().model_copy(update={
        "storage_backend": "memory", "procurement_mcp_url": None, "external_mcp_url": None, "sandbox_url": None,
    })
    return AppServices.from_settings(settings)


class FakeInvestigator:
    """Stands in for the Deep Agent: returns a report recommending plan 1."""

    def invoke(self, *args, **kwargs):
        report = InvestigationReport(
            summary="Test investigation.", key_numbers=[], supplier_verdicts=[], conflicts=[],
            recommended_plan_id="plan_independent_upstream", recommendation_rationale="Independent upstream.",
        )
        return {"structured_response": report, "messages": []}


def build_graph(checkpointer, interrupt_before=None, **facts):
    """The real workflow graph with fake LLM parts. `facts` is what the fake intake model extracts."""
    extracted = IncidentFacts(**(facts or ATLAS_MC17))
    deps = Dependencies(
        intake_chain_factory=lambda repo: build_intake_chain(repo, extractor=RunnableLambda(lambda _: extracted)),
        investigator=FakeInvestigator(),
    )
    return build_incident_graph(deps, checkpointer=checkpointer, interrupt_before=interrupt_before)


def context(services: AppServices, user_id: str) -> SupplyGuardContext:
    return SupplyGuardContext(user=services.identity.authenticate(user_id), services=services, case_id=CASE_ID)


def start_case(graph, services, user_id: str, message: str = "Atlas Components: MC-17 is 12 days late."):
    """Run a new case until the workflow stops (normally at the approval step)."""
    graph.invoke(new_case_input(CASE_ID, message), config=case_config(CASE_ID), context=context(services, user_id))
    return graph.get_state(case_config(CASE_ID))


def decide(graph, services, user_id: str, decision: dict):
    """Resume the paused case with a human decision, acting as `user_id`."""
    graph.invoke(Command(resume=decision), config=case_config(CASE_ID), context=context(services, user_id))
    return graph.get_state(case_config(CASE_ID))
