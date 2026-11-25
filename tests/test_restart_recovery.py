"""Restart and idempotency (red-team RT-14, RT-15).

A "restart" is simulated by building a NEW graph object on the same
checkpointer and services - what the app has after a process restart, since
checkpoints, approvals and the procurement ledger live in Postgres.
Here they live in memory, so this checks the workflow logic, not Postgres.
"""

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from supplyguard_ai.graph.builder import case_config
from workflow_setup import CASE_ID, build_graph, context, decide, memory_services, start_case

MANAGER, FINANCE = "u-pm-01", "u-fc-01"


@pytest.fixture
def services():
    return memory_services()


def approve_by_both(graph, services, approval_id):
    """The reference plan (~EUR 30,378) needs the manager and the finance controller."""
    decide(graph, services, MANAGER, {"action": "approve", "approval_id": approval_id})
    return decide(graph, services, FINANCE, {"action": "approve", "approval_id": approval_id})


def test_rt15_restart_while_waiting_for_approval(services):
    checkpoints = InMemorySaver()
    before = start_case(build_graph(checkpoints), services, MANAGER)

    restarted = build_graph(checkpoints)
    state = restarted.get_state(case_config(CASE_ID))
    assert state.next == ("await_approval",)
    assert state.values["approval_id"] == before.values["approval_id"]  # same payload, same approval

    state = approve_by_both(restarted, services, state.values["approval_id"])
    assert state.values["phase"] == "closed"
    assert len(services.procurement.created_purchase_orders()) == 1


def test_restart_right_before_execution(services):
    checkpoints = InMemorySaver()
    graph = build_graph(checkpoints, interrupt_before=["execute"])
    state = start_case(graph, services, MANAGER)
    state = approve_by_both(graph, services, state.values["approval_id"])
    assert state.next == ("execute",)  # approved, but the app "died" before executing

    restarted = build_graph(checkpoints)
    restarted.invoke(None, config=case_config(CASE_ID), context=context(services, MANAGER))
    assert restarted.get_state(case_config(CASE_ID)).values["phase"] == "closed"
    assert len(services.procurement.created_purchase_orders()) == 1


def test_rt14_replaying_executed_actions_creates_nothing_new(services):
    graph = build_graph(InMemorySaver())
    state = start_case(graph, services, MANAGER)
    approval_id = state.values["approval_id"]
    state = approve_by_both(graph, services, approval_id)

    # Run every approved action again with the same idempotency keys.
    gateway = services.procurement_gateway
    for i, action in enumerate(state.values["selected_plan"]["actions"]):
        again = gateway.execute(CASE_ID, f"{approval_id}:{i}", approval_id, action["kind"], action["payload"], MANAGER)
        assert again.status == "duplicate"
    assert len(services.procurement.created_purchase_orders()) == 1
