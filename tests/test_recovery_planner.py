"""The recovery planner (services/recovery.py) on the MC-17 golden scenario.

Expected plans come from feasible_reference_plans in golden_scenario.json.
"""

import pytest

from helpers import load_json
from supplyguard_ai.config import get_settings
from supplyguard_ai.services.impact import ImpactService
from supplyguard_ai.services.recovery import PlannerOverrides, RecoveryPlanner
from supplyguard_ai.services.repository import CsvRepository

GOLDEN = load_json("evaluation/golden_scenario.json")
REFERENCE = GOLDEN["feasible_reference_plans"]
CAP_EUR = 75_000


@pytest.fixture(scope="module")
def planner() -> RecoveryPlanner:
    return RecoveryPlanner(CsvRepository(get_settings().support_data_dir))


@pytest.fixture(scope="module")
def impact(planner):
    return ImpactService(planner.repo).assess("S001", "MC-17", 12)


@pytest.fixture(scope="module")
def plans(planner, impact) -> dict:
    return {p.plan_id: p for p in planner.build_plans(impact, "S001", CAP_EUR)}


def suppliers_bought_from(plan) -> set[str]:
    return {a.payload["supplier_id"] for a in plan.actions if a.kind == "create_purchase_order"}


def test_borealis_and_helios_are_rejected(planner, impact):
    suppliers = {s.supplier_id: s for s in planner.assess_suppliers(impact, "S001")}
    # Borealis: ERP says "approved" but the certificate expired; the date wins.
    assert not suppliers["S003"].certification_valid
    # Helios: 21-day lead time, so it arrives after Atlas anyway.
    assert not suppliers["S002"].arrives_before_stockout
    assert not suppliers["S002"].feasible and not suppliers["S003"].feasible


def test_plan_without_shared_upstream_matches_reference(plans):
    plan = plans["plan_independent_upstream"]
    expected = REFERENCE["plan_1_without_shared_upstream"]
    assert round(plan.total_cost_eur) == expected["cost_eur"]
    assert plan.uncovered_units == expected["uncovered_units"]
    assert suppliers_bought_from(plan) == {"S004"}
    assert plan.within_cap


def test_plan_with_delta_matches_reference_and_warns(plans):
    plan = plans["plan_full_cover"]
    expected = REFERENCE["plan_2_with_delta"]
    assert round(plan.total_cost_eur) == expected["cost_eur"]
    assert plan.uncovered_units == expected["uncovered_units"]
    assert suppliers_bought_from(plan) == {"S004", "S005"}
    # Delta shares an upstream manufacturer with Atlas.
    assert any("weak diversification" in c for c in plan.caveats)


def test_every_plan_combines_at_least_three_strategies(plans):
    for plan in plans.values():
        assert {"A", "B", "C"} <= set(plan.strategies)


def test_hub_stock_goes_to_the_plants_that_run_out_first(plans):
    transfers = [a for a in plans["plan_independent_upstream"].actions if a.kind == "request_stock_transfer"]
    assert {a.payload["to_location"] for a in transfers} == {"W-KAT", "W-LYO"}
    assert sum(a.payload["qty"] for a in transfers) == GOLDEN["usable_by_location"]["W-HUB"]


def test_fee_from_email_changes_the_expedite_cost(planner, impact):
    # Contract says 6%, an e-mail says 10%; the resolved value is passed in.
    overrides = PlannerOverrides(expedite_fee_pct={"PO-2026-9002": 10.0})
    plan = planner.build_plans(impact, "S001", CAP_EUR, overrides)[0]
    expedite = next(a for a in plan.actions if a.kind == "expedite_purchase_order")
    assert expedite.cost_eur == GOLDEN["cost_sensitivity"]["expedite_fee_if_10pct_email"]


def test_plan_over_the_cap_is_marked(planner, impact):
    for plan in planner.build_plans(impact, "S001", spending_cap_eur=20_000):
        assert not plan.within_cap
