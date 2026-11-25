"""The app's impact service (services/impact.py) on the MC-17 golden scenario.

test_golden_scenario.py proves the expected numbers with small helper
functions; this file checks that the service the agents actually call
produces the same numbers.
"""

import pytest

from helpers import load_json
from supplyguard_ai.config import get_settings
from supplyguard_ai.services.impact import ImpactError, ImpactService
from supplyguard_ai.services.repository import CsvRepository

GOLDEN = load_json("evaluation/golden_scenario.json")


@pytest.fixture(scope="module")
def service() -> ImpactService:
    return ImpactService(CsvRepository(get_settings().support_data_dir))


@pytest.fixture(scope="module")
def impact(service):
    facts = GOLDEN["facts"]
    return service.assess(facts["supplier"], facts["component"], facts["stated_delay_days"])


def test_usable_stock_per_location(impact):
    usable = {s.location_id: s.usable for s in impact.stock}
    assert usable == GOLDEN["usable_by_location"]
    assert impact.total_usable == GOLDEN["mc17_inventory"]["usable"]


def test_daily_demand(impact):
    assert impact.daily_demand_total == GOLDEN["daily_mc17_demand"]


def test_delayed_purchase_order_and_timeline(impact):
    assert impact.affected_po_id == GOLDEN["facts"]["affected_po"]
    assert impact.original_arrival_day == 7
    assert impact.delayed_arrival_day == 7 + GOLDEN["gap_days"]
    assert impact.stock_at_original_arrival == GOLDEN["stock_at_original_arrival"]


def test_shortfall(impact):
    assert impact.shortfall_units == GOLDEN["expected_shortfall_units"]


def test_affected_products_and_orders(impact):
    assert impact.affected_products == GOLDEN["affected_products"]
    assert len(impact.at_risk_orders) == GOLDEN["open_orders_due_within_60d_on_affected_products"]


def test_tier_one_customers_come_first(impact):
    tiers = [o.priority_tier for o in impact.at_risk_orders]
    assert tiers == sorted(tiers)


def test_unknown_component_is_rejected(service):
    with pytest.raises(ImpactError):
        service.assess("S001", "MC-999", 5)
