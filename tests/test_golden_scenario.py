"""Runs the business calculations on the real MC-17 data.

The expected answers are given in support-data/evaluation/golden_scenario.json.
Each test reads the raw CSV files, passes them to the functions in
supplyguard_ai.calculations, and checks that the result matches the golden answer.
This proves both that the golden answers are right and that our code finds them.
"""

from datetime import date, timedelta

import pytest

from helpers import load_csv, load_json
from supplyguard_ai.calculations import (
    daily_demand,
    expedite_fee,
    is_certificate_valid,
    required_approver,
    shares_upstream,
    shortfall_units,
    usable_stock,
)

# The component affected by the incident.
COMPONENT = "MC-17"

# Expected answers and business rules, loaded once for the whole file.
GOLDEN = load_json("evaluation/golden_scenario.json")
POLICIES = load_json("config/policies.json")

# "Today" in the scenario: 2026-10-19.
SCENARIO_DATE = date.fromisoformat(POLICIES["scenario_date"])


# ---------------------------------------------------------------------------
# Small data lookups used by the tests below
# ---------------------------------------------------------------------------

def mc17_inventory() -> dict[str, dict[str, str]]:
    """MC-17 inventory rows keyed by warehouse, e.g. {"W-HUB": {...}}."""
    rows = load_csv("structured/inventory.csv")
    return {row["location_id"]: row for row in rows if row["component_id"] == COMPONENT}


def mc17_boards_per_product() -> dict[str, int]:
    """How many MC-17 boards one unit of each product needs (from the bill of materials)."""
    rows = load_csv("structured/bill_of_materials.csv")
    return {
        row["product_id"]: int(row["qty_per_unit"])
        for row in rows
        if row["component_id"] == COMPONENT
    }


def supplier(supplier_id: str) -> dict[str, str]:
    """One supplier from the supplier master (upstream manufacturer, certificate...)."""
    rows = load_csv("structured/suppliers.csv")
    return next(row for row in rows if row["supplier_id"] == supplier_id)


def mc17_offer(supplier_id: str) -> dict[str, str]:
    """What one supplier offers for MC-17: price, lead time and monthly capacity."""
    rows = load_csv("structured/supplier_components.csv")
    return next(
        row
        for row in rows
        if row["supplier_id"] == supplier_id and row["component_id"] == COMPONENT
    )


def total_usable_stock() -> int:
    """Usable MC-17 stock summed over all warehouses."""
    return sum(
        usable_stock(int(row["on_hand"]), int(row["reserved"]), int(row["quarantined"]))
        for row in mc17_inventory().values()
    )


def mc17_daily_demand() -> float:
    """Daily MC-17 demand from the first week of the production schedule."""
    boards = mc17_boards_per_product()
    weekly_units = {
        row["product_id"]: int(row["planned_units"])
        for row in load_csv("structured/production_schedule.csv")
        if row["product_id"] in boards and row["week_start"] == POLICIES["scenario_date"]
    }
    return daily_demand(weekly_units, boards)


# ---------------------------------------------------------------------------
# 1. Usable stock and business impact
# ---------------------------------------------------------------------------

def test_usable_stock_is_3200():
    # 5,200 on hand - 1,400 reserved - 600 quarantined = 3,200 usable.
    assert total_usable_stock() == GOLDEN["mc17_inventory"]["usable"]


def test_six_products_use_mc17():
    assert set(mc17_boards_per_product()) == set(GOLDEN["affected_products"])


def test_77_open_customer_orders_are_at_risk():
    # Definition from the README: open orders, due within 60 days,
    # for products that contain MC-17.
    affected_products = set(mc17_boards_per_product())
    horizon = SCENARIO_DATE + timedelta(days=60)

    at_risk = [
        row
        for row in load_csv("structured/customer_orders.csv")
        if row["product_id"] in affected_products
        and row["status"] == "open"
        and date.fromisoformat(row["due_date"]) <= horizon
    ]
    assert len(at_risk) == GOLDEN["open_orders_due_within_60d_on_affected_products"]


def test_daily_demand_is_450_units():
    # P058 and P066 need 3 boards per unit, so planned units are not enough on their own.
    assert mc17_daily_demand() == GOLDEN["daily_mc17_demand"]


def test_shortfall_is_5350_units():
    # Atlas should have delivered on day 7 (2026-10-26) and is now 12 days late.
    original_due = date.fromisoformat(GOLDEN["facts"]["original_due"])
    days_until_original_arrival = (original_due - SCENARIO_DATE).days

    shortfall = shortfall_units(
        usable=total_usable_stock(),
        demand_per_day=mc17_daily_demand(),
        days_until_original_arrival=days_until_original_arrival,
        delay_days=GOLDEN["facts"]["stated_delay_days"],
    )
    # 12 x 450 = 5,400 needed during the delay; 50 units are left; 5,350 missing.
    assert shortfall == GOLDEN["expected_shortfall_units"]


# ---------------------------------------------------------------------------
# 2. Supplier rules that reject alternatives
# ---------------------------------------------------------------------------

def test_borealis_cannot_receive_orders():
    # Borealis (S003) is the fastest option, but its ISO 9001 certificate expired.
    # Trap in the data: the ERP still says "approved", so the system must check the
    # certificate date and not trust the ERP status.
    borealis = supplier("S003")
    expiry = date.fromisoformat(borealis["certification_expiry"])

    assert borealis["erp_status"] == "approved"
    assert is_certificate_valid(expiry, SCENARIO_DATE) is False


def test_delta_is_not_real_diversification_from_atlas():
    # Delta (S005) buys from the same upstream manufacturer as Atlas (S001),
    # so a problem upstream would hit both suppliers.
    atlas_upstream = supplier("S001")["upstream_manufacturer_id"]
    delta_upstream = supplier("S005")["upstream_manufacturer_id"]

    assert shares_upstream(atlas_upstream, delta_upstream) is True


# ---------------------------------------------------------------------------
# 3. Cost and approval of the reference recovery plan
# ---------------------------------------------------------------------------

def test_reference_plan_costs_30378_eur():
    # The plan combines three actions:
    #   1. expedite the open Cordillera order PO-2026-9002,
    #   2. buy Cordillera's full monthly capacity as a new order,
    #   3. transfer all usable stock from the Rotterdam hub to the plants.
    purchase_orders = load_csv("structured/purchase_orders.csv")
    open_po = next(row for row in purchase_orders if row["po_id"] == "PO-2026-9002")
    cordillera = mc17_offer("S004")
    hub = mc17_inventory()["W-HUB"]

    fee = expedite_fee(
        int(open_po["qty"]), float(open_po["unit_price_eur"]), float(open_po["expedite_fee_pct"])
    )
    purchase = int(cordillera["monthly_capacity_units"]) * float(cordillera["unit_price_eur"])
    transfers = int(hub["usable"]) * POLICIES["transfer_cost_eur_per_unit"]

    # EUR 882 + EUR 29,400 + EUR 96 = EUR 30,378.
    # pytest.approx ignores tiny floating-point differences such as 30378.000000000004.
    plan = GOLDEN["feasible_reference_plans"]["plan_1_without_shared_upstream"]
    assert fee + purchase + transfers == pytest.approx(plan["cost_eur"])


def test_reference_plan_needs_two_approvers_and_stays_under_the_cap():
    cost = GOLDEN["feasible_reference_plans"]["plan_1_without_shared_upstream"]["cost_eur"]

    # Above EUR 25,000 the procurement manager alone may not approve.
    approver = required_approver(cost, POLICIES["approval_tiers_eur"])
    assert approver == "procurement_manager + finance_controller"
    assert cost <= POLICIES["incident_spending_cap_eur"]
