"""Unit tests for the business calculations, using small hand-made numbers.

Each function is checked on a simple case we can verify by hand,
on its edge cases, and on invalid input that must raise an error.
"""

from datetime import date

import pytest

from supplyguard_ai.calculations import (
    daily_demand,
    expedite_fee,
    is_certificate_valid,
    required_approver,
    shares_upstream,
    shortfall_units,
    usable_stock,
)

# Same approval tiers as config/policies.json, written out so the test is readable.
TIERS = [
    {"up_to": 25000, "approver": "procurement_manager"},
    {"up_to": 75000, "approver": "procurement_manager + finance_controller"},
    {"up_to": 90000, "approver": "finance_director"},
]


def test_usable_stock_subtracts_reserved_and_quarantined():
    assert usable_stock(on_hand=100, reserved=30, quarantined=20) == 50


def test_usable_stock_rejects_impossible_numbers():
    # More stock reserved than physically exists must be reported, not hidden.
    with pytest.raises(ValueError):
        usable_stock(on_hand=10, reserved=20, quarantined=0)


def test_daily_demand_multiplies_by_component_quantity():
    # 70 units of a product that needs 1 board + 70 units of one that needs 3 boards
    # = 280 boards per week = 40 per day.
    assert daily_demand({"A": 70, "B": 70}, {"A": 1, "B": 3}) == 40


def test_shortfall_uses_leftover_stock_first():
    # 100 in stock, 10 per day, delivery due in 5 days -> 50 left.
    # 8-day delay needs 80; 50 are covered, so 30 are missing.
    shortfall = shortfall_units(
        usable=100, demand_per_day=10, days_until_original_arrival=5, delay_days=8
    )
    assert shortfall == 30


def test_no_shortfall_when_stock_lasts_through_the_delay():
    shortfall = shortfall_units(
        usable=1000, demand_per_day=10, days_until_original_arrival=5, delay_days=8
    )
    assert shortfall == 0


def test_expedite_fee_is_a_percentage_of_order_value():
    # 100 units x EUR 10 = EUR 1,000 order value; 6% fee = EUR 60.
    assert expedite_fee(qty=100, unit_price_eur=10, fee_pct=6) == pytest.approx(60)


def test_certificate_is_valid_until_and_including_expiry_day():
    expiry = date(2026, 9, 30)
    assert is_certificate_valid(expiry, on_date=date(2026, 9, 30)) is True
    assert is_certificate_valid(expiry, on_date=date(2026, 10, 1)) is False


def test_unknown_upstream_is_never_treated_as_shared():
    assert shares_upstream("U01", "U01") is True
    assert shares_upstream("U01", "U02") is False
    assert shares_upstream("", "") is False


# Amounts exactly on a limit belong to the lower tier.
@pytest.mark.parametrize(
    "amount, expected_approver",
    [
        (25000, "procurement_manager"),
        (25001, "procurement_manager + finance_controller"),
        (75000, "procurement_manager + finance_controller"),
        (75001, "finance_director"),
        (90000, "finance_director"),
    ],
)
def test_required_approver_by_amount(amount, expected_approver):
    assert required_approver(amount, TIERS) == expected_approver


def test_amount_above_every_limit_is_refused():
    with pytest.raises(ValueError):
        required_approver(90001, TIERS)
