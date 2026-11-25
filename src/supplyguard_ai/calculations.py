"""Deterministic business calculations used by SupplyGuard AI.

Numbers such as stock, shortfall and cost must never be produced by the LLM.
The agent calls these plain functions instead (for example wrapped as tools),
so every number is reproducible and covered by tests.
"""

from datetime import date


def usable_stock(on_hand: int, reserved: int, quarantined: int) -> int:
    """Stock that can actually be used (procedure PRO-06).

    Reserved units belong to specific customer orders and quarantined units
    failed inspection, so neither can be used freely.
    """
    usable = on_hand - reserved - quarantined
    if usable < 0:
        raise ValueError("reserved plus quarantined stock exceeds stock on hand")
    return usable


def daily_demand(weekly_units: dict[str, int], boards_per_product: dict[str, int]) -> float:
    """Average daily component demand from one week of planned production.

    weekly_units:       planned units per product for the week, e.g. {"P012": 350}
    boards_per_product: component quantity in one unit of each product, e.g. {"P058": 3}
    """
    weekly_need = sum(
        units * boards_per_product[product] for product, units in weekly_units.items()
    )
    return weekly_need / 7


def shortfall_units(
    usable: int,
    demand_per_day: float,
    days_until_original_arrival: int,
    delay_days: int,
) -> float:
    """Demand that cannot be covered while waiting for the delayed delivery.

    1. Until the original arrival date we consume our usable stock.
    2. Whatever is left covers part of the demand during the delay.
    3. The rest of the demand during the delay is the shortfall.
    """
    stock_left = max(usable - days_until_original_arrival * demand_per_day, 0)
    demand_during_delay = delay_days * demand_per_day
    return max(demand_during_delay - stock_left, 0)


def expedite_fee(qty: int, unit_price_eur: float, fee_pct: float) -> float:
    """Cost of expediting an open purchase order: order value x fee % (procedure PRO-08)."""
    return qty * unit_price_eur * fee_pct / 100


def is_certificate_valid(expiry: date, on_date: date) -> bool:
    """True if the supplier certificate is still valid on the given date (procedure PRO-04)."""
    return expiry >= on_date


def shares_upstream(upstream_a: str, upstream_b: str) -> bool:
    """True if two suppliers depend on the same upstream manufacturer (procedure PRO-05).

    An empty upstream id means "unknown", which we never treat as shared.
    """
    return bool(upstream_a) and upstream_a == upstream_b


def required_approver(amount_eur: float, approval_tiers: list[dict]) -> str:
    """Who must approve a recovery plan of this cost (procedure PRO-03).

    approval_tiers comes from config/policies.json, for example:
    [{"up_to": 25000, "approver": "procurement_manager"}, ...]
    """
    # Check the tiers from the lowest limit to the highest.
    for tier in sorted(approval_tiers, key=lambda t: t["up_to"]):
        if amount_eur <= tier["up_to"]:
            return tier["approver"]

    raise ValueError(f"EUR {amount_eur:,.0f} exceeds the highest approval limit")
