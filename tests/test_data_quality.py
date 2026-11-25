"""Planted data-quality problems (support-data/structured/data_quality_manifest.csv).

The data package contains deliberate problems: ambiguous supplier names,
name variants, expired certificates behind an "approved" status, missing
identifiers and impossible dates. These tests check that the repository
(the layer every agent reads data through) exposes each problem in a way
the app can detect, instead of hiding it or crashing.

Data is read from the CSV files with CsvRepository. No database, no LLM.
"""

import pytest

from supplyguard_ai.config import get_settings
from supplyguard_ai.services.repository import CsvRepository


@pytest.fixture(scope="module")
def repo() -> CsvRepository:
    return CsvRepository(get_settings().support_data_dir)


def ids(suppliers: list[dict]) -> list[str]:
    """Supplier rows -> sorted supplier ids, for easy comparison."""
    return sorted(s["supplier_id"] for s in suppliers)


# ---------------------------------------------------------------------------
# ambiguous_supplier_name: "Atlas" can mean two suppliers
# ---------------------------------------------------------------------------

def test_atlas_alone_returns_both_candidates(repo):
    # The app must ask the user which Atlas is meant, so the repository
    # has to return both, not pick one.
    assert ids(repo.resolve_supplier("Atlas")) == ["S001", "S006"]


def test_full_name_returns_one_supplier(repo):
    assert ids(repo.resolve_supplier("Atlas Components")) == ["S001"]


# ---------------------------------------------------------------------------
# supplier_name_variant: names typed differently in purchase orders
# ---------------------------------------------------------------------------

def test_lower_case_name_is_resolved(repo):
    assert ids(repo.resolve_supplier("quanta electronics")) == ["S048"]


@pytest.mark.xfail(strict=True, reason="Known gap: suffixes like 'Ltd' and 'Comp.' are not normalised")
@pytest.mark.parametrize("typed, expected", [
    ("Orion Supply Ltd", "S016"),
    ("Argo Dynamics Ltd", "S035"),
])
def test_names_with_company_suffix_are_resolved(repo, typed, expected):
    assert ids(repo.resolve_supplier(typed)) == [expected]


# ---------------------------------------------------------------------------
# stale_status: ERP says "approved" but the certificate has expired
# ---------------------------------------------------------------------------

def test_borealis_is_approved_in_erp_but_certificate_expired(repo):
    borealis = repo.supplier("S003")
    assert borealis["erp_status"] == "approved"
    # So the app must check the expiry date, not only the ERP status.
    assert borealis["certification_expiry"] < repo.scenario_date()


# ---------------------------------------------------------------------------
# shared_upstream: two "different" suppliers depend on the same factory
# ---------------------------------------------------------------------------

def test_atlas_and_delta_share_the_same_upstream_manufacturer(repo):
    # Delta Micro (S005) is not a safe alternative to Atlas (S001):
    # both buy from Veltro Semiconductor (U01).
    atlas, delta = repo.supplier("S001"), repo.supplier("S005")
    assert atlas["upstream_manufacturer_id"] == delta["upstream_manufacturer_id"] == "U01"


# ---------------------------------------------------------------------------
# missing_identifier and conflicting_dates in purchase orders
# ---------------------------------------------------------------------------

def test_blank_component_id_is_loaded_as_none(repo):
    # A blank value must become None, not an empty string, so code can test for it.
    assert repo.purchase_order("PO-2026-1281")["component_id"] is None


def test_delivery_before_order_date_is_detectable(repo):
    po = repo.purchase_order("PO-2026-1273")
    assert po["actual_delivery_date"] < po["order_date"]
