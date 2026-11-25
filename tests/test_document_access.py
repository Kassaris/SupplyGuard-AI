"""Access-scope rules for documents (DocumentService).

Rule from CONTRIBUTING.md: users only see documents within their scopes, and
finance-confidential data never reaches a user without that scope.

These tests read the real 212 documents from support-data/ and use the real
users from support-data/config/users.csv. No database, no LLM.
"""

import pytest

from supplyguard_ai.config import get_settings
from supplyguard_ai.domain.models import Role, User
from supplyguard_ai.services.documents import DocumentService

# The only finance-confidential document in the data package.
FINANCE_DOC = "DOC-FIN-01-v1"


# ---------------------------------------------------------------------------
# Test users, copied from support-data/config/users.csv
# ---------------------------------------------------------------------------

def make_user(user_id: str, role: Role, scopes: str) -> User:
    """Build a User the same way the app does after login."""
    return User(
        user_id=user_id,
        name=user_id,
        role=role,
        document_scopes=frozenset(s for s in scopes.split(";") if s),
        factory_access=frozenset(),
    )


PROCUREMENT_MANAGER = make_user("u-pm-01", Role.PROCUREMENT_MANAGER, "public;internal;procurement")
FINANCE_CONTROLLER = make_user(
    "u-fc-01", Role.FINANCE_CONTROLLER, "public;internal;procurement;finance-confidential"
)
VIEWER = make_user("u-ro-01", Role.VIEWER, "public")


@pytest.fixture(scope="module")
def documents() -> DocumentService:
    """Load all documents once for the whole file."""
    return DocumentService(get_settings().support_data_dir)


# ---------------------------------------------------------------------------
# Finance-confidential documents
# ---------------------------------------------------------------------------

def test_finance_document_is_hidden_from_procurement_manager(documents):
    assert documents.get(PROCUREMENT_MANAGER, FINANCE_DOC) is None


def test_finance_document_is_visible_to_finance_controller(documents):
    doc = documents.get(FINANCE_CONTROLLER, FINANCE_DOC)
    assert doc is not None
    assert doc.access_scope == "finance-confidential"


# ---------------------------------------------------------------------------
# Every user sees only their own scopes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("user", [PROCUREMENT_MANAGER, FINANCE_CONTROLLER, VIEWER])
def test_user_sees_only_documents_in_their_scopes(documents, user):
    for doc in documents.find(user):
        assert doc.access_scope in user.document_scopes


def test_viewer_sees_only_public_documents(documents):
    visible = documents.find(VIEWER)
    assert visible, "the viewer should see at least one public document"
    assert {d.access_scope for d in visible} == {"public"}


def test_user_without_scopes_sees_nothing(documents):
    nobody = make_user("u-none", Role.VIEWER, "")
    assert documents.find(nobody) == []


# ---------------------------------------------------------------------------
# Old document versions (planted problem: stale_document)
# ---------------------------------------------------------------------------

def test_old_atlas_contract_is_marked_as_superseded(documents):
    # The Atlas contract v2 (10-day lead time) was replaced by v3 (14 days).
    old = documents.get(PROCUREMENT_MANAGER, "DOC-CON-S001-v2")
    assert old.superseded_by == "DOC-CON-S001-v3"


def test_contract_search_can_hide_old_versions(documents):
    contracts = documents.find(
        PROCUREMENT_MANAGER, supplier_id="S001", doc_type="contract", include_superseded=False
    )
    assert [d.source_id for d in contracts] == ["DOC-CON-S001-v3"]
