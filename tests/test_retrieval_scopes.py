"""Access-scope and safety rules for document search (RetrievalService).

The real app searches pgvector, ChromaDB and Postgres full-text. Here we use
the in-memory version of the same service (build_in_memory_retrieval), so the
tests run without Docker. The search logic, the scope checks and the fusion
step are the same code as in production; only the storage is different.

The embeddings are fake (random but repeatable vectors), so we do NOT test
how good the ranking is. That is the job of the retrieval benchmark.
Here we test the rules that must hold whatever the ranking is.
"""

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from supplyguard_ai.config import get_settings
from supplyguard_ai.domain.models import Role, User
from supplyguard_ai.retrieval.service import build_in_memory_retrieval
from supplyguard_ai.services.documents import DocumentService
from supplyguard_ai.services.repository import CsvRepository

# Every search mode that works without Neo4j.
MODES = ["pgvector", "chroma", "keyword", "hybrid"]

# A question whose best answer is the finance-confidential document.
FINANCE_QUESTION = "margin and penalty rate sheet"

# A question whose best answers are procurement documents (not public).
PROCUREMENT_QUESTION = "MC-17 delay Atlas Components contract lead time"


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
def retrieval():
    """Index all 212 documents and 40 memories in memory, once for this file."""
    data_dir = get_settings().support_data_dir
    documents = DocumentService(data_dir).all_documents()
    memories = CsvRepository(data_dir).approved_memories()
    return build_in_memory_retrieval(documents, memories, DeterministicFakeEmbedding(size=64))


# ---------------------------------------------------------------------------
# Scopes are enforced in every search mode
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", MODES)
def test_procurement_manager_never_gets_finance_confidential(retrieval, mode):
    hits = retrieval.search(PROCUREMENT_MANAGER, FINANCE_QUESTION, k=10, mode=mode)
    assert all(hit.access_scope != "finance-confidential" for hit in hits)


@pytest.mark.parametrize("mode", MODES)
def test_viewer_gets_only_public_documents(retrieval, mode):
    hits = retrieval.search(VIEWER, PROCUREMENT_QUESTION, k=10, mode=mode)
    assert all(hit.access_scope == "public" for hit in hits)


def test_finance_controller_can_find_the_finance_document(retrieval):
    # The opposite check: the scope filter must not hide documents from
    # the one user who is allowed to read them.
    hits = retrieval.search(FINANCE_CONTROLLER, FINANCE_QUESTION, k=10, mode="keyword")
    assert "DOC-FIN-01-v1" in {hit.source_id for hit in hits}


def test_user_without_scopes_gets_no_results(retrieval):
    nobody = make_user("u-none", Role.VIEWER, "")
    assert retrieval.search(nobody, "MC-17 delay", mode="hybrid") == []


def test_memory_search_respects_scopes(retrieval):
    # All 40 approved memories are "internal", so a public-only user sees none.
    assert retrieval.search_memories(VIEWER, "certification lapse on MC-17") == []
    assert retrieval.search_memories(PROCUREMENT_MANAGER, "certification lapse on MC-17")


# ---------------------------------------------------------------------------
# Old versions and injected documents
# ---------------------------------------------------------------------------

def test_superseded_contract_is_hidden_by_default(retrieval):
    hits = retrieval.search(PROCUREMENT_MANAGER, "Atlas Components contract lead time", k=20,
                            mode="keyword", supplier_id="S001")
    assert "DOC-CON-S001-v2" not in {hit.source_id for hit in hits}


def test_superseded_contract_can_be_requested_explicitly(retrieval):
    hits = retrieval.search(PROCUREMENT_MANAGER, "Atlas Components contract lead time", k=20,
                            mode="keyword", supplier_id="S001", include_superseded=True)
    assert "DOC-CON-S001-v2" in {hit.source_id for hit in hits}


def test_helios_notice_is_flagged_as_containing_instructions(retrieval):
    # DOC-NOT-HELIOS-PRICE hides instructions to bypass approval. The search
    # result must carry a flag, so the agent treats the text as data only.
    hits = retrieval.search(VIEWER, "Helios price update notice", k=10, mode="keyword")
    helios = [hit for hit in hits if hit.source_id == "DOC-NOT-HELIOS-PRICE"]
    assert helios, "the Helios notice should be found"
    assert any(hit.contains_instructions for hit in helios)
