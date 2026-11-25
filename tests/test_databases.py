"""Integration tests for the loaded databases.

Requires the Docker services with the data loaded:
    docker compose up -d
    uv run pytest -m integration
"""

import csv
from datetime import timedelta

import pytest

from helpers import DATA_DIR, load_json
from supplyguard_ai.calculations import usable_stock
from supplyguard_ai.config import get_settings
from supplyguard_ai.persistence.db import make_pool
from supplyguard_ai.persistence.neo4j_graph import Neo4jGraph
from supplyguard_ai.persistence.repository import PostgresRepository
from supplyguard_ai.retrieval import stores

pytestmark = pytest.mark.integration

GOLDEN = load_json("evaluation/golden_scenario.json")


def csv_row_count(path) -> int:
    with open(path, encoding="utf-8", newline="") as file:
        return sum(1 for _ in csv.DictReader(file))


@pytest.fixture(scope="module")
def pool():
    pool = make_pool(get_settings(), max_size=2)
    yield pool
    pool.close()


@pytest.fixture(scope="module")
def repo(pool):
    return PostgresRepository(get_settings().support_data_dir, pool)


@pytest.fixture(scope="module")
def graph():
    graph = Neo4jGraph.from_settings()
    yield graph
    graph.close()


def sql(pool, query: str) -> list[dict]:
    with pool.connection() as conn:
        return conn.execute(query).fetchall()


def cypher(graph, query: str) -> list[dict]:
    records, _, _ = graph.driver.execute_query(query)
    return [record.data() for record in records]


# --- PostgreSQL -------------------------------------------------------------

def test_all_csv_files_are_loaded(pool):
    expected, loaded = {}, {}
    for path in sorted((DATA_DIR / "structured").glob("*.csv")):
        expected[path.stem] = csv_row_count(path)
        loaded[path.stem] = sql(pool, f"SELECT count(*) AS n FROM ops.{path.stem}")[0]["n"]
    assert loaded == expected


def test_mc17_usable_stock_is_3200(repo):
    total = sum(usable_stock(r["on_hand"], r["reserved"], r["quarantined"]) for r in repo.inventory("MC-17"))
    assert total == GOLDEN["mc17_inventory"]["usable"]


def test_mc17_orders_at_risk_is_77(repo):
    products = {r["product_id"] for r in repo.bom_for_component("MC-17")}
    horizon = repo.scenario_date() + timedelta(days=60)
    at_risk = [
        o for o in repo.customer_orders(products)
        if o["status"] == "open" and o["due_date"] <= horizon
    ]
    assert len(at_risk) == GOLDEN["open_orders_due_within_60d_on_affected_products"]


# --- pgvector / ChromaDB ----------------------------------------------------

def test_all_documents_are_indexed(pool):
    docs = sql(pool, "SELECT count(DISTINCT source_id) AS n FROM vec.documents")[0]["n"]
    assert docs == csv_row_count(DATA_DIR / "documents" / "documents_index.csv")


def test_chroma_and_pgvector_hold_the_same_chunks(pool):
    # The retrieval benchmark compares the two stores, so they must index the same chunks.
    pg_ids = {r["id"] for r in sql(pool, "SELECT id FROM vec.documents")}
    collection = stores.chroma_client().get_collection(stores.CHROMA_COLLECTION)
    assert set(collection.get(include=[])["ids"]) == pg_ids


def test_finance_document_keeps_its_access_scope(pool):
    scopes = sql(pool, "SELECT DISTINCT access_scope FROM vec.documents WHERE source_id = 'DOC-FIN-01-v1'")
    assert scopes == [{"access_scope": "finance-confidential"}]


def test_all_memories_are_indexed(pool):
    assert sql(pool, "SELECT count(*) AS n FROM vec.incident_memories")[0]["n"] == 40


# --- Neo4j ------------------------------------------------------------------

NODE_FILES = {
    "Component": "nodes_component.csv",
    "Customer": "nodes_customer.csv",
    "Factory": "nodes_factory.csv",
    "Product": "nodes_product.csv",
    "Supplier": "nodes_supplier.csv",
    "Upstream": "nodes_upstream.csv",
}
RELATIONSHIP_FILES = {
    "ORDERED_BY": "rel_ordered_by.csv",
    "PRODUCED_AT": "rel_produced_at.csv",
    "SOURCES_FROM": "rel_sources_from.csv",
    "SUPPLIES": "rel_supplies.csv",
    "USED_IN": "rel_used_in.csv",
}


def test_all_nodes_are_loaded(graph):
    rows = cypher(graph, "MATCH (n) RETURN labels(n)[0] AS label, count(*) AS n")
    expected = {label: csv_row_count(DATA_DIR / "graph" / f) for label, f in NODE_FILES.items()}
    assert {r["label"]: r["n"] for r in rows} == expected


def test_all_relationships_are_loaded(graph):
    rows = cypher(graph, "MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS n")
    expected = {rel: csv_row_count(DATA_DIR / "graph" / f) for rel, f in RELATIONSHIP_FILES.items()}
    assert {r["type"]: r["n"] for r in rows} == expected


def test_mc17_suppliers(graph):
    suppliers = [r["supplier_id"] for r in graph.component_suppliers("MC-17")]
    assert suppliers == ["S001", "S002", "S003", "S004", "S005"]


def test_suppliers_without_components_are_detected(graph):
    # Planted in the data: S058-S060 exist but supply nothing.
    found = graph.disconnected()["suppliers_without_components"]
    assert {"S058", "S059", "S060"} <= set(found)
