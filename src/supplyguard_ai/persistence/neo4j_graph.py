"""
Neo4jGraph: the GraphStore protocol in Cypher (services/graph.py).

SAFETY
------
* Every query is a fixed string with $parameters - values never become part
  of the query text (no Cypher injection).
* Every query runs in `session.execute_read(...)`: a READ transaction.  The
  database refuses writes in it, so even a bug here cannot change the graph.
  (Neo4j Community has no role-based access control; read transactions are
  the guard we have.  With Enterprise, use a read-only user as well.)
* `default_access_mode=READ` routes sessions to read replicas in a cluster.

DATA TYPES
----------
Neo4j returns `neo4j.time.Date` for date properties; `_plain()` converts them
to ISO strings so results are JSON-serialisable (LangGraph state, tool output)
and identical to the in-memory twin (parity test).
"""

from __future__ import annotations

from neo4j import READ_ACCESS, Driver, GraphDatabase

from supplyguard_ai.config import Settings, get_settings

Q_COMPONENT_PRODUCTS = """
MATCH (c:Component {componentId: $cid})-[u:USED_IN]->(p:Product)
OPTIONAL MATCH (p)-[:PRODUCED_AT]->(f:Factory)
RETURN p.productId AS product_id, p.name AS product_name, u.qtyPerUnit AS qty_per_unit,
       f.factoryId AS factory_id, f.name AS factory_name,
       u.sourceId AS source_id, u.effectiveDate AS effective_date
ORDER BY product_id
"""

Q_PRODUCT_CUSTOMERS = """
MATCH (p:Product)-[o:ORDERED_BY]->(c:Customer)
WHERE p.productId IN $pids
RETURN p.productId AS product_id, c.customerId AS customer_id, c.name AS customer_name,
       c.priorityTier AS priority_tier, o.openOrders AS open_orders, o.sourceId AS source_id
ORDER BY product_id, customer_id
"""

Q_COMPONENT_SUPPLIERS = """
MATCH (s:Supplier)-[x:SUPPLIES]->(:Component {componentId: $cid})
OPTIONAL MATCH (s)-[:SOURCES_FROM]->(u:Upstream)
RETURN s.supplierId AS supplier_id, s.name AS supplier_name, s.country AS country,
       s.erpStatus AS erp_status, s.certificationExpiry AS certification_expiry,
       x.leadTimeDays AS lead_time_days, x.unitPriceEur AS unit_price_eur,
       u.upstreamId AS upstream_id, u.name AS upstream_name, u.country AS upstream_country,
       x.sourceId AS source_id, x.effectiveDate AS effective_date
ORDER BY supplier_id
"""

Q_SUPPLIER_UPSTREAM = """
MATCH (:Supplier {supplierId: $sid})-[r:SOURCES_FROM]->(u:Upstream)
RETURN u.upstreamId AS upstream_id, u.name AS name, u.country AS country, r.sourceId AS source_id
"""

Q_UPSTREAM_SUPPLIERS = """
MATCH (s:Supplier)-[:SOURCES_FROM]->(:Upstream {upstreamId: $uid})
RETURN s.supplierId AS supplier_id, s.name AS supplier_name ORDER BY supplier_id
"""

Q_SUPPLIER_COMPONENTS = """
MATCH (:Supplier {supplierId: $sid})-[:SUPPLIES]->(c:Component) RETURN c.componentId AS component_id ORDER BY component_id
"""

Q_DISCONNECTED_SUPPLIERS = "MATCH (s:Supplier) WHERE NOT (s)-[:SUPPLIES]->() RETURN s.supplierId AS id ORDER BY id"
Q_DISCONNECTED_COMPONENTS = """
MATCH (c:Component) WHERE NOT (c)-[:USED_IN]->()
RETURN c.componentId AS id ORDER BY toInteger(split(c.componentId, '-')[1])
"""


def _plain(v):
    """neo4j.time.Date/DateTime -> ISO string; everything else unchanged."""
    return v.iso_format() if hasattr(v, "iso_format") else v


class Neo4jGraph:
    def __init__(self, driver: Driver, database: str | None = None) -> None:
        self.driver = driver
        self.database = database

    @classmethod
    def from_settings(cls, s: Settings | None = None) -> "Neo4jGraph":
        s = s or get_settings()
        driver = GraphDatabase.driver(s.neo4j_uri, auth=(s.neo4j_user, s.neo4j_password.get_secret_value()))
        driver.verify_connectivity()  # fail fast with a clear error
        return cls(driver)

    def close(self) -> None:
        self.driver.close()

    def _read(self, query: str, **params) -> list[dict]:
        def work(tx):
            return [{k: _plain(v) for k, v in rec.items()} for rec in tx.run(query, **params)]

        with self.driver.session(database=self.database, default_access_mode=READ_ACCESS) as session:
            return session.execute_read(work)

    # ------------------------------------------------------------ protocol
    def component_products(self, component_id):
        return self._read(Q_COMPONENT_PRODUCTS, cid=component_id)

    def product_customers(self, product_ids):
        return self._read(Q_PRODUCT_CUSTOMERS, pids=list(product_ids)) if product_ids else []

    def component_suppliers(self, component_id):
        return self._read(Q_COMPONENT_SUPPLIERS, cid=component_id)

    def supplier_upstream(self, supplier_id):
        rows = self._read(Q_SUPPLIER_UPSTREAM, sid=supplier_id)
        return rows[0] if rows else None

    def upstream_suppliers(self, upstream_id):
        return self._read(Q_UPSTREAM_SUPPLIERS, uid=upstream_id)

    def supplier_components(self, supplier_id):
        return [r["component_id"] for r in self._read(Q_SUPPLIER_COMPONENTS, sid=supplier_id)]

    def disconnected(self):
        return {
            "suppliers_without_components": [r["id"] for r in self._read(Q_DISCONNECTED_SUPPLIERS)],
            "components_without_products": [r["id"] for r in self._read(Q_DISCONNECTED_COMPONENTS)],
        }
