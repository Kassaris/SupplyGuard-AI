"""
Knowledge-graph service: supplier -> component -> product -> factory -> customer.

ASSIGNMENT §05: "Neo4j - Supplier, component, product, factory, and customer
dependency graph."  §03 step 4: "Investigate dependencies, across suppliers,
components, factories, and customers."

WHAT THE GRAPH ANSWERS BETTER THAN SQL
--------------------------------------
Multi-hop questions with variable paths:
  * "Which customers are hit if MC-17 is late?"   component -USED_IN-> product -ORDERED_BY-> customer
  * "Is Delta a real second source for Atlas?"    supplier -SOURCES_FROM-> upstream <-SOURCES_FROM- supplier
  * "What else depends on Veltro?"                upstream <- suppliers -> components -> products
In SQL these are chains of joins whose shape changes per question; in Cypher
they are one readable pattern.

DESIGN: FIXED, PARAMETERIZED TRAVERSALS - NO LLM-WRITTEN CYPHER
---------------------------------------------------------------
LangChain offers GraphCypherQAChain (the LLM writes Cypher from a question).
We deliberately don't use it:
  * a prompt-injected document could steer generated Cypher into
    `MATCH (n) DETACH DELETE n` or into reading data the user may not see,
  * generated queries are non-deterministic -> untestable numbers,
  * every question SupplyGuard needs is known in advance.
So the graph is exposed as a small library of read-only traversals; the
agent chooses WHICH traversal to run (a tool call), never writes the query.

TWO IMPLEMENTATIONS, ONE PROTOCOL
---------------------------------
  Neo4jGraph   (persistence/neo4j_graph.py)  production, Cypher over bolt
  InMemoryGraph (below)                      unit tests, built from the SAME
                                             graph CSVs; a parity test checks
                                             both return identical results.
`GraphService` holds the business logic on top (shared-upstream detection,
blast radius, data-quality warnings) and works with either.

PROVENANCE: every node/relationship carries sourceId + effectiveDate
(assignment §06); results keep them so claims stay citable.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field

Row = dict


class GraphStore(Protocol):
    def component_products(self, component_id: str) -> list[Row]: ...
    def product_customers(self, product_ids: list[str]) -> list[Row]: ...
    def component_suppliers(self, component_id: str) -> list[Row]: ...
    def supplier_upstream(self, supplier_id: str) -> Row | None: ...
    def upstream_suppliers(self, upstream_id: str) -> list[Row]: ...
    def supplier_components(self, supplier_id: str) -> list[str]: ...
    def disconnected(self) -> dict[str, list[str]]: ...


# =========================================================================== in-memory twin
def _read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _col(row: dict, prefix: str) -> str:
    """Neo4j import headers look like 'supplierId:ID(Supplier)' or ':START_ID(Supplier)'."""
    return next(v for k, v in row.items() if k.startswith(prefix))


class InMemoryGraph:
    """Adjacency lists built from support-data/graph/*.csv (same files as load.cypher)."""

    def __init__(self, graph_dir: Path) -> None:
        g = graph_dir
        self.suppliers = {_col(r, "supplierId"): r for r in _read(g / "nodes_supplier.csv")}
        self.upstreams = {_col(r, "upstreamId"): r for r in _read(g / "nodes_upstream.csv")}
        self.products = {_col(r, "productId"): r for r in _read(g / "nodes_product.csv")}
        self.factories = {_col(r, "factoryId"): r for r in _read(g / "nodes_factory.csv")}
        self.customers = {_col(r, "customerId"): r for r in _read(g / "nodes_customer.csv")}
        self.components = {_col(r, "componentId"): r for r in _read(g / "nodes_component.csv")}

        self.supplies = defaultdict(list)       # component -> [edge]
        self.supplies_by_supplier = defaultdict(list)
        for r in _read(g / "rel_supplies.csv"):
            e = {"supplier_id": _col(r, ":START_ID"), "component_id": _col(r, ":END_ID"),
                 "lead_time_days": int(r["leadTimeDays:int"]), "unit_price_eur": float(r["unitPriceEur:float"]),
                 "source_id": r["sourceId"], "effective_date": r["effectiveDate"]}
            self.supplies[e["component_id"]].append(e)
            self.supplies_by_supplier[e["supplier_id"]].append(e)
        self.sources_from = {}                  # supplier -> edge
        for r in _read(g / "rel_sources_from.csv"):
            self.sources_from[_col(r, ":START_ID")] = {"upstream_id": _col(r, ":END_ID"), "source_id": r["sourceId"],
                                                      "effective_date": r["effectiveDate"]}
        self.used_in = defaultdict(list)        # component -> [edge]
        for r in _read(g / "rel_used_in.csv"):
            self.used_in[_col(r, ":START_ID")].append({"product_id": _col(r, ":END_ID"), "qty_per_unit": int(r["qtyPerUnit:int"]),
                                                       "source_id": r["sourceId"], "effective_date": r["effectiveDate"]})
        self.produced_at = {_col(r, ":START_ID"): _col(r, ":END_ID") for r in _read(g / "rel_produced_at.csv")}
        self.ordered_by = defaultdict(list)     # product -> [edge]
        for r in _read(g / "rel_ordered_by.csv"):
            self.ordered_by[_col(r, ":START_ID")].append({"customer_id": _col(r, ":END_ID"), "open_orders": int(r["openOrders:int"]),
                                                          "source_id": r["sourceId"]})

    def component_products(self, component_id):
        rows = []
        for e in self.used_in.get(component_id, []):
            f = self.produced_at.get(e["product_id"])
            rows.append({"product_id": e["product_id"], "product_name": self.products[e["product_id"]]["name"],
                         "qty_per_unit": e["qty_per_unit"], "factory_id": f,
                         "factory_name": self.factories[f]["name"] if f else None,
                         "source_id": e["source_id"], "effective_date": e["effective_date"]})
        return sorted(rows, key=lambda r: r["product_id"])

    def product_customers(self, product_ids):
        rows = [{"product_id": p, "customer_id": e["customer_id"], "customer_name": self.customers[e["customer_id"]]["name"],
                 "priority_tier": int(self.customers[e["customer_id"]]["priorityTier:int"]),
                 "open_orders": e["open_orders"], "source_id": e["source_id"]}
                for p in product_ids for e in self.ordered_by.get(p, [])]
        return sorted(rows, key=lambda r: (r["product_id"], r["customer_id"]))

    def component_suppliers(self, component_id):
        rows = []
        for e in self.supplies.get(component_id, []):
            s = self.suppliers[e["supplier_id"]]
            up = self.sources_from.get(e["supplier_id"])
            rows.append({"supplier_id": e["supplier_id"], "supplier_name": s["name"], "country": s["country"],
                         "erp_status": s["erpStatus"], "certification_expiry": s["certificationExpiry"],
                         "lead_time_days": e["lead_time_days"], "unit_price_eur": e["unit_price_eur"],
                         "upstream_id": up["upstream_id"] if up else None,
                         "upstream_name": self.upstreams[up["upstream_id"]]["name"] if up else None,
                         "upstream_country": self.upstreams[up["upstream_id"]]["country"] if up else None,
                         "source_id": e["source_id"], "effective_date": e["effective_date"]})
        return sorted(rows, key=lambda r: r["supplier_id"])

    def supplier_upstream(self, supplier_id):
        up = self.sources_from.get(supplier_id)
        if not up:
            return None
        u = self.upstreams[up["upstream_id"]]
        return {"upstream_id": up["upstream_id"], "name": u["name"], "country": u["country"], "source_id": up["source_id"]}

    def upstream_suppliers(self, upstream_id):
        return sorted(({"supplier_id": s, "supplier_name": self.suppliers[s]["name"]}
                       for s, e in self.sources_from.items() if e["upstream_id"] == upstream_id),
                      key=lambda r: r["supplier_id"])

    def supplier_components(self, supplier_id):
        return sorted(e["component_id"] for e in self.supplies_by_supplier.get(supplier_id, []))

    def disconnected(self):
        return {
            "suppliers_without_components": sorted(s for s in self.suppliers if not self.supplies_by_supplier.get(s)),
            "components_without_products": sorted((c for c in self.components if not self.used_in.get(c)),
                                                  key=lambda c: int(c.split("-")[1])),
        }


# =========================================================================== business layer
class GraphSupplier(BaseModel):
    supplier_id: str
    supplier_name: str
    lead_time_days: int
    unit_price_eur: float
    upstream_id: str | None
    upstream_name: str | None
    upstream_country: str | None
    shares_upstream_with_disrupted: bool
    source_id: str


class ComponentTrace(BaseModel):
    component_id: str
    products: list[dict]
    factories: list[str]
    customers: list[dict]                 # aggregated per customer
    customers_by_tier: dict[str, int]
    open_order_lines: int
    suppliers: list[GraphSupplier]
    shared_upstream_groups: dict[str, list[str]]   # upstream -> suppliers of THIS component
    warnings: list[str] = Field(default_factory=list)
    sources: list[str]


class SupplierTrace(BaseModel):
    supplier_id: str
    components: list[str]
    upstream: dict | None
    upstream_peers: list[dict]            # other suppliers with the same upstream
    overlapping_components: dict[str, list[str]]   # peer -> components BOTH supply (real concentration)
    downstream_products: int
    downstream_customers: int
    warnings: list[str] = Field(default_factory=list)
    sources: list[str]


@dataclass
class GraphService:
    store: GraphStore

    def trace_component(self, component_id: str, disrupted_supplier_id: str | None = None) -> ComponentTrace:
        """Downstream impact + who else can supply + hidden shared dependencies."""
        products = self.store.component_products(component_id)
        customers_raw = self.store.product_customers([p["product_id"] for p in products])
        agg: dict[str, dict] = {}
        for c in customers_raw:
            a = agg.setdefault(c["customer_id"], {"customer_id": c["customer_id"], "customer_name": c["customer_name"],
                                                  "priority_tier": c["priority_tier"], "open_orders": 0, "products": []})
            a["open_orders"] += c["open_orders"]
            a["products"].append(c["product_id"])
        customers = sorted(agg.values(), key=lambda c: (c["priority_tier"], -c["open_orders"], c["customer_id"]))
        tiers: dict[str, int] = defaultdict(int)
        for c in customers:
            tiers[f"tier_{c['priority_tier']}"] += 1

        raw_suppliers = self.store.component_suppliers(component_id)
        disrupted_up = next((s["upstream_id"] for s in raw_suppliers if s["supplier_id"] == disrupted_supplier_id), None)
        if disrupted_supplier_id and disrupted_up is None:
            disrupted_up = (self.store.supplier_upstream(disrupted_supplier_id) or {}).get("upstream_id")
        suppliers = [GraphSupplier(**{k: s[k] for k in GraphSupplier.model_fields if k in s},
                                   shares_upstream_with_disrupted=bool(disrupted_up) and s["upstream_id"] == disrupted_up
                                   and s["supplier_id"] != disrupted_supplier_id)
                     for s in raw_suppliers]
        groups: dict[str, list[str]] = defaultdict(list)
        for s in suppliers:
            if s.upstream_id:
                groups[s.upstream_id].append(s.supplier_id)

        warnings = []
        if not products:
            warnings.append(f"{component_id} is used in no product in the graph (disconnected entity) - verify the BOM.")
        if not raw_suppliers:
            warnings.append(f"No supplier SUPPLIES {component_id} in the graph.")
        for up, members in groups.items():
            if len(members) > 1:
                warnings.append(f"Suppliers {', '.join(members)} share upstream {up}: not independent sources.")
        sources = sorted({p["source_id"] for p in products} | {c["source_id"] for c in customers_raw}
                         | {s["source_id"] for s in raw_suppliers})
        return ComponentTrace(
            component_id=component_id, products=products,
            factories=sorted({p["factory_id"] for p in products if p["factory_id"]}),
            customers=customers, customers_by_tier=dict(tiers),
            open_order_lines=sum(c["open_orders"] for c in customers),
            suppliers=suppliers, shared_upstream_groups=dict(groups), warnings=warnings, sources=sources,
        )

    def trace_supplier(self, supplier_id: str) -> SupplierTrace:
        """Footprint of one supplier and its hidden concentration risk."""
        comps = self.store.supplier_components(supplier_id)
        up = self.store.supplier_upstream(supplier_id)
        peers, overlap = [], {}
        if up:
            peers = [p for p in self.store.upstream_suppliers(up["upstream_id"]) if p["supplier_id"] != supplier_id]
            mine = set(comps)
            for p in peers:
                shared = sorted(mine & set(self.store.supplier_components(p["supplier_id"])))
                if shared:
                    overlap[p["supplier_id"]] = shared
        products = {p["product_id"] for c in comps for p in self.store.component_products(c)}
        customers = {c["customer_id"] for c in self.store.product_customers(sorted(products))}
        warnings = []
        if not comps:
            warnings.append(f"{supplier_id} supplies no component in the graph (disconnected entity).")
        for peer, shared in overlap.items():
            warnings.append(f"{peer} shares upstream {up['upstream_id']} AND supplies {', '.join(shared)}: "
                            "switching to it does not remove the upstream risk.")
        return SupplierTrace(
            supplier_id=supplier_id, components=comps, upstream=up, upstream_peers=peers,
            overlapping_components=overlap, downstream_products=len(products), downstream_customers=len(customers),
            warnings=warnings, sources=sorted({"SRC-SUPPLIER-QUESTIONNAIRE-2026"} if up else set()),
        )

    def related_for_retrieval(self, component_id: str | None, supplier_id: str | None) -> dict[str, list[str]]:
        """Graph neighbourhood used to EXPAND a document search (GraphRAG).

        Returns supplier ids worth searching documents for, plus document
        source ids that graph edges cite directly (SUPPLIES edges point at the
        contract that defines lead time and price).
        """
        suppliers, doc_ids = set(), set()
        if component_id:
            for s in self.store.component_suppliers(component_id):
                suppliers.add(s["supplier_id"])
                if s["source_id"].startswith("DOC-"):
                    doc_ids.add(s["source_id"])
        if supplier_id:
            suppliers.add(supplier_id)
            up = self.store.supplier_upstream(supplier_id)
            if up:
                suppliers |= {p["supplier_id"] for p in self.store.upstream_suppliers(up["upstream_id"])}
        return {"suppliers": sorted(suppliers)[:12], "documents": sorted(doc_ids)[:12]}
