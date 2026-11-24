"""
PostgresRepository: the operational repository backed by schema `ops`.

WHY SUBCLASS CsvRepository
--------------------------
`CsvRepository` already implements every query of the
`OperationalRepository` protocol on top of one primitive: `_table(name)`.
Overriding that primitive to `SELECT * FROM ops.<name>` makes the whole
repository Postgres-backed at once.  The hot queries (called on every
impact/plan computation) are additionally overridden with real `WHERE`
clauses, so Postgres - with the indexes created by the loader - does the
filtering instead of Python.

Business services don't change at all: they still receive `Row` dicts with
int / float / date / None values (see the type mapping in load_structured).

Policies, the provenance registry fallback and approved memories stay
file-based: they are configuration, not operational records (the memories
move to pgvector in the retrieval step).
"""

from __future__ import annotations

from pathlib import Path

from psycopg import sql
from psycopg_pool import ConnectionPool

from supplyguard_ai.services.repository import CsvRepository, Row


class PostgresRepository(CsvRepository):
    # A plain subclass (no @dataclass): the parent's generated __init__ sets
    # data_dir + cache; we only add the pool.
    def __init__(self, data_dir: Path, pool: ConnectionPool) -> None:
        super().__init__(data_dir)
        self.pool = pool

    # ------------------------------------------------------------ primitive
    def _q(self, query: str | sql.Composed, params: tuple | dict = ()) -> list[Row]:
        with self.pool.connection() as conn:
            return list(conn.execute(query, params).fetchall())

    def _table(self, name: str) -> list[Row]:
        """No caching: ops.* is reloaded by data-load; always read the current data."""
        return self._q(sql.SQL("SELECT * FROM {}").format(sql.Identifier("ops", name)))

    # --------------------------------------------------- hot, filtered queries
    def supplier(self, supplier_id: str) -> Row | None:
        rows = self._q("SELECT * FROM ops.suppliers WHERE supplier_id = %s", (supplier_id,))
        return rows[0] if rows else None

    def inventory(self, component_id: str) -> list[Row]:
        return self._q("SELECT * FROM ops.inventory WHERE component_id = %s ORDER BY location_id", (component_id,))

    def bom_for_component(self, component_id: str) -> list[Row]:
        return self._q("SELECT * FROM ops.bill_of_materials WHERE component_id = %s ORDER BY product_id", (component_id,))

    def production_schedule(self, product_ids: set[str]) -> list[Row]:
        return self._q("SELECT * FROM ops.production_schedule WHERE product_id = ANY(%s)", (list(product_ids),))

    def customer_orders(self, product_ids: set[str]) -> list[Row]:
        return self._q("SELECT * FROM ops.customer_orders WHERE product_id = ANY(%s) ORDER BY order_id", (list(product_ids),))

    def open_purchase_orders(self, component_id: str) -> list[Row]:
        return self._q("SELECT * FROM ops.purchase_orders WHERE component_id = %s AND status = 'open' ORDER BY po_id",
                       (component_id,))

    def purchase_order(self, po_id: str) -> Row | None:
        rows = self._q("SELECT * FROM ops.purchase_orders WHERE po_id = %s", (po_id,))
        return rows[0] if rows else None

    def supplier_catalog(self, component_id: str) -> list[Row]:
        return self._q("SELECT * FROM ops.supplier_components WHERE component_id = %s ORDER BY supplier_id", (component_id,))

    def shipment_updates(self, po_id: str) -> list[Row]:
        return self._q("SELECT * FROM ops.shipment_updates WHERE po_id = %s ORDER BY event_date, shipment_event_id", (po_id,))
