"""
Load support-data/structured/*.csv into PostgreSQL schema `ops`.

Run by the `data-load` service in docker-compose.yml:
    python -m supplyguard_ai.ingestion.load_structured

BEHAVIOUR: truncate + reload (idempotent)
-----------------------------------------
Every run drops and recreates each ops table and copies the CSV in again.
The data is provided and read-only (CONTRIBUTING: never edit support-data),
so the CSVs are the source of truth and reloading can never lose anything.
Each table is replaced inside ONE transaction: readers see either the old or
the new table, never a half-loaded one.

TYPES
-----
Column types come from the SAME column-name lists the CSV repository uses
(services/repository.py), so both backends agree on what is an int, a float
or a date.  Floats become `double precision` (not `numeric`) so psycopg
returns Python floats, not Decimals - business code sees identical values.

DATA QUALITY
------------
We load the data AS IS, problems included (blank supplier ids, duplicate
messages, delivered-before-ordered dates ...).  Cleaning it here would hide
exactly the problems the agents are graded on detecting.
"""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path

from psycopg import sql
from psycopg_pool import ConnectionPool

from supplyguard_ai.config import get_settings
from supplyguard_ai.persistence.db import ensure_app_schema, make_pool
from supplyguard_ai.services.repository import _DATE_COLS, _FLOAT_COLS, _INT_COLS

log = logging.getLogger("load_structured")

EXTRA_DATE_COLS = {"date", "received"}  # historical_disruption_cases.date, supplier_message_inbox.received
INDEX_COLS = ("component_id", "supplier_id", "product_id", "po_id", "order_id", "location_id")


def column_type(name: str) -> str:
    if name in _INT_COLS:
        return "integer"
    if name in _FLOAT_COLS:
        return "double precision"
    if name in _DATE_COLS or name in EXTRA_DATE_COLS:
        return "date"
    return "text"


def load_table(pool: ConnectionPool, csv_path: Path) -> int:
    """Recreate ops.<csv name> and bulk-load it with COPY (fast: ~1 s for all files)."""
    table = csv_path.stem
    with csv_path.open(encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows = [[(v.strip() or None) for v in row] for row in reader]  # "" -> NULL

    # sql.Identifier quotes names safely - never build SQL with f-strings from data.
    cols = sql.SQL(", ").join(
        sql.SQL("{} {}").format(sql.Identifier(c), sql.SQL(column_type(c))) for c in header
    )
    ident = sql.Identifier("ops", table)
    with pool.connection() as conn, conn.transaction():
        conn.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(ident))
        conn.execute(sql.SQL("CREATE TABLE {} ({})").format(ident, cols))
        with conn.cursor().copy(
            sql.SQL("COPY {} ({}) FROM STDIN").format(ident, sql.SQL(", ").join(map(sql.Identifier, header)))
        ) as copy:
            for r in rows:
                copy.write_row(r)
        for c in INDEX_COLS:
            if c in header:
                conn.execute(sql.SQL("CREATE INDEX ON {} ({})").format(ident, sql.Identifier(c)))
    return len(rows)


def load_all(pool: ConnectionPool, data_dir: Path) -> dict[str, int]:
    ensure_app_schema(pool)  # creates the ops schema if the init script did not run
    counts = {}
    for path in sorted((data_dir / "structured").glob("*.csv")):
        counts[path.stem] = load_table(pool, path)
        log.info("loaded ops.%s: %d rows", path.stem, counts[path.stem])
    return counts


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = get_settings()
    pool = make_pool(settings, max_size=2)
    try:
        counts = load_all(pool, settings.support_data_dir)
    finally:
        pool.close()
    print(f"load_structured: {len(counts)} tables, {sum(counts.values())} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
