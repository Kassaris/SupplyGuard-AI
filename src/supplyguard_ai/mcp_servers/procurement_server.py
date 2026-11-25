"""
MCP server of the simulated procurement system (assignment §05 "MCP
integrations: Access to procurement operations").

Run by docker compose (service `procurement-mcp`):
    python -m supplyguard_ai.mcp_servers.procurement_server        # PORT=8101

WHY A SEPARATE MCP SERVER
-------------------------
In the real world the procurement system (ERP) is another team's system.
Putting it behind MCP makes the integration boundary explicit and testable:
  * the app can ONLY change procurement data through these tools,
  * the server enforces its own rules - it does not trust the caller,
  * any MCP client (our agents, Claude Desktop, a test) sees the same contract.
The tool list mirrors support-data/config/procurement_system_operations.json.

MCP PRIMITIVES USED (Week 3, day 11)
------------------------------------
  tools     read operations + approval-bound write operations
  resource  procurement://operations - the system's contract (JSON)
  prompt    approval_brief - a reusable prompt to summarise an approval for a human

SECURITY MODEL OF THE WRITE TOOLS
---------------------------------
The server cannot approve anything.  A write needs an `approval_id` whose
record in Postgres (app.approvals) is APPROVED and contains EXACTLY this
payload (`execute_approved`).  So a leaked token or a prompt-injected agent
calling `create_purchase_order` with 20,000 units gets "rejected".  Every write
also carries case_id + step -> idempotency key: retries never duplicate.
The compose file publishes the port on 127.0.0.1 only; inside the compose
network only our services can reach it.

FastMCP turns each decorated function into an MCP tool: the function name is
the tool name, the docstring its description, the type hints its JSON schema.
"""

from __future__ import annotations

import json
import os
from datetime import date
from functools import lru_cache

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from supplyguard_ai.config import get_settings
from supplyguard_ai.mcp_servers import allowed_hosts, bind_host

PORT = int(os.environ.get("PORT", "8101"))

mcp = FastMCP(
    "supplyguard-procurement",
    instructions=("Simulated procurement system. Read operations are free. Write operations require an "
                  "approval_id of an APPROVED approval containing exactly the same payload."),
    host=bind_host(get_settings().mcp_host),
    port=PORT,
    stateless_http=True,   # no per-client session state: any replica can serve any request
    json_response=True,
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=True,
                                                 allowed_hosts=allowed_hosts(PORT), allowed_origins=[]),
)


@lru_cache(maxsize=1)
def services():
    """The same service layer as the app, on the same Postgres (approvals, ledger)."""
    from supplyguard_ai.services.container import AppServices

    return AppServices.from_settings()


def _rows(rows: list[dict]) -> list[dict]:
    """JSON-safe rows (dates -> ISO strings)."""
    return [{k: (v.isoformat() if isinstance(v, date) else v) for k, v in r.items()} for r in rows]


# =========================================================================== read operations
@mcp.tool()
def get_inventory(component_id: str) -> list[dict]:
    """Stock of a component per location: on_hand, reserved, quarantined, usable, last_counted_date."""
    return _rows(services().procurement.get_inventory(component_id))


@mcp.tool()
def get_open_purchase_orders(component_id: str | None = None, supplier_id: str | None = None) -> list[dict]:
    """Open purchase orders for a component and/or supplier, incl. POs created by approved executions."""
    if not component_id and not supplier_id:
        raise ValueError("give component_id and/or supplier_id")
    svc = services()
    if component_id:
        rows = svc.procurement.get_open_purchase_orders(component_id)
    else:
        rows = [r for r in svc.repo._table("purchase_orders") if r["status"] == "open"]
    if supplier_id:
        rows = [r for r in rows if r.get("supplier_id") == supplier_id]
    return _rows(rows)


@mcp.tool()
def get_supplier(supplier_id: str) -> dict:
    """Supplier master record: ERP status, certification and its expiry, upstream manufacturer."""
    row = services().repo.supplier(supplier_id)
    if row is None:
        raise ValueError(f"unknown supplier {supplier_id}")
    return _rows([row])[0]


@mcp.tool()
def get_supplier_catalog(component_id: str) -> list[dict]:
    """Who can supply a component: price, lead time, monthly capacity, MOQ, approved_source."""
    return _rows(services().procurement.get_supplier_catalog(component_id))


@mcp.tool()
def get_customer_orders(product_id: str, due_before: str | None = None) -> list[dict]:
    """Customer orders for a product, optionally only those due before an ISO date (YYYY-MM-DD)."""
    rows = services().repo.customer_orders({product_id})
    if due_before:
        limit = date.fromisoformat(due_before)
        rows = [r for r in rows if isinstance(r["due_date"], date) and r["due_date"] < limit]
    return _rows(rows)


@mcp.tool()
def get_production_schedule(product_id: str) -> list[dict]:
    """Planned weekly production units for a product."""
    return _rows(services().repo.production_schedule({product_id}))


@mcp.tool()
def get_shipment_updates(po_id: str) -> list[dict]:
    """Carrier / supplier events for a purchase order (status, event date, ETA, note), oldest first."""
    return _rows(services().repo.shipment_updates(po_id))


# =========================================================================== write operations
def _write(kind: str, case_id: str, step: str, approval_id: str, actor: str, payload: dict) -> dict:
    r = services().procurement.execute_approved(case_id, step, approval_id, kind, payload, actor)
    return r.model_dump(mode="json")


@mcp.tool()
def create_purchase_order(case_id: str, step: str, approval_id: str, actor: str, supplier_id: str, component_id: str,
                          qty: int, unit_price_eur: float, requested_delivery_date: str) -> dict:
    """Create a PO. Requires an APPROVED approval containing exactly this payload. Idempotent per case_id+step."""
    return _write("create_purchase_order", case_id, step, approval_id, actor, {
        "supplier_id": supplier_id, "component_id": component_id, "qty": qty,
        "unit_price_eur": unit_price_eur, "requested_delivery_date": requested_delivery_date})


@mcp.tool()
def expedite_purchase_order(case_id: str, step: str, approval_id: str, actor: str, po_id: str, days_gain: int,
                            fee_pct: float) -> dict:
    """Expedite an open PO. Requires an APPROVED approval containing exactly this payload."""
    return _write("expedite_purchase_order", case_id, step, approval_id, actor,
                  {"po_id": po_id, "days_gain": days_gain, "fee_pct": fee_pct})


@mcp.tool()
def request_stock_transfer(case_id: str, step: str, approval_id: str, actor: str, component_id: str,
                           from_location: str, to_location: str, qty: int) -> dict:
    """Transfer usable stock between locations. Requires an APPROVED approval containing exactly this payload."""
    return _write("request_stock_transfer", case_id, step, approval_id, actor, {
        "component_id": component_id, "from_location": from_location, "to_location": to_location, "qty": qty})


@mcp.tool()
def update_production_schedule(case_id: str, step: str, approval_id: str, actor: str, product_id: str,
                               week_start: str, planned_units: int) -> dict:
    """Change planned units for a product week. Requires an APPROVED approval containing exactly this payload."""
    return _write("update_production_schedule", case_id, step, approval_id, actor,
                  {"product_id": product_id, "week_start": week_start, "planned_units": planned_units})


# =========================================================================== resource & prompt
@mcp.resource("procurement://operations", mime_type="application/json")
def operations_contract() -> str:
    """The contract of this system (operations, required conditions, idempotency rule)."""
    path = get_settings().support_data_dir / "config" / "procurement_system_operations.json"
    return path.read_text(encoding="utf-8")


@mcp.prompt()
def approval_brief(approval_id: str) -> str:
    """Prompt that asks a model to summarise an approval request for a human approver."""
    rec = services().approvals.get(approval_id)
    return ("Summarise this approval request for a procurement approver in 5 bullet points: what will be "
            "bought/moved, quantities, total cost, who must still sign, and the main risk.\n\n"
            + json.dumps(rec.model_dump(mode="json"), indent=2))


def main() -> None:
    mcp.run(transport="streamable-http")  # serves http://<host>:<PORT>/mcp


if __name__ == "__main__":
    main()
