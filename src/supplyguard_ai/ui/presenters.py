"""
Presenters: case data -> Markdown for the chat window.

Pure functions (dict in, str out): easy to test, and the Chainlit file
stays free of formatting code.  Every number shown comes from the
deterministic state; every claim keeps its source id in [brackets].
"""

from __future__ import annotations

ACTION_LABELS = {
    "request_stock_transfer": "Transfer stock",
    "expedite_purchase_order": "Expedite PO",
    "create_purchase_order": "New purchase order",
    "update_production_schedule": "Reschedule production",
}


def eur(x: float) -> str:
    return f"EUR {x:,.2f}"


def welcome(user, open_cases: list[dict]) -> str:
    scopes = ", ".join(sorted(user.document_scopes)) or "none"
    lines = [f"Hi **{user.name}**! I'm SupplyGuard AI. Role: {user.role.value} · document access: {scopes}.", ""]
    if user.role.value == "viewer":
        # Viewers are read-only (services/identity.py): don't invite them to open cases.
        lines.append("You have read-only access: you can open the cases shared with you below, "
                     "but you can't open new incident cases or approve plans.")
    else:
        lines.append("Describe the supplier disruption (the supplier, the component and the announced delay) "
                     "to open a new incident case, or resume a case below.")
    if open_cases:
        lines += ["", "**Your cases**", *[f"- `{c['case_id']}` · {c['phase']} · {c['title']}" for c in open_cases]]
    return "\n".join(lines)


def clarification(value: dict) -> str:
    out = [f"**I need a clarification before investigating.**", "", value["question"]]
    if value.get("candidates"):
        out += ["", *[f"- {c['name']} ({c['supplier_id']}, {c['city']})" for c in value["candidates"]]]
    return "\n".join(out)


def approval(value: dict) -> str:
    plan, ch, ev = value["plan"], value.get("challenge", {}), value.get("evidence", {}) or {}
    signed = value.get("signed_by", {})
    out = [
        f"### Approval required · `{value['approval_id']}`",
        f"Plan **{plan['plan_id']}** · total **{eur(plan['total_cost_eur'])}** · "
        f"{plan['units_covered']} units covered · **{plan['uncovered_units']} uncovered**",
        "",
        "| # | Action | Exact payload | Cost |",
        "|---|---|---|---|",
    ]
    for i, a in enumerate(plan["actions"], 1):
        payload = ", ".join(f"{k}={v}" for k, v in a["payload"].items())
        out.append(f"| {i} | {ACTION_LABELS.get(a['kind'], a['kind'])} | `{payload}` | {eur(a['cost_eur'])} |")
    roles = value.get("required_roles", [])
    out += ["", "**Signatures:** " + " · ".join(f"{r} {'✅ ' + signed[r] if r in signed else '⏳ pending'}" for r in roles)]
    if ch.get("blockers"):
        out += ["", "**Blockers**", *[f"- ⛔ {b}" for b in ch["blockers"]]]
    if ch.get("warnings"):
        out += ["", "**Warnings**", *[f"- ⚠️ {w}" for w in ch["warnings"]]]
    if ch.get("notes"):
        # Plain Markdown: Chainlit does not render raw HTML such as <details> (it showed as text).
        out += ["", "**Notes**", *[f"- {n}" for n in ch["notes"]]]
    if ev.get("supplier_verdicts"):
        out += ["", "**Supplier verdicts**", *[f"- {v['supplier_name']} ({v['supplier_id']}): **{v['verdict']}** - "
                                              f"{v['reason']} [{v['source_id']}]" for v in ev["supplier_verdicts"]]]
    if ev.get("conflicts"):
        out += ["", "**Resolved conflicts**", *[f"- {c['supplier_id']} {c['topic']}: used **{c['resolved_value']}** "
                                               f"[{c['resolved_source_id']}] - {c['reason']}" for c in ev["conflicts"]]]
    out += ["", f"_Payload hash `{value['payload_hash'][:12]}…`. No purchase order exists until this request is "
                f"approved; any change to supplier, quantity or price requires a new approval._"]
    return "\n".join(out)


def investigation_summary(state: dict) -> str:
    imp, inv = state.get("impact", {}), state.get("investigation", {})
    if not imp:
        return ""
    cover = ", ".join(f"{s['location_id']} {s['days_of_cover']} d" for s in imp["stock"] if s.get("days_of_cover"))
    out = [
        "### Impact",
        f"- Usable stock **{imp['total_usable']}** of {imp['total_on_hand']} on hand "
        f"({imp['total_reserved']} reserved, {imp['total_quarantined']} quarantined) [SRC-WMS-INVENTORY]",
        f"- Days of cover: {cover}",
        f"- Shortfall **{imp['shortfall_units']} units**; {len(imp['affected_products'])} products, "
        f"{len(imp['at_risk_orders'])} open orders at risk [SRC-ERP-BOM, SRC-ERP-ORDERS]",
        f"- Revenue at risk (exposure, not a loss): {eur(imp['revenue_at_risk_eur'])}",
    ]
    if inv.get("security_findings"):
        out += ["", "**Security findings**", *[f"- 🛡️ `{s['source_id']}`: {s['description']} - ignored" for s in inv["security_findings"]]]
    if inv.get("external_alerts"):
        out += ["", "**External risk alerts** (untrusted feed)",
                *[f"- {a['supplier_name']} via {a['via']} · {a['region']} · {a['severity']}: {a['text']}" for a in inv["external_alerts"]]]
    return "\n".join(out)


def outcome(event: dict) -> str:
    rows = event.get("execution_results") or []
    out = [f"### Case `{event['case_id']}` · {event['phase']}"]
    if rows:
        out += ["", *[f"- {r['operation']}: **{r['status']}** {r.get('reference') or r.get('message')}" for r in rows]]
    return "\n".join(out)
