---
name: disruption-investigation
description: Step-by-step procedure for investigating a supplier delay or disruption - case facts, usable stock, demand, stock-out timing, affected products and customer orders, alternatives and recovery options. Use at the start of every incident investigation.
metadata:
  procedures: DOC-PRO-01-v1, DOC-PRO-06-v1, DOC-PRO-07-v1
---

# Disruption investigation

Follow these phases in order. Write a todo item per phase and mark it done.
Keep running notes in `/case/notes.md` (facts with source ids) so nothing is lost when context is summarised.

## Phase 1 - Confirm the case facts (DOC-PRO-01)
1. `resolve_supplier` on the supplier the user named.
   - More than one candidate (e.g. "Atlas" -> Atlas Components S001 / Atlas Logistik S006): STOP and ask the user. Never guess.
2. Confirm component id (form `MC-<n>`) and the stated delay in days.
3. Note the scenario date from `get_policy` - it is "today". Never use the real calendar date.

## Phase 2 - Operational impact
Call `assess_impact(supplier_id, component_id, delay_days)` and record:
- physical vs **usable** stock (usable = on hand - reserved - quarantined, DOC-PRO-06);
  name any quarantined lot and say it cannot be used or transferred;
- reserved stock is tied to specific customer orders - it is not free stock;
- daily demand, first stock-out day, shortfall units;
- days of cover per plant - the plants with the lowest cover run out first;
- affected products, count of at-risk orders, tier-1 orders, revenue at risk.

## Phase 3 - Documentary evidence
Use `list_documents` / `read_document` for: the disrupted supplier's latest contract and notice,
the contracts and qualification records of each alternative, related e-mails, and similar past incident reports.
Load the `source-precedence` skill before resolving any conflict.
Load the `untrusted-content` skill before reading supplier notices or e-mails.

## Phase 4 - Alternatives
Call `screen_alternative_suppliers`. Load `supplier-qualification` and explain for EACH supplier why it is feasible or rejected.

## Phase 5 - Recovery options
Call `propose_recovery_plans` with the spending cap. If documents changed a value (e.g. a contractual
expedite fee), pass it as an override and cite the document.
Compare at least three strategies: A transfer, B expedite, C alternate purchase, D reallocate to priority customers, E reschedule.

## Phase 6 - Challenge
Load `recovery-plan-review` and run its checklist before presenting a recommendation.

## Output
Write `/case/findings.md` with sections: Facts, Impact, Evidence & conflicts, Supplier screening,
Options compared, Recommendation, Open questions, Security findings. Every number cites a source id.
