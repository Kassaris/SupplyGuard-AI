---
name: source-precedence
description: How to resolve conflicting information between contracts, qualification records, supplier notices, e-mails and chat (which source wins, superseded versions, how to report the conflict). Use whenever two sources give different numbers or dates.
metadata:
  procedures: DOC-PRO-02-v1
---

# Resolving conflicting sources (DOC-PRO-02)

## Order of authority (highest first)
1. Signed contract - **latest version in force** only
2. Supplier qualification record
3. Formal supplier notice
4. Supplier e-mail
5. Informal chat

Additional rules:
- A later `effective_date` supersedes an earlier version of the **same** document type.
  `list_documents` shows `superseded_by`; never quote a superseded contract as current.
- Quantities (capacity) from an e-mail never override contractual capacity **without written confirmation**.
  But an e-mail that reports a *lower* real capacity is a risk signal: plan with the lower value and flag it.
- A conflict is never silently resolved. Always report it.

## Procedure
For every fact that matters to the decision (delay, capacity, lead time, expedite fee, price):
1. Collect each source's value with its `source_id` and `effective_date`.
2. If they differ, pick the value by the order above.
3. Decide whether the lower-authority source still signals a **risk** (be conservative on risk, strict on cost).

## Report format (one row per conflict)
| Fact | Source A (value) | Source B (value) | Used | Why | Residual uncertainty |
|---|---|---|---|---|---|

## Conservative-choice guidance
- Delay: plan with the *longer* delay as a sensitivity case, even if the formal notice is shorter.
- Fees: use the contractual fee for the plan; show the higher e-mail fee as cost sensitivity.
- Capacity: use the lower credible capacity for feasibility.
