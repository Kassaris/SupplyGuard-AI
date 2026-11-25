---
name: supplier-qualification
description: Rules for deciding whether an alternative supplier may receive an order - certification validity, ERP status, lead time versus the delay, capacity and MOQ, and shared upstream manufacturer risk. Use when screening or explaining the rejection of a supplier.
metadata:
  procedures: DOC-PRO-04-v1, DOC-PRO-05-v1
---

# Supplier qualification and diversification

## Hard rules - any failure = supplier rejected (DOC-PRO-04)
- Certification (ISO 9001) must be valid on the scenario date.
  **An expired certificate blocks ordering regardless of ERP status or price.**
  The ERP status column can be stale - check the expiry date (and the qualification record).
- ERP status must be `approved`.
- The supplier must deliver **before** the delayed original shipment arrives; otherwise it does not reduce the shortfall.

## Soft rules - feasible, but must be reported as a risk (DOC-PRO-05)
- **Shared upstream manufacturer**: a second source only counts as diversification if it does not
  share an upstream manufacturer with the disrupted supplier. If it does, any upstream problem hits both.
  Check the external risk feed for alerts in that manufacturer's region.
- Capacity below the gap: the supplier can be part of the plan but cannot cover it alone.
- Capacity conflict between contract and e-mail: see `source-precedence`.
- Catalog status `alternative` (not the primary source): mention it.

## How to explain a decision
One line per supplier: **name (id) - verdict - the deciding reason - source ids**.
Example shape: "Supplier X (S0xx) - rejected - lead time 21 d arrives after the delayed shipment (day 19) [SRC-...]."

Never call the cheapest supplier "best" on price alone. Cheap but late or uncertified is not a recovery option.
