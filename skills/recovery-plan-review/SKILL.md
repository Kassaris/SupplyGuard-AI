---
name: recovery-plan-review
description: Critic checklist to challenge a proposed recovery plan before it goes to human approval - missing evidence, policy violations, weak assumptions, double-counted penalties, cap and approval tier. Use after plans are built and before recommending one.
metadata:
  procedures: DOC-PRO-03-v1, DOC-PRO-06-v1, DOC-PRO-07-v1, DOC-PRO-08-v1, DOC-PRO-12-v1
---

# Recovery plan review (the "challenge" step)

Play devil's advocate. For the recommended plan answer every question; any "no" is a finding.

## Evidence
- [ ] Every quantity, price, date and cost comes from a tool result or document, with a source id?
- [ ] Every conflict between sources is reported with the value used (see `source-precedence`)?
- [ ] Superseded contract versions were not used?

## Policy
- [ ] No supplier with an expired certificate or non-approved ERP status receives an order (DOC-PRO-04)?
- [ ] Shared-upstream suppliers are flagged, not presented as diversification (DOC-PRO-05)?
- [ ] Transfers use usable stock only; no quarantined lot is moved; reserved stock is only reallocated via customer priority (DOC-PRO-06/07)?
- [ ] Expedites only on open POs with expedite allowed; fee = PO value x fee% (DOC-PRO-08)?
- [ ] Total additional spend is within the cap, and the approval tier is named (DOC-PRO-03):
      up to EUR 25k procurement manager; 25k-75k procurement manager + finance controller; above 75k finance director?

## Business reasoning
- [ ] Revenue at risk is reported **separately** from recovery cost and not called a loss (DOC-PRO-12)?
- [ ] Late penalties counted once, only for orders that actually miss their date, capped per clause?
- [ ] Timing checked, not just totals: does stock run out at a plant before the first delivery arrives?
- [ ] Plants with the lowest days of cover are protected first?
- [ ] Residual uncovered units have a stated handling (reschedule E or reallocate to tier-1 orders D)?
- [ ] Sensitivity stated: longer delay from e-mail, higher fee from e-mail, lower real capacity?

## Output
List findings as: severity (blocker / warning / note) - finding - fix.
A blocker means the plan must be rebuilt before approval is requested.
