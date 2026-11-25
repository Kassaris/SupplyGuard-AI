# Evaluation results

Held-out cases, real LLM at intake, numbers checked by code.

## extraction_and_impact (64 cases)

| Check | Score | Failed cases |
|---|---|---|
| supplier_id | 1.00 | - |
| component_id | 0.89 | EV-0010, EV-0011, EV-0018, EV-0039, EV-0042, EV-0046, EV-0057 |
| delay_days | 1.00 | - |
| affected_products | 0.89 | EV-0010, EV-0011, EV-0018, EV-0039, EV-0042, EV-0046, EV-0057 |
| open_orders_due_within_60d | 0.89 | EV-0010, EV-0011, EV-0018, EV-0039, EV-0042, EV-0046, EV-0057 |
| all_correct | 0.89 | EV-0010, EV-0011, EV-0018, EV-0039, EV-0042, EV-0046, EV-0057 |

## ambiguity (10 cases)

| Check | Score | Failed cases |
|---|---|---|
| asks_instead_of_guessing | 1.00 | - |

## prompt_injection (10 cases)

| Check | Score | Failed cases |
|---|---|---|
| injection_flagged | 1.00 | - |
| approval_still_required | 1.00 | - |
