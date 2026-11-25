# Red-team scenarios

`scenarios.jsonl` lists the attacks the application must withstand, one per line.

| Field | Meaning |
|---|---|
| `category` | prompt_injection, jailbreak, data_exfiltration, tool_abuse, approval_bypass, duplicate_action, misinformation, denial_of_service, authorization |
| `user` | who sends the input (ids from `support-data/config/users.csv`) |
| `input` | the message or call used in the attack |
| `expected` | the safe behaviour |
| `defense` | the layer that should stop it |
| `covered_by` | the automated test that already checks it, or `pending` |

Scenarios marked "workflow, fake LLM" run through the real LangGraph workflow with the LLM parts replaced by fakes (tests/workflow_setup.py). Scenarios marked `pending` depend on what the model itself says or does, so they need a run with the real LLM.

The inputs are written for red-teaming; they are not taken from the held-out evaluation cases.
