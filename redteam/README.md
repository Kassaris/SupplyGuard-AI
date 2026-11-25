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

Scenarios marked `pending` need the full workflow (LangGraph + approvals service) and will be run end to end once it is in `main`. Already covered scenarios are checked at the service level today; the end-to-end run checks them again through the agent.

The inputs are written for red-teaming; they are not taken from the held-out evaluation cases.
