# SupplyGuard AI

AI assistant for supply-chain disruptions. A user reports a supplier delay in a chat; SupplyGuard AI measures the impact, gathers evidence from the company's documents and data, proposes costed recovery plans, and asks the responsible people for approval. No purchase order or other change is made without a bound human approval.

All numbers (stock, shortfall, costs, approval tiers) are calculated by deterministic code, not by the language model.

## Technology

| Area | Technology |
|---|---|
| Workflow | LangGraph (checkpointed in PostgreSQL) |
| Agents | Deep Agents, LangChain `create_agent` |
| Language model | Azure OpenAI (OpenAI and Ollama also supported) |
| Data | PostgreSQL + pgvector, ChromaDB, Neo4j |
| Integrations | MCP servers for procurement and external information |
| Sandbox | Isolated container without internet access |
| Interface | Chainlit |
| Tracing | LangSmith |

## How it works

```
Report -> Intake -> Investigate -> Plan -> Challenge -> Approval -> Execute -> Report
            |                                 |            |
            +-> asks the user if facts        +-> stops    +-> rejected: report only
                are missing                       on policy    modified: re-plan,
                                                  violations   new approval
```

Each case is a LangGraph workflow saved in PostgreSQL after every step, so a case can be resumed after a restart. There is no path from planning to execution that skips the approval step.

## Security and control

- **Bound approvals:** an approval covers one exact action (supplier, quantity, price). If anything changes, a new approval is needed. Every write to the procurement system carries an idempotency key, so a repeated call never creates a duplicate order.
- **Role-based access:** each user sees only the documents allowed by their role's access scopes; this is enforced in code, not in the prompt.
- **Isolated sandbox:** code written by the agent runs in a separate container with no internet access, a read-only file system and resource limits.
- **Prompt-injection checks:** the user's message and retrieved documents are scanned; document content is treated as data, never as instructions.

## Getting started

**Requirements:** Docker with Docker Compose, Python 3.12, [uv](https://docs.astral.sh/uv/), Azure OpenAI chat and embedding deployments.

1. Create the configuration file:

   ```bash
   cp .env.example .env
   ```

2. Fill in `.env`: the Azure OpenAI keys and endpoints, `POSTGRES_PASSWORD`, `NEO4J_PASSWORD` (at least 8 characters), `CHAINLIT_AUTH_SECRET` (generate with `uv run chainlit create-secret`), and one login password per user (`USER_PASSWORD_U_PM_01`, `USER_PASSWORD_U_AN_01`, `USER_PASSWORD_U_IC_01`, `USER_PASSWORD_U_FC_01`, `USER_PASSWORD_U_RO_01`). LangSmith keys are optional.

3. Start all services:

   ```bash
   docker compose up -d --build
   ```

   The first start loads the data and embeds the documents; the application starts when this has finished.

| Service | Address |
|---|---|
| Application | http://localhost:8000 |
| pgAdmin | http://localhost:5050 (`admin@example.com` / `admin`) |
| Neo4j Browser | http://localhost:7474 |

To stop: `docker compose down`.

## Usage

1. Open http://localhost:8000 and sign in with a user's **name** and the password you set in `.env`.

   | Name | Role |
   |---|---|
   | Maria Keller | Procurement manager |
   | Jon Weber | Operations analyst |
   | Ines Duarte | Incident coordinator |
   | Paul Novak | Finance controller |
   | Read Only Guest | Viewer |

2. Describe the disruption, for example:

   > Supplier Atlas Components has announced a 12-day delay for component MC-17. Determine which customer commitments are at risk, investigate alternative sources, and recommend a recovery plan. Additional spending must not exceed €75,000. Do not create a purchase order without approval.

3. Review the impact and the recommended plan. On the approval card, approve it, reject it, or switch to another proposed plan (which then needs a new approval). Plans up to EUR 25,000 need the procurement manager; plans up to EUR 75,000 also need the finance controller.

4. After approval, the actions are executed in the procurement system and an incident report is produced. You can ask follow-up questions about the case, for example *"Why did you reject Supplier Helios?"*.

## Key results

For the reference scenario (Atlas Components, MC-17, 12-day delay), verified by the automated tests:

| Measure | Result |
|---|---|
| Usable stock | 3,200 units |
| Shortfall | 5,350 units |
| Customer orders at risk | 77 |
| Recommended plan cost | about EUR 30,378 (within the EUR 75,000 cap) |

Document retrieval benchmark (31 questions, `reports/retrieval_benchmark.md`): hybrid search found the relevant document in the top 5 for every question (Recall@5 = 1.0), with 0 access-scope leaks.

## Tests

```bash
uv sync
uv run pytest                  # unit tests, no Docker or LLM needed
uv run pytest -m integration   # database tests, needs the running services
```

Retrieval benchmark results are in `reports/retrieval_benchmark.md`, red-team scenarios in `redteam/`, and quality notes in `reports/quality_notes.txt`.

## Team

| Member | Responsibility |
|---|---|
| Nikos Timplalexis | Platform |
| Nikos Kassaris | Workflow |
| Dimitris Delis | Interface |
| Maria Makri | Quality |

Knowledge (data and retrieval) was shared across the team.
