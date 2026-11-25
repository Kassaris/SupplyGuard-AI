# Contributing to SupplyGuard AI

This guide explains how the team works on this repository: how to set up the project, how to name branches and commits, how changes get reviewed, and which quality rules every change must follow.

## 1. Setup

Requirements: Python 3.12, [uv](https://docs.astral.sh/uv/), Docker, and Git.

```bash
git clone https://github.com/Kassaris/SupplyGuard-AI.git
cd SupplyGuard-AI
uv sync                      # creates .venv and installs dependencies
cp .env.example .env         # then fill in your own keys
```

On Windows PowerShell, use `Copy-Item .env.example .env` instead of `cp`.

- Never commit `.env` or any file that contains API keys, passwords, or tokens.
- Each team member uses their own LangSmith API key, and everyone sends traces to the same project (`LANGSMITH_PROJECT=supplyguard-ai`).
- Add dependencies with `uv add <package>` (or `uv add --dev <package>` for test tools), so `pyproject.toml` and `uv.lock` stay in sync. Commit both files together.

## 2. Repository layout

| Folder | Content | Owner |
|---|---|---|
| `src/supplyguard_ai/` | Application code (agents, graph, tools, UI) | App team |
| `support-data/` | Provided synthetic data. **Read-only: never edit these files.** | - |
| `tests/` | Automated tests (pytest) | Quality |
| `eval/` | Evaluation harness for the 105 cases and LangSmith experiments | Quality |
| `benchmarks/` | Retrieval benchmark (pgvector, ChromaDB, hybrid) | Quality |
| `redteam/` | Adversarial test scenarios and findings | Quality |
| `reports/` | Generated results (evaluation, benchmark, red-team) | Quality |

## 3. Branches

- `main` always works. Nobody pushes directly to `main`.
- Everyone works on their own branch and merges into `main` through a pull request.

## 4. Commit messages

Format: `<type>: <short description>` in the imperative mood, under 72 characters.

| Type | Meaning |
|---|---|
| `feat` | New functionality |
| `fix` | Bug fix |
| `test` | Tests or evaluation |
| `docs` | Documentation |
| `refactor` | Code change without behaviour change |
| `chore` | Configuration, dependencies, tooling |

Examples:

```
feat: add usable stock calculation tool
test: verify MC-17 shortfall against golden scenario
fix: exclude quarantined lots from transfer options
```

## 5. Pull requests

1. Pull the latest `main` and rebase or merge it into your branch.
2. Run the tests locally: `uv run pytest`.
3. Open a pull request to `main` with:
   - what changed and why,
   - how it was tested,
   - any known limitations.
4. At least **one other team member** reviews and approves before merging.
5. A pull request that breaks existing tests is not merged.

## 6. Quality rules

These rules come from the project requirements. Every change is expected to respect them.

1. **Numbers come from deterministic tools, never from the LLM.** Stock, shortfall, costs, and approval tiers are calculated in code and covered by tests.
2. **The golden scenario must always pass.** Any change that alters the MC-17 results (usable stock 3,200, shortfall 5,350, 77 orders at risk, reference plan about EUR 30,378) must be explained in the pull request.
3. **Retrieved content is data, not instructions.** Text from documents, e-mails, or tool results never changes policies, approvals, or tool calls.
4. **No write action without a bound approval.** Purchase orders, expedites, transfers, and schedule changes require an `approval_id` for the exact payload and an idempotency key.
5. **Access scopes are enforced in retrieval.** Users only see documents within their scopes; finance-confidential data never reaches users without that scope.
6. **The evaluation cases are held out.** Do not tune prompts on `support-data/evaluation/eval_cases.jsonl`; these cases are only measured.
7. **Every new tool or business rule comes with a test.**

## 7. Running tests and evaluation

```bash
uv run pytest                 # unit and data tests (no LLM, no network)
uv run pytest -m integration  # tests that need the Docker services running
```

The evaluation and benchmark scripts in `eval/` and `benchmarks/` describe how to run them at the top of each file; the red-team scenarios are described in `redteam/README.md`.

## 8. Reporting issues

Open a GitHub issue with: what you expected, what happened, steps to reproduce, and (for agent behaviour) the LangSmith trace link.
