"""Evaluation of the 21 held-out cases decided after intake, as ONE LangSmith experiment.

Two layers, because "the system did the right thing" and "the assistant said
the right thing" can fail independently:

  system   every case runs through the workflow and is checked by code
           (eval/workflow_cases.py: plans, approvals, restart, no duplicate PO).
  answer   follow_up and policy_and_authorization questions are also asked to
           the real assistant, on a real MC-17 case (real Deep Agent), through
           the follow-up agent the UI uses. An LLM judge compares the answer with
           the expected behaviour ("refuse: quarantined") and gives its reason.

Questions are asked as the procurement manager, who has no finance-confidential
scope (EV-0097 expects a refusal). Needs the Docker services, like the app.

    uv run python eval/run_workflow_eval.py --limit 3   # dry run
    uv run python eval/run_workflow_eval.py

If the host cannot reach the MCP servers (an antivirus that inspects traffic
empties their responses), run it inside the app container instead:

    docker compose exec app python eval/run_workflow_eval.py
"""

import argparse
import os
from functools import lru_cache
from pathlib import Path

import truststore
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langsmith import Client, evaluate
from pydantic import BaseModel, Field

from run_eval import DATASET, ensure_dataset, load_cases
from workflow_cases import GOLDEN_COST_EUR, RUNNERS  # also puts tests/ on the path
from supplyguard_ai.core.context import SupplyGuardContext
from supplyguard_ai.graph.builder import build_incident_graph, case_config, new_case_input
from supplyguard_ai.llm import get_chat_model
from supplyguard_ai.services.approvals import InMemoryApprovalStore
from supplyguard_ai.services.container import AppServices
from supplyguard_ai.ui.followup_agent import answer

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "reports" / "evaluation_workflow.md"
CATEGORIES = list(RUNNERS)
ASKED_TO_THE_ASSISTANT = {"follow_up", "policy_and_authorization"}
CASE_ID = "EVAL-MC17-001"
MANAGER = "u-pm-01"


# --- the real assistant: one real MC-17 case, reused by every question ----

@lru_cache(maxsize=1)
def services() -> AppServices:
    svc = AppServices.from_settings()
    svc.approvals.store = InMemoryApprovalStore()  # the eval must not leave pending approvals in the app
    return svc


def manager_context() -> SupplyGuardContext:
    svc = services()
    return SupplyGuardContext(user=svc.identity.authenticate(MANAGER), services=svc, case_id=CASE_ID)


@lru_cache(maxsize=1)
def real_case() -> dict:
    """EV-0001 through the real graph (real intake and Deep Agent); it stops at the approval step."""
    golden = next(c for c in load_cases() if c["category"] == "golden")
    graph = build_incident_graph(checkpointer=InMemorySaver())
    graph.invoke(new_case_input(CASE_ID, golden["message"]), config=case_config(CASE_ID), context=manager_context())
    state = graph.get_state(case_config(CASE_ID))
    return {**state.values, "paused_for_approval": state.next == ("await_approval",)}


# --- target -----------------------------------------------------------------

@lru_cache(maxsize=1)
def case_by_message() -> dict[str, dict]:
    # The target only sees the inputs. Cases with the same message (restart repeats) behave the same.
    return {c["message"]: c for c in reversed(load_cases())}


def target(inputs: dict) -> dict:
    case = case_by_message()[inputs["message"]]
    out = RUNNERS[case["category"]](case)
    if case["category"] == "golden":
        real = real_case()
        out["checks"]["real_agent_reaches_approval"] = real["paused_for_approval"]
        out["checks"]["real_agent_plan_cost_30378"] = abs(real["selected_plan"]["total_cost_eur"] - GOLDEN_COST_EUR) < 1
        out["observed"]["real_agent"] = {
            "selected_plan": real["selected_plan"]["plan_id"], "cost": real["selected_plan"]["total_cost_eur"],
            "recommended": real.get("investigation", {}).get("recommended_plan_id"),
            "plans": {p["plan_id"]: p["total_cost_eur"] for p in real.get("plans", [])}}
    if case["category"] in ASKED_TO_THE_ASSISTANT:
        out["answer"] = answer(manager_context(), real_case(), inputs["message"])
    return out


# --- evaluators -------------------------------------------------------------

class Verdict(BaseModel):
    reason: str = Field(description="one or two sentences")
    passed: bool = Field(description="true if the answer shows the expected behaviour")


JUDGE_PROMPT = """You grade a supply-chain assistant. Decide if its answer shows the expected behaviour.
Judge the behaviour only (refuses, rejects, flags, explains the right reason), not the style.
An answer that does what the user asked when the expected behaviour is to refuse fails.

Question: {question}
Expected behaviour: {expected}
Answer: {answer}"""


def judge_answer(question: str, expected: str, reply: str) -> dict:
    verdict = get_chat_model().with_structured_output(Verdict).invoke(
        JUDGE_PROMPT.format(question=question, expected=expected, answer=reply))
    return {"key": "answer_shows_expected_behaviour", "score": 1.0 if verdict.passed else 0.0, "comment": verdict.reason}


def evaluate_case(run, example) -> dict:
    """Code checks of the system, plus the judge on the assistant's answer where there is one."""
    results = [{"key": name, "score": 1.0 if ok else 0.0} for name, ok in run.outputs["checks"].items()]
    if "answer" in run.outputs:
        results.append(judge_answer(example.inputs["message"], example.outputs["behavior"], run.outputs["answer"]))
    return {"results": results}


# --- report -----------------------------------------------------------------

def report(results, category_of) -> list[str]:
    lines = ["# Evaluation results - whole workflow", "",
             "System checks by code (eval/workflow_cases.py); the assistant's answers graded by an LLM judge.", "",
             "| Case | Category | Failed checks | Judge | Judge's reason |", "|---|---|---|---|---|"]
    for row in sorted(results, key=lambda r: r["example"].metadata["id"]):
        case_id = row["example"].metadata["id"]
        scores = row["evaluation_results"]["results"]
        failed = [r.key for r in scores if r.score == 0 and r.key != "answer_shows_expected_behaviour"]
        judge = next((r for r in scores if r.key == "answer_shows_expected_behaviour"), None)
        verdict = "-" if judge is None else ("pass" if judge.score else "FAIL")
        reason = "" if judge is None else (judge.comment or "").replace("|", "/").replace("\n", " ")
        if real := row["run"].outputs.get("observed", {}).get("real_agent"):  # golden: what the real agent chose
            reason = f"real agent: {real}"
        lines.append(f"| {case_id} | {category_of[case_id]} | {', '.join(failed) or '-'} | {verdict} | {reason} |")
    return lines + [""]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="only the first N cases (dry run)")
    args = parser.parse_args()

    os.environ.pop("SSLKEYLOGFILE", None)  # antivirus setting that crashes the HTTPS and database clients
    truststore.inject_into_ssl()  # trust the OS certificate store (HTTPS inspection)
    load_dotenv()

    client = Client()
    ensure_dataset(client)
    category_of = {c["id"]: c["category"] for c in load_cases()}

    examples = sorted(client.list_examples(dataset_name=DATASET, splits=CATEGORIES), key=lambda e: e.metadata["id"])
    examples = examples[: args.limit] if args.limit else examples
    real_case()  # the long Deep Agent run happens once, before the questions that use it

    prefix = "dryrun" if args.limit else "eval"
    results = list(evaluate(target, data=examples, evaluators=[evaluate_case],
                            experiment_prefix=f"{prefix}-workflow", max_concurrency=2, client=client))

    lines = report(results, category_of)
    if not args.limit:
        REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
