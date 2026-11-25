"""
Chainlit app: the only thing the end user sees (assignment §04).

Started by the Dockerfile:
    chainlit run src/supplyguard_ai/ui/chainlit_app.py -h --host 0.0.0.0 --port 8000
Look and feel: .chainlit/config.toml loads public/custom.css, public/custom.js and the
logos; chainlit.md is the welcome page.

THIS FILE IS DELIBERATELY THIN
------------------------------
All logic lives in tested modules:
  auth.authenticate      login (name + password)    runner.CaseRunner   graph driving
  presenters.*           Markdown                   followup_agent      Q&A on a case
Here we only map user actions to those calls and events to messages.

CONVERSATION STATE (cl.user_session, per browser tab)
-----------------------------------------------------
  sg_user   the authenticated domain User (from the login callback)
  case_id   the case this chat is working on
  waiting   None | "clarification" | "approval"   (what the graph is paused on)
The CASE state itself is in the Postgres checkpoints - so after an app restart
the user picks the case from the list and it continues exactly where it stopped.
Cases are shared through the case registry, so a finance controller can open
and co-sign a case that a procurement manager started.

SYNC GRAPH IN AN ASYNC APP
--------------------------
Chainlit handlers are async; the runner is sync (see runner.py).
`_stream()` runs the runner in a worker thread and hands each Event back to
the event loop through an asyncio.Queue, so progress appears live while the
deep agent is still working.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from functools import lru_cache

import chainlit as cl

from supplyguard_ai.core.context import SupplyGuardContext
from supplyguard_ai.graph.builder import build_incident_graph
from supplyguard_ai.services.container import AppServices
from supplyguard_ai.ui import presenters as P
from supplyguard_ai.ui.auth import authenticate
from supplyguard_ai.ui.followup_agent import answer
from supplyguard_ai.ui.runner import CaseRunner, Event

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- process-wide objects
@lru_cache(maxsize=1)
def services() -> AppServices:
    return AppServices.from_settings()


@lru_cache(maxsize=1)
def runner() -> CaseRunner:
    svc = services()
    return CaseRunner(svc, build_incident_graph(checkpointer=svc.make_checkpointer()))


def _user():
    # Our domain User. NOT the session key "user": Chainlit keeps its own
    # cl.User (from the login callback) there and relies on it internally.
    return cl.user_session.get("sg_user")


async def _stream(gen_factory) -> list[Event]:
    """Run a sync event generator in a thread; render events as they arrive."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    done = object()

    def worker():
        try:
            for ev in gen_factory():
                loop.call_soon_threadsafe(queue.put_nowait, ev)
        except Exception as exc:  # noqa: BLE001 - shown to the user, logged here
            logger.exception("Error while running the case")
            loop.call_soon_threadsafe(queue.put_nowait, Event("error", {"message": f"{type(exc).__name__}: {exc}"}))
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, done)

    threading.Thread(target=worker, daemon=True).start()
    events, holder = [], {"step": None}  # one collapsible "Working on the case" step per run
    while (ev := await queue.get()) is not done:
        events.append(ev)
        await _render(ev, holder)
    if holder["step"] is not None:
        await holder["step"].__aexit__(None, None, None)
    return events


async def _render(ev: Event, holder: dict) -> None:
    if ev.kind in ("progress", "node"):
        if holder["step"] is None:
            holder["step"] = cl.Step(name="Working on the case", type="run", default_open=False)
            await holder["step"].__aenter__()
        label = ev.data.get("label") or f"{ev.data.get('phase')}: {ev.data.get('detail', '')}"
        holder["step"].output = (holder["step"].output or "") + f"✔ {label}\n"
        await holder["step"].update()
        if ev.kind == "node" and ev.data["node"] == "plan":
            state = await asyncio.to_thread(runner().state, cl.user_session.get("case_id"))
            summary = P.investigation_summary(state)
            if summary:
                await cl.Message(summary).send()
    elif ev.kind == "message" and ev.data["node"] in ("investigate", "await_approval", "execute"):
        await cl.Message(ev.data["text"]).send()
    elif ev.kind == "clarification":
        cl.user_session.set("waiting", "clarification")
        await cl.Message(P.clarification(ev.data)).send()
    elif ev.kind == "approval":
        cl.user_session.set("waiting", "approval")
        await _send_approval_card(ev.data)
    elif ev.kind == "done":
        cl.user_session.set("waiting", None)
        elements = [cl.Text(name="incident_report.md", content=ev.data["report_md"], display="inline")] if ev.data.get("report_md") else []
        await cl.Message(P.outcome(ev.data) + "\n\nAsk me follow-up questions about this case, or describe a new incident.",
                         elements=elements).send()
    elif ev.kind == "error":
        await cl.Message(f"⛔ {ev.data['message']}").send()


async def _send_approval_card(value: dict) -> None:
    """Show the exact proposal. Only users whose role must still sign get the buttons;
    everyone else sees who the request is waiting for."""
    await _clear_card_buttons()  # only ONE live approval card per chat
    signed = value.get("signed_by", {})
    missing = [r for r in value.get("required_roles", []) if r not in signed]
    if _user().role.value not in missing:
        await cl.Message(P.approval(value) + "\n\nWaiting for approval by: **"
                         + (", ".join(missing) or "nobody") + "**.").send()
        return

    case_id = value["case_id"]
    state = await asyncio.to_thread(runner().state, case_id)
    other = [p["plan_id"] for p in state.get("plans", []) if p["plan_id"] != value["plan"]["plan_id"]]
    ref = {"case_id": case_id, "approval_id": value["approval_id"]}  # the card names the approval it shows
    actions = [
        # Labels must stay exactly "Approve" / "Reject": public/custom.js finds the buttons by
        # their text and custom.css colours them green / red.
        cl.Action(name="approve", payload=ref, label="Approve"),
        cl.Action(name="reject", payload=ref, label="Reject"),
        *[cl.Action(name="choose_plan", payload={**ref, "plan_id": pid}, label=f"↺ Switch to {pid}") for pid in other],
    ]
    cl.user_session.set("card_actions", actions)
    await cl.Message(P.approval(value), actions=actions).send()


async def _clear_card_buttons():
    """Remove all buttons of the previous approval card (stale cards must not be clickable).
    The graph refuses a stale approval_id anyway - this just keeps the UI honest."""
    for a in cl.user_session.get("card_actions") or []:
        await a.remove()
    cl.user_session.set("card_actions", [])


# --------------------------------------------------------------------------- auth & start
@cl.password_auth_callback
async def login(username: str, password: str):
    """Login: the user's name (e.g. Maria Keller) + personal password. None = rejected."""
    try:
        user = await asyncio.to_thread(authenticate, services(), username, password)
    except Exception:
        # A database problem must never let anyone in.
        logger.exception("Login check failed")
        return None
    if user is None:
        return None
    # Role and scopes come from the database, never from what the user types.
    return cl.User(identifier=user.user_id, display_name=user.name, metadata={"role": user.role.value})


@cl.on_chat_start
async def start():
    app_user = cl.user_session.get("user")  # Chainlit's cl.User from the login callback
    user = services().identity.authenticate(app_user.identifier)
    cl.user_session.set("sg_user", user)
    cl.user_session.set("case_id", None)
    cl.user_session.set("waiting", None)
    cases = await asyncio.to_thread(services().cases.visible_to, user)
    actions = [cl.Action(name="open_case", payload={"case_id": c["case_id"]}, label=f"Open {c['case_id']} ({c['phase']})")
               for c in cases[:5]]
    await cl.Message(P.welcome(user, cases), actions=actions).send()


# --------------------------------------------------------------------------- messages
@cl.on_message
async def on_message(msg: cl.Message):
    user, case_id, waiting = _user(), cl.user_session.get("case_id"), cl.user_session.get("waiting")
    text = msg.content.strip()

    if waiting == "clarification" and case_id:                       # answer to the graph's question
        await _stream(lambda: runner().resume(user, case_id, text))
        return

    state = await asyncio.to_thread(runner().state, case_id) if case_id else {}
    if case_id and state and not text.lower().startswith(("new incident", "new case")):
        # follow-up question on the current case (also while an approval is pending)
        ctx = SupplyGuardContext(user=user, services=services(), case_id=case_id)
        try:
            async with cl.Step(name="Looking into the case", type="tool"):
                reply = await asyncio.to_thread(answer, ctx, state, text)
        except Exception as exc:  # noqa: BLE001
            # The user must always get an answer, even when the LLM or a data service
            # is down. The case itself is unaffected (a follow-up never changes it).
            logger.exception("Follow-up question failed")
            reply = (f"⚠️ I couldn't answer right now ({type(exc).__name__}). The language model or a data "
                     "service is unreachable - please ask again in a moment. Your case is unchanged.")
        await cl.Message(reply or "I found no answer to that - please rephrase the question.").send()
        return

    try:
        new_id, gen = await asyncio.to_thread(runner().start, user, text)  # a new incident case
    except PermissionError as exc:  # e.g. a viewer: read-only, no case is created
        await cl.Message(f"⛔ {exc}. You can't open incident cases.").send()
        return
    cl.user_session.set("case_id", new_id)
    await cl.Message(f"Opened case `{new_id}`. Investigating - this takes about a minute.").send()
    await _stream(lambda: gen)


# --------------------------------------------------------------------------- buttons
async def _allowed(case_id: str) -> bool:
    """Button payloads come from the browser and can be tampered with: every
    case id is checked against the registry's access rule on the SERVER."""
    if await asyncio.to_thread(services().cases.can_access, _user(), case_id):
        return True
    await cl.Message(f"⛔ You don't have access to case `{case_id}`.").send()
    return False


@cl.action_callback("open_case")
async def open_case(action: cl.Action):
    # Loading a case takes a moment; extra clicks meanwhile used to print
    # "Resumed case ..." once per click (found in the UI test). Ignore them.
    if cl.user_session.get("opening"):
        return
    cl.user_session.set("opening", True)
    try:
        await _open_case(action.payload["case_id"])
    finally:
        cl.user_session.set("opening", False)


async def _open_case(case_id: str):
    if not await _allowed(case_id):
        return
    cl.user_session.set("case_id", case_id)
    pending = await asyncio.to_thread(runner().pending, case_id)
    state = await asyncio.to_thread(runner().state, case_id)
    await cl.Message(f"Resumed case `{case_id}` · phase **{state.get('phase')}**.\n\n" + P.investigation_summary(state)).send()
    if pending and pending["type"] == "approval":
        cl.user_session.set("waiting", "approval")
        await _send_approval_card({"case_id": case_id, **pending})
    elif pending and pending["type"] == "clarification":
        cl.user_session.set("waiting", "clarification")
        await cl.Message(P.clarification(pending)).send()


async def _decide(action: cl.Action, decision: dict):
    case_id = action.payload["case_id"]
    if not await _allowed(case_id):
        return
    cl.user_session.set("case_id", case_id)
    await _clear_card_buttons()  # single-use; a fresh card comes if the case is still pending
    # Read the user HERE, on the event loop: cl.user_session is not available in the worker thread.
    user = _user()
    await _stream(lambda: runner().resume(user, case_id, {**decision, "approval_id": action.payload["approval_id"]}))


@cl.action_callback("approve")
async def on_approve(action: cl.Action):
    await _decide(action, {"action": "approve"})


@cl.action_callback("reject")
async def on_reject(action: cl.Action):
    res = await cl.AskUserMessage(content="Reason for rejecting this plan?", timeout=300).send()
    await _decide(action, {"action": "reject", "reason": (res or {}).get("output", "no reason given")})


@cl.action_callback("choose_plan")
async def on_choose_plan(action: cl.Action):
    await _decide(action, {"action": "modify", "changes": {"preferred_plan_id": action.payload["plan_id"]}})
