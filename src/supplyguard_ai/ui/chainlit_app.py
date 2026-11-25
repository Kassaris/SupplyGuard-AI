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

CHAT HISTORY (Chainlit data layer, tables chat.* - persistence/chat_history.py)
------------------------------------------------------------------------------
With DATABASE_URL set, Chainlit saves every chat per user and lists them in its
sidebar. Each chat remembers its case in the thread metadata ("case_id",
`_remember_case`). Reopening a chat calls `resume()`: it restores the user and
the case, and shows a pending approval card or question again. Chainlit itself
only lets a user reopen their own chats.

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
from chainlit.data import get_data_layer
from chainlit.types import ThreadDict

from supplyguard_ai.core.context import SupplyGuardContext
from supplyguard_ai.graph.builder import build_incident_graph
from supplyguard_ai.persistence.chat_history import ensure_chat_history_schema
from supplyguard_ai.services.container import AppServices
from supplyguard_ai.ui import presenters as P
from supplyguard_ai.ui.auth import authenticate
from supplyguard_ai.ui.followup_agent import answer
from supplyguard_ai.ui.runner import CaseRunner, Event

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- process-wide objects
@lru_cache(maxsize=1)
def services() -> AppServices:
    svc = AppServices.from_settings()
    if svc.pool is not None:
        # Chat history tables (chat.*) for Chainlit's data layer: they must exist before
        # Chainlit saves a user or a chat (see app_startup below).
        ensure_chat_history_schema(svc.pool)
    return svc


@cl.on_app_startup
async def app_startup():
    """Build the services when the server starts, which also creates the chat history
    tables - before any login or reopened chat needs them. If the database is not up yet,
    log it and try again on the first login (services() is only cached once it succeeds)."""
    try:
        await asyncio.to_thread(services)
    except Exception:  # noqa: BLE001
        logger.exception("Services not ready at start-up; retrying on first use")


@lru_cache(maxsize=1)
def runner() -> CaseRunner:
    svc = services()
    return CaseRunner(svc, build_incident_graph(checkpointer=svc.make_checkpointer()))


def _user():
    # Our domain User. NOT the session key "user": Chainlit keeps its own
    # cl.User (from the login callback) there and relies on it internally.
    return cl.user_session.get("sg_user")


async def _remember_case(case_id: str) -> None:
    """Make this chat work on `case_id`, and save it in the chat's metadata so that
    reopening the chat from the history sidebar brings the case back (see resume()).
    The chat is also named after its case: otherwise a chat opened with the
    "Open case" button was listed in the sidebar as "open_case"."""
    cl.user_session.set("case_id", case_id)
    data_layer = get_data_layer()
    if data_layer is None:  # no DATABASE_URL: chat history is off
        return
    try:
        cases = await asyncio.to_thread(services().cases.visible_to, _user(), 50)
        title = next((c["title"] for c in cases if c["case_id"] == case_id), "")
        name = f"{case_id} · {title}" if title else case_id
        await data_layer.update_thread(thread_id=cl.context.session.thread_id, name=name[:100],
                                       metadata={"case_id": case_id})
    except Exception:  # noqa: BLE001 - history is a convenience; never break the case for it
        logger.exception("Could not save the case id in the chat history")


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
        await cl.Message(f"{ev.data['message']}").send()


async def _send_approval_card(value: dict) -> cl.Message:
    """Show the exact proposal. Only users whose role must still sign get the buttons;
    everyone else sees who the request is waiting for. Returns the sent message."""
    await _clear_card_buttons()  # only ONE live approval card per chat
    signed = value.get("signed_by", {})
    missing = [r for r in value.get("required_roles", []) if r not in signed]
    if _user().role.value not in missing:
        return await cl.Message(P.approval(value) + "\n\nWaiting for approval by: **"
                                + (", ".join(missing) or "nobody") + "**.").send()

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
    return await cl.Message(P.approval(value), actions=actions).send()


async def _clear_card_buttons():
    """Remove all buttons of the previous approval card (stale cards must not be clickable).
    The graph refuses a stale approval_id anyway - this just keeps the UI honest."""
    for a in cl.user_session.get("card_actions") or []:
        if isinstance(a, cl.Action):  # skip leftovers restored from a saved chat (None)
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


def _init_session():
    """Put our domain User in the session (from Chainlit's cl.User of the login)."""
    app_user = cl.user_session.get("user")  # Chainlit's cl.User from the login callback
    user = services().identity.authenticate(app_user.identifier)
    cl.user_session.set("sg_user", user)
    cl.user_session.set("case_id", None)
    cl.user_session.set("waiting", None)
    # Chainlit saves the session in the chat's metadata and restores it when the chat is
    # reopened. Buttons can't be saved, so old "card_actions" come back as [None, ...].
    cl.user_session.set("card_actions", [])
    return user


@cl.on_chat_start
async def start():
    user = _init_session()
    cases = await asyncio.to_thread(services().cases.visible_to, user)
    actions = [cl.Action(name="open_case", payload={"case_id": c["case_id"]}, label=f"Open {c['case_id']} ({c['phase']})")
               for c in cases[:5]]
    await cl.Message(P.welcome(user, cases), actions=actions).send()


@cl.on_chat_resume
async def resume(thread: ThreadDict):
    """A chat reopened from the history sidebar (Chainlit has checked that it belongs to
    this user). We restore the user and the chat's case; if the case is waiting for an
    approval or an answer, it is shown again with fresh buttons (the old card's buttons
    are not live any more)."""
    _init_session()
    case_id = (thread.get("metadata") or {}).get("case_id")
    if not case_id:
        return  # a chat that never opened a case: nothing to restore
    if not await asyncio.to_thread(services().cases.can_access, _user(), case_id):
        return  # access may have changed since the chat was saved: restore nothing
    await _remember_case(case_id)  # also renames chats saved before they were named after their case
    pending = await asyncio.to_thread(runner().pending, case_id)
    if not pending:
        # Nothing to act on: the old messages already show everything. (Sending a
        # notice here added one more saved message to the chat on every reopen.)
        return
    cl.user_session.set("waiting", "approval" if pending["type"] == "approval" else "clarification")
    # The "still waiting" messages sent by the previous reopen of this chat (if any).
    previous = (thread.get("metadata") or {}).get(RESUME_MESSAGES_KEY) or []
    # Chainlit calls this hook BEFORE it sends the saved messages to the browser, and that
    # replaces whatever we sent here (found in the UI test: the fresh card and its buttons
    # vanished). So send them from a task that runs after the history has been shown.
    asyncio.create_task(_show_pending(case_id, pending, previous))


# Thread-metadata key: ids of the "still waiting" line + card sent by the last reopen.
RESUME_MESSAGES_KEY = "resume_message_ids"


async def _show_pending(case_id: str, pending: dict, previous: list[str]) -> None:
    """Show the step a reopened case is waiting for, REPLACING the copy sent by the
    previous reopen: Chainlit saves every message, so without this each reopen added
    two more messages to the chat for good."""
    await asyncio.sleep(1)  # let Chainlit finish replaying the chat history first
    try:
        data_layer = get_data_layer()
        for message_id in previous:
            # Remove it from the screen ...
            await cl.Message(content="", id=message_id).remove()
            # ... and from chat."Step". remove() also deletes it, but in a fire-and-forget
            # task that did not always run (found in the UI test), so we wait for it here.
            if data_layer is not None:
                await data_layer.delete_step(message_id)
        # Neutral wording: the user reopening the chat may not be one of the approvers.
        sent = [await cl.Message(f"Case `{case_id}` is still waiting for "
                                 f"{'an approval decision' if pending['type'] == 'approval' else 'an answer'}:").send()]
        if pending["type"] == "approval":
            sent.append(await _send_approval_card({"case_id": case_id, **pending}))
        else:
            sent.append(await cl.Message(P.clarification(pending)).send())
        ids = [m.id for m in sent]
        # Save the new ids in BOTH places: Chainlit restored the session from the chat's
        # metadata (with the OLD ids) and writes the session back when the browser leaves
        # the chat - which overwrote our new ids with the old ones (found in the UI test).
        cl.user_session.set(RESUME_MESSAGES_KEY, ids)
        if data_layer is not None:
            await data_layer.update_thread(thread_id=cl.context.session.thread_id,
                                           metadata={RESUME_MESSAGES_KEY: ids})
    except Exception:  # noqa: BLE001 - a background task: log instead of failing silently
        logger.exception("Could not show the pending step of a reopened chat")


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
            reply = (f"I couldn't answer right now ({type(exc).__name__}). The language model or a data "
                     "service is unreachable - please ask again in a moment. Your case is unchanged.")
        await cl.Message(reply or "I found no answer to that - please rephrase the question.").send()
        return

    try:
        new_id, gen = await asyncio.to_thread(runner().start, user, text)  # a new incident case
    except PermissionError as exc:  # e.g. a viewer: read-only, no case is created
        await cl.Message(f"{exc}. You can't open incident cases.").send()
        return
    await _remember_case(new_id)
    await cl.Message(f"Opened case `{new_id}`. Investigating - this takes about a minute.").send()
    await _stream(lambda: gen)


# --------------------------------------------------------------------------- buttons
async def _allowed(case_id: str) -> bool:
    """Button payloads come from the browser and can be tampered with: every
    case id is checked against the registry's access rule on the SERVER."""
    if await asyncio.to_thread(services().cases.can_access, _user(), case_id):
        return True
    await cl.Message(f"You don't have access to case `{case_id}`.").send()
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
    await _remember_case(case_id)
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
    await _remember_case(case_id)
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
