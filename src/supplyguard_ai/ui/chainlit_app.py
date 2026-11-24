"""Chainlit chat interface for the SupplyGuard AI incident graph.

The graph is loaded from supplyguard_ai.graph.runtime. That module documents the contract
between this UI and the graph (input, config, pause payloads, resume values).

Two kinds of pause can come back from the graph:
  * "clarify"  -> show the question; the user's NEXT message answers it.
  * "approval" -> show the exact proposal; users whose role must still approve get
                  Approve / Reject buttons, everyone else sees who it is waiting for.
"""

import logging

import chainlit as cl
from langchain_core.messages import AIMessageChunk, HumanMessage
from langgraph.types import Command

from supplyguard_ai.auth.users import authenticate
from supplyguard_ai.graph.runtime import get_graph, pending_interrupt

logger = logging.getLogger(__name__)

# GRAPH TEAM: nodes whose LLM output is streamed to the screen word by word.
# Other nodes may call the LLM too (extraction, critic...), but that output is internal.
STREAM_NODES = {"respond"}

# How long the Approve / Reject buttons wait for a click before the request just stays
# pending (it is shown again the next time the user opens the app).
APPROVAL_WAIT_SECONDS = 86400


@cl.password_auth_callback
async def auth(username: str, password: str):
    """Login: the user's name (e.g. Maria Keller) + personal password, checked against the
    `users` table. None = rejected."""
    try:
        account = await authenticate(username, password)
    except Exception:
        # A database problem must never let anyone in.
        logger.exception("Login check failed")
        return None
    if account is None:
        return None
    # Role, scopes and factories come from the database, never from what the user types.
    return cl.User(
        identifier=account["user_id"],
        display_name=account["name"],
        metadata={
            "role": account["role"],
            "scopes": account["scopes"],
            "factories": account["factories"],
        },
    )


def _user() -> dict:
    """The signed-in user, in the shape the graph expects (see graph/runtime.py)."""
    user = cl.user_session.get("user")
    return {
        "user_id": user.identifier,
        "name": user.display_name,
        "role": user.metadata["role"],
        "scopes": user.metadata["scopes"],
        "factories": user.metadata["factories"],
    }


def _config() -> dict:
    """The graph config: thread_id is the key to the saved conversation.

    For now there is one conversation per user. GRAPH TEAM: when cases exist, use the
    case's thread here instead, so a finance controller can open and co-sign a case that a
    procurement manager started.
    """
    user = _user()
    return {"configurable": {"thread_id": f"user-{user['user_id']}", "user": user}}


@cl.on_chat_start
async def start():
    """Runs on every page load, refresh and reconnect: look for unfinished work first."""
    cl.user_session.set("awaiting_clarify", False)
    config = _config()
    graph = await get_graph()

    # Is this user's saved conversation waiting for a clarification or an approval?
    if await pending_interrupt(graph, config):
        await cl.Message(content="Welcome back. You have an unfinished incident.").send()
        # Input None = "continue from the saved checkpoint". The paused node asks again,
        # and _drive() shows the question or the approval request.
        await _drive(None, config)
        return

    user = _user()
    await cl.Message(content=(
        f"Hi {user['name']}! I'm SupplyGuard AI. Describe the supplier disruption: "
        "the supplier, the component and the announced delay."
    )).send()


def _eur(value) -> str:
    return "n/a" if value is None else f"EUR {value:,.0f}"


def _describe(proposal: dict) -> str:
    """Turn the approval proposal into the text shown next to the buttons."""
    lines = [f"**Approval request {proposal['approval_id']}: plan {proposal['plan_id']}**",
             proposal.get("title", ""), ""]
    for i, action in enumerate(proposal.get("actions", []), start=1):
        lines.append(f"{i}. {action['description']} ({_eur(action.get('cost_eur'))})")
    lines += [
        "",
        f"- Total additional spend: **{_eur(proposal.get('total_cost_eur'))}** "
        f"(cap {_eur(proposal.get('cap_eur'))})",
    ]
    if proposal.get("revenue_at_risk_eur") is not None:
        lines.append(f"- Revenue at risk after this plan: {_eur(proposal['revenue_at_risk_eur'])} "
                     "(separate from cost)")
    if proposal.get("uncovered_units") is not None:
        lines.append(f"- Units still uncovered: {proposal['uncovered_units']:,}")
    lines.append(f"- Required approvers: {', '.join(proposal.get('required_roles', [])) or 'none'}")
    if proposal.get("signed_roles"):
        lines.append(f"- Already approved by: {', '.join(proposal['signed_roles'])}")
    for cite in proposal.get("evidence", []):
        lines.append(f"- Evidence: {cite['source_id']} {cite.get('title', '')}")
    lines += ["", "No purchase order exists until this request is approved. Any change to "
                  "supplier, quantity or price needs a new approval."]
    return "\n".join(lines)


async def _run(graph, graph_input, config):
    """Like graph.ainvoke, but shows the answer's words on screen as the LLM writes them.

    Returns (result, streamed):
      * result   - the final state, plus "__interrupt__" when the graph paused
      * streamed - the Chainlit message that received the words, or None if nothing was streamed
    """
    streamed = None
    result = {}
    async for mode, data in graph.astream(
        graph_input, config, stream_mode=["messages", "updates", "values"]
    ):
        if mode == "messages":
            chunk, metadata = data
            # AIMessageChunk = a piece of text coming from the LLM right now.
            if (
                isinstance(chunk, AIMessageChunk)
                and metadata.get("langgraph_node") in STREAM_NODES
                and isinstance(chunk.content, str)
                and chunk.content
            ):
                if streamed is None:
                    streamed = cl.Message(content="")
                await streamed.stream_token(chunk.content)
        elif mode == "values":
            result = dict(data)  # the latest full state
        elif mode == "updates" and "__interrupt__" in data:
            result["__interrupt__"] = data["__interrupt__"]  # the graph paused
    return result, streamed


async def _drive(graph_input, config):
    """Run the graph, then handle every pause until it finishes or needs the user."""
    graph = await get_graph()
    result, streamed = await _run(graph, graph_input, config)
    user = _user()

    # While the graph is paused, "__interrupt__" is present in the result.
    while "__interrupt__" in result:
        pause = result["__interrupt__"][0].value

        if pause["type"] == "clarify":
            # Show the question and stop. The user's next message resumes the graph.
            cl.user_session.set("awaiting_clarify", True)
            await cl.Message(content=pause["question"]).send()
            return

        if pause["type"] == "approval":
            proposal = pause["proposal"]
            missing = [r for r in proposal.get("required_roles", [])
                       if r not in proposal.get("signed_roles", [])]
            if user["role"] not in missing:
                # This user cannot sign: show the request and who it is waiting for.
                await cl.Message(content=_describe(proposal) + "\n\nWaiting for approval by: **"
                                 + (", ".join(missing) or "nobody") + "**.").send()
                return
            choice = await cl.AskActionMessage(
                content=_describe(proposal),
                actions=[
                    cl.Action(name="approve", payload={}, label="Approve"),
                    cl.Action(name="reject", payload={}, label="Reject"),
                ],
                timeout=APPROVAL_WAIT_SECONDS,
            ).send()
            if not choice:
                # No click before the timeout: nothing is decided, the request stays pending.
                await cl.Message(content="The approval request is still pending.").send()
                return
            # The decision is bound to the exact proposal the user saw (approval_id + hash).
            decision = {
                "decision": "approved" if choice.get("name") == "approve" else "rejected",
                "approval_id": proposal["approval_id"],
                "payload_hash": proposal["payload_hash"],
                "user_id": user["user_id"],
                "role": user["role"],
            }
            result, streamed = await _run(graph, Command(resume=decision), config)
            continue

        logger.error("Unknown pause type from the graph: %s", pause.get("type"))
        await cl.Message(content="The assistant paused in an unexpected way. Please try again.").send()
        return

    # No more pauses: the last message is the assistant's final reply.
    final = str(result["messages"][-1].content)
    if streamed:
        # The words are already on screen; replace them with the final text.
        streamed.content = final
        await streamed.send()
    else:
        await cl.Message(content=final).send()


@cl.on_message
async def on_message(message: cl.Message):
    """Route each user message: an answer to a clarification, or a new message."""
    config = _config()
    try:
        if cl.user_session.get("awaiting_clarify"):
            # The graph is paused on a question: this message is the answer.
            cl.user_session.set("awaiting_clarify", False)
            await _drive(Command(resume=message.content), config)
        else:
            await _drive({"messages": [HumanMessage(content=message.content)], "user": _user()}, config)
    except Exception:
        logger.exception("Error while processing chat message")
        await cl.Message(content="Sorry, something went wrong. Please try again.").send()
