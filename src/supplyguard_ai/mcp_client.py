"""
MCP client side: how the app and the agents consume the MCP servers

TWO CONSUMERS
-------------
1. AGENTS get MCP tools as normal LangChain tools via
   `langchain_mcp_adapters.MultiServerMCPClient.get_tools()`.
2. The WORKFLOW's execute node talks to procurement through a
   `ProcurementGateway` - in-process for tests / local dev, MCP in docker
   compose.  Same interface, so the node never knows which one it uses.

THE SYNC/ASYNC BRIDGE (why `_Loop` exists)
------------------------------------------
MCP clients are async (`await session.call_tool(...)`); the adapter builds
tools with only a coroutine.  Our agents and graph nodes run synchronously
(`agent.invoke`), and calling `asyncio.run()` inside a thread that may
already run an event loop (Chainlit!) raises "event loop is running".
So one daemon thread owns ONE long-lived event loop; sync code submits
coroutines to it with `run_coroutine_threadsafe(...).result()`.  Each MCP
call opens a short streamable-HTTP session (the servers are stateless).
"""

from __future__ import annotations

import asyncio
import threading
from functools import lru_cache
from typing import Any, Protocol

from langchain_core.tools import BaseTool, StructuredTool

from supplyguard_ai.config import Settings, get_settings
from supplyguard_ai.domain.models import ExecutionResult

CALL_TIMEOUT_S = 60


# --------------------------------------------------------------------------- async bridge
class _Loop:
    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, name="mcp-loop", daemon=True).start()

    def run(self, coro, timeout: float = CALL_TIMEOUT_S):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)


@lru_cache(maxsize=1)
def _loop() -> _Loop:
    return _Loop()


def run_sync(coro, timeout: float = CALL_TIMEOUT_S):
    """Run a coroutine from sync code, whatever thread / loop we are in."""
    return _loop().run(coro, timeout)


# --------------------------------------------------------------------------- client
def connections(s: Settings | None = None) -> dict[str, dict]:
    s = s or get_settings()
    conns = {}
    if s.procurement_mcp_url:
        conns["procurement"] = {"url": s.procurement_mcp_url, "transport": "streamable_http"}
    if s.external_mcp_url:
        conns["external"] = {"url": s.external_mcp_url, "transport": "streamable_http"}
    return conns


def make_client(conns: dict[str, dict] | None = None):
    from langchain_mcp_adapters.client import MultiServerMCPClient

    return MultiServerMCPClient(conns if conns is not None else connections())


def as_sync_tool(tool: BaseTool) -> StructuredTool:
    """Give an async-only MCP tool a sync entry point (keeps the async one too)."""
    def _call(**kwargs: Any):
        # Call the adapter's coroutine directly: it returns (content, artifact),
        # exactly what response_format="content_and_artifact" expects.
        # (tool.ainvoke(...) would already strip the artifact.)
        return run_sync(tool.coroutine(**kwargs))

    return StructuredTool(
        name=tool.name, description=tool.description, args_schema=tool.args_schema,
        func=_call, coroutine=tool.coroutine, response_format=getattr(tool, "response_format", "content"),
        metadata={**(tool.metadata or {}), "mcp": True},
    )


def load_mcp_tools(server: str, names: set[str] | None = None, conns: dict[str, dict] | None = None) -> list[StructuredTool]:
    """LangChain tools from one MCP server, optionally only `names` (least privilege:
    agents get READ tools only - never the procurement write tools)."""
    client = make_client(conns)
    tools = run_sync(client.get_tools(server_name=server))
    return [as_sync_tool(t) for t in tools if names is None or t.name in names]


# --------------------------------------------------------------------------- procurement gateway
class ProcurementGateway(Protocol):
    def execute(self, case_id: str, step: str, approval_id: str, kind: str, payload: dict, actor: str) -> ExecutionResult: ...


class InProcessGateway:
    """Calls the procurement service directly (unit tests, local dev without MCP)."""

    def __init__(self, procurement) -> None:
        self.procurement = procurement

    def execute(self, case_id, step, approval_id, kind, payload, actor):
        return self.procurement.execute_approved(case_id, step, approval_id, kind, payload, actor)


class McpProcurementGateway:
    """Calls the procurement MCP server's write tool (docker compose).

    The tool name IS the operation kind (create_purchase_order, ...), and the
    payload fields are the tool arguments - so the server rebuilds exactly the
    payload the human approved and checks it against app.approvals.
    """

    def __init__(self, url: str) -> None:
        self.url = url

    async def _call(self, name: str, args: dict) -> dict:
        import json

        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client

        async with streamable_http_client(self.url) as (read, write, _), ClientSession(read, write) as session:
            await session.initialize()
            res = await session.call_tool(name, args)
        if res.isError:
            raise RuntimeError(f"MCP tool {name} failed: {res.content}")
        if res.structuredContent is not None:
            data = res.structuredContent
            return data.get("result", data) if isinstance(data, dict) else data
        return json.loads(res.content[0].text)

    def execute(self, case_id, step, approval_id, kind, payload, actor):
        data = run_sync(self._call(kind, {"case_id": case_id, "step": step, "approval_id": approval_id,
                                          "actor": actor, **payload}))
        return ExecutionResult.model_validate(data)


# --------------------------------------------------------------------------- external risk feed
class RiskFeed(Protocol):
    def alerts(self, region: str) -> list[dict]: ...


class FileRiskFeed:
    """Offline twin of the external service: reads the same risk_feed.json the
    provided external_info_service.py serves (unit tests, local dev)."""

    def __init__(self, path) -> None:
        import json

        self._alerts = json.loads(path.read_text(encoding="utf-8"))["alerts"]

    def alerts(self, region: str) -> list[dict]:
        return [dict(a) for a in self._alerts if a["region"] == region]


class McpRiskFeed:
    """The external information service through the external MCP server (docker compose)."""

    def __init__(self, url: str) -> None:
        self.url = url

    async def _call(self, region: str) -> list[dict]:
        import json

        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client

        async with streamable_http_client(self.url) as (read, write, _), ClientSession(read, write) as session:
            await session.initialize()
            res = await session.call_tool("get_risk_alerts", {"region": region})
        if res.isError:
            raise RuntimeError(f"get_risk_alerts failed: {res.content}")
        data = res.structuredContent if res.structuredContent is not None else json.loads(res.content[0].text)
        return data.get("result", data)["alerts"]

    def alerts(self, region: str) -> list[dict]:
        return run_sync(self._call(region), timeout=15)
