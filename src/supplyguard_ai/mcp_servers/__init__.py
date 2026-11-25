"""MCP servers run as separate processes by docker compose (procurement-mcp, external-mcp)."""

import os
from pathlib import Path


def bind_host(configured: str | None) -> str:
    """Where an MCP server listens.

    Inside a container it must listen on 0.0.0.0 (other containers connect
    via the compose network; the host only sees 127.0.0.1 because compose
    publishes the port as "127.0.0.1:8101:8101").  Outside Docker we bind to
    127.0.0.1 so a dev laptop never exposes the server on the LAN.
    """
    if configured:
        return configured
    return "0.0.0.0" if Path("/.dockerenv").exists() else "127.0.0.1"


def allowed_hosts(port: int) -> list[str]:
    """Host headers accepted by the DNS-rebinding protection of the MCP SDK:
    local names plus the compose service names."""
    names = ["localhost", "127.0.0.1", "procurement-mcp", "external-mcp", os.environ.get("HOSTNAME", "")]
    return [f"{n}:{port}" for n in names if n] + [n for n in names if n]
