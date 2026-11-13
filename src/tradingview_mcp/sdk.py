"""The one place that knows which MCP SDK is installed.

This server was written against the 1.x SDK, whose ``FastMCP`` lives in
``mcp.server.fastmcp``. That line stops at protocol 2025-11-25. On
2026-11-13 Claude's connector began opening with the 2026-07-28 revision
(``server/discover``, no handshake, no sessions) and, read off the hosted
server's log, did not fall back when refused - not for the SDK's own
refusal, and not for the UnsupportedProtocolVersionError the specification
defines. To that client an older server is simply down.

The 2.x SDK speaks 2026-07-28 and every earlier revision from one server.
It renamed ``FastMCP`` to ``MCPServer`` and kept the decorator API, so the
tools themselves do not change. What moved is listed here, and nothing
outside this module imports the SDK's server classes directly:

  1.x                                   2.x
  mcp.server.fastmcp.FastMCP            mcp.server.mcpserver.MCPServer
  server._mcp_server                    server._lowlevel_server
  server.get_context()                  gone: a tool is handed a Context
  server.settings.host / .port / ...    arguments to streamable_http_app()

Both lines are supported so that a local install pinned to 1.x keeps
working exactly as it did while the hosted server runs 2.x.
"""
from __future__ import annotations

from typing import Any, Optional

try:  # 2.x
    from mcp.server.mcpserver import Context, MCPServer as FastMCP

    MAJOR = 2
except ModuleNotFoundError:  # 1.x
    from mcp.server.fastmcp import Context, FastMCP

    MAJOR = 1

__all__ = ["Context", "FastMCP", "MAJOR", "lowlevel", "current_request",
           "http_app", "log_level", "field", "result_text", "capabilities"]


def lowlevel(server: Any) -> Any:
    """The low-level server a FastMCP/MCPServer drives."""
    found = getattr(server, "_lowlevel_server", None)
    return found if found is not None else server._mcp_server


def current_request(server: Any) -> Optional[Any]:
    """The HTTP request carrying the call being handled, where the SDK can say.

    1.x exposes it through ``get_context()``. 2.x has no such accessor, and
    answers None: there the request's space is carried by a context variable
    instead (see core/spaces.py).
    """
    getter = getattr(server, "get_context", None)
    if getter is None:
        return None
    try:
        return getter().request_context.request
    except Exception:  # noqa: BLE001 - outside a request, or not HTTP
        return None


def http_app(server: Any, *, host: str, stateless: bool, security: Any) -> Any:
    """The Starlette app for streamable-http, configured the way each line wants."""
    if MAJOR >= 2:
        return server.streamable_http_app(
            stateless_http=stateless, transport_security=security, host=host,
        )
    server.settings.host = host
    server.settings.stateless_http = stateless
    server.settings.transport_security = security
    return server.streamable_http_app()


def log_level(server: Any) -> str:
    return str(getattr(getattr(server, "settings", None), "log_level", "INFO")).lower()


def field(obj: Any, wire_name: str) -> Any:
    """Read a protocol field by the name it has on the wire.

    1.x models use the wire's camelCase as the attribute (``inputSchema``);
    2.x models use snake_case (``input_schema``) and keep camelCase for the
    wire only. Code that inspects a tool or a result asks for the wire name
    and gets the right attribute on either.
    """
    if hasattr(obj, wire_name):
        return getattr(obj, wire_name)
    import re

    return getattr(obj, re.sub(r"(?<!^)(?=[A-Z])", "_", wire_name).lower())


def result_text(result: Any) -> str:
    """The text a client is shown for a tool result, whatever call_tool returned.

    1.x returns a list of content blocks, or a (blocks, structured) pair;
    2.x returns a CallToolResult.
    """
    if isinstance(result, tuple):
        result = result[0]
    blocks = getattr(result, "content", result)
    return "".join(getattr(block, "text", "") or "" for block in blocks)


def capabilities(server: Any) -> dict:
    """The capabilities this server advertises, as a client receives them."""
    low = lowlevel(server)
    try:
        caps = low.get_capabilities()
    except TypeError:  # 1.x wants its notification options and experimental map
        from mcp.server.lowlevel import NotificationOptions

        caps = low.get_capabilities(NotificationOptions(), {})
    return caps.model_dump(by_alias=True, exclude_none=True)
