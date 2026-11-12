"""MCP Apps: the three wire-format facts a host needs to draw a panel.

An interactive view is not framework magic. A host that supports the
``io.modelcontextprotocol/ui`` extension looks for:

  1. ``_meta.ui.resourceUri`` on a tool in tools/list,
  2. a resource at that ``ui://`` uri whose mimeType is exactly
     ``text/html;profile=mcp-app`` (with an optional ``_meta.ui.csp``
     allow-list, read from the resources/read content item first and the
     resources/list entry second),
  3. a page that speaks the bridge protocol to its parent frame.

This server's first attempt did none of the three: it embedded an HTML
resource in the tool result, and the client printed it as text. The
capability advertisement and the general shape are taken from the upstream
project's ``apps_shim`` (branch feat/price-chart-app), which needed to
patch request handlers; the SDK version locked here takes ``meta`` on
``tool()`` and ``resource()`` directly, so only the capability remains.
"""
from __future__ import annotations

import inspect
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

UI_EXTENSION = "io.modelcontextprotocol/ui"
APP_MIME = "text/html;profile=mcp-app"


def supports_meta() -> bool:
    """Whether the installed SDK can carry ``_meta`` on tools and resources.

    The dependency floor is older than the feature. An install on the floor
    keeps every tool and simply draws nothing, rather than failing to start.
    """
    return (
        "meta" in inspect.signature(FastMCP.tool).parameters
        and "meta" in inspect.signature(FastMCP.resource).parameters
    )


def tool_kwargs(resource_uri: str) -> dict:
    """Keyword arguments that point a tool at its view. Empty if unsupported."""
    if not supports_meta():
        return {}
    return {"meta": {"ui": {"resourceUri": resource_uri}}}


def advertise(mcp: FastMCP) -> None:
    """Add the ui extension to the capabilities sent at initialize.

    ``extensions`` is not a declared field on this SDK's ServerCapabilities;
    it rides as a pydantic extra, so it is read with getattr and assigned
    plainly.
    """
    server = mcp._mcp_server
    original = server.get_capabilities
    if getattr(original, "_advertises_ui", False):
        return

    def get_capabilities(*args: Any, **kwargs: Any):
        caps = original(*args, **kwargs)
        existing = getattr(caps, "extensions", None) or {}
        caps.extensions = {**existing, UI_EXTENSION: {}}
        return caps

    get_capabilities._advertises_ui = True  # type: ignore[attr-defined]
    server.get_capabilities = get_capabilities  # type: ignore[method-assign]


def client_name(ctx: Any) -> Optional[str]:
    """The connecting software's own name for itself, for the audit line.

    It names a program, not a person: every user of one app sends the
    same string.
    """
    try:
        info = ctx.session.client_params.clientInfo
        return f"{info.name}/{info.version}"[:60]
    except Exception:  # noqa: BLE001
        return None


def client_supports_ui(ctx: Any) -> Optional[bool]:
    """Whether the connected client said it can draw views.

    True or False when the client's capabilities could be read, None when
    they could not - outside a request, or on a transport that keeps no
    session. None is not "no": the caller should then behave as it would
    for a client that cannot draw, because that answer is right for both.
    """
    try:
        caps = ctx.session.client_params.capabilities
    except Exception:  # noqa: BLE001
        return None
    if caps is None:
        return None
    extensions = getattr(caps, "extensions", None)
    if extensions is None:
        extensions = (getattr(caps, "model_extra", None) or {}).get("extensions")
    if extensions is None:
        try:
            extensions = caps.model_dump().get("extensions")
        except Exception:  # noqa: BLE001
            extensions = None
    if not isinstance(extensions, dict):
        return False
    return UI_EXTENSION in extensions
