"""Answering a client that speaks a newer protocol than this server does.

The 2026-07-28 revision of MCP dropped the ``initialize`` handshake and
sessions: a client opens with ``server/discover`` and the header
``MCP-Protocol-Version: 2026-07-28``. The 1.x SDK this server is built on
stops at 2025-11-25, and the 2.x line that speaks the new revision removed
the API the whole server is written against.

A client meeting an older server is meant to fall back to ``initialize``.
The specification says how it decides (Streamable HTTP, "Backward
Compatibility"): on ``400 Bad Request`` it inspects the body, and "if the
body is empty or is not a recognized modern JSON-RPC error, fall back to
initialize". The SDK's own refusal is neither: it is a JSON-RPC error, with
a code no revision defines for this and an id of "server-error" that matches
no request. Measured on the hosted server, Claude's connector fell back
after it 56 times in 60 and gave up the other 4, each of which a person saw
as a tool that failed.

This answers such a request before the SDK sees it, in one of the two ways
the specification names, chosen by ``TRADINGVIEW_MCP_NEWER_PROTOCOL_REPLY``:

  empty    400 with no body: the plainest "I am an older server"
  modern   400 with UnsupportedProtocolVersionError (-32022), the request's
           own id, and ``data.supported`` listing what this server speaks
  sdk      leave it to the SDK, as before

Which one a given client handles best is an empirical question, so the
choice is a setting and every such exchange is logged.
"""
from __future__ import annotations

import json
import os
from typing import List, Optional

HEADER = b"mcp-protocol-version"
UNSUPPORTED_PROTOCOL_VERSION = -32022
_MAX_BODY = 64 * 1024
DEFAULT_REPLY = "empty"


def supported_versions() -> List[str]:
    try:
        from mcp.shared.version import SUPPORTED_PROTOCOL_VERSIONS

        return list(SUPPORTED_PROTOCOL_VERSIONS)
    except Exception:  # noqa: BLE001
        return []


def reply_mode() -> str:
    mode = os.environ.get("TRADINGVIEW_MCP_NEWER_PROTOCOL_REPLY", "").strip().lower()
    return mode if mode in ("empty", "modern", "sdk") else DEFAULT_REPLY


def refusal(mode: str, requested: str, request_id, supported: List[str]) -> bytes:
    """The body sent back for a protocol version this server does not speak."""
    if mode != "modern":
        return b""
    return json.dumps({
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {
            "code": UNSUPPORTED_PROTOCOL_VERSION,
            "message": "Unsupported protocol version",
            "data": {"supported": supported, "requested": requested},
        },
    }).encode()


def _request_id(body: bytes):
    try:
        parsed = json.loads(body)
    except ValueError:
        return None
    return parsed.get("id") if isinstance(parsed, dict) else None


class NewerProtocolGate:
    """ASGI middleware: answer a too-new protocol version in a way that is
    unambiguous to the client, instead of passing it to an SDK that is not."""

    def __init__(self, app, supported: Optional[List[str]] = None):
        self.app = app
        self.supported = supported if supported is not None else supported_versions()

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http" or scope.get("method") != "POST":
            await self.app(scope, receive, send)
            return
        headers = {k.lower(): v for k, v in scope.get("headers", [])}
        requested = headers.get(HEADER, b"").decode("latin-1").strip()
        mode = reply_mode()
        if not requested or requested in self.supported or mode == "sdk" \
                or not self.supported:
            await self.app(scope, receive, send)
            return

        body = b""
        while len(body) < _MAX_BODY:
            message = await receive()
            if message.get("type") != "http.request":
                break
            body += message.get("body") or b""
            if not message.get("more_body"):
                break
        answer = refusal(mode, requested, _request_id(body), self.supported)
        response_headers = [(b"content-length", str(len(answer)).encode())]
        if answer:
            response_headers.append((b"content-type", b"application/json"))
        await send({"type": "http.response.start", "status": 400,
                    "headers": response_headers})
        await send({"type": "http.response.body", "body": answer})
