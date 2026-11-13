"""Say why a request was refused.

The transport answers a request it will not serve with 400, 404 or 406 and
logs nothing but the status. On the hosted server that left 60 refusals in
a day with no way to tell a forgotten session from a malformed body from a
protocol version the SDK does not speak - and the first explanation reached
for turned out to be only part of it.

For every answer of 400 or above this writes one line: the status, the
JSON-RPC method that was asked, which of the headers the transport judges a
request by were present, and the start of the refusal the server sent back.
It never logs a request body, an argument, a cookie or an authorisation
header, and it never logs a successful request at all.
"""
from __future__ import annotations

import re

from tradingview_mcp.core import limits

_METHOD = re.compile(rb'"method"\s*:\s*"([A-Za-z_/]{1,60})"')
_SEEN = (b"mcp-session-id", b"mcp-protocol-version", b"accept", b"content-type",
         b"content-length", b"user-agent")


def describe(scope: dict, body_start: bytes, status: int, answer: bytes) -> str:
    headers = {k.lower(): v for k, v in scope.get("headers", [])}
    method = _METHOD.search(body_start or b"")
    parts = [
        f"[http] status={status}",
        f"verb={scope.get('method')}",
        f"rpc={method.group(1).decode() if method else 'unreadable'}",
        f"session_header={'yes' if b'mcp-session-id' in headers else 'no'}",
        f"protocol={headers.get(b'mcp-protocol-version', b'-').decode('latin-1')[:20]}",
        f"accept={headers.get(b'accept', b'-').decode('latin-1')[:60]!r}",
        f"content_type={headers.get(b'content-type', b'-').decode('latin-1')[:40]!r}",
        f"body_bytes={headers.get(b'content-length', b'?').decode('latin-1')[:10]}",
        f"client={headers.get(b'user-agent', b'-').decode('latin-1')[:40]!r}",
        f"said={answer[:160].decode('utf-8', 'replace')!r}",
    ]
    return " ".join(parts)


class RefusalLog:
    """ASGI middleware: one line for each request answered with 400 or above."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        seen = {"body": b"", "status": 0, "answer": b""}

        async def watching_receive():
            message = await receive()
            if message.get("type") == "http.request" and len(seen["body"]) < 2048:
                seen["body"] += (message.get("body") or b"")[:2048]
            return message

        async def watching_send(message):
            if message.get("type") == "http.response.start":
                seen["status"] = message.get("status", 0)
            elif message.get("type") == "http.response.body" and seen["status"] >= 400:
                if len(seen["answer"]) < 200:
                    seen["answer"] += (message.get("body") or b"")[:200]
                if not message.get("more_body"):
                    try:
                        limits.note(describe(scope, seen["body"], seen["status"],
                                             seen["answer"]))
                    except Exception:  # noqa: BLE001 - never fail a request to log it
                        pass
            await send(message)

        await self.app(scope, watching_receive, watching_send)
