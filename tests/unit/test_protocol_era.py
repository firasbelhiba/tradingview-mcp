"""A client that speaks a newer protocol than the server. Offline.

Read off the hosted server's log: Claude's connector opened with
server/discover and MCP-Protocol-Version 2026-07-28, was refused by the SDK
with a reply the specification does not define, and in 4 of 60 cases did
not fall back - which a person saw as a tool that failed.
"""
import asyncio
import json

import pytest

from tradingview_mcp import sdk
from tradingview_mcp.core import protocol_era as era

SUPPORTED = ["2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25"]
DISCOVER = json.dumps({"jsonrpc": "2.0", "id": 41, "method": "server/discover",
                       "params": {}}).encode()


def _send(version, body=DISCOVER, method="POST"):
    """-> (status, headers, body) from the gate, or 'passed' if it reached the app."""
    out = {"reached": False, "status": None, "headers": {}, "body": b""}

    async def app(scope, receive, send):
        out["reached"] = True

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message):
        if message["type"] == "http.response.start":
            out["status"] = message["status"]
            out["headers"] = dict(message["headers"])
        else:
            out["body"] += message.get("body", b"")

    headers = [(b"MCP-Protocol-Version", version.encode())] if version else []
    scope = {"type": "http", "method": method, "path": "/mcp", "headers": headers}
    asyncio.run(era.NewerProtocolGate(app, SUPPORTED)(scope, receive, send))
    return out


@pytest.mark.parametrize("version", ["2025-11-25", "2025-06-18", "2024-11-05", ""])
def test_a_version_the_server_speaks_and_a_request_naming_none_pass_through(version, monkeypatch):
    monkeypatch.delenv("TRADINGVIEW_MCP_NEWER_PROTOCOL_REPLY", raising=False)
    assert _send(version)["reached"] is True


def test_a_newer_version_gets_a_bare_400_by_default(monkeypatch):
    """'If the body is empty ... fall back to initialize.'"""
    monkeypatch.delenv("TRADINGVIEW_MCP_NEWER_PROTOCOL_REPLY", raising=False)
    out = _send("2026-07-28")
    assert out["reached"] is False
    assert out["status"] == 400 and out["body"] == b""
    assert out["headers"][b"content-length"] == b"0"


def test_the_modern_reply_is_the_error_the_specification_defines(monkeypatch):
    monkeypatch.setenv("TRADINGVIEW_MCP_NEWER_PROTOCOL_REPLY", "modern")
    out = _send("2026-07-28")
    assert out["status"] == 400
    reply = json.loads(out["body"])
    assert reply["id"] == 41, "the request's own id, not a made-up one"
    assert reply["error"]["code"] == -32022
    assert reply["error"]["data"] == {"supported": SUPPORTED, "requested": "2026-07-28"}
    assert out["headers"][b"content-type"] == b"application/json"


def test_the_sdk_can_be_left_to_answer_as_before(monkeypatch):
    monkeypatch.setenv("TRADINGVIEW_MCP_NEWER_PROTOCOL_REPLY", "sdk")
    assert _send("2026-07-28")["reached"] is True


def test_an_unknown_setting_falls_back_to_the_default(monkeypatch):
    monkeypatch.setenv("TRADINGVIEW_MCP_NEWER_PROTOCOL_REPLY", "whatever")
    assert era.reply_mode() == "empty"


def test_a_body_that_is_not_json_still_gets_an_answer(monkeypatch):
    monkeypatch.setenv("TRADINGVIEW_MCP_NEWER_PROTOCOL_REPLY", "modern")
    out = _send("2026-07-28", body=b"not json")
    assert out["status"] == 400 and json.loads(out["body"])["id"] is None


def test_only_posts_are_judged(monkeypatch):
    monkeypatch.delenv("TRADINGVIEW_MCP_NEWER_PROTOCOL_REPLY", raising=False)
    assert _send("2026-07-28", method="GET")["reached"] is True


def test_the_list_of_supported_versions_comes_from_the_sdk_in_use():
    versions = era.supported_versions()
    if sdk.MAJOR >= 2:
        # The 2.x SDK speaks 2026-07-28 itself and publishes no such list,
        # so the gate has nothing to judge by and lets everything through.
        assert versions == []
    else:
        assert "2025-06-18" in versions and "2026-07-28" not in versions
