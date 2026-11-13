"""Why a request was refused. Offline."""
import asyncio

from tradingview_mcp.core import http_log


def _run(status, answer, body=b'{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"x","arguments":{"secret":"hunter2"}}}',
         headers=((b"Accept", b"application/json"), (b"Authorization", b"Bearer abc"),
                  (b"mcp-session-id", b"s" * 32), (b"Cookie", b"a=b"))):
    async def app(scope, receive, send):
        await receive()
        await send({"type": "http.response.start", "status": status, "headers": []})
        await send({"type": "http.response.body", "body": answer})

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message):
        pass

    scope = {"type": "http", "method": "POST", "path": "/mcp", "headers": list(headers)}
    asyncio.run(http_log.RefusalLog(app)(scope, receive, send))


def test_a_refusal_is_explained_in_one_line(capsys):
    _run(400, b'{"error":{"message":"Bad Request: Missing session ID"}}')
    line = capsys.readouterr().err.strip()
    assert line.startswith("[http] status=400 verb=POST rpc=tools/call session_header=yes")
    assert "Missing session ID" in line
    assert "\n" not in line


def test_a_request_that_was_served_is_not_logged_at_all(capsys):
    _run(200, b'{"result":{}}')
    assert capsys.readouterr().err == ""


def test_nothing_private_is_ever_written(capsys):
    """Not the body, not an argument, not a cookie, not a credential, and not
    the session id itself - only whether one was sent."""
    _run(404, b"Session not found")
    line = capsys.readouterr().err
    for secret in ("hunter2", "Bearer", "abc", "a=b", "s" * 32, "secret"):
        assert secret not in line, secret


def test_a_body_that_is_not_json_rpc_is_called_unreadable(capsys):
    _run(400, b"Parse error", body=b"not json at all")
    assert "rpc=unreadable" in capsys.readouterr().err


def test_logging_can_never_fail_the_request(monkeypatch, capsys):
    monkeypatch.setattr(http_log.limits, "note", lambda line: 1 / 0)
    _run(400, b"x")          # must not raise
