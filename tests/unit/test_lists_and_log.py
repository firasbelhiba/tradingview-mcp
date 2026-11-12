"""Symbol lists as people type them, and a usage log that survives. Offline."""
import asyncio
import os

import pytest

from tradingview_mcp import server
from tradingview_mcp.core import limits


# ── A list of symbols, however it was typed ──────────────────────────────────

@pytest.mark.parametrize("typed, expected", [
    ("BTC,ETH", ["BTC", "ETH"]),
    ("BTC, ETH, SOL", ["BTC", "ETH", "SOL"]),
    ("btc eth  sol", ["btc", "eth", "sol"]),
    ("BTC;ETH", ["BTC", "ETH"]),
    (" BTCUSDT ", ["BTCUSDT"]),
    ("", []),
    (["BTC", "ETH"], ["BTC", "ETH"]),
])
def test_one_string_is_split_and_a_list_is_left_alone(typed, expected):
    assert server._split_symbols(typed) == expected


def _tool(name):
    return next(t for t in server.mcp._tool_manager.list_tools() if t.name == name)


@pytest.mark.parametrize("name", ["crypto_watchlist", "correlation",
                                  "supertrend_scan", "donchian_breakout_scan"])
def test_the_schema_still_promises_a_list_of_strings(name):
    """Accepting a string must not make the schema vaguer for clients that
    read it: an assistant should still be told to pass a list."""
    prop = _tool(name).parameters["properties"]["symbols"]
    shape = prop if "type" in prop else next(
        option for option in prop["anyOf"] if option.get("type") == "array")
    assert shape["type"] == "array" and shape["items"] == {"type": "string"}


def test_a_comma_string_passes_the_tools_own_validation():
    """It was refused with 'Input should be a valid list [type=list_type'.
    Checked against the model FastMCP validates arguments with."""
    tool = _tool("crypto_watchlist")
    args = tool.fn_metadata.arg_model.model_validate({"symbols": "BTC, eth,SOL/USDT"})
    assert args.symbols == ["BTC", "eth", "SOL/USDT"]
    again = tool.fn_metadata.arg_model.model_validate({"symbols": ["BTC"]})
    assert again.symbols == ["BTC"]


def test_an_optional_list_can_still_be_left_out():
    args = _tool("supertrend_scan").fn_metadata.arg_model.model_validate({})
    assert args.symbols is None


# ── A usage log that outlives the container ───────────────────────────────────

def test_nothing_is_written_to_disk_unless_a_file_is_named(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("TRADINGVIEW_MCP_AUDIT_FILE", raising=False)
    monkeypatch.chdir(tmp_path)
    limits.audit("market_brief", 1.25, "ok")
    assert capsys.readouterr().err.strip() == "[audit] tool=market_brief sec=1.25 outcome=ok"
    assert list(tmp_path.iterdir()) == []


def test_a_named_file_gets_the_line_with_a_time_on_it(tmp_path, monkeypatch, capsys):
    """After a redeploy, two lines were left of everything recorded before."""
    log = tmp_path / "audit.log"
    monkeypatch.setenv("TRADINGVIEW_MCP_AUDIT_FILE", str(log))
    limits.audit("paper_trade_open", 0.5, "ok")
    limits.audit("coin_analysis", 2.0, "error:SYMBOL_NOT_FOUND")
    lines = log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    stamp, rest = lines[0].split(" ", 1)
    assert stamp.endswith("Z") and "T" in stamp and len(stamp) == 20
    assert rest == "[audit] tool=paper_trade_open sec=0.50 outcome=ok"
    assert "[audit] tool=paper_trade_open" in capsys.readouterr().err, "stderr still gets it"


def test_the_file_never_carries_arguments(tmp_path, monkeypatch):
    log = tmp_path / "audit.log"
    monkeypatch.setenv("TRADINGVIEW_MCP_AUDIT_FILE", str(log))
    limits.audit("paper_trade_open", 0.5, "ok")
    text = log.read_text(encoding="utf-8")
    assert set(text.split()[1:]) == {"[audit]", "tool=paper_trade_open", "sec=0.50",
                                     "outcome=ok"}


def test_a_full_file_is_moved_aside_not_grown_forever(tmp_path, monkeypatch):
    log = tmp_path / "audit.log"
    monkeypatch.setenv("TRADINGVIEW_MCP_AUDIT_FILE", str(log))
    monkeypatch.setattr(limits, "_MAX_LOG_BYTES", 200)
    for i in range(12):
        limits.audit(f"tool_{i}", 0.1, "ok")
    assert (tmp_path / "audit.log.1").exists()
    assert log.stat().st_size <= 200 + 80, "the live file starts again"
    kept = (tmp_path / "audit.log.1").read_text() + log.read_text()
    assert "tool_11" in kept, "the newest line is never the one dropped"
    assert len(list(tmp_path.iterdir())) == 2, "one spare, not a growing pile"


def test_a_log_that_cannot_be_written_does_not_fail_the_call(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADINGVIEW_MCP_AUDIT_FILE",
                       str(tmp_path / "no-such-folder" / "audit.log"))
    limits.audit("market_brief", 0.1, "ok")      # must not raise


def test_a_tool_call_lands_in_the_file_end_to_end(tmp_path, monkeypatch):
    registry = server.mcp._tool_manager.list_tools()
    saved = {t.name: t.fn for t in registry}
    log = tmp_path / "audit.log"
    monkeypatch.setenv("TRADINGVIEW_MCP_AUDIT_LOG", "1")
    monkeypatch.setenv("TRADINGVIEW_MCP_AUDIT_FILE", str(log))
    monkeypatch.setenv("TRADINGVIEW_MCP_ALERTS_DB", str(tmp_path / "alerts.sqlite3"))
    monkeypatch.delenv("TRADINGVIEW_MCP_RATE_LIMIT_PER_MIN", raising=False)
    try:
        server._guard_tools()
        tool = next(t for t in registry if t.name == "alert_metrics")
        asyncio.run(tool.fn())
    finally:
        for t in registry:
            t.fn = saved[t.name]
    assert "[audit] tool=alert_metrics" in log.read_text(encoding="utf-8")
    assert os.path.getsize(log) < 200
