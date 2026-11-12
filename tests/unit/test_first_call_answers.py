"""What the first call actually returned. Offline.

Two models were given 35 everyday questions and the tool list. Both chose the
right tool every time, so the descriptions were left alone. Running the calls
they chose is what found something: a made-up trader code described as a
trader who hides, a scan that matched nothing answering with nothing at all,
and a refusal that named only the destructive way forward.
"""
import asyncio

import pytest

from tradingview_mcp.core.services import copytrading_service as copysvc
from tradingview_mcp.core.services import copytrading_trades_service as trades
from tradingview_mcp.core.services import paper_copy
from tradingview_mcp.core.services import paper_store as store
from tradingview_mcp.core.services import paper_trading_service as svc

_GONE = "OKX error 59269: This futures lead trader doesn't exist"


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.delenv("TRADINGVIEW_MCP_PAPER_DB", raising=False)
    monkeypatch.setenv("TRADINGVIEW_MCP_DATA_DIR", str(tmp_path))
    store._initialised.clear()
    store.select_wallet(None)
    copysvc._cache.clear()
    yield
    store.select_wallet(None)
    store._initialised.clear()


def _okx(monkeypatch, hides=(), on_board=()):
    """An OKX with one leaderboard and a set of traders who hide.

    Anyone in neither set does not exist, and is refused everywhere with the
    same error OKX uses for a trader who hides - which is the whole problem.
    """
    board = [{"nickName": f"Trader {c}", "uniqueCode": c, "pnlRatio": "1.0",
              "winRatio": "0.6", "leadDays": "400", "copyTraderNum": "10",
              "maxCopyTraderNum": "300", "aum": "1000", "pnl": "10"}
             for c in on_board]

    def fake(path):
        if "public-lead-traders" in path:
            return {"code": "0", "data": [{"ranks": board}]}
        code = path.split("uniqueCode=")[1].split("&")[0]
        if code in hides and "public-stats" in path:
            return {"code": "0", "data": [{"winRatio": "0.6"}]}
        if code in on_board and code not in hides:
            return {"code": "0", "data": []}
        raise RuntimeError(_GONE)

    monkeypatch.setattr(copysvc, "_get", fake)
    monkeypatch.setattr(trades, "_get", fake)


# ── A typo is not a trader who hides ──────────────────────────────────────────

def test_a_code_nobody_has_is_reported_as_unknown_not_hidden(monkeypatch):
    """Measured live: the profile of a made-up code came back as a trader
    with a null name and 'position-level data is hidden'."""
    _okx(monkeypatch)
    out = copysvc.get_lead_trader("3A1B2C4D5E6F7890")
    assert out["error"]["code"] == "INVALID_PARAMETER"
    assert "no lead trader with the code" in out["error"]["message"]
    assert out["error"]["retryable"] is False
    assert out["error"]["next_step"] == "lead_traders", "say where codes come from"


def test_a_trader_who_hides_is_still_reported_as_hidden(monkeypatch):
    """The fix must not turn a private trader into a missing one."""
    _okx(monkeypatch, hides={"HIDES"})
    out = copysvc.get_lead_trader("HIDES")
    assert "error" not in out
    assert out["trade_history_state"] == "hidden"
    assert out["okx_published_stats"] is not None


def test_a_failed_lookup_concludes_nothing(monkeypatch):
    """If OKX cannot be asked, 'unknown code' would be a guess."""
    def flaky(path):
        if "public-lead-traders" in path:
            return {"code": "0", "data": [{"ranks": []}]}
        if "public-stats" in path:
            raise RuntimeError("timed out")
        raise RuntimeError(_GONE)

    monkeypatch.setattr(copysvc, "_get", flaky)
    assert copysvc.trader_is_known("X", "SWAP") is None
    out = copysvc.get_lead_trader("X")
    assert "error" not in out, "not enough to call the code wrong"


def test_the_trades_tool_makes_the_same_distinction(monkeypatch):
    _okx(monkeypatch, hides={"HIDES"})
    unknown = trades.get_lead_trader_trades("TYPO")
    hidden = trades.get_lead_trader_trades("HIDES")
    assert unknown["error"]["code"] == "INVALID_PARAMETER"
    assert "typo" in unknown["error"]["message"]
    assert hidden["error"]["code"] == "NO_DATA"
    assert hidden["error"]["trade_history_state"] == "hidden"


def test_copying_a_code_nobody_has_says_so(monkeypatch):
    """It used to answer that the positions 'could not be read'."""
    for mod in (svc.px, paper_copy.px):
        monkeypatch.setattr(mod, "mark_price", lambda s, m: (100.0, "test"))
    svc.reset_wallet(1000.0)
    _okx(monkeypatch)
    out = paper_copy.copy_trader("3A1B2C4D5E6F7890")
    assert out["error"]["code"] == "INVALID_PARAMETER"
    assert "no lead trader" in out["error"]["message"]


# ── A scan that matches nothing still answers ─────────────────────────────────

def _tool(name):
    from tradingview_mcp import server

    return next(t for t in server.mcp._tool_manager.list_tools() if t.name == name)


def test_six_screeners_are_covered_and_only_those():
    """Every tool that can hand back a bare list, and none that cannot."""
    from tradingview_mcp import server

    assert server._EXPLAINED_TOOL_COUNT == 6
    assert set(server._EMPTY_ADVICE) == {
        "top_gainers", "top_losers", "bollinger_scan", "rating_filter",
        "volume_breakout_scanner", "smart_volume_scanner",
    }


def test_an_empty_scan_says_what_was_asked_and_what_to_loosen(monkeypatch):
    """'Any coins breaking out right now?' was answered with no content at
    all, leaving the assistant to guess between none, broken and wrong."""
    async def nothing(*args, **kwargs):
        return []

    result = asyncio.run(_call_with_stub(
        _tool("volume_breakout_scanner"), nothing,
        exchange="BINANCE", timeframe="15m"))
    assert result["matches"] == 0 and result["rows"] == []
    assert "not a failure" in result["meaning"]
    assert result["asked"]["exchange"] == "BINANCE"
    assert result["asked"]["volume_multiplier"] == 2.0, "defaults are part of what was asked"
    assert "volume_multiplier" in result["to_widen"]


def _empty_wrapper(tool):
    """The layer that explains an empty result, wherever it sits in the stack."""
    wrapper = tool.fn
    while "__sig" not in (wrapper.__kwdefaults__ or {}):
        wrapper = wrapper.__wrapped__
    return wrapper


async def _call_with_stub(tool, stub, **kwargs):
    """Call a tool's empty-result wrapper with its inner function replaced."""
    wrapper = _empty_wrapper(tool)
    saved = wrapper.__kwdefaults__["__inner"]
    wrapper.__kwdefaults__["__inner"] = stub
    try:
        return await wrapper(**kwargs)
    finally:
        wrapper.__kwdefaults__["__inner"] = saved


def test_rows_are_passed_through_untouched(monkeypatch):
    rows = [{"symbol": "BINANCE:BTCUSDT", "changePercent": 2.0}]

    async def found(*args, **kwargs):
        return rows

    out = asyncio.run(_call_with_stub(_tool("top_gainers"), found, exchange="BINANCE"))
    assert out is rows


def test_an_error_is_passed_through_untouched():
    envelope = {"error": {"code": "UPSTREAM_ERROR", "message": "x"}}

    async def failed(*args, **kwargs):
        return envelope

    out = asyncio.run(_call_with_stub(_tool("bollinger_scan"), failed))
    assert out is envelope


def test_an_empty_scan_survives_the_real_call_path(monkeypatch):
    """Through FastMCP itself, because these six carry an output schema and
    an object where a list was promised could have been rejected there."""
    from tradingview_mcp import server

    wrapper = _empty_wrapper(_tool("rating_filter"))

    async def nothing(*args, **kwargs):
        return []

    saved = wrapper.__kwdefaults__["__inner"]
    wrapper.__kwdefaults__["__inner"] = nothing
    try:
        out = asyncio.run(server.mcp.call_tool("rating_filter", {"rating": 3}))
    finally:
        wrapper.__kwdefaults__["__inner"] = saved
    content = out[0] if isinstance(out, tuple) else out
    text = "".join(getattr(c, "text", "") for c in content)
    assert '"matches": 0' in text, "the client must receive something to read"
    assert "neighbouring rating" in text


# ── The refusal names the way out that destroys nothing ───────────────────────

def test_refusing_to_wipe_a_wallet_offers_a_separate_one(monkeypatch):
    """Asked to 'create a wallet' when one exists, the only path named was
    wiping it. The person usually wanted a second wallet."""
    for mod in (svc.px,):
        monkeypatch.setattr(mod, "mark_price", lambda s, m: (100.0, "test"))
        monkeypatch.setattr(mod, "bars_since",
                            lambda s, ms, m, now_ms=None: ([], "1m"))
    monkeypatch.setattr(svc.px, "funding_cost", lambda *a, **k: (0.0, 0))
    svc.reset_wallet(1000.0)
    svc.open_trade("BTCUSDT", "long", 100.0)
    out = svc.reset_wallet(5000.0)
    assert out["error"]["code"] == "INVALID_PARAMETER"
    assert "wallet='<a name>'" in out["error"]["message"]
    assert "nothing here is touched" in out["error"]["message"]
    assert out["error"]["or_instead"].startswith("wallet=")
    assert svc.wallet_status()["open_count"] == 1, "and nothing was wiped"
