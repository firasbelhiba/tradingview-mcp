"""What an audit of the hosted server found. Offline.

Every test here pins something measured against the live deployment rather
than read from the code: a URL-only-protected endpoint that let any caller
create without limit, a copy-trading tool that misread traders whose
positions OKX half-hides, a default answer of 95-150 KB, and an upstream
that throttled the server's address on two calls in five.
"""
import json
import urllib.error

import pytest

from tradingview_mcp.core import limits
from tradingview_mcp.core.services import alert_service
from tradingview_mcp.core.services import alert_store
from tradingview_mcp.core.services import bitcoin_market_service as pulse
from tradingview_mcp.core.services import bubbles_service as bubbles
from tradingview_mcp.core.services import copytrading_service as copysvc
from tradingview_mcp.core.services import copytrading_trades_service as trades
from tradingview_mcp.core.services import paper_copy
from tradingview_mcp.core.services import paper_orders as orders
from tradingview_mcp.core.services import paper_store as store
from tradingview_mcp.core.services import paper_trading_service as svc


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.delenv("TRADINGVIEW_MCP_PAPER_DB", raising=False)
    monkeypatch.setenv("TRADINGVIEW_MCP_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("TRADINGVIEW_MCP_ALERTS_DB", str(tmp_path / "alerts.sqlite3"))
    store._initialised.clear()
    alert_store._initialised.clear()
    store.select_wallet(None)
    copysvc._cache.clear()
    bubbles._cache.clear()
    pulse._cache.clear()
    yield
    store.select_wallet(None)
    store._initialised.clear()
    alert_store._initialised.clear()


def _prices(monkeypatch, price=100.0):
    for mod in (svc.px, orders.px, paper_copy.px):
        monkeypatch.setattr(mod, "mark_price", lambda s, m: (price, "test"))
        monkeypatch.setattr(mod, "bars_since",
                            lambda s, ms, m, now_ms=None: ([], "1m"))
    monkeypatch.setattr(svc.px, "funding_cost", lambda *a, **k: (0.0, 0))


# ── 1. A caller cannot create without end ─────────────────────────────────────

def test_wallets_stop_at_the_limit(monkeypatch):
    """A wallet is a file, so an unbounded loop over names fills the disk."""
    _prices(monkeypatch)
    monkeypatch.setattr(limits, "MAX_WALLETS", 2)
    assert "wallet" in svc.reset_wallet(100.0, wallet="one")
    assert "wallet" in svc.reset_wallet(100.0, wallet="two")
    out = svc.reset_wallet(100.0, wallet="three")
    assert out["error"]["code"] == "LIMIT_REACHED"
    assert out["error"]["retryable"] is False
    assert not store.db_path("three").exists(), "a refused wallet leaves no file"


def test_an_existing_wallet_can_still_be_reset_at_the_limit(monkeypatch):
    """The cap is on making new ones, not on using the ones there are."""
    _prices(monkeypatch)
    monkeypatch.setattr(limits, "MAX_WALLETS", 1)
    svc.reset_wallet(100.0, wallet="one")
    assert "wallet" in svc.reset_wallet(250.0, wallet="one")


def test_open_positions_stop_at_the_limit(monkeypatch):
    _prices(monkeypatch)
    monkeypatch.setattr(limits, "MAX_OPEN_POSITIONS", 1)
    svc.reset_wallet(10000.0)
    assert "opened" in svc.open_trade("BTCUSDT", "long", 100.0)
    out = svc.open_trade("ETHUSDT", "long", 100.0)
    assert out["error"]["code"] == "LIMIT_REACHED"
    assert svc.wallet_status()["open_count"] == 1


def test_resting_orders_stop_at_the_limit(monkeypatch):
    _prices(monkeypatch)
    monkeypatch.setattr(limits, "MAX_PENDING_ORDERS", 1)
    svc.reset_wallet(10000.0)
    assert "placed" in orders.place_order("BTCUSDT", "long", 100.0, 90.0)
    out = orders.place_order("BTCUSDT", "long", 100.0, 80.0)
    assert out["error"]["code"] == "LIMIT_REACHED"
    assert store.get_wallet()["cash"] == pytest.approx(9900.0), (
        "a refused order must not hold margin"
    )


def test_alerts_stop_at_the_limit(monkeypatch):
    monkeypatch.setattr(limits, "MAX_ALERTS", 1)
    assert "id" in alert_service.create_alert("BTCUSDT", "price", "above", 1.0)
    out = alert_service.create_alert("BTCUSDT", "price", "above", 2.0)
    assert out["error"]["code"] == "LIMIT_REACHED"
    assert "alert_manage" in out["error"]["message"], "say how to make room"


# ── 1. The call budget ────────────────────────────────────────────────────────

def test_the_budget_allows_up_to_its_rate_then_says_how_long_to_wait():
    budget = limits.CallBudget(3)
    assert [budget.take(now=100.0 + i) for i in range(3)] == [None, None, None]
    wait = budget.take(now=103.0)
    assert wait == pytest.approx(57.0), "until the oldest call leaves the window"


def test_the_budget_frees_up_as_the_minute_passes():
    budget = limits.CallBudget(1)
    assert budget.take(now=0.0) is None
    assert budget.take(now=30.0) is not None
    assert budget.take(now=60.0) is None


def test_a_budget_of_zero_never_refuses():
    """Off by default: a stdio install has one user and nothing to protect."""
    budget = limits.CallBudget(0)
    assert all(budget.take(now=1.0) is None for _ in range(1000))


def test_nothing_is_wrapped_unless_the_deployment_asks(monkeypatch):
    from tradingview_mcp import server

    monkeypatch.delenv("TRADINGVIEW_MCP_RATE_LIMIT_PER_MIN", raising=False)
    monkeypatch.delenv("TRADINGVIEW_MCP_AUDIT_LOG", raising=False)
    assert server._guard_tools() == 0


def test_a_hosted_deployment_refuses_past_its_budget_and_logs_each_call(
        monkeypatch, capsys):
    """End to end on a tool that touches no network."""
    import asyncio

    from tradingview_mcp import server

    registry = server.mcp._tool_manager.list_tools()
    saved = {t.name: t.fn for t in registry}
    monkeypatch.setenv("TRADINGVIEW_MCP_RATE_LIMIT_PER_MIN", "1")
    monkeypatch.setenv("TRADINGVIEW_MCP_AUDIT_LOG", "1")
    try:
        assert server._guard_tools() == len(registry)
        tool = next(t for t in registry if t.name == "alert_metrics")
        first = asyncio.run(tool.fn())
        second = asyncio.run(tool.fn())
    finally:
        for t in registry:
            t.fn = saved[t.name]
    assert "error" not in first
    assert second["error"]["code"] == "RATE_LIMITED"
    assert second["error"]["retryable"] is True
    assert second["error"]["retry_after_s"] > 0
    log = capsys.readouterr().err
    assert "[audit] tool=alert_metrics" in log and "outcome=ok" in log
    assert "outcome=refused:RATE_LIMITED" in log


def test_the_audit_line_never_carries_arguments(capsys):
    """Tool, duration, outcome. A wallet name or a symbol is nobody's
    business in a log, and the proxy's own log would have held the secret."""
    limits.audit("paper_trade_open", 0.5, "ok")
    line = capsys.readouterr().err.strip()
    assert line == "[audit] tool=paper_trade_open sec=0.50 outcome=ok"


# ── 2. Traders whose instruments OKX blanks ───────────────────────────────────

def _okx(monkeypatch, positions):
    routes = {
        "public-lead-traders": {"code": "0", "data": [{"ranks": [{
            "nickName": "Half", "uniqueCode": "H1", "pnlRatio": "1.0",
            "winRatio": "0.6", "leadDays": "400", "copyTraderNum": "10",
            "maxCopyTraderNum": "300", "aum": "1000", "pnl": "10"}]}]},
        "public-subpositions-history": {"code": "0", "data": []},
        "public-weekly-pnl": {"code": "0", "data": [{"pnl": "10"}]},
        "public-current-subpositions": {"code": "0", "data": positions},
        "public-preference-currency": {"code": "0", "data": []},
        "public-stats": {"code": "0", "data": []},
    }

    def fake_get(path):
        for fragment, payload in routes.items():
            if fragment in path:
                return payload
        raise RuntimeError(f"unrouted {path}")

    monkeypatch.setattr(copysvc, "_get", fake_get)


_BLANK = {"instId": "", "posSide": "short", "lever": "10", "margin": "3856.78",
          "openAvgPx": "", "markPx": "", "openTime": "", "mgnMode": "cross"}
_SHOWN = {"instId": "BTC-USDT-SWAP", "posSide": "long", "lever": "3",
          "margin": "100", "openAvgPx": "100", "markPx": "101",
          "openTime": "1700000000000", "mgnMode": "cross"}


def test_blanked_instruments_are_a_state_of_their_own(monkeypatch):
    """Measured live: 5 of the top 20 traders return positions with side,
    leverage, margin and unrealised PnL but no instrument, price or size.
    They were being presented as ordinary positions."""
    _okx(monkeypatch, [_BLANK, _BLANK])
    out = copysvc.get_lead_trader("H1")
    assert out["open_positions_state"] == "instruments_hidden"
    assert out["open_position_count"] == 2, "the count is real and worth keeping"
    assert out["open_max_leverage"] == 10, "leverage is still readable"
    assert "cannot be mirrored" in out["instrument_hidden_note"]
    assert all(p["instrument"] is None and p["instrument_hidden"]
               for p in out["open_positions"])


def test_a_mixed_book_stays_measured_and_counts_the_blanks(monkeypatch):
    _okx(monkeypatch, [_SHOWN, _BLANK])
    out = copysvc.get_lead_trader("H1")
    assert out["open_positions_state"] == "measured"
    assert out["open_positions_instrument_hidden"] == 1


def test_a_fully_visible_book_carries_no_note(monkeypatch):
    _okx(monkeypatch, [_SHOWN])
    out = copysvc.get_lead_trader("H1")
    assert out["open_positions_state"] == "measured"
    assert out["open_positions_instrument_hidden"] is None
    assert "instrument_hidden_note" not in out


def test_copying_a_blanked_trader_gives_the_real_reason(monkeypatch):
    """It used to say "not a USDT pair this wallet can price", which sent
    the reader looking for an exotic instrument that was never there."""
    _prices(monkeypatch)
    svc.reset_wallet(10000.0)
    _okx(monkeypatch, [_BLANK])
    out = paper_copy.copy_trader("H1")
    assert out["error"]["code"] == "NO_DATA"
    assert "blanks which instruments" in out["error"]["message"]
    assert out["error"]["retryable"] is False


def test_copying_a_mixed_book_skips_only_the_blanked_position(monkeypatch):
    _prices(monkeypatch)
    svc.reset_wallet(10000.0)
    _okx(monkeypatch, [_SHOWN, _BLANK])
    out = paper_copy.copy_trader("H1")
    assert [r["symbol"] for r in out["plan"]] == ["BTCUSDT"]
    assert out["skipped"] == [{
        "instrument": None,
        "reason": "instrument hidden by OKX for this position",
    }]


# ── 3. A default answer that fits in a conversation ───────────────────────────

def _raw_trade(i):
    return {"subPosId": str(i), "pnl": "100", "pnlRatio": "0.1", "lever": "2E+1",
            "instId": "BTC-USDT-SWAP", "posSide": "long",
            "openTime": "1700000000000", "closeTime": "1700003600000",
            "margin": "1000", "subPos": "400", "openAvgPx": "70000",
            "closeAvgPx": "70700", "ccy": "USDT", "mgnMode": "cross"}


def _serve_trades(monkeypatch, n):
    page = [_raw_trade(i) for i in range(n)]
    monkeypatch.setattr(
        trades, "_get",
        lambda path: {"code": "0", "data": [] if "after=" in path else page})


def test_rows_are_capped_while_statistics_cover_everything(monkeypatch):
    """300 rows measured 95-150 KB in one answer. The figures still have to
    describe all of them, or the cap would quietly change the conclusion."""
    _serve_trades(monkeypatch, 90)
    out = trades.get_lead_trader_trades("C", max_trades=300)
    assert out["trades_shown"] == 40
    assert out["trades_omitted"] == 50
    assert out["statistics"]["trades"] == 90
    assert "statistics cover all of them" in out["trades_note"]


def test_the_cap_can_be_raised_when_the_rows_are_the_point(monkeypatch):
    _serve_trades(monkeypatch, 90)
    out = trades.get_lead_trader_trades("C", max_trades=300, trade_rows=90)
    assert out["trades_shown"] == 90
    assert "trades_omitted" not in out


def test_the_default_answer_is_a_fraction_of_the_old_one(monkeypatch):
    _serve_trades(monkeypatch, 100)
    capped = len(json.dumps(trades.get_lead_trader_trades("C")))
    full = len(json.dumps(trades.get_lead_trader_trades("C", trade_rows=100)))
    assert capped < full * 0.55, f"{capped} vs {full}"


# ── 4. An upstream that throttles the server's address ────────────────────────

class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return json.dumps(self._payload).encode()


def _throttle(monkeypatch, failures, status=429):
    """urlopen that fails `failures` times before answering."""
    state = {"calls": 0, "slept": []}

    def fake(req, timeout=None):
        state["calls"] += 1
        if state["calls"] <= failures:
            raise urllib.error.HTTPError("u", status, "x", {}, None)
        return _Resp([{"symbol": "btc"}])

    monkeypatch.setattr(bubbles.urllib.request, "urlopen", fake)
    monkeypatch.setattr(bubbles, "_sleep", lambda s: state["slept"].append(s))
    return state


def test_a_throttled_request_is_waited_out(monkeypatch):
    """From the hosted server CoinGecko answered 429 on 2 calls in 5, each
    straight after another CoinGecko-backed tool, and worked seconds later."""
    state = _throttle(monkeypatch, failures=1)
    assert bubbles._get_with_retry("https://example.invalid/m") == [{"symbol": "btc"}]
    assert state["slept"] == [2.5]


def test_it_gives_up_after_its_retries_rather_than_hanging(monkeypatch):
    state = _throttle(monkeypatch, failures=99)
    with pytest.raises(urllib.error.HTTPError):
        bubbles._get_with_retry("https://example.invalid/m")
    assert state["slept"] == [2.5, 6.0]
    assert state["calls"] == 3


def test_only_a_throttle_is_retried(monkeypatch):
    """A 500 is a different problem, and waiting will not fix it."""
    state = _throttle(monkeypatch, failures=99, status=500)
    with pytest.raises(urllib.error.HTTPError):
        bubbles._get_with_retry("https://example.invalid/m")
    assert state["slept"] == []
    assert state["calls"] == 1


def test_the_bitcoin_pulse_asks_once_a_minute_not_once_a_call(monkeypatch):
    """Two requests on every pulse spent the allowance fast enough that the
    next CoinGecko-backed tool was the one refused."""
    calls = []

    def fake(req, timeout=None):
        calls.append(req.full_url)
        return _Resp({"data": {}})

    monkeypatch.setattr(pulse.urllib.request, "urlopen", fake)
    pulse._http_get_json("https://example.invalid/a")
    pulse._http_get_json("https://example.invalid/a")
    pulse._http_get_json("https://example.invalid/b")
    assert len(calls) == 2, "the repeat of a is served from memory"


# ── 4b. The retry was not enough, measured after it shipped ───────────────────
#
# Five boards of different sizes back to back still failed three times on the
# hosted server: each size was its own cache entry, so its own request.

def _coin(rank, symbol=None):
    row = {"market_cap_rank": rank, "symbol": symbol or f"c{rank}",
           "name": f"Coin {rank}", "current_price": 1.0,
           "market_cap": 1_000_000 - rank, "total_volume": 10.0, "image": None}
    for field in bubbles.WINDOWS.values():
        row[field] = 1.0
    return row


def _coingecko(monkeypatch, refuse_markets=lambda n: False, stable_fails=False):
    """A CoinGecko that counts what it is asked. -> the counters."""
    seen = {"markets": 0, "stables": 0, "slept": []}
    page = [_coin(i) for i in range(1, 251)]

    def fake(req, timeout=None):
        if "category=stablecoins" in req.full_url:
            seen["stables"] += 1
            if stable_fails:
                raise urllib.error.HTTPError("u", 429, "x", {}, None)
            return _Resp([{"symbol": "usdt"}])
        seen["markets"] += 1
        if refuse_markets(seen["markets"]):
            raise urllib.error.HTTPError("u", 429, "x", {}, None)
        return _Resp(page)

    monkeypatch.setattr(bubbles.urllib.request, "urlopen", fake)
    monkeypatch.setattr(bubbles, "_sleep", lambda s: seen["slept"].append(s))
    return seen


def test_boards_of_every_size_share_one_request(monkeypatch):
    seen = _coingecko(monkeypatch)
    for limit in (10, 30, 50, 11, 12):
        assert bubbles.crypto_bubbles(limit)["count"] == limit
    assert seen["markets"] == 1, "CoinGecko counts requests, not rows"
    assert seen["stables"] == 1


def test_a_board_of_a_hundred_is_a_hundred(monkeypatch):
    """Capping the over-fetch at 100 meant the stablecoins came out of the
    hundred and nothing replaced them."""
    seen = _coingecko(monkeypatch)
    page_with_stables = [_coin(i, "usdt" if i % 10 == 0 else None)
                         for i in range(1, 251)]
    monkeypatch.setattr(bubbles.urllib.request, "urlopen",
                        lambda req, timeout=None: _Resp(
                            [{"symbol": "usdt"}]
                            if "category=stablecoins" in req.full_url
                            else page_with_stables))
    out = bubbles.crypto_bubbles(100)
    assert out["count"] == 100
    assert seen["slept"] == []


def test_a_refused_refresh_shows_the_older_board_and_says_how_old(monkeypatch):
    seen = _coingecko(monkeypatch, refuse_markets=lambda n: n > 1)
    fresh = bubbles.crypto_bubbles(10)
    assert fresh["stale"] is None and fresh["data_age_s"] == 0

    # Age the page past its minute without waiting one.
    with bubbles._cache_lock:
        at, rows = bubbles._cache[bubbles._MARKETS_KEY]
        bubbles._cache[bubbles._MARKETS_KEY] = (at - 200, rows)

    out = bubbles.crypto_bubbles(10)
    assert "error" not in out, "an older board beats no board"
    assert out["count"] == 10
    assert 199 <= out["data_age_s"] <= 202
    assert "refused a refresh" in out["stale"]
    assert seen["slept"] == [], "nothing to wait for when there is a fallback"


def test_a_board_too_old_to_trust_is_not_shown(monkeypatch):
    """Fifteen minutes is the limit: past that a mover has moved."""
    _coingecko(monkeypatch, refuse_markets=lambda n: n > 1)
    bubbles.crypto_bubbles(10)
    with bubbles._cache_lock:
        at, rows = bubbles._cache[bubbles._MARKETS_KEY]
        bubbles._cache[bubbles._MARKETS_KEY] = (at - 1000, rows)
    out = bubbles.crypto_bubbles(10)
    assert out["error"]["code"] == "UPSTREAM_ERROR"
    assert out["error"]["retryable"] is True


def test_a_failed_stablecoin_lookup_is_not_repeated_on_the_next_call(monkeypatch):
    """Asking again every time spent the allowance the board itself needed."""
    seen = _coingecko(monkeypatch, stable_fails=True)
    for _ in range(4):
        assert bubbles.crypto_bubbles(10)["count"] == 10
    assert seen["stables"] == 1


# ── 2b. Found while verifying: the trades tool repeated OKX's wording ─────────

def test_a_hidden_history_is_not_reported_as_a_trader_who_does_not_exist(
        monkeypatch):
    """OKX answers 59269 "This futures lead trader doesn't exist" for a
    trader the profile tool can read. It was passed on as a retryable
    upstream error, so the reader retried something that cannot change."""
    def refuse(path):
        raise RuntimeError(
            "OKX error 59269: This futures lead trader doesn't exist")

    monkeypatch.setattr(trades, "_get", refuse)
    out = trades.get_lead_trader_trades("C")
    assert out["error"]["code"] == "NO_DATA"
    assert out["error"]["retryable"] is False
    assert out["error"]["trade_history_state"] == "hidden"
    assert "hidden" in out["error"]["message"]
    assert "doesn't exist" not in out["error"]["message"]


def test_any_other_okx_failure_is_still_worth_retrying(monkeypatch):
    def refuse(path):
        raise RuntimeError("OKX error 50011: Too Many Requests")

    monkeypatch.setattr(trades, "_get", refuse)
    out = trades.get_lead_trader_trades("C")
    assert out["error"]["code"] == "UPSTREAM_ERROR"
    assert out["error"]["retryable"] is True
