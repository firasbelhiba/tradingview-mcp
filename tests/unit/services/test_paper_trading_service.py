"""Unit tests for the simulated wallet. Offline.

The settlement rules get the most attention, because they are what makes a
practice wallet teach anything: a stop that price recovered from must still
count, an ambiguous bar must resolve against the trade, and costs must be
charged. A simulator that gets these wrong flatters every strategy.
"""
import pytest

from tradingview_mcp.core.services import paper_pricing as px
from tradingview_mcp.core.services import paper_store as store
from tradingview_mcp.core.services import paper_trading_service as svc


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADINGVIEW_MCP_PAPER_DB", str(tmp_path / "paper.sqlite3"))
    store._initialised.clear()
    yield
    store._initialised.clear()


def _bar(high, low, open_ms=0):
    return {"high": high, "low": low, "open": (high + low) / 2,
            "close": (high + low) / 2, "open_ms": open_ms}


def _patch_price(monkeypatch, price=100.0, source="test"):
    monkeypatch.setattr(svc.px, "mark_price", lambda s, m: (price, source))


def _patch_bars(monkeypatch, bars, interval="1m"):
    monkeypatch.setattr(svc.px, "bars_since", lambda s, ms, m, now_ms=None: (bars, interval))


def _patch_funding(monkeypatch, cost=0.0, payments=0):
    monkeypatch.setattr(svc.px, "funding_cost",
                        lambda *a, **k: (cost, payments))


# ── Pricing primitives ────────────────────────────────────────────────────────

def test_liquidation_price_is_where_loss_equals_margin():
    # 5x long from 100: a 20% adverse move wipes the margin.
    assert px.liquidation_price("long", 100.0, 5.0) == pytest.approx(80.0)
    assert px.liquidation_price("short", 100.0, 5.0) == pytest.approx(120.0)
    assert px.liquidation_price("long", 100.0, 1.0) == pytest.approx(0.0)


def test_liquidation_price_is_none_without_inputs():
    assert px.liquidation_price("long", 0, 5.0) is None
    assert px.liquidation_price("long", 100.0, 0) is None


def test_position_pnl_direction():
    assert px.position_pnl("long", 100.0, 110.0, 2.0) == pytest.approx(20.0)
    assert px.position_pnl("long", 100.0, 90.0, 2.0) == pytest.approx(-20.0)
    assert px.position_pnl("short", 100.0, 90.0, 2.0) == pytest.approx(20.0)
    assert px.position_pnl("short", 100.0, 110.0, 2.0) == pytest.approx(-20.0)


@pytest.mark.parametrize("span_ms,expected", [
    (60_000, "1m"),
    (60_000 * 1500, "1m"),
    (60_000 * 1501, "5m"),
    (3_600_000 * 24 * 3, "5m"),
    (3_600_000 * 24 * 30, "1h"),
    (3_600_000 * 24 * 3000, "1d"),
])
def test_interval_picked_covers_the_span(span_ms, expected):
    assert px.pick_interval(span_ms)[0] == expected


def test_every_interval_the_settler_picks_is_a_real_kline_interval():
    """The two maps have to agree, and they silently did not.

    pick_interval chose "1m" for a position opened seconds ago, but "1m" was
    missing from binance_service's kline map, so the settler's very first
    call failed for every new position until it aged past 25 hours — and the
    spot price fallback, which also reads 1m klines, was broken with it. The
    unit tests could not see it because they mock bars_since; only a live run
    did. This pins the relationship instead of the symptom.
    """
    from tradingview_mcp.core.services import binance_service as bn

    for name, _step in px._INTERVAL_SPAN_MS:
        assert bn.kline_interval_for(name) is not None, (
            f"pick_interval can return {name!r} but get_klines rejects it"
        )


def test_symbol_normalisation():
    assert px.normalise_symbol("kucoin:btc-usdt") == "BTCUSDT"
    assert px.normalise_symbol(" eth/usdt ") == "ETHUSDT"


# ── Settlement rules ──────────────────────────────────────────────────────────

def test_stop_fills_even_though_price_recovered():
    """The rule that makes the simulator honest.

    Price dipped through the stop and came back. A simulator that only
    compares the CURRENT price would show this trade still open and winning.
    """
    bars = [_bar(105, 104), _bar(104, 89), _bar(110, 108)]  # middle bar hit 89
    hit = px.scan_bars_for_exit(bars, "long", stop_loss=90.0,
                                take_profit=None, liquidation=None)
    assert hit is not None
    assert hit["reason"] == "stop_loss"
    assert hit["price"] == 90.0


def test_target_fills_for_a_short_on_a_low():
    bars = [_bar(101, 100), _bar(100, 88)]
    hit = px.scan_bars_for_exit(bars, "short", stop_loss=None,
                                take_profit=90.0, liquidation=None)
    assert hit["reason"] == "take_profit"


def test_stop_wins_when_one_bar_holds_both_levels():
    """The bar does not record order, so the unfavourable level is taken."""
    bars = [_bar(120, 80)]  # spans both a 90 stop and a 110 target
    hit = px.scan_bars_for_exit(bars, "long", stop_loss=90.0,
                                take_profit=110.0, liquidation=None)
    assert hit["reason"] == "stop_loss"
    assert hit["ambiguous"] is True


def test_liquidation_outranks_the_stop():
    bars = [_bar(120, 70)]
    hit = px.scan_bars_for_exit(bars, "long", stop_loss=90.0,
                                take_profit=110.0, liquidation=80.0)
    assert hit["reason"] == "liquidated"
    assert hit["ambiguous"] is True


def test_earliest_bar_wins():
    bars = [_bar(105, 104, open_ms=1), _bar(100, 89, open_ms=2),
            _bar(130, 129, open_ms=3)]
    hit = px.scan_bars_for_exit(bars, "long", stop_loss=90.0,
                                take_profit=125.0, liquidation=None)
    assert hit["reason"] == "stop_loss"
    assert hit["bar"]["open_ms"] == 2


def test_no_level_reached_returns_none():
    bars = [_bar(105, 95), _bar(106, 96)]
    assert px.scan_bars_for_exit(bars, "long", 90.0, 110.0, None) is None


def test_bars_without_high_or_low_are_skipped():
    bars = [{"high": None, "low": None}, _bar(100, 89)]
    hit = px.scan_bars_for_exit(bars, "long", 90.0, None, None)
    assert hit is not None


# ── Wallet lifecycle ──────────────────────────────────────────────────────────

def test_reset_creates_the_wallet():
    out = svc.reset_wallet(5000.0, "usdt", 0.04, 10.0, note="test")
    assert out["wallet"]["starting_balance"] == 5000.0
    assert out["wallet"]["currency"] == "USDT"
    assert out["wallet"]["cash"] == 5000.0
    assert "SIMULATED" in out["simulation"]


def test_reset_wipes_positions_and_counts_them(monkeypatch):
    svc.reset_wallet(10000.0)
    _patch_price(monkeypatch, 100.0)
    _patch_bars(monkeypatch, [])
    svc.open_trade("BTCUSDT", "long", 1000.0, "futures", 2.0)
    out = svc.reset_wallet(10000.0)
    assert out["positions_discarded"] == 1
    assert out["wallet"]["reset_count"] == 1
    assert store.list_positions() == []


@pytest.mark.parametrize("kwargs", [
    {"balance": 1.0}, {"balance": 10 ** 10}, {"fee_pct": -1},
    {"fee_pct": 2}, {"max_leverage": 0}, {"max_leverage": 500},
])
def test_reset_rejects_bad_params(kwargs):
    assert svc.reset_wallet(**kwargs)["error"]["code"] == "INVALID_PARAMETER"


def test_tools_refuse_to_act_without_a_wallet():
    for call in (lambda: svc.open_trade("BTCUSDT", "long", 100.0),
                 lambda: svc.close_trade(1),
                 lambda: svc.wallet_status(),
                 lambda: svc.trade_history()):
        assert call()["error"]["code"] == "NO_DATA"


# ── Opening ───────────────────────────────────────────────────────────────────

def test_open_charges_margin_and_fee(monkeypatch):
    svc.reset_wallet(10000.0, "USDT", 0.05, 20.0)
    _patch_price(monkeypatch, 50000.0)
    _patch_bars(monkeypatch, [])
    out = svc.open_trade("BTCUSDT", "long", 2000.0, "futures", 5.0,
                         stop_loss=48000.0, take_profit=55000.0)
    opened = out["opened"]
    assert opened["margin"] == 400.0            # 2000 / 5
    assert opened["entry_fee"] == 1.0           # 2000 * 0.05%
    assert opened["quantity"] == pytest.approx(0.04)
    assert opened["liquidation_price"] == pytest.approx(40000.0)
    assert out["cash_remaining"] == 9599.0      # 10000 - 400 - 1


def test_open_reports_risk_and_reward(monkeypatch):
    svc.reset_wallet(10000.0, "USDT", 0.0, 20.0)
    _patch_price(monkeypatch, 100.0)
    _patch_bars(monkeypatch, [])
    out = svc.open_trade("BTCUSDT", "long", 1000.0, "futures", 1.0,
                         stop_loss=95.0, take_profit=110.0)
    risk = out["risk"]
    assert risk["loss_at_stop"] == pytest.approx(50.0)      # 5% of 1000
    assert risk["pct_of_wallet"] == pytest.approx(0.5)
    assert risk["reward_at_target"] == pytest.approx(100.0)
    assert risk["risk_reward"] == pytest.approx(2.0)


def test_open_without_a_stop_says_so(monkeypatch):
    svc.reset_wallet(10000.0)
    _patch_price(monkeypatch, 100.0)
    _patch_bars(monkeypatch, [])
    out = svc.open_trade("BTCUSDT", "long", 1000.0, "futures", 2.0)
    assert out["risk"] is None
    assert "no defined loss" in out["risk_note"]


def test_spot_cannot_be_shorted(monkeypatch):
    svc.reset_wallet(10000.0)
    _patch_price(monkeypatch, 100.0)
    out = svc.open_trade("BTCUSDT", "short", 1000.0, "spot", 1.0)
    assert out["error"]["code"] == "INVALID_PARAMETER"
    assert "margin trade" in out["error"]["message"]


def test_spot_cannot_be_levered(monkeypatch):
    svc.reset_wallet(10000.0)
    _patch_price(monkeypatch, 100.0)
    out = svc.open_trade("BTCUSDT", "long", 1000.0, "spot", 3.0)
    assert out["error"]["code"] == "INVALID_PARAMETER"
    assert "unlevered" in out["error"]["message"]


@pytest.mark.parametrize("side,stop,target", [
    ("long", 110.0, None),    # stop above entry
    ("long", None, 90.0),     # target below entry
    ("short", 90.0, None),    # stop below entry
    ("short", None, 110.0),   # target above entry
])
def test_levels_on_the_wrong_side_are_refused(monkeypatch, side, stop, target):
    """A typo, not a strategy: these would fill on the next settle."""
    svc.reset_wallet(10000.0)
    _patch_price(monkeypatch, 100.0)
    _patch_bars(monkeypatch, [])
    out = svc.open_trade("BTCUSDT", side, 1000.0, "futures", 2.0,
                         stop_loss=stop, take_profit=target)
    assert out["error"]["code"] == "INVALID_PARAMETER"


def test_open_refuses_beyond_wallet_leverage(monkeypatch):
    svc.reset_wallet(10000.0, "USDT", 0.05, 5.0)
    _patch_price(monkeypatch, 100.0)
    _patch_bars(monkeypatch, [])
    out = svc.open_trade("BTCUSDT", "long", 1000.0, "futures", 10.0)
    assert out["error"]["code"] == "INVALID_PARAMETER"
    assert out["error"]["wallet_max_leverage"] == 5.0


def test_open_refuses_without_enough_cash(monkeypatch):
    svc.reset_wallet(100.0)
    _patch_price(monkeypatch, 100.0)
    _patch_bars(monkeypatch, [])
    out = svc.open_trade("BTCUSDT", "long", 10000.0, "futures", 1.0)
    assert out["error"]["code"] == "INVALID_PARAMETER"
    assert out["error"]["available"] == 100.0


def test_open_rejects_bad_side_and_market(monkeypatch):
    svc.reset_wallet(10000.0)
    _patch_price(monkeypatch, 100.0)
    assert svc.open_trade("B", "sideways", 10.0)["error"]["code"] == "INVALID_PARAMETER"
    assert svc.open_trade("B", "long", 10.0, "options")["error"]["code"] == \
        "INVALID_PARAMETER"


def test_open_surfaces_an_unknown_symbol(monkeypatch):
    svc.reset_wallet(10000.0)

    def boom(symbol, market):
        raise px.PriceUnavailable("No live price for NOPEUSDT")

    monkeypatch.setattr(svc.px, "mark_price", boom)
    _patch_bars(monkeypatch, [])
    out = svc.open_trade("NOPEUSDT", "long", 100.0)
    assert out["error"]["code"] == "SYMBOL_NOT_FOUND"


# ── Settlement through the service ────────────────────────────────────────────

def test_settlement_closes_and_returns_margin(monkeypatch):
    svc.reset_wallet(10000.0, "USDT", 0.0, 20.0)
    _patch_price(monkeypatch, 100.0)
    _patch_bars(monkeypatch, [])
    _patch_funding(monkeypatch, 0.0, 0)
    svc.open_trade("BTCUSDT", "long", 1000.0, "futures", 2.0, stop_loss=95.0)
    # Now a bar that takes the stop out.
    _patch_bars(monkeypatch, [_bar(101, 94)])
    status = svc.wallet_status()
    fills = status["settled_this_check"]
    assert len(fills) == 1
    assert fills[0]["reason"] == "stop_loss"
    assert fills[0]["realised_pnl"] == pytest.approx(-50.0)
    # margin (500) back, minus the 50 loss
    assert status["wallet"]["cash"] == pytest.approx(9950.0)
    assert status["open_count"] == 0


def test_fills_and_failures_are_separate_lists(monkeypatch):
    """The shape bug this caught in its first live run.

    A position whose bars cannot be fetched is not a fill. Returning both in
    one list gave consumers two different dict shapes.
    """
    svc.reset_wallet(10000.0)
    _patch_price(monkeypatch, 100.0)
    _patch_bars(monkeypatch, [])
    svc.open_trade("BTCUSDT", "long", 1000.0, "futures", 2.0, stop_loss=95.0)

    def boom(symbol, since_ms, market, now_ms=None):
        raise RuntimeError("klines unavailable")

    monkeypatch.setattr(svc.px, "bars_since", boom)
    status = svc.wallet_status()
    assert status["settled_this_check"] is None
    failures = status["unsettled_positions"]
    assert len(failures) == 1
    assert "klines unavailable" in failures[0]["unsettled"]
    assert "may already have hit" in failures[0]["consequence"]
    # The position must still be open — an outage is not an exit.
    assert status["open_count"] == 1


def test_liquidation_cannot_lose_more_than_margin(monkeypatch):
    svc.reset_wallet(10000.0, "USDT", 0.0, 20.0)
    _patch_price(monkeypatch, 100.0)
    _patch_bars(monkeypatch, [])
    _patch_funding(monkeypatch, 0.0, 0)
    svc.open_trade("BTCUSDT", "long", 2000.0, "futures", 10.0)
    # 10x long from 100 liquidates at 90; this bar crashes far below.
    _patch_bars(monkeypatch, [_bar(101, 50)])
    status = svc.wallet_status()
    fill = status["settled_this_check"][0]
    assert fill["reason"] == "liquidated"
    assert fill["realised_pnl"] == pytest.approx(-200.0)  # the margin, no more


def test_funding_is_charged_on_futures(monkeypatch):
    svc.reset_wallet(10000.0, "USDT", 0.0, 20.0)
    _patch_price(monkeypatch, 100.0)
    _patch_bars(monkeypatch, [])
    _patch_funding(monkeypatch, 7.5, 3)
    svc.open_trade("BTCUSDT", "long", 1000.0, "futures", 2.0, stop_loss=95.0)
    _patch_bars(monkeypatch, [_bar(101, 94)])
    fill = svc.wallet_status()["settled_this_check"][0]
    assert fill["funding_paid"] == 7.5
    assert fill["funding_payments"] == 3
    assert fill["realised_pnl"] == pytest.approx(-57.5)  # -50 loss - 7.5 funding


def test_spot_pays_no_funding(monkeypatch):
    svc.reset_wallet(10000.0, "USDT", 0.0, 20.0)
    _patch_price(monkeypatch, 100.0)
    _patch_bars(monkeypatch, [])

    def should_not_run(*a, **k):
        raise AssertionError("spot must not be charged funding")

    monkeypatch.setattr(svc.px, "funding_cost", should_not_run)
    svc.open_trade("BTCUSDT", "long", 1000.0, "spot", 1.0, stop_loss=95.0)
    _patch_bars(monkeypatch, [_bar(101, 94)])
    fill = svc.wallet_status()["settled_this_check"][0]
    assert fill["funding_paid"] == 0.0


def test_ambiguous_fill_is_flagged_through_the_service(monkeypatch):
    svc.reset_wallet(10000.0, "USDT", 0.0, 20.0)
    _patch_price(monkeypatch, 100.0)
    _patch_bars(monkeypatch, [])
    _patch_funding(monkeypatch, 0.0, 0)
    svc.open_trade("BTCUSDT", "long", 1000.0, "futures", 2.0,
                   stop_loss=95.0, take_profit=105.0)
    _patch_bars(monkeypatch, [_bar(106, 94)])
    fill = svc.wallet_status()["settled_this_check"][0]
    assert fill["reason"] == "stop_loss"
    assert fill["ambiguous_bar"] is True
    assert "which came first" in fill["ambiguity_note"]


# ── Closing and modifying ─────────────────────────────────────────────────────

def test_manual_close_realises_pnl(monkeypatch):
    svc.reset_wallet(10000.0, "USDT", 0.0, 20.0)
    _patch_price(monkeypatch, 100.0)
    _patch_bars(monkeypatch, [])
    _patch_funding(monkeypatch, 0.0, 0)
    opened = svc.open_trade("BTCUSDT", "long", 1000.0, "futures", 2.0)
    _patch_price(monkeypatch, 110.0)
    out = svc.close_trade(opened["opened"]["position_id"], note="taking it")
    closed = out["closed"]
    assert closed["realised_pnl"] == pytest.approx(100.0)
    assert closed["return_on_margin_pct"] == pytest.approx(20.0)
    assert closed["reason"] == "manual"
    assert out["cash_after"] == pytest.approx(10100.0)


def test_close_uses_the_settled_level_not_the_current_price(monkeypatch):
    """A stop already reached must not be overwritten by a nicer price."""
    svc.reset_wallet(10000.0, "USDT", 0.0, 20.0)
    _patch_price(monkeypatch, 100.0)
    _patch_bars(monkeypatch, [])
    _patch_funding(monkeypatch, 0.0, 0)
    opened = svc.open_trade("BTCUSDT", "long", 1000.0, "futures", 2.0,
                            stop_loss=95.0)
    # Price dipped through the stop, then recovered above entry.
    _patch_bars(monkeypatch, [_bar(101, 94)])
    _patch_price(monkeypatch, 120.0)
    out = svc.close_trade(opened["opened"]["position_id"])
    assert "closed_by_settlement" in out
    assert out["closed_by_settlement"]["reason"] == "stop_loss"
    assert "not the current price" in out["note"]


def test_closing_twice_is_refused(monkeypatch):
    svc.reset_wallet(10000.0, "USDT", 0.0, 20.0)
    _patch_price(monkeypatch, 100.0)
    _patch_bars(monkeypatch, [])
    _patch_funding(monkeypatch, 0.0, 0)
    pid = svc.open_trade("BTCUSDT", "long", 1000.0, "futures", 2.0)["opened"]["position_id"]
    svc.close_trade(pid)
    assert svc.close_trade(pid)["error"]["code"] == "INVALID_PARAMETER"


def test_closing_an_unknown_position(monkeypatch):
    svc.reset_wallet(10000.0)
    assert svc.close_trade(999)["error"]["code"] == "SYMBOL_NOT_FOUND"


def test_modify_moves_the_levels(monkeypatch):
    svc.reset_wallet(10000.0)
    _patch_price(monkeypatch, 100.0)
    _patch_bars(monkeypatch, [])
    pid = svc.open_trade("BTCUSDT", "long", 1000.0, "futures", 2.0,
                         stop_loss=95.0)["opened"]["position_id"]
    _patch_price(monkeypatch, 110.0)
    out = svc.modify_trade(pid, stop_loss=105.0)   # trailing it up
    assert out["modified"]["stop_loss"] == 105.0


def test_modify_refuses_a_level_that_would_fill_at_once(monkeypatch):
    svc.reset_wallet(10000.0)
    _patch_price(monkeypatch, 100.0)
    _patch_bars(monkeypatch, [])
    pid = svc.open_trade("BTCUSDT", "long", 1000.0, "futures", 2.0)["opened"]["position_id"]
    out = svc.modify_trade(pid, stop_loss=105.0)   # above current price
    assert out["error"]["code"] == "INVALID_PARAMETER"
    assert "fill immediately" in out["error"]["message"]


def test_modify_needs_at_least_one_level(monkeypatch):
    svc.reset_wallet(10000.0)
    assert svc.modify_trade(1)["error"]["code"] == "INVALID_PARAMETER"


# ── Status and history ────────────────────────────────────────────────────────

def test_equity_is_cash_plus_margin_plus_unrealised(monkeypatch):
    svc.reset_wallet(10000.0, "USDT", 0.0, 20.0)
    _patch_price(monkeypatch, 100.0)
    _patch_bars(monkeypatch, [])
    svc.open_trade("BTCUSDT", "long", 1000.0, "futures", 2.0)
    _patch_price(monkeypatch, 110.0)
    status = svc.wallet_status()
    w = status["wallet"]
    assert w["cash"] == pytest.approx(9500.0)
    assert w["margin_posted"] == pytest.approx(500.0)
    assert w["unrealised_pnl"] == pytest.approx(100.0)
    assert w["equity"] == pytest.approx(10100.0)
    assert w["total_return_pct"] == pytest.approx(1.0)


def test_unpriceable_position_is_reported_not_silently_zeroed(monkeypatch):
    svc.reset_wallet(10000.0, "USDT", 0.0, 20.0)
    _patch_price(monkeypatch, 100.0)
    _patch_bars(monkeypatch, [])
    svc.open_trade("BTCUSDT", "long", 1000.0, "futures", 2.0)

    def boom(symbol, market):
        raise RuntimeError("feed down")

    monkeypatch.setattr(svc.px, "mark_price", boom)
    status = svc.wallet_status()
    assert status["price_errors"]
    assert status["open_positions"][0]["unrealised_pnl"] is None


def test_history_statistics(monkeypatch):
    svc.reset_wallet(10000.0, "USDT", 0.0, 20.0)
    _patch_bars(monkeypatch, [])
    _patch_funding(monkeypatch, 0.0, 0)
    # One winner, one loser.
    _patch_price(monkeypatch, 100.0)
    a = svc.open_trade("BTCUSDT", "long", 1000.0, "futures", 2.0)["opened"]["position_id"]
    _patch_price(monkeypatch, 110.0)
    svc.close_trade(a)
    _patch_price(monkeypatch, 100.0)
    b = svc.open_trade("ETHUSDT", "long", 1000.0, "futures", 2.0)["opened"]["position_id"]
    _patch_price(monkeypatch, 95.0)
    svc.close_trade(b)

    stats = svc.trade_history()["statistics"]
    assert stats["trades"] == 2
    assert stats["wins"] == 1 and stats["losses"] == 1
    assert stats["net_pnl"] == pytest.approx(50.0)
    assert stats["profit_factor"] == pytest.approx(2.0)
    assert stats["realised_drawdown"] <= 0
    assert stats["closed_by"] == {"manual": 2}


def test_history_without_trades():
    svc.reset_wallet(10000.0)
    out = svc.trade_history()
    assert out["trades"] == []
    assert out["statistics"] is None


def test_history_limit_is_bounded():
    svc.reset_wallet(10000.0)
    assert svc.trade_history(0)["error"]["code"] == "INVALID_PARAMETER"
    assert svc.trade_history(9999)["error"]["code"] == "INVALID_PARAMETER"


def test_every_result_declares_it_is_simulated(monkeypatch):
    svc.reset_wallet(10000.0)
    _patch_price(monkeypatch, 100.0)
    _patch_bars(monkeypatch, [])
    opened = svc.open_trade("BTCUSDT", "long", 1000.0, "futures", 2.0)
    assert "SIMULATED" in opened["simulation"]
    assert "slippage is not modelled" in opened["slippage"]
    status = svc.wallet_status()
    assert "SIMULATED" in status["simulation"]
    assert "SIMULATED" in svc.trade_history()["simulation"]
