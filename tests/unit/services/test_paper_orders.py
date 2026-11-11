"""Unit tests for resting limit orders. Offline.

The wallet could only fill at the current market price, so "buy at support"
could not be expressed — you had to sit and watch for the level. An order
rests instead, and is resolved retroactively the way stops already are: the
bars since you placed it are scanned on the next check.

What gets the attention here is the seam between the two. An order that
filled and then stopped out between two checks has to resolve both, in the
right order, with the existing bias intact — a bar that holds the fill and
the stop ends as a loss, because the bar does not record which came first.
"""
import pytest

from tradingview_mcp.core.services import paper_orders as orders
from tradingview_mcp.core.services import paper_store as store
from tradingview_mcp.core.services import paper_trading_service as svc


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADINGVIEW_MCP_PAPER_DB", str(tmp_path / "paper.sqlite3"))
    store._initialised.clear()
    yield
    store._initialised.clear()


def _bar(high, low, open_ms):
    return {"high": high, "low": low, "open": (high + low) / 2,
            "close": (high + low) / 2, "open_ms": open_ms}


def _patch_price(monkeypatch, price=100.0):
    for mod in (svc.px, orders.px):
        monkeypatch.setattr(mod, "mark_price", lambda s, m: (price, "test"))


def _patch_bars(monkeypatch, bars):
    """Time-aware, because the seam depends on it.

    A fill stamps its position with the bar that reached it, and the position
    scan that follows asks for bars since that moment. A mock that ignored
    `since_ms` would hide whether the two steps line up.
    """
    def fake(symbol, since_ms, market, now_ms=None):
        return [b for b in bars if b["open_ms"] >= since_ms], "1m"

    for mod in (svc.px, orders.px):
        monkeypatch.setattr(mod, "bars_since", fake)


def _patch_funding(monkeypatch, cost=0.0):
    for mod in (svc.px, orders.px):
        monkeypatch.setattr(mod, "funding_cost", lambda *a, **k: (cost, 0))


def _wallet(fee=0.0):
    svc.reset_wallet(10000.0, "USDT", fee, 20.0)


# ── Placing ───────────────────────────────────────────────────────────────────

def test_a_buy_rests_below_the_market(monkeypatch):
    _wallet()
    _patch_price(monkeypatch, 100.0)
    out = orders.place_order("BTCUSDT", "long", 1000.0, 90.0)
    assert out["placed"]["limit_price"] == 90.0
    assert out["placed"]["status"] == store.PENDING
    assert out["distance_to_fill_pct"] == pytest.approx(10.0)


def test_an_order_that_would_fill_instantly_is_refused(monkeypatch):
    """Otherwise "wait for support" quietly becomes "buy now"."""
    _wallet()
    _patch_price(monkeypatch, 100.0)
    out = orders.place_order("BTCUSDT", "long", 1000.0, 105.0)
    assert out["error"]["code"] == "INVALID_PARAMETER"
    assert "below the market" in out["error"]["message"]
    assert "paper_trade_open" in out["error"]["message"], "say what to use"


def test_a_resting_sell_must_sit_above_the_market(monkeypatch):
    _wallet()
    _patch_price(monkeypatch, 100.0)
    assert orders.place_order("BTCUSDT", "short", 1000.0, 95.0)["error"][
        "code"] == "INVALID_PARAMETER"
    assert orders.place_order("BTCUSDT", "short", 1000.0, 110.0)["placed"]


def test_levels_are_checked_against_the_limit_not_todays_price(monkeypatch):
    """A stop at 92 is below today's 100 but above a limit of 90 — which
    makes it not a stop at all for the trade that would actually open."""
    _wallet()
    _patch_price(monkeypatch, 100.0)
    out = orders.place_order("BTCUSDT", "long", 1000.0, 90.0, stop_loss=92.0)
    assert out["error"]["code"] == "INVALID_PARAMETER"
    assert "below its entry" in out["error"]["message"]


def test_risk_is_measured_from_the_limit(monkeypatch):
    _wallet()
    _patch_price(monkeypatch, 100.0)
    out = orders.place_order("BTCUSDT", "long", 1000.0, 90.0,
                             stop_loss=85.5, take_profit=99.0)
    # 1000 notional at 90 is 11.11 units; a 4.5 move against is 50.
    assert out["risk_if_filled"]["loss_at_stop"] == pytest.approx(50.0, abs=0.05)
    assert out["risk_if_filled"]["risk_reward"] == pytest.approx(2.0, abs=0.02)


# ── Margin is held from the moment it rests ───────────────────────────────────

def test_margin_leaves_the_cash_when_the_order_is_placed(monkeypatch):
    _wallet()
    _patch_price(monkeypatch, 100.0)
    orders.place_order("BTCUSDT", "long", 1000.0, 90.0, leverage=2.0)
    assert store.get_wallet()["cash"] == pytest.approx(9500.0)


def test_two_orders_cannot_rest_on_the_same_cash(monkeypatch):
    """Without holding margin, both would fill and the wallet would carry
    risk it never had the money for."""
    _wallet()
    _patch_price(monkeypatch, 100.0)
    assert orders.place_order("BTCUSDT", "long", 9000.0, 90.0)["placed"]
    second = orders.place_order("ETHUSDT", "long", 9000.0, 90.0)
    assert second["error"]["code"] == "INVALID_PARAMETER"
    assert "Not enough cash" in second["error"]["message"]


def test_cancelling_gives_the_margin_back(monkeypatch):
    _wallet()
    _patch_price(monkeypatch, 100.0)
    oid = orders.place_order("BTCUSDT", "long", 1000.0, 90.0)["placed"]["order_id"]
    out = orders.cancel_order(oid)
    assert out["margin_released"] == pytest.approx(1000.0)
    assert store.get_wallet()["cash"] == pytest.approx(10000.0)
    assert out["cancelled"]["status"] == store.CANCELLED


def test_cancelling_twice_is_refused(monkeypatch):
    _wallet()
    _patch_price(monkeypatch, 100.0)
    oid = orders.place_order("BTCUSDT", "long", 1000.0, 90.0)["placed"]["order_id"]
    orders.cancel_order(oid)
    assert orders.cancel_order(oid)["error"]["code"] == "INVALID_PARAMETER"


def test_cancelling_an_unknown_order_names_the_right_failure(monkeypatch):
    _wallet()
    assert orders.cancel_order(4242)["error"]["code"] == "RECORD_NOT_FOUND"


# ── Filling ───────────────────────────────────────────────────────────────────

def test_a_reached_level_fills_at_the_level(monkeypatch):
    _wallet()
    _patch_price(monkeypatch, 100.0)
    _patch_funding(monkeypatch)
    oid = orders.place_order("BTCUSDT", "long", 1000.0, 90.0,
                             stop_loss=80.0)["placed"]["order_id"]
    ms = store.now_ms()
    _patch_bars(monkeypatch, [_bar(99, 95, ms + 1), _bar(96, 88, ms + 2)])

    status = svc.wallet_status()
    done = status["orders_resolved_this_check"]
    assert done and done[0]["outcome"] == "filled"
    assert done[0]["fill_price"] == 90.0, "filled at the level, not the low"
    assert status["open_positions"][0]["entry_price"] == 90.0
    assert store.get_order(oid)["status"] == store.FILLED


def test_a_level_never_reached_keeps_resting(monkeypatch):
    _wallet()
    _patch_price(monkeypatch, 100.0)
    orders.place_order("BTCUSDT", "long", 1000.0, 90.0)
    ms = store.now_ms()
    _patch_bars(monkeypatch, [_bar(99, 95, ms + 1), _bar(98, 93, ms + 2)])
    status = svc.wallet_status()
    assert status["orders_resolved_this_check"] is None
    assert len(status["pending_orders"]) == 1


def test_the_entry_fee_is_charged_at_the_fill_not_at_placement(monkeypatch):
    _wallet(fee=0.1)
    _patch_price(monkeypatch, 100.0)
    _patch_funding(monkeypatch)
    orders.place_order("BTCUSDT", "long", 1000.0, 90.0)
    assert store.get_wallet()["cash"] == pytest.approx(9000.0), "margin only"
    ms = store.now_ms()
    _patch_bars(monkeypatch, [_bar(95, 88, ms + 1)])
    svc.wallet_status()
    assert store.get_wallet()["cash"] == pytest.approx(8999.0), "1.0 fee at fill"


# ── The seam: filled and stopped out between two checks ───────────────────────

def test_an_order_can_fill_and_stop_out_in_one_check(monkeypatch):
    """The case this design exists to get right.

    Place a buy at 90 with a stop at 85, then come back after a bar that
    traded 95 down to 84. The bar reached the limit AND the stop. The fill
    stamps the position with that bar, the position scan starts from it, and
    the existing "ambiguous bar goes against the trade" rule finishes it.
    """
    _wallet()
    _patch_price(monkeypatch, 100.0)
    _patch_funding(monkeypatch)
    orders.place_order("BTCUSDT", "long", 1000.0, 90.0, stop_loss=85.0)
    ms = store.now_ms()
    _patch_bars(monkeypatch, [_bar(95, 84, ms + 1)])

    status = svc.wallet_status()
    assert status["orders_resolved_this_check"][0]["outcome"] == "filled"
    assert status["settled_this_check"], "the position must settle too"
    assert status["settled_this_check"][0]["reason"] == "stop_loss"
    assert status["open_count"] == 0

    closed = store.list_positions(store.CLOSED)[0]
    assert closed["entry_price"] == 90.0
    assert closed["exit_price"] == 85.0
    assert closed["realised_pnl"] < 0


def test_a_fill_in_an_earlier_bar_is_not_stopped_by_a_later_one_it_survived(
        monkeypatch):
    """The scan must start at the fill, not at placement."""
    _wallet()
    _patch_price(monkeypatch, 100.0)
    _patch_funding(monkeypatch)
    orders.place_order("BTCUSDT", "long", 1000.0, 90.0, stop_loss=85.0)
    ms = store.now_ms()
    _patch_bars(monkeypatch, [_bar(95, 89, ms + 1), _bar(94, 88, ms + 2)])
    status = svc.wallet_status()
    assert status["orders_resolved_this_check"][0]["outcome"] == "filled"
    assert status["open_count"] == 1, "nothing reached 85"


# ── Expiry ────────────────────────────────────────────────────────────────────

def test_an_order_expires_if_the_price_never_came(monkeypatch):
    _wallet()
    _patch_price(monkeypatch, 100.0)
    orders.place_order("BTCUSDT", "long", 1000.0, 90.0, good_for_hours=1)
    oid = store.list_orders(store.PENDING)[0]["id"]
    store.resolve_order(oid, store.PENDING, None)  # keep pending, clear fields
    with store._connect() as conn:
        conn.execute("UPDATE orders SET expires_ms = ? WHERE id = ?",
                     (store.now_ms() - 1000, oid))
    ms = store.now_ms()
    _patch_bars(monkeypatch, [_bar(99, 95, ms + 1)])

    status = svc.wallet_status()
    assert status["orders_resolved_this_check"][0]["outcome"] == "expired"
    assert store.get_wallet()["cash"] == pytest.approx(10000.0), "margin back"


def test_a_fill_beats_an_expiry_that_came_later(monkeypatch):
    """Reached on Tuesday and expired on Wednesday means filled on Tuesday."""
    _wallet()
    _patch_price(monkeypatch, 100.0)
    _patch_funding(monkeypatch)
    orders.place_order("BTCUSDT", "long", 1000.0, 90.0, good_for_hours=1)
    oid = store.list_orders(store.PENDING)[0]["id"]
    with store._connect() as conn:
        conn.execute("UPDATE orders SET expires_ms = ? WHERE id = ?",
                     (store.now_ms() - 1000, oid))
    ms = store.now_ms()
    _patch_bars(monkeypatch, [_bar(95, 88, ms + 1)])
    status = svc.wallet_status()
    assert status["orders_resolved_this_check"][0]["outcome"] == "filled"


# ── Honesty about what a simulated fill is worth ──────────────────────────────

def test_the_result_says_a_touched_level_is_a_best_case(monkeypatch):
    """The stop rule makes a touch pessimistic; the same rule on an entry is
    optimistic, and that asymmetry has to be stated rather than buried."""
    _wallet()
    _patch_price(monkeypatch, 100.0)
    out = orders.place_order("BTCUSDT", "long", 1000.0, 90.0)
    assert "BEST CASE" in out["fill_realism"]
    assert "queue" in out["fill_realism"]
    assert "no real order" in out["simulation"].lower() or "SIMULATED" in out["simulation"]


def test_a_bar_feed_that_fails_leaves_the_order_resting_and_says_so(monkeypatch):
    _wallet()
    _patch_price(monkeypatch, 100.0)
    orders.place_order("BTCUSDT", "long", 1000.0, 90.0)

    def boom(*a, **k):
        raise RuntimeError("klines unavailable")

    for mod in (svc.px, orders.px):
        monkeypatch.setattr(mod, "bars_since", boom)
    status = svc.wallet_status()
    stuck = status["orders_unchecked"]
    assert stuck and "klines unavailable" in stuck[0]["unchecked"]
    assert "still resting" in stuck[0]["consequence"]
    assert len(status["pending_orders"]) == 1


# ── The wallet shows what is resting ──────────────────────────────────────────

def test_the_wallet_shows_pending_orders_and_the_cash_they_hold(monkeypatch):
    """Margin on a resting order has left `cash` without being in a position,
    so a wallet that did not show it would look like money had vanished."""
    _wallet()
    _patch_price(monkeypatch, 100.0)
    orders.place_order("BTCUSDT", "long", 1000.0, 90.0, note="at support")
    status = svc.wallet_status()
    resting = status["pending_orders"]
    assert len(resting) == 1
    assert resting[0]["limit_price"] == 90.0
    assert resting[0]["margin_held"] == pytest.approx(1000.0)
    assert resting[0]["note"] == "at support"


def test_listing_reports_what_is_holding_margin(monkeypatch):
    _wallet()
    _patch_price(monkeypatch, 100.0)
    _patch_bars(monkeypatch, [])
    orders.place_order("BTCUSDT", "long", 1000.0, 90.0)
    orders.place_order("BTCUSDT", "long", 500.0, 80.0)
    out = orders.list_orders(store.PENDING)
    assert out["counts"]["pending"] == 2
    assert out["margin_held_by_pending"] == pytest.approx(1500.0)
    assert out["nearest_fill"]["limit_price"] == 90.0, "closest to the market"
