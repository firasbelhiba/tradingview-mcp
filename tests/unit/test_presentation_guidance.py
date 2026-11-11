"""The server's guidance to the assistant that reads it.

The server cannot render anything — a local host ignores the ui:// resource
it can emit. What it CAN do is tell the assistant, through two channels the
assistant actually reads: the connection-level instructions, and the tool
results themselves. These tests keep both channels honest.
"""
import pytest

from tradingview_mcp.core.services import paper_trading_service as ps
from tradingview_mcp.core.services import paper_store as store
from tradingview_mcp.server import mcp


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADINGVIEW_MCP_PAPER_DB", str(tmp_path / "paper.sqlite3"))
    store._initialised.clear()
    yield
    store._initialised.clear()


# ── Connection-level instructions ─────────────────────────────────────────────

def test_instructions_ask_for_a_rendered_panel():
    instructions = mcp.instructions or ""
    assert "PRESENTATION" in instructions
    for fragment in ("visual panel", "without asking first", "display_hint"):
        assert fragment in instructions, fragment


def test_instructions_name_the_failure_fields_a_glance_could_miss():
    """A dashboard that shows only healthy numbers is worse than a table."""
    instructions = mcp.instructions or ""
    for field in ("unsettled_positions", "unavailable_feeds",
                  "not_evaluated", "price_errors"):
        assert field in instructions, field


def test_instructions_leave_an_out():
    """An assistant with no rendering tool must ignore this without ceremony,
    and a narrow question should still get a one-line answer."""
    instructions = mcp.instructions or ""
    assert "Skip all of this if you have no rendering tool" in instructions
    assert "one number answers it" in instructions


def test_instructions_stay_about_presentation_only():
    """The server may say how to SHOW a finding, never what to conclude.

    Guidance that drifted into "recommend", "suggest the trade" or similar
    would be the server steering the assistant's judgement on someone's
    money, which is a different thing from formatting.
    """
    block = (mcp.instructions or "").split("PRESENTATION")[-1].lower()
    for forbidden in ("recommend", "advise", "you should buy", "tell them to",
                      "profitable", "best trade"):
        assert forbidden not in block, f"guidance drifted into advice: {forbidden}"


def test_instructions_stay_small_enough_to_carry():
    # Sent on every connection — a page of prose would be a tax on each one.
    assert len(mcp.instructions or "") < 2500


# ── Tool-result hints ─────────────────────────────────────────────────────────

def _patch(monkeypatch, price=100.0):
    monkeypatch.setattr(ps.px, "mark_price", lambda s, m: (price, "test"))
    monkeypatch.setattr(ps.px, "bars_since", lambda s, ms, m, now_ms=None: ([], "1m"))
    monkeypatch.setattr(ps.px, "funding_cost", lambda *a, **k: (0.0, 0))


def test_wallet_status_carries_the_hint(monkeypatch):
    ps.reset_wallet(1000.0, "USDT", 0.0, 20.0)
    _patch(monkeypatch)
    hint = ps.wallet_status()["display_hint"]
    assert "draw it as a panel" in hint
    assert "PRESENTATION" in hint, "it must point at the full policy"


def test_history_carries_the_hint_once_there_is_something_to_draw(monkeypatch):
    ps.reset_wallet(1000.0, "USDT", 0.0, 20.0)
    _patch(monkeypatch)
    # Empty history has nothing to render, so no hint is needed there.
    assert "display_hint" not in ps.trade_history()
    pid = ps.open_trade("BTCUSDT", "long", 100.0, "futures", 1.0)["opened"]["position_id"]
    ps.close_trade(pid)
    assert "display_hint" in ps.trade_history()


def test_hint_tells_the_assistant_to_surface_failures(monkeypatch):
    ps.reset_wallet(1000.0, "USDT", 0.0, 20.0)
    _patch(monkeypatch)
    hint = ps.wallet_status()["display_hint"].lower()
    assert "unsettled_positions" in hint
    assert "not just the healthy numbers" in hint


def test_hint_is_a_trigger_not_a_restatement(monkeypatch):
    """It rides on EVERY call, so it has to stay small.

    The first version restated the whole policy and ran 890 bytes — 52% of an
    empty wallet's payload. The policy belongs in the connection
    instructions, which travel once.
    """
    import json

    ps.reset_wallet(1000.0, "USDT", 0.0, 20.0)
    _patch(monkeypatch)
    status = ps.wallet_status()
    hint_size = len(json.dumps(status["display_hint"]))
    total = len(json.dumps(status, default=str))
    assert hint_size < 320, f"hint is {hint_size} bytes; keep it a pointer"
    assert hint_size < total * 0.25, "the hint should not dominate the result"


# ── Discovery ─────────────────────────────────────────────────────────────────

def test_trade_plan_points_at_the_paper_wallet(monkeypatch):
    """Discovery at the moment of relevance.

    Someone holding an entry, a stop and a target is the person who should
    learn the simulated wallet exists. A README line does not reach them
    here; the tool result does.
    """
    from tradingview_mcp.core.services import crypto_analysis_service as cas

    class _A:
        indicators = {
            "close": 100.0, "high": 102.0, "low": 98.0, "ATR": 3.0,
            "EMA20": 99.0, "EMA50": 95.0, "EMA200": 90.0, "RSI": 55.0,
            "Pivot.M.Classic.S1": 96.0, "Pivot.M.Classic.R1": 108.0,
        }

    monkeypatch.setattr(
        cas, "resilient_get_multiple_analysis",
        lambda screener, interval, symbols: {symbols[0]: _A()},
    )
    out = cas.generate_crypto_trade_plan("BTCUSDT", "KUCOIN", "4h")
    assert "paper_trade_open" in out["next_step"]
    assert "no real order" in out["next_step"]
    assert len(out["next_step"]) < 250, "a pointer, not a pitch"


def test_server_instructions_name_the_wallet_as_a_capability():
    """So "what can you do?" gets an accurate answer without tool search."""
    instructions = mcp.instructions or ""
    assert "paper-trading wallet" in instructions
    assert "alerts" in instructions


# ── A live panel has to be re-readable ────────────────────────────────────────

def test_wallet_status_says_when_it_was_read(monkeypatch):
    """A panel drawn from live marks is a photograph of a moving thing.

    Without a time on it there is no telling a fresh panel from one left
    open for an hour, and the equity on it is quietly wrong.
    """
    import re

    ps.reset_wallet(1000.0, "USDT", 0.0, 20.0)
    _patch(monkeypatch)
    as_of = ps.wallet_status()["as_of"]
    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", as_of), as_of
    assert as_of.endswith("+00:00"), "a bare local time is not a timestamp"


def test_a_wallet_with_live_marks_asks_for_the_stamp_and_the_refresh(monkeypatch):
    ps.reset_wallet(1000.0, "USDT", 0.0, 20.0)
    _patch(monkeypatch)
    ps.open_trade("BTCUSDT", "long", 100.0, "futures", 1.0)
    hint = ps.wallet_status()["display_hint"]
    assert "as_of" in hint
    assert "refresh" in hint


def test_an_empty_wallet_is_not_told_to_offer_a_refresh(monkeypatch):
    """Nothing is marked live, so the control would be furniture — and the
    sentence would be a quarter of an otherwise tiny payload."""
    ps.reset_wallet(1000.0, "USDT", 0.0, 20.0)
    _patch(monkeypatch)
    hint = ps.wallet_status()["display_hint"]
    assert "refresh" not in hint
    assert "draw it as a panel" in hint, "it is still worth drawing"


def test_the_refresh_is_asked_for_conditionally(monkeypatch):
    """The server cannot draw a button, and a host that proved it cannot
    render ui:// resources will not grow one because we insist."""
    ps.reset_wallet(1000.0, "USDT", 0.0, 20.0)
    _patch(monkeypatch)
    ps.open_trade("BTCUSDT", "long", 100.0, "futures", 1.0)
    hint = ps.wallet_status()["display_hint"].lower()
    assert "if you can" in hint, "it must stay conditional, not an order"


def test_the_instructions_carry_the_policy_not_the_hint():
    """The hint rides on every call; the instructions travel once."""
    instructions = mcp.instructions or ""
    assert "photograph" in instructions
    assert "re-asks for the same view" in instructions
    assert "closes it" in instructions, "a wallet panel's other obvious control"


# ── Every wallet result, not only the ones that read it ───────────────────────

def test_every_paper_result_says_how_to_draw_itself(monkeypatch):
    """The gap a person found by using it.

    Of eight paper results exactly one carried a hint, and it was a read. So
    "how is my wallet?" drew a panel and "create me a wallet" came back as a
    bulleted list — even though somebody who just created a wallet is the
    person most certainly looking at it. The hint belonged to the moment
    state CHANGES, which is precisely where it was missing.
    """
    from tradingview_mcp.core.services import paper_orders as po

    _patch(monkeypatch)
    monkeypatch.setattr(po.px, "mark_price", lambda s, m: (100.0, "test"))
    monkeypatch.setattr(po.px, "bars_since",
                        lambda s, ms, m, now_ms=None: ([], "1m"))

    results = {"reset": ps.reset_wallet(1000.0, "USDT", 0.0, 20.0)}
    results["open"] = ps.open_trade("BTCUSDT", "long", 100.0, "futures", 1.0,
                                    stop_loss=90.0)
    results["order_place"] = po.place_order("BTCUSDT", "long", 50.0, 80.0)
    results["order_list"] = po.list_orders()
    results["status"] = ps.wallet_status()
    pid = results["status"]["open_positions"][0]["position_id"]
    results["modify"] = ps.modify_trade(pid, stop_loss=95.0)
    results["close"] = ps.close_trade(pid)
    results["history"] = ps.trade_history()

    missing = [k for k, v in results.items() if not v.get("display_hint")]
    assert missing == [], f"results with no display_hint: {missing}"


def test_each_wallet_hint_names_that_surface_and_not_another(monkeypatch):
    """Identical hints everywhere would be noise with extra steps."""
    from tradingview_mcp.core.services import display_hints as d

    surfaces = [d.WALLET_CREATED, d.TRADE_OPENED, d.TRADE_CLOSED,
                d.TRADE_MODIFIED, d.ORDER_PLACED, d.ORDERS_LISTED,
                d.TRADE_HISTORY]
    assert len(set(surfaces)) == len(surfaces), "a hint was reused verbatim"
    assert "max leverage" in d.WALLET_CREATED
    assert "risk_note" in d.TRADE_OPENED, "a position with no stop must show"
    assert "fees and funding" in d.TRADE_CLOSED
    assert "fill_realism" in d.ORDER_PLACED


def test_the_instructions_no_longer_name_only_the_read_tools():
    instructions = mcp.instructions or ""
    assert "every paper wallet result" in instructions
    assert "most certainly looking" in instructions
