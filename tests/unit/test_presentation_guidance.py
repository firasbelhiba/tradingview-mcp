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
