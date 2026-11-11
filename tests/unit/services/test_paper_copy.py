"""Mirroring a lead trader into a practice wallet. Offline.

Most of what is pinned here is refusal and disclosure, because the dangerous
version of this tool is the one that works smoothly. Copying a leaderboard
is not copy trading: you enter at today's price, and a position most of the
way to its target is one where you buy what is left of the move while
carrying the whole distance to the stop.
"""
import pytest

from tradingview_mcp.core.services import paper_copy as copy
from tradingview_mcp.core.services import paper_store as store
from tradingview_mcp.core.services import paper_trading_service as svc


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.delenv("TRADINGVIEW_MCP_PAPER_DB", raising=False)
    monkeypatch.setenv("TRADINGVIEW_MCP_DATA_DIR", str(tmp_path))
    store._initialised.clear()
    store.select_wallet(None)
    yield
    store.select_wallet(None)
    store._initialised.clear()


def _patch_prices(monkeypatch, price=100.0):
    for mod in (svc.px, copy.px):
        monkeypatch.setattr(mod, "mark_price", lambda s, m: (price, "test"))
        monkeypatch.setattr(mod, "bars_since",
                            lambda s, ms, m, now_ms=None: ([], "1m"))
    monkeypatch.setattr(svc.px, "funding_cost", lambda *a, **k: (0.0, 0))


def _patch_profile(monkeypatch, profile):
    monkeypatch.setattr(
        "tradingview_mcp.core.services.copytrading_service.get_lead_trader",
        lambda code, inst="SWAP", history_limit=50: profile,
    )


def _profile(positions, state="measured", nickname="Hronos"):
    return {
        "nickname": nickname,
        "open_positions": positions or None,
        "open_positions_state": state,
        "open_positions_note": "could not read" if state != "measured" else None,
    }


def _pos(inst="BTC-USDT-SWAP", side="long", lever=3.0, open_price=100.0):
    return {"instrument": inst, "side": side, "leverage": lever,
            "open_price": open_price, "mark_price": open_price}


# ── Symbol mapping ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("inst,expected", [
    ("BTC-USDT-SWAP", "BTCUSDT"),
    ("ETH-USDT-SWAP", "ETHUSDT"),
    ("SOL-USDT", "SOLUSDT"),
    ("BTC-USD-SWAP", None),
    ("BTC-USD-241227", None),
    ("", None),
    (None, None),
])
def test_only_usdt_instruments_convert(inst, expected):
    """A coin-margined instrument has different economics; copying it into a
    USDT wallet would invent a trade the trader did not take."""
    assert copy.okx_symbol(inst) == expected


# ── Refusals ──────────────────────────────────────────────────────────────────

def test_an_unreadable_book_is_refused_rather_than_copied_as_empty(monkeypatch):
    """The reason open_position_count stopped being 0 on a failed lookup.

    Copying a trader whose positions could not be read would open nothing
    and look exactly like a trader holding nothing.
    """
    _patch_prices(monkeypatch)
    svc.reset_wallet(10000.0)
    _patch_profile(monkeypatch, _profile([], state="unknown"))
    out = copy.copy_trader("X1")
    assert out["error"]["code"] == "NO_DATA"
    assert "could not be read" in out["error"]["message"]
    assert out["error"]["open_positions_state"] == "unknown"


def test_a_genuinely_flat_trader_is_a_different_message(monkeypatch):
    _patch_prices(monkeypatch)
    svc.reset_wallet(10000.0)
    _patch_profile(monkeypatch, _profile([], state="measured"))
    out = copy.copy_trader("X1")
    assert out["error"]["code"] == "NO_DATA"
    assert "holding nothing" in out["error"]["message"]


def test_a_wallet_that_does_not_exist_is_refused(monkeypatch):
    _patch_prices(monkeypatch)
    _patch_profile(monkeypatch, _profile([_pos()]))
    assert "error" in copy.copy_trader("X1", wallet="never-made")


@pytest.mark.parametrize("kwargs", [
    {"margin_pct": 0}, {"margin_pct": 50}, {"max_positions": 0},
    {"max_positions": 99},
])
def test_impossible_sizing_is_refused(monkeypatch, kwargs):
    _patch_prices(monkeypatch)
    svc.reset_wallet(10000.0)
    _patch_profile(monkeypatch, _profile([_pos()]))
    assert copy.copy_trader("X1", **kwargs)["error"]["code"] == "INVALID_PARAMETER"


# ── Preview is the default ────────────────────────────────────────────────────

def test_nothing_is_opened_without_confirmation(monkeypatch):
    _patch_prices(monkeypatch)
    svc.reset_wallet(10000.0)
    _patch_profile(monkeypatch, _profile([_pos(), _pos("ETH-USDT-SWAP")]))
    out = copy.copy_trader("X1")
    assert out["opened"] is None
    assert "confirm=true" in out["preview"]
    assert svc.wallet_status()["open_count"] == 0, "a preview must not write"


def test_confirming_opens_them(monkeypatch):
    _patch_prices(monkeypatch)
    svc.reset_wallet(10000.0)
    _patch_profile(monkeypatch, _profile([_pos(), _pos("ETH-USDT-SWAP")]))
    out = copy.copy_trader("X1", confirm=True)
    assert len(out["opened"]) == 2
    assert svc.wallet_status()["open_count"] == 2
    assert "copied from Hronos" in svc.wallet_status()["open_positions"][0]["note"]


def test_it_copies_into_the_wallet_it_was_told_to(monkeypatch):
    _patch_prices(monkeypatch)
    svc.reset_wallet(10000.0, wallet=None)
    svc.reset_wallet(10000.0, wallet="copied")
    _patch_profile(monkeypatch, _profile([_pos()]))
    copy.copy_trader("X1", wallet="copied", confirm=True)
    assert svc.wallet_status(wallet="copied")["open_count"] == 1
    assert svc.wallet_status(wallet=None)["open_count"] == 0


# ── The disclosure that makes it honest ───────────────────────────────────────

def test_a_position_already_run_is_flagged_as_late(monkeypatch):
    """They opened a long at 100 and it is now 130: you buy the last leg."""
    _patch_prices(monkeypatch, price=130.0)
    svc.reset_wallet(10000.0)
    _patch_profile(monkeypatch, _profile([_pos(open_price=100.0, side="long")]))
    row = copy.copy_trader("X1")["plan"][0]
    assert row["already_moved_pct"] == pytest.approx(30.0)
    assert "what is left of the move" in row["late_note"]


def test_a_position_that_went_against_them_is_not_flagged(monkeypatch):
    """Copying there is cheaper than it looks, not more expensive."""
    _patch_prices(monkeypatch, price=90.0)
    svc.reset_wallet(10000.0)
    _patch_profile(monkeypatch, _profile([_pos(open_price=100.0, side="long")]))
    row = copy.copy_trader("X1")["plan"][0]
    assert row["already_moved_pct"] == pytest.approx(-10.0)
    assert "late_note" not in row


def test_drift_is_measured_in_the_traders_direction(monkeypatch):
    """A short that has fallen is winning, not losing."""
    _patch_prices(monkeypatch, price=80.0)
    svc.reset_wallet(10000.0)
    _patch_profile(monkeypatch, _profile([_pos(open_price=100.0, side="short")]))
    assert copy.copy_trader("X1")["plan"][0]["already_moved_pct"] == pytest.approx(20.0)


def test_the_result_leads_with_the_furthest_along(monkeypatch):
    _patch_prices(monkeypatch, price=110.0)
    svc.reset_wallet(10000.0)
    _patch_profile(monkeypatch, _profile([
        _pos("BTC-USDT-SWAP", open_price=100.0),
        _pos("ETH-USDT-SWAP", open_price=50.0),
    ]))
    out = copy.copy_trader("X1")
    assert out["furthest_along"]["symbol"] == "ETHUSDT"


def test_leverage_above_the_wallet_cap_is_capped_and_named(monkeypatch):
    """Not skipped silently, and not pretended to be the same trade."""
    _patch_prices(monkeypatch)
    svc.reset_wallet(10000.0, "USDT", 0.0, 5.0)
    _patch_profile(monkeypatch, _profile([_pos(lever=20.0)]))
    row = copy.copy_trader("X1")["plan"][0]
    assert row["your_leverage"] == 5.0
    assert row["leverage_capped"] is True
    assert "different trade from theirs" in row["leverage_note"]


def test_the_result_says_the_copies_have_no_stop(monkeypatch):
    _patch_prices(monkeypatch)
    svc.reset_wallet(10000.0)
    _patch_profile(monkeypatch, _profile([_pos()]))
    out = copy.copy_trader("X1")
    assert "NO STOP" in out["no_stops"]
    assert "liquidation" in out["no_stops"]
    assert "today's price, not theirs" in out["not_copy_trading"]


def test_sizing_is_the_wallets_not_the_traders(monkeypatch):
    _patch_prices(monkeypatch)
    svc.reset_wallet(10000.0)
    _patch_profile(monkeypatch, _profile([_pos(lever=3.0)]))
    row = copy.copy_trader("X1", margin_pct=2.0)["plan"][0]
    assert row["your_margin"] == pytest.approx(200.0)
    assert row["your_notional"] == pytest.approx(600.0)


def test_an_instrument_this_wallet_cannot_price_is_skipped_with_a_reason(
        monkeypatch):
    _patch_prices(monkeypatch)
    svc.reset_wallet(10000.0)
    _patch_profile(monkeypatch, _profile([
        _pos("BTC-USD-SWAP"), _pos("BTC-USDT-SWAP"),
    ]))
    out = copy.copy_trader("X1")
    assert len(out["plan"]) == 1
    assert out["skipped"][0]["reason"].startswith("not a USDT pair")


def test_a_plan_bigger_than_the_cash_is_refused_before_opening_half(monkeypatch):
    """Opening three of five and failing on the fourth would leave a wallet
    holding a strategy nobody chose."""
    _patch_prices(monkeypatch)
    svc.reset_wallet(1000.0)
    # Five at 25% of a 1,000 wallet is 1,250 of margin against 1,000 of cash.
    _patch_profile(monkeypatch, _profile([
        _pos("BTC-USDT-SWAP"), _pos("ETH-USDT-SWAP"), _pos("SOL-USDT-SWAP"),
        _pos("ARB-USDT-SWAP"), _pos("OP-USDT-SWAP"),
    ]))
    out = copy.copy_trader("X1", margin_pct=25.0, max_positions=5, confirm=True)
    assert out["error"]["code"] == "INVALID_PARAMETER"
    assert svc.wallet_status()["open_count"] == 0


def test_max_positions_bounds_what_is_copied(monkeypatch):
    _patch_prices(monkeypatch)
    svc.reset_wallet(10000.0)
    _patch_profile(monkeypatch, _profile([
        _pos("BTC-USDT-SWAP"), _pos("ETH-USDT-SWAP"), _pos("SOL-USDT-SWAP"),
    ]))
    assert len(copy.copy_trader("X1", max_positions=2)["plan"]) == 2
