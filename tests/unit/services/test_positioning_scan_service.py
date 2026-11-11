"""Unit tests for the board-wide positioning scans.

Offline: the three bulk reads and the per-symbol detail reads are
monkeypatched. What is pinned is the per-symbol funding interval (the thing
that makes an annualised figure right or 8x wrong), the stage-1 shortlist
rule, and that grading is delegated rather than duplicated.
"""
import pytest

from tradingview_mcp.core.services import binance_service as bn
from tradingview_mcp.core.services import positioning_scan_service as svc
from tradingview_mcp.core.services import positioning_service as ps


def _premium(symbol, rate, mark=100.0):
    return {"symbol": symbol, "lastFundingRate": str(rate), "markPrice": str(mark)}


def _ticker(symbol, change, volume=1e9):
    return {"symbol": symbol, "priceChangePercent": str(change),
            "quoteVolume": str(volume)}


def _patch_board(monkeypatch, premium, tickers, intervals=None):
    monkeypatch.setattr(bn, "get_premium_index_all", lambda: premium)
    monkeypatch.setattr(bn, "get_ticker_24h_all", lambda: tickers)
    monkeypatch.setattr(bn, "funding_intervals", lambda: intervals or {})


# ── Funding interval handling ─────────────────────────────────────────────────

def test_annualisation_uses_the_symbols_own_interval(monkeypatch):
    # Same raw rate, different intervals: a 1h symbol costs eight times what
    # an 8h symbol does. Annualising everything at 8h is the bug this guards.
    _patch_board(
        monkeypatch,
        [_premium("EIGHTH", 0.0001), _premium("ONEH", 0.0001)],
        [_ticker("EIGHTH", 1.0), _ticker("ONEH", 1.0)],
        intervals={"ONEH": 1.0},
    )
    out = svc.screen_funding(limit=5, min_quote_volume_usd=None)
    rows = {r["symbol"]: r for r in out["longs_pay_most"]}
    assert rows["EIGHTH"]["funding_interval_hours"] == 8.0
    assert rows["ONEH"]["funding_interval_hours"] == 1.0
    assert rows["EIGHTH"]["funding_annualised_pct"] == pytest.approx(10.95, abs=0.05)
    assert rows["ONEH"]["funding_annualised_pct"] == pytest.approx(87.6, abs=0.1)


def test_symbols_absent_from_funding_info_default_to_eight_hours(monkeypatch):
    _patch_board(monkeypatch, [_premium("X", 0.0001)], [_ticker("X", 1.0)], intervals={})
    row = svc.screen_funding(min_quote_volume_usd=None)["longs_pay_most"][0]
    assert row["funding_interval_hours"] == 8.0


def test_funding_intervals_returns_empty_on_failure(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("down")
    monkeypatch.setattr(bn, "_fetch", boom)
    bn._cache.clear()
    assert bn.funding_intervals() == {}


# ── Screener ──────────────────────────────────────────────────────────────────

def test_screener_splits_payers_and_summarises(monkeypatch):
    _patch_board(
        monkeypatch,
        [_premium("POS", 0.001), _premium("NEG", -0.002), _premium("FLAT", 0.0)],
        [_ticker("POS", 1.0), _ticker("NEG", -1.0), _ticker("FLAT", 0.0)],
    )
    out = svc.screen_funding(limit=3, min_quote_volume_usd=None)
    assert out["longs_pay_most"][0]["symbol"] == "POS"
    assert out["longs_pay_most"][0]["payer"] == "longs"
    assert out["shorts_pay_most"][0]["symbol"] == "NEG"
    assert out["shorts_pay_most"][0]["payer"] == "shorts"
    assert out["summary"]["positive_funding"] == 1
    assert out["summary"]["negative_funding"] == 1


def test_screener_volume_filter_drops_thin_contracts(monkeypatch):
    _patch_board(
        monkeypatch,
        [_premium("THICK", 0.001), _premium("THIN", 0.05)],
        [_ticker("THICK", 1.0, volume=5e8), _ticker("THIN", 1.0, volume=1000.0)],
    )
    out = svc.screen_funding(min_quote_volume_usd=1e6)
    symbols = [r["symbol"] for r in out["longs_pay_most"]]
    assert symbols == ["THICK"], "a huge rate on a dead market is noise"
    assert out["after_volume_filter"] == 1


@pytest.mark.parametrize("direction,expect_long,expect_short", [
    ("both", True, True), ("longs_pay", True, False), ("shorts_pay", False, True),
])
def test_screener_direction(monkeypatch, direction, expect_long, expect_short):
    _patch_board(monkeypatch, [_premium("A", 0.001)], [_ticker("A", 1.0)])
    out = svc.screen_funding(direction=direction, min_quote_volume_usd=None)
    assert ("longs_pay_most" in out) is expect_long
    assert ("shorts_pay_most" in out) is expect_short


@pytest.mark.parametrize("kwargs", [{"limit": 0}, {"limit": 500}, {"direction": "sideways"}])
def test_screener_rejects_bad_params(kwargs):
    assert svc.screen_funding(**kwargs)["error"]["code"] == "INVALID_PARAMETER"


def test_screener_reports_empty_board(monkeypatch):
    _patch_board(monkeypatch, [], [])
    assert svc.screen_funding()["error"]["code"] == "NO_DATA"


def test_screener_reports_everything_filtered_out(monkeypatch):
    _patch_board(monkeypatch, [_premium("A", 0.001)], [_ticker("A", 1.0, volume=1.0)])
    out = svc.screen_funding(min_quote_volume_usd=1e9)
    assert out["error"]["code"] == "NO_DATA"
    assert out["error"]["perpetuals_seen"] == 1


def test_screener_surfaces_upstream_failure(monkeypatch):
    def boom():
        raise RuntimeError("binance down")
    monkeypatch.setattr(bn, "get_premium_index_all", boom)
    out = svc.screen_funding()
    assert out["error"]["code"] == "UPSTREAM_ERROR"
    assert out["error"]["retryable"] is True


# ── Squeeze scan: the stage-1 rule ────────────────────────────────────────────

def _patch_detail(monkeypatch, mapping):
    def fake(symbol, period):
        return symbol, mapping.get(symbol)
    monkeypatch.setattr(svc, "_detail", fake)


def test_shortlist_requires_funding_against_the_move(monkeypatch):
    # WITH: longs pay while price falls. WITHOUT: longs pay while price rises
    # — expensive, but the payers are winning, so it is not a cascade setup.
    _patch_board(
        monkeypatch,
        [_premium("WITH", 0.002), _premium("WITHOUT", 0.002)],
        [_ticker("WITH", -5.0), _ticker("WITHOUT", +5.0)],
    )
    _patch_detail(monkeypatch, {
        "WITH": {"crowd_long": 0.80, "crowd_ratio": 4.0, "top_long": 0.70,
                 "oi_change_pct": 10.0},
    })
    out = svc.scan_squeeze_risk(limit=5, candidates=5, min_quote_volume_usd=None)
    assert [g["symbol"] for g in out["ranked"]] == ["WITH"]
    assert out["shortlisted"] == 1


def test_negative_funding_against_a_rising_price_also_shortlists(monkeypatch):
    # Shorts paying while price rises: the mirror case.
    _patch_board(monkeypatch, [_premium("S", -0.002)], [_ticker("S", +5.0)])
    _patch_detail(monkeypatch, {
        "S": {"crowd_long": 0.30, "crowd_ratio": 0.43, "top_long": 0.40,
              "oi_change_pct": 10.0},
    })
    out = svc.scan_squeeze_risk(min_quote_volume_usd=None)
    assert out["ranked"][0]["squeeze_risk"].endswith("SHORT_SQUEEZE")


def test_board_with_no_pressure_says_so(monkeypatch):
    _patch_board(monkeypatch, [_premium("A", 0.002)], [_ticker("A", +5.0)])
    out = svc.scan_squeeze_risk(min_quote_volume_usd=None)
    assert out["ranked"] == []
    assert out["shortlisted"] == 0
    assert "precondition for a cascade is absent" in out["note"]


def test_grading_is_delegated_to_the_single_symbol_logic(monkeypatch):
    """The scan must not reimplement the verdict.

    If it did, a symbol could read ELEVATED in `positioning` and LOW in the
    scan. This asserts the shared function is what produces the grade.
    """
    calls = []
    real = ps._squeeze_risk

    def spy(*args):
        calls.append(args)
        return real(*args)

    monkeypatch.setattr(ps, "_squeeze_risk", spy)
    _patch_board(monkeypatch, [_premium("A", 0.002)], [_ticker("A", -5.0)])
    _patch_detail(monkeypatch, {
        "A": {"crowd_long": 0.80, "crowd_ratio": 4.0, "top_long": 0.70,
              "oi_change_pct": 10.0},
    })
    svc.scan_squeeze_risk(min_quote_volume_usd=None)
    assert calls, "_squeeze_risk from positioning_service was never called"


def test_low_grade_carries_an_explanation(monkeypatch):
    # A stressed market whose account split is not one-sided grades LOW; the
    # result has to say that LOW is not "nothing happening".
    _patch_board(monkeypatch, [_premium("A", 0.002)], [_ticker("A", -50.0)])
    _patch_detail(monkeypatch, {
        "A": {"crowd_long": 0.64, "crowd_ratio": 1.79, "top_long": 0.61,
              "oi_change_pct": 10.0},
    })
    out = svc.scan_squeeze_risk(min_quote_volume_usd=None)
    assert out["ranked"][0]["squeeze_risk"] == "LOW"
    assert "does NOT mean nothing is happening" in out["reading_low"]


def test_symbols_whose_detail_fails_are_listed(monkeypatch):
    _patch_board(
        monkeypatch,
        [_premium("OK", 0.002), _premium("BAD", 0.002)],
        [_ticker("OK", -5.0), _ticker("BAD", -5.0)],
    )
    _patch_detail(monkeypatch, {
        "OK": {"crowd_long": 0.80, "crowd_ratio": 4.0, "top_long": 0.70,
               "oi_change_pct": 10.0},
    })  # BAD returns None
    out = svc.scan_squeeze_risk(min_quote_volume_usd=None)
    assert out["unavailable"] == ["BAD"]
    assert [g["symbol"] for g in out["ranked"]] == ["OK"]


def test_elevated_rows_are_ranked_first(monkeypatch):
    _patch_board(
        monkeypatch,
        [_premium("ELEV", 0.002), _premium("CALM", 0.002)],
        [_ticker("ELEV", -5.0), _ticker("CALM", -5.0)],
    )
    _patch_detail(monkeypatch, {
        "ELEV": {"crowd_long": 0.85, "crowd_ratio": 5.7, "top_long": 0.70,
                 "oi_change_pct": 20.0},
        "CALM": {"crowd_long": 0.55, "crowd_ratio": 1.2, "top_long": 0.54,
                 "oi_change_pct": 0.0},
    })
    out = svc.scan_squeeze_risk(min_quote_volume_usd=None)
    assert out["ranked"][0]["symbol"] == "ELEV"
    # Symbols, not rows: `ranked` already holds the detail for anything it
    # shows, and repeating those rows here cost 26% of a live payload.
    assert out["elevated"] == ["ELEV"]
    assert out["elevated_beyond_limit"] is None


@pytest.mark.parametrize("kwargs", [
    {"limit": 0}, {"limit": 99}, {"candidates": 0}, {"candidates": 99},
])
def test_squeeze_scan_rejects_bad_params(kwargs):
    assert svc.scan_squeeze_risk(**kwargs)["error"]["code"] == "INVALID_PARAMETER"


def test_squeeze_scan_rejects_bad_period():
    out = svc.scan_squeeze_risk(period="7s")
    assert out["error"]["code"] == "INVALID_TIMEFRAME"


# ── The payload a person actually receives ────────────────────────────────────

def test_elevated_names_survive_the_limit_without_repeating_rows(monkeypatch):
    """`elevated` exists so a grade cannot hide past `limit`.

    It used to keep that promise by sending whole rows, which at limit=20
    meant 5,753 of 21,710 live characters were duplicates of rows already in
    `ranked`. Symbols keep the promise; `elevated_beyond_limit` says when a
    row was not included, so "not shown" never looks like "not elevated".
    """
    symbols = ["E1", "E2", "E3"]
    _patch_board(
        monkeypatch,
        [_premium(s, 0.002) for s in symbols],
        [_ticker(s, -5.0) for s in symbols],
    )
    _patch_detail(monkeypatch, {
        s: {"crowd_long": 0.85, "crowd_ratio": 5.7, "top_long": 0.70,
            "oi_change_pct": 20.0} for s in symbols
    })
    out = svc.scan_squeeze_risk(limit=1, min_quote_volume_usd=None)

    assert len(out["ranked"]) == 1, "limit must still bound the rows"
    assert sorted(out["elevated"]) == symbols, "no elevated name may vanish"
    assert all(isinstance(s, str) for s in out["elevated"])
    beyond = out["elevated_beyond_limit"]
    assert sorted(beyond["symbols"]) == sorted(symbols[1:])
    assert "limit=1" in beyond["note"], "it must say how to see the rest"


def test_nothing_is_beyond_the_limit_when_everything_fits(monkeypatch):
    _patch_board(monkeypatch, [_premium("E1", 0.002)], [_ticker("E1", -5.0)])
    _patch_detail(monkeypatch, {
        "E1": {"crowd_long": 0.85, "crowd_ratio": 5.7, "top_long": 0.70,
               "oi_change_pct": 20.0},
    })
    assert svc.scan_squeeze_risk(min_quote_volume_usd=None)[
        "elevated_beyond_limit"] is None


def test_the_scans_carry_their_rendering_pointer(monkeypatch):
    """Both are named in the server's PRESENTATION block, so both must
    carry the pointer that tells the assistant what not to bury."""
    _patch_board(monkeypatch, [_premium("X", 0.002)], [_ticker("X", -5.0)])
    _patch_detail(monkeypatch, {
        "X": {"crowd_long": 0.85, "crowd_ratio": 5.7, "top_long": 0.70,
              "oi_change_pct": 20.0},
    })
    squeeze = svc.scan_squeeze_risk(min_quote_volume_usd=None)
    assert "unavailable" in squeeze["display_hint"]
    funding = svc.screen_funding(min_quote_volume_usd=None)
    assert "interval_note" in funding["display_hint"]
