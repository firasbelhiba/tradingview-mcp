"""Unit tests for the copy-trading lead-trader tools. Offline.

The emphasis is on the risk metrics this module derives, because the
leaderboard does not publish them and a wrong one would be worse than none.
"""
import pytest

from tradingview_mcp.core.services import copytrading_service as svc


@pytest.fixture(autouse=True)
def _clear_cache():
    svc._cache.clear()
    yield
    svc._cache.clear()


def _trader(nick="T", code="C1", pnl_ratio=1.0, win=0.6, days=400,
            followers=100, aum=50000.0):
    return {
        "nickName": nick, "uniqueCode": code, "pnlRatio": str(pnl_ratio),
        "winRatio": str(win), "leadDays": str(days),
        "copyTraderNum": str(followers), "accCopyTraderNum": str(followers * 2),
        "aum": str(aum), "pnl": "1234.5", "copyState": "1",
        "portLink": "https://example.invalid/t",
    }


def _closed(lever="2E+1", ratio=0.1, inst="BTC-USDT-SWAP"):
    return {"lever": lever, "pnlRatio": str(ratio), "instId": inst,
            "posSide": "long", "pnl": "100"}


def _patch(monkeypatch, routes):
    """Route by URL fragment; values may be payloads or exceptions."""
    def fake_get(path):
        for fragment, payload in routes.items():
            if fragment in path:
                if isinstance(payload, Exception):
                    raise payload
                return payload
        raise RuntimeError(f"unrouted path {path}")
    monkeypatch.setattr(svc, "_get", fake_get)


def _ranks(traders):
    return {"code": "0", "data": [{"ranks": traders}]}


def _plain(rows):
    return {"code": "0", "data": rows}


# ── Field parsing ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ("2E+1", 20.0), ("3E+1", 30.0), ("5E+1", 50.0),
    ("10", 10.0), ("7.5", 7.5), ("", None), (None, None), ("abc", None),
])
def test_leverage_parses_scientific_notation(raw, expected):
    """OKX sends leverage as "2E+1" for 20x.

    int() raises on it and a string comparison sorts it wrongly, so the
    parser has to go through float().
    """
    assert svc._f(raw) == expected


# ── Drawdown ──────────────────────────────────────────────────────────────────

def test_max_drawdown_is_peak_to_trough():
    # +100 then -60 then +20: peak 100, trough 40 -> -60.
    assert svc._max_drawdown([100.0, -60.0, 20.0]) == pytest.approx(-60.0)


def test_max_drawdown_is_zero_for_a_rising_curve():
    assert svc._max_drawdown([10.0, 20.0, 30.0]) == pytest.approx(0.0)


def test_max_drawdown_handles_an_early_loss():
    assert svc._max_drawdown([-50.0, 10.0]) == pytest.approx(-50.0)


def test_max_drawdown_of_nothing_is_none():
    assert svc._max_drawdown([]) is None


def test_weekly_series_is_reversed_into_chronological_order():
    # OKX returns newest first. A drawdown computed on that order measures a
    # curve running backwards through time.
    newest_first = [{"pnl": "20"}, {"pnl": "-60"}, {"pnl": "100"}]
    stats = svc._weekly_stats(newest_first)
    assert stats["max_drawdown_usd"] == pytest.approx(-60.0)


# ── Trade statistics ──────────────────────────────────────────────────────────

def test_trade_stats_measures_leverage_and_worst_trade():
    history = [
        _closed(lever="2E+1", ratio=0.5),
        _closed(lever="5E+1", ratio=-2.6),
        _closed(lever="1E+1", ratio=0.1),
    ]
    stats = svc._trade_stats(history)
    assert stats["avg_leverage"] == pytest.approx(26.7, abs=0.1)
    assert stats["max_leverage"] == 50
    assert stats["worst_trade_pct"] == -260.0
    assert stats["best_trade_pct"] == 50.0
    assert stats["measured_win_rate_pct"] == pytest.approx(66.7, abs=0.1)
    assert stats["closed_trades_sampled"] == 3


def test_most_traded_instruments_are_counted():
    history = [_closed(inst="BTC-USDT-SWAP"), _closed(inst="BTC-USDT-SWAP"),
               _closed(inst="ETH-USDT-SWAP")]
    assert svc._trade_stats(history)["most_traded"][0] == ("BTC-USDT-SWAP", 2)


def test_empty_stats_blocks_keep_the_same_keys():
    """The regression the first live run raised.

    When a fetch fails the block used to come back with a different key set
    and consumers hit KeyError. Shape must not depend on success.
    """
    populated = set(svc._trade_stats([_closed()]))
    assert set(svc._trade_stats([])) == populated
    populated_weekly = set(svc._weekly_stats([{"pnl": "1"}]))
    assert set(svc._weekly_stats([])) == populated_weekly


# ── Risk flags ────────────────────────────────────────────────────────────────

def test_extreme_leverage_is_flagged():
    flags = svc._risk_flags(_trader(days=400), svc._trade_stats(
        [_closed(lever="3E+1")]), {"weeks_sampled": 12, "losing_weeks": 1})
    assert any("extreme leverage" in f for f in flags)


def test_high_but_not_extreme_leverage_is_flagged_differently():
    flags = svc._risk_flags(_trader(days=400), svc._trade_stats(
        [_closed(lever="1.3E+1")]), {"weeks_sampled": 12, "losing_weeks": 1})
    assert any("high leverage" in f for f in flags)
    assert not any("extreme" in f for f in flags)


def test_low_leverage_is_not_flagged():
    flags = svc._risk_flags(_trader(days=400, win=0.6), svc._trade_stats(
        [_closed(lever="3", ratio=0.1)]), {"weeks_sampled": 12, "losing_weeks": 1})
    assert not any("leverage" in f for f in flags)


def test_catastrophic_single_trade_is_flagged():
    flags = svc._risk_flags(_trader(days=400), svc._trade_stats(
        [_closed(lever="2", ratio=-0.93)]), {"weeks_sampled": 12, "losing_weeks": 1})
    assert any("lost 93%" in f for f in flags)


def test_short_track_record_is_flagged():
    flags = svc._risk_flags(_trader(days=66), svc._trade_stats([_closed(lever="2")]),
                            {"weeks_sampled": 12, "losing_weeks": 1})
    assert any("short track record" in f for f in flags)


def test_long_track_record_is_not_flagged():
    flags = svc._risk_flags(_trader(days=1152), svc._trade_stats([_closed(lever="2")]),
                            {"weeks_sampled": 12, "losing_weeks": 1})
    assert not any("track record" in f for f in flags)


def test_advertised_versus_measured_win_rate_gap_is_flagged():
    # Advertised 62%, measured 100% over the sample: worth surfacing either way.
    flags = svc._risk_flags(
        _trader(days=400, win=0.62),
        svc._trade_stats([_closed(lever="2", ratio=0.1) for _ in range(5)]),
        {"weeks_sampled": 12, "losing_weeks": 1},
    )
    assert any("advertised win rate" in f for f in flags)


def test_mostly_losing_weeks_is_flagged():
    flags = svc._risk_flags(_trader(days=400), svc._trade_stats([_closed(lever="2")]),
                            {"weeks_sampled": 12, "losing_weeks": 7})
    assert any("losing weeks" in f for f in flags)


# ── Listing ───────────────────────────────────────────────────────────────────

def test_default_sort_is_track_record_not_return(monkeypatch):
    _patch(monkeypatch, {
        "public-lead-traders": _ranks([
            _trader("Young", "A", pnl_ratio=10.0, days=60),
            _trader("Veteran", "B", pnl_ratio=1.0, days=1200),
        ]),
        "public-subpositions-history": _plain([_closed(lever="2")]),
        "public-weekly-pnl": _plain([{"pnl": "10"}]),
    })
    out = svc.list_lead_traders()
    assert [t["nickname"] for t in out["traders"]] == ["Veteran", "Young"]
    assert out["sort"] == "track_record"


def test_sort_by_return(monkeypatch):
    _patch(monkeypatch, {
        "public-lead-traders": _ranks([
            _trader("Young", "A", pnl_ratio=10.0, days=60),
            _trader("Veteran", "B", pnl_ratio=1.0, days=1200),
        ]),
        "public-subpositions-history": _plain([_closed(lever="2")]),
        "public-weekly-pnl": _plain([{"pnl": "10"}]),
    })
    assert [t["nickname"] for t in svc.list_lead_traders(sort="return")["traders"]] \
        == ["Young", "Veteran"]


def test_sort_by_lowest_leverage(monkeypatch):
    def fake_get(path):
        if "public-lead-traders" in path:
            return _ranks([_trader("Hi", "HI"), _trader("Lo", "LO")])
        if "public-weekly-pnl" in path:
            return _plain([{"pnl": "10"}])
        if "uniqueCode=HI" in path:
            return _plain([_closed(lever="5E+1")])
        return _plain([_closed(lever="3")])
    monkeypatch.setattr(svc, "_get", fake_get)
    out = svc.list_lead_traders(sort="lowest_leverage")
    assert [t["nickname"] for t in out["traders"]] == ["Lo", "Hi"]


def test_leverage_filter_excludes_and_counts(monkeypatch):
    def fake_get(path):
        if "public-lead-traders" in path:
            return _ranks([_trader("Hi", "HI"), _trader("Lo", "LO")])
        if "public-weekly-pnl" in path:
            return _plain([{"pnl": "10"}])
        if "uniqueCode=HI" in path:
            return _plain([_closed(lever="5E+1")])
        return _plain([_closed(lever="3")])
    monkeypatch.setattr(svc, "_get", fake_get)
    out = svc.list_lead_traders(max_avg_leverage=10.0)
    assert [t["nickname"] for t in out["traders"]] == ["Lo"]
    assert out["excluded_by_leverage_filter"] == 1


def test_min_lead_days_filter(monkeypatch):
    _patch(monkeypatch, {
        "public-lead-traders": _ranks([
            _trader("Young", "A", days=60), _trader("Veteran", "B", days=1200),
        ]),
        "public-subpositions-history": _plain([_closed(lever="2")]),
        "public-weekly-pnl": _plain([{"pnl": "10"}]),
    })
    out = svc.list_lead_traders(min_lead_days=365)
    assert [t["nickname"] for t in out["traders"]] == ["Veteran"]


def test_min_lead_days_excluding_everyone_is_no_data(monkeypatch):
    _patch(monkeypatch, {"public-lead-traders": _ranks([_trader(days=60)])})
    out = svc.list_lead_traders(min_lead_days=5000)
    assert out["error"]["code"] == "NO_DATA"


def test_return_per_day_normalises_the_track_record(monkeypatch):
    _patch(monkeypatch, {
        "public-lead-traders": _ranks([_trader("A", "A", pnl_ratio=6.0, days=60)]),
        "public-subpositions-history": _plain([_closed(lever="2")]),
        "public-weekly-pnl": _plain([{"pnl": "10"}]),
    })
    out = svc.list_lead_traders()
    # +600% over 60 days -> 10%/day
    assert out["traders"][0]["return_pct_per_day"] == pytest.approx(10.0, abs=0.01)


def test_failed_enrichment_still_returns_the_trader(monkeypatch):
    _patch(monkeypatch, {
        "public-lead-traders": _ranks([_trader("Spot", "S")]),
        "public-subpositions-history": RuntimeError(
            "OKX error 59269: This futures lead trader doesn't exist"),
        "public-weekly-pnl": _plain([{"pnl": "10"}]),
    })
    out = svc.list_lead_traders()
    row = out["traders"][0]
    assert row["measured"]["avg_leverage"] is None
    # 59269 is OKX withholding the data, which is a different fact from a
    # request that failed, and is reported as one.
    assert row["trade_history_state"] == "hidden"
    assert any("hidden by OKX" in m for m in row["incomplete"])


def test_result_states_its_own_limits(monkeypatch):
    _patch(monkeypatch, {
        "public-lead-traders": _ranks([_trader()]),
        "public-subpositions-history": _plain([_closed()]),
        "public-weekly-pnl": _plain([{"pnl": "1"}]),
    })
    out = svc.list_lead_traders()
    assert any("survivors" in s for s in out["what_this_cannot_tell_you"])
    assert "not place, follow or size" in out["read_only_note"]


@pytest.mark.parametrize("kwargs", [
    {"sort": "vibes"}, {"limit": 0}, {"limit": 99},
    {"instrument_type": "FUTURES"}, {"history_limit": 1}, {"history_limit": 500},
])
def test_listing_rejects_bad_params(kwargs):
    assert svc.list_lead_traders(**kwargs)["error"]["code"] == "INVALID_PARAMETER"


def test_listing_surfaces_upstream_failure(monkeypatch):
    _patch(monkeypatch, {"public-lead-traders": RuntimeError("HTTP 503")})
    out = svc.list_lead_traders()
    assert out["error"]["code"] == "UPSTREAM_ERROR"
    assert out["error"]["retryable"] is True


def test_empty_leaderboard_is_no_data(monkeypatch):
    _patch(monkeypatch, {"public-lead-traders": _ranks([])})
    assert svc.list_lead_traders()["error"]["code"] == "NO_DATA"


# ── Single profile ────────────────────────────────────────────────────────────

def test_profile_includes_open_positions_with_leverage(monkeypatch):
    _patch(monkeypatch, {
        "public-lead-traders": _ranks([_trader("A", "C1")]),
        "public-subpositions-history": _plain([_closed(lever="2E+1")]),
        "public-weekly-pnl": _plain([{"pnl": "10"}]),
        "public-current-subpositions": _plain([{
            "instId": "BTC-USDT-SWAP", "posSide": "long", "lever": "5E+1",
            "margin": "1000", "openAvgPx": "70000", "markPx": "71000",
            "mgnMode": "cross", "openTime": "1700000000000",
        }]),
        "public-preference-currency": _plain([{"ccy": "BTC", "ratio": "0.6"}]),
    })
    out = svc.get_lead_trader("C1")
    assert out["open_position_count"] == 1
    assert out["open_positions"][0]["leverage"] == 50.0
    assert out["open_max_leverage"] == 50
    assert out["instrument_preferences"][0] == {"currency": "BTC", "share_pct": 60.0}


def test_profile_for_a_code_not_on_the_leaderboard(monkeypatch):
    _patch(monkeypatch, {
        "public-lead-traders": _ranks([_trader("Other", "OTHER")]),
        "public-subpositions-history": _plain([_closed()]),
        "public-weekly-pnl": _plain([{"pnl": "1"}]),
        "public-current-subpositions": _plain([]),
        "public-preference-currency": _plain([]),
    })
    out = svc.get_lead_trader("UNLISTED")
    assert out["unique_code"] == "UNLISTED"
    assert "not on the leaderboard page" in out["note"]


def test_profile_requires_a_code():
    assert svc.get_lead_trader("")["error"]["code"] == "INVALID_PARAMETER"
    assert svc.get_lead_trader("   ")["error"]["code"] == "INVALID_PARAMETER"


def test_profile_rejects_bad_instrument_type():
    assert svc.get_lead_trader("C1", instrument_type="X")["error"]["code"] == \
        "INVALID_PARAMETER"


def test_profile_records_unavailable_positions(monkeypatch):
    _patch(monkeypatch, {
        "public-lead-traders": _ranks([_trader("A", "C1")]),
        "public-subpositions-history": _plain([_closed()]),
        "public-weekly-pnl": _plain([{"pnl": "1"}]),
        "public-current-subpositions": RuntimeError("rate limited"),
        "public-preference-currency": _plain([]),
    })
    out = svc.get_lead_trader("C1")
    assert out["open_positions"] is None
    assert any("open positions unavailable" in m for m in out["incomplete"])


# ── A measurement that failed must not look like a good reading ───────────────

def test_a_trader_whose_history_is_missing_is_named(monkeypatch):
    """The live run that found this returned 2 of 10 traders unmeasured.

    Their avg_leverage, worst_trade_pct and measured_win_rate_pct were all
    null, nothing at the top level said so, and a blank leverage sitting
    beside measured ones reads as a LOW one - the opposite of what this tool
    exists to show. It is the same distinction the alert layer keeps between
    "not evaluated" and "condition not met".
    """
    _patch(monkeypatch, {
        "public-lead-traders": _ranks([
            _trader(nick="Measured", code="M1", days=500),
            _trader(nick="Blank", code="B1", days=400),
        ]),
        "public-subpositions-history": _plain([]),      # no history for anyone
        "public-weekly-pnl": _plain([{"pnl": "10"}]),
    })
    out = svc.list_lead_traders(limit=2)
    unmeasured = out["unmeasured"]
    assert sorted(unmeasured["traders"]) == ["Blank", "Measured"]
    assert unmeasured["count"] == 2
    assert unmeasured["of_returned"] == 2
    assert "MISSING, not low" in unmeasured["meaning"]


def test_the_flag_is_absent_when_everyone_was_measured(monkeypatch):
    """A field that is always present stops being a warning."""
    _patch(monkeypatch, {
        "public-lead-traders": _ranks([_trader(nick="Measured", code="M1")]),
        "public-subpositions-history": _plain([_closed(lever="2E+1")]),
        "public-weekly-pnl": _plain([{"pnl": "10"}]),
    })
    out = svc.list_lead_traders(limit=1)
    assert out["unmeasured"] is None
    assert out["traders"][0]["measured"]["avg_leverage"] == 20.0


def test_only_the_returned_traders_are_counted(monkeypatch):
    """`unmeasured` describes the rows in hand, not the whole candidate set."""
    _patch(monkeypatch, {
        "public-lead-traders": _ranks([
            _trader(nick="A", code="A1", days=900),
            _trader(nick="B", code="B1", days=800),
            _trader(nick="C", code="C1", days=700),
        ]),
        "public-subpositions-history": _plain([]),
        "public-weekly-pnl": _plain([{"pnl": "10"}]),
    })
    out = svc.list_lead_traders(limit=1)
    assert out["unmeasured"]["of_returned"] == 1
    assert out["unmeasured"]["count"] == 1


def test_the_list_carries_its_rendering_pointer(monkeypatch):
    _patch(monkeypatch, {
        "public-lead-traders": _ranks([_trader()]),
        "public-subpositions-history": _plain([_closed()]),
        "public-weekly-pnl": _plain([{"pnl": "10"}]),
    })
    hint = svc.list_lead_traders(limit=1)["display_hint"]
    assert "advertised next to measured" in hint
    assert "unmeasured" in hint


def test_one_traders_profile_says_so_too(monkeypatch):
    """The same silence, one trader at a time."""
    _patch(monkeypatch, {
        "public-lead-traders": _ranks([_trader(nick="Blank", code="B1")]),
        "public-subpositions-history": _plain([]),
        "public-weekly-pnl": _plain([{"pnl": "10"}]),
        "public-current-subpositions": _plain([]),
        "public-preference-currency": _plain([]),
    })
    out = svc.get_lead_trader("B1")
    assert out["measured"]["closed_trades_sampled"] == 0
    assert any("missing, not low" in w for w in out["incomplete"])


# ── Silence is not an empty book ──────────────────────────────────────────────

def test_a_failed_position_lookup_is_not_reported_as_zero(monkeypatch):
    """Found in use: OKX returns nothing for a code that is not a futures
    lead trader, and the profile still said "0 open positions". The note
    was there, but anything reading the number saw a trader holding
    nothing. The number itself has to refuse to answer.
    """
    _patch(monkeypatch, {
        "public-lead-traders": _ranks([_trader(nick="Plym", code="P1")]),
        "public-subpositions-history": _plain([_closed()]),
        "public-weekly-pnl": _plain([{"pnl": "10"}]),
        "public-current-subpositions": RuntimeError("OKX error 51000: not found"),
        "public-preference-currency": _plain([]),
    })
    out = svc.get_lead_trader("P1")
    assert out["open_position_count"] is None, "0 would read as 'holds nothing'"
    assert out["open_positions_state"] == "unknown"
    assert "NOT 'no open trades'" in out["open_positions_note"]
    assert any("open positions unavailable" in w for w in out["incomplete"])


def test_a_genuine_empty_book_says_it_was_measured(monkeypatch):
    """The other half: a real zero must still be usable as a real zero."""
    _patch(monkeypatch, {
        "public-lead-traders": _ranks([_trader(nick="Flat", code="F1")]),
        "public-subpositions-history": _plain([_closed()]),
        "public-weekly-pnl": _plain([{"pnl": "10"}]),
        "public-current-subpositions": _plain([]),
        "public-preference-currency": _plain([]),
    })
    out = svc.get_lead_trader("F1")
    assert out["open_position_count"] == 0
    assert out["open_positions_state"] == "measured"
    assert "holding nothing right now" in out["open_positions_note"]


def test_positions_that_came_back_are_counted_normally(monkeypatch):
    _patch(monkeypatch, {
        "public-lead-traders": _ranks([_trader(nick="Live", code="L1")]),
        "public-subpositions-history": _plain([_closed()]),
        "public-weekly-pnl": _plain([{"pnl": "10"}]),
        "public-current-subpositions": _plain([
            {"instId": "BTC-USDT-SWAP", "posSide": "long", "lever": "3",
             "margin": "100", "openAvgPx": "70000", "markPx": "71000"},
        ]),
        "public-preference-currency": _plain([]),
    })
    out = svc.get_lead_trader("L1")
    assert out["open_position_count"] == 1
    assert out["open_positions_state"] == "measured"
    assert out["open_max_leverage"] == 3
    assert "open_positions_note" not in out, "nothing to explain when it worked"


def test_a_failed_preference_lookup_is_flagged_rather_than_empty(monkeypatch):
    """The same silence, one field over: an exception was swallowed whole."""
    _patch(monkeypatch, {
        "public-lead-traders": _ranks([_trader(nick="Pref", code="R1")]),
        "public-subpositions-history": _plain([_closed()]),
        "public-weekly-pnl": _plain([{"pnl": "10"}]),
        "public-current-subpositions": _plain([]),
        "public-preference-currency": RuntimeError("timed out"),
    })
    out = svc.get_lead_trader("R1")
    assert out["instrument_preferences"] is None
    assert any("preferences unavailable" in w for w in out["incomplete"])


# ── Hidden is not unknown, and seats are not copyState ────────────────────────
#
# All three found by probing OKX directly after a profile said a trader on
# the leaderboard "doesn't exist" and "isn't accepting followers".

_HIDDEN = RuntimeError("OKX error 59269: This futures lead trader doesn't exist")


def test_a_hidden_trader_is_told_apart_from_a_failed_request(monkeypatch):
    """3 of the top 20 answered 59269 on both position endpoints while their
    aggregate endpoints answered normally. A retry cannot fix that, so it
    must not be reported the way a timeout is."""
    _patch(monkeypatch, {
        "public-lead-traders": _ranks([_trader("Hidden", "H1")]),
        "public-subpositions-history": _HIDDEN,
        "public-weekly-pnl": _plain([{"pnl": "10"}]),
    })
    out = svc.list_lead_traders(limit=1)
    row = out["traders"][0]
    assert row["trade_history_state"] == "hidden"
    assert "cannot be verified" in row["hidden_note"]
    assert any("hidden" in f for f in row["risk_flags"]), (
        "unverifiable leverage is itself a risk worth flagging"
    )
    assert out["unmeasured"]["hidden_by_okx"] == ["Hidden"]


def test_an_ordinary_failure_stays_unknown(monkeypatch):
    _patch(monkeypatch, {
        "public-lead-traders": _ranks([_trader("Flaky", "F1")]),
        "public-subpositions-history": RuntimeError("timed out"),
        "public-weekly-pnl": _plain([{"pnl": "10"}]),
    })
    out = svc.list_lead_traders(limit=1)
    row = out["traders"][0]
    assert row["trade_history_state"] == "unknown"
    assert "hidden_note" not in row
    assert any("trade history unavailable" in m for m in row["incomplete"])
    assert out["unmeasured"]["hidden_by_okx"] is None


def test_a_measured_trader_says_so(monkeypatch):
    _patch(monkeypatch, {
        "public-lead-traders": _ranks([_trader("Open", "O1")]),
        "public-subpositions-history": _plain([_closed()]),
        "public-weekly-pnl": _plain([{"pnl": "10"}]),
    })
    assert svc.list_lead_traders(limit=1)["traders"][0][
        "trade_history_state"] == "measured"


def _seats(now, cap, copy_state="0"):
    row = _trader("Seats", "S1")
    row["copyTraderNum"] = str(now)
    row["maxCopyTraderNum"] = str(cap)
    row["copyState"] = copy_state
    return row


@pytest.mark.parametrize("now,cap,expected", [
    (116, 300, True), (600, 600, False), (0, 300, True),
])
def test_accepting_followers_is_read_from_the_seats(monkeypatch, now, cap, expected):
    """copyState was "0" for all 20 traders probed, including 18 with open
    seats: it is the CALLER's copy relationship, and an anonymous caller
    copies nobody. Read as "accepting followers" it said no trader on OKX
    accepts anyone."""
    _patch(monkeypatch, {
        "public-lead-traders": _ranks([_seats(now, cap)]),
        "public-subpositions-history": _plain([_closed()]),
        "public-weekly-pnl": _plain([{"pnl": "10"}]),
    })
    adv = svc.list_lead_traders(limit=1)["traders"][0]["advertised"]
    assert adv["accepting_followers"] is expected
    assert adv["max_followers"] == cap


def test_copy_state_no_longer_decides_anything(monkeypatch):
    for state in ("0", "1"):
        _patch(monkeypatch, {
            "public-lead-traders": _ranks([_seats(10, 300, copy_state=state)]),
            "public-subpositions-history": _plain([_closed()]),
            "public-weekly-pnl": _plain([{"pnl": "10"}]),
        })
        svc._cache.clear()
        assert svc.list_lead_traders(limit=1)["traders"][0]["advertised"][
            "accepting_followers"] is True


def test_seats_unknown_is_none_not_false(monkeypatch):
    _patch(monkeypatch, {
        "public-lead-traders": _ranks([_trader("NoCap", "N1")]),
        "public-subpositions-history": _plain([_closed()]),
        "public-weekly-pnl": _plain([{"pnl": "10"}]),
    })
    assert svc.list_lead_traders(limit=1)["traders"][0]["advertised"][
        "accepting_followers"] is None


def test_okx_published_stats_survive_a_hidden_history(monkeypatch):
    """The aggregate endpoint answers when the position ones will not, and
    it was never being called."""
    _patch(monkeypatch, {
        "public-lead-traders": _ranks([_trader("Hidden", "H1")]),
        "public-subpositions-history": _HIDDEN,
        "public-weekly-pnl": _plain([{"pnl": "10"}]),
        "public-stats": _plain([{
            "winRatio": "0.7372", "profitDays": "115", "lossDays": "41",
            "avgSubPosNotional": "1135.945", "investAmt": "8065.77",
            "curCopyTraderPnl": "11092.82",
        }]),
    })
    stats = svc.list_lead_traders(limit=1)["traders"][0]["okx_published_stats"]
    assert stats["win_rate_pct"] == 73.7
    assert (stats["profit_days"], stats["loss_days"]) == (115, 41)
    assert stats["avg_position_notional"] == pytest.approx(1135.945, abs=0.01)
    assert "not measured" in stats["source"], "published is not verified"


def test_missing_published_stats_are_none(monkeypatch):
    _patch(monkeypatch, {
        "public-lead-traders": _ranks([_trader("T", "T1")]),
        "public-subpositions-history": _plain([_closed()]),
        "public-weekly-pnl": _plain([{"pnl": "10"}]),
    })
    assert svc.list_lead_traders(limit=1)["traders"][0][
        "okx_published_stats"] is None


def test_a_hidden_book_says_a_retry_will_not_help(monkeypatch):
    _patch(monkeypatch, {
        "public-lead-traders": _ranks([_trader("Hidden", "H1")]),
        "public-subpositions-history": _HIDDEN,
        "public-weekly-pnl": _plain([{"pnl": "10"}]),
        "public-current-subpositions": _HIDDEN,
        "public-preference-currency": _plain([]),
    })
    out = svc.get_lead_trader("H1")
    assert out["open_positions_state"] == "hidden"
    assert out["open_position_count"] is None
    assert "Retrying will not change it" in out["open_positions_note"]


def test_a_failed_book_says_a_retry_might(monkeypatch):
    _patch(monkeypatch, {
        "public-lead-traders": _ranks([_trader("Flaky", "F1")]),
        "public-subpositions-history": _plain([_closed()]),
        "public-weekly-pnl": _plain([{"pnl": "10"}]),
        "public-current-subpositions": RuntimeError("timed out"),
        "public-preference-currency": _plain([]),
    })
    out = svc.get_lead_trader("F1")
    assert out["open_positions_state"] == "unknown"
    assert "may work on a retry" in out["open_positions_note"]
