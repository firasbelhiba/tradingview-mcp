"""Unit tests for the positioning service and the Binance futures client.

Offline: every HTTP read is monkeypatched. What is pinned here is the
arithmetic (funding annualisation on a measured interval, OI notional, the
crowd/top divergence), the quadrant logic, the verdict thresholds, and the
partial-failure behaviour.
"""
import pytest

from tradingview_mcp.core.services import binance_futures_service as bn
from tradingview_mcp.core.services import positioning_service as svc

_PERP = "BTCUSDT"


@pytest.fixture(autouse=True)
def _clear_cache():
    bn._cache.clear()
    yield
    bn._cache.clear()


def _feeds(monkeypatch, *, funding=0.0001, mark=70000.0, oi=100000.0,
           oi_hist=None, fund_hist=None, crowd_long=0.60, top_long=0.60,
           taker=1.0, price_change=1.0, fail=()):
    """Patch every client read. Names in ``fail`` raise instead."""
    eight_h = 8 * 3_600_000

    def maybe(name, value):
        def inner(*a, **k):
            if name in fail:
                raise RuntimeError(f"{name} unavailable")
            return value
        return inner

    monkeypatch.setattr(bn, "perpetual_symbols", maybe("universe", [_PERP, "ETHUSDT"]))
    monkeypatch.setattr(bn, "get_premium_index", maybe("premium", {
        "markPrice": str(mark), "indexPrice": str(mark - 5),
        "lastFundingRate": str(funding), "nextFundingTime": 1794355200000,
    }))
    monkeypatch.setattr(bn, "get_open_interest", maybe("oi", {"openInterest": str(oi)}))
    monkeypatch.setattr(bn, "get_open_interest_history", maybe("oi_hist", oi_hist if oi_hist is not None else [
        {"sumOpenInterest": "100000.0"}, {"sumOpenInterest": "100000.0"},
    ]))
    monkeypatch.setattr(bn, "get_funding_history", maybe("fund_hist", fund_hist if fund_hist is not None else [
        {"fundingRate": str(funding), "fundingTime": eight_h * i} for i in range(1, 5)
    ]))
    monkeypatch.setattr(bn, "get_global_long_short", maybe("crowd", [{
        "longShortRatio": str(crowd_long / (1 - crowd_long)), "longAccount": str(crowd_long),
    }]))
    monkeypatch.setattr(bn, "get_top_trader_positions", maybe("tops", [{
        "longShortRatio": str(top_long / (1 - top_long)), "longAccount": str(top_long),
    }]))
    monkeypatch.setattr(bn, "get_taker_flow", maybe("taker", [{"buySellRatio": str(taker)}]))
    monkeypatch.setattr(bn, "get_ticker_24h", maybe("ticker", {
        "priceChangePercent": str(price_change), "quoteVolume": "1000000",
    }))


# ── Symbol handling ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("given,expected", [
    ("BTCUSDT", "BTCUSDT"),
    ("btcusdt", "BTCUSDT"),
    ("KUCOIN:BTCUSDT", "BTCUSDT"),
    ("BTC-USDT", "BTCUSDT"),
    ("BTC/USDT", "BTCUSDT"),
])
def test_perp_symbol_normalisation(given, expected):
    assert bn.perp_symbol_for(given) == expected


def test_spot_only_pair_is_reported_with_suggestions(monkeypatch):
    _feeds(monkeypatch)
    out = svc.analyze_positioning("BTCUSD")  # not in the patched universe
    assert out["error"]["code"] == "SYMBOL_NOT_FOUND"
    assert "perpetual" in out["error"]["message"].lower()
    assert out["error"]["perpetuals_available"] == 2


def test_unreachable_universe_is_retryable(monkeypatch):
    _feeds(monkeypatch, fail=("universe",))
    out = svc.analyze_positioning("BTCUSDT")
    assert out["error"]["code"] == "UPSTREAM_ERROR"
    assert out["error"]["retryable"] is True


# ── Funding arithmetic ────────────────────────────────────────────────────────

def test_funding_is_annualised_on_the_measured_interval(monkeypatch):
    # 0.01% per 8h -> 3 payments a day -> 0.03%/day -> 10.95%/yr
    eight_h = 8 * 3_600_000
    _feeds(monkeypatch, funding=0.0001, fund_hist=[
        {"fundingRate": "0.0001", "fundingTime": eight_h * i} for i in range(1, 5)
    ])
    f = svc.analyze_positioning("BTCUSDT")["funding"]
    assert f["interval_hours"] == 8.0
    assert f["pct_per_interval"] == pytest.approx(0.01, abs=1e-9)
    assert f["pct_per_day"] == pytest.approx(0.03, abs=1e-6)
    assert f["annualised_pct"] == pytest.approx(10.95, abs=0.05)
    assert f["payer"] == "longs pay shorts"
    assert "measured" in f["interval_source"]


def test_four_hour_funding_interval_doubles_the_annualised_cost(monkeypatch):
    # Some perps pay every 4h. Annualising those with a hardcoded 8 would
    # understate the cost by half, so the interval is measured.
    four_h = 4 * 3_600_000
    _feeds(monkeypatch, funding=0.0001, fund_hist=[
        {"fundingRate": "0.0001", "fundingTime": four_h * i} for i in range(1, 5)
    ])
    f = svc.analyze_positioning("BTCUSDT")["funding"]
    assert f["interval_hours"] == 4.0
    assert f["annualised_pct"] == pytest.approx(21.9, abs=0.1)


def test_funding_interval_snaps_to_a_real_schedule(monkeypatch):
    # A missed payment or clock skew must not yield a 7.4-hour "interval".
    skewed = [
        {"fundingRate": "0.0001", "fundingTime": 0},
        {"fundingRate": "0.0001", "fundingTime": int(7.4 * 3_600_000)},
        {"fundingRate": "0.0001", "fundingTime": int(15.1 * 3_600_000)},
    ]
    assert bn.infer_funding_interval_hours(skewed) == 8.0


def test_funding_interval_falls_back_when_history_is_missing(monkeypatch):
    _feeds(monkeypatch, funding=0.0001, fund_hist=[])
    f = svc.analyze_positioning("BTCUSDT")["funding"]
    assert f["interval_hours"] == 8.0
    assert "assumed" in f["interval_source"]


def test_negative_funding_means_shorts_pay(monkeypatch):
    _feeds(monkeypatch, funding=-0.0003)
    out = svc.analyze_positioning("BTCUSDT")
    assert out["funding"]["payer"] == "shorts pay longs"
    assert out["assessment"]["funding"] == "SHORTS_PAYING"


# ── Open interest ─────────────────────────────────────────────────────────────

def test_open_interest_notional_uses_the_mark_price(monkeypatch):
    _feeds(monkeypatch, oi=1000.0, mark=70000.0)
    oi = svc.analyze_positioning("BTCUSDT")["open_interest"]
    assert oi["base_units"] == 1000.0
    assert oi["notional_usd"] == 70_000_000.0


def test_open_interest_change_is_measured_across_the_window(monkeypatch):
    _feeds(monkeypatch, oi_hist=[
        {"sumOpenInterest": "100000"}, {"sumOpenInterest": "110000"},
    ])
    oi = svc.analyze_positioning("BTCUSDT")["open_interest"]
    assert oi["change_pct"] == 10.0
    assert oi["state"] == "BUILDING"


def test_open_interest_unwinding(monkeypatch):
    _feeds(monkeypatch, oi_hist=[
        {"sumOpenInterest": "100000"}, {"sumOpenInterest": "80000"},
    ])
    assert svc.analyze_positioning("BTCUSDT")["open_interest"]["state"] == "UNWINDING"


# ── The OI / price quadrant ───────────────────────────────────────────────────

@pytest.mark.parametrize("oi_delta,price_change,expected", [
    (10.0, 2.0, "new_longs"),
    (10.0, -2.0, "new_shorts"),
    (-10.0, 2.0, "short_covering"),
    (-10.0, -2.0, "long_liquidation"),
])
def test_oi_price_quadrant(monkeypatch, oi_delta, price_change, expected):
    start = 100000.0
    _feeds(
        monkeypatch,
        oi_hist=[{"sumOpenInterest": str(start)},
                 {"sumOpenInterest": str(start * (1 + oi_delta / 100))}],
        price_change=price_change,
    )
    out = svc.analyze_positioning("BTCUSDT")
    assert out["oi_price_read"]["state"] == expected


def test_quadrant_is_null_without_both_inputs():
    assert svc._oi_price_read(None, 1.0) is None
    assert svc._oi_price_read(1.0, None) is None


# ── Crowd versus top traders ──────────────────────────────────────────────────

def test_divergence_is_the_point_of_the_tool(monkeypatch):
    # Crowd 75% long, top traders 68% long -> crowd is 7 points more long.
    _feeds(monkeypatch, crowd_long=0.75, top_long=0.68)
    out = svc.analyze_positioning("BTCUSDT")
    assert out["crowd"]["long_pct"] == 75.0
    assert out["top_traders"]["long_pct"] == 68.0
    assert out["crowd_vs_top_divergence_pts"] == pytest.approx(7.0, abs=0.1)
    assert any("top traders are less long" in r for r in out["assessment"]["reasons"])


def test_short_and_long_percentages_sum_to_one_hundred(monkeypatch):
    _feeds(monkeypatch, crowd_long=0.62)
    crowd = svc.analyze_positioning("BTCUSDT")["crowd"]
    assert crowd["long_pct"] + crowd["short_pct"] == pytest.approx(100.0, abs=0.01)


def test_small_divergence_is_not_reported_as_a_reason(monkeypatch):
    _feeds(monkeypatch, crowd_long=0.60, top_long=0.58)
    out = svc.analyze_positioning("BTCUSDT")
    assert not any("top traders" in r for r in out["assessment"]["reasons"])


# ── Verdicts ──────────────────────────────────────────────────────────────────

def test_crowded_long_with_positive_funding_into_a_fall_flags_squeeze(monkeypatch):
    # The signal the price tools are blind to: crowded longs paying to hold
    # while price falls is a liquidation setup, not a contrarian long.
    _feeds(monkeypatch, crowd_long=0.78, funding=0.0002, price_change=-3.0)
    a = svc.analyze_positioning("BTCUSDT")["assessment"]
    assert a["crowding"] == "CROWDED_LONG"
    assert a["squeeze_risk"] == "ELEVATED_LONG_SQUEEZE"
    assert "cascade" in a["squeeze_reasoning"]


def test_crowded_long_with_rising_price_is_only_a_watch(monkeypatch):
    # Crowded, paying funding, and more long than the top traders — but price
    # is RISING, so these longs are not under pressure. The adverse move gates
    # the ELEVATED grade however many other factors pile up.
    _feeds(monkeypatch, crowd_long=0.78, funding=0.0002, price_change=+3.0)
    a = svc.analyze_positioning("BTCUSDT")["assessment"]
    assert a["squeeze_risk"] == "WATCH_LONG_SQUEEZE"
    assert "price is falling" not in a["squeeze_factors"]


def test_crowding_plus_falling_price_and_building_oi_is_elevated_without_rich_funding(monkeypatch):
    # The case live BTC exposed: 75% long, OI building, price down, funding
    # only mildly positive. An earlier version gated squeeze risk on funding
    # cost and graded this LOW while its own reasons said crowded-long,
    # OI-building, price-falling.
    _feeds(monkeypatch, crowd_long=0.755, top_long=0.685, funding=0.000076,
           price_change=-1.2,
           oi_hist=[{"sumOpenInterest": "100000"}, {"sumOpenInterest": "106500"}])
    a = svc.analyze_positioning("BTCUSDT")["assessment"]
    assert a["funding"] == "NEUTRAL", "funding must not be elevated for this case"
    assert a["squeeze_risk"] == "ELEVATED_LONG_SQUEEZE"
    assert "price is falling" in a["squeeze_factors"]
    assert "open interest is building into the move" in a["squeeze_factors"]


def test_squeeze_factors_are_listed_with_the_grade(monkeypatch):
    _feeds(monkeypatch, crowd_long=0.80, top_long=0.60, funding=0.0002,
           price_change=-2.0)
    a = svc.analyze_positioning("BTCUSDT")["assessment"]
    assert a["squeeze_factors"][0] == "the crowd is long"
    assert len(a["squeeze_factors"]) >= svc._SQUEEZE_ELEVATED_FACTORS


def test_crowded_short_with_negative_funding_flags_short_squeeze(monkeypatch):
    _feeds(monkeypatch, crowd_long=0.40, funding=-0.0003, price_change=2.0)
    a = svc.analyze_positioning("BTCUSDT")["assessment"]
    assert a["crowding"] == "CROWDED_SHORT"
    assert a["squeeze_risk"] == "ELEVATED_SHORT_SQUEEZE"


def test_balanced_positioning_reports_low_risk(monkeypatch):
    # top_long must match crowd_long here: any gap of 5 points or more is a
    # real finding and would correctly show up in reasons.
    _feeds(monkeypatch, crowd_long=0.55, top_long=0.55, funding=0.000001,
           price_change=0.2)
    a = svc.analyze_positioning("BTCUSDT")["assessment"]
    assert a["crowding"] == "BALANCED"
    assert a["squeeze_risk"] == "LOW"
    assert a["reasons"] == ["no threshold was crossed"]


def test_thresholds_are_reported_with_the_verdict(monkeypatch):
    _feeds(monkeypatch)
    th = svc.analyze_positioning("BTCUSDT")["thresholds"]
    assert th["crowded_long_ratio"] == svc._CROWDED_LONG_RATIO
    assert th["funding_rich_pct_per_interval"] == svc._FUNDING_RICH_PCT
    assert th["oi_shift_pct"] == svc._OI_SHIFT_PCT


# ── Partial failure ───────────────────────────────────────────────────────────

def test_one_dead_feed_does_not_sink_the_report(monkeypatch):
    _feeds(monkeypatch, fail=("taker", "tops"))
    out = svc.analyze_positioning("BTCUSDT")
    assert "error" not in out
    assert out["funding"]["pct_per_interval"] is not None
    assert out["taker_flow"]["buy_sell_ratio"] is None
    assert set(out["unavailable_feeds"]) == {"taker_flow", "top_trader_positions"}


def test_both_core_feeds_dead_is_an_error(monkeypatch):
    _feeds(monkeypatch, fail=("premium", "oi"))
    out = svc.analyze_positioning("BTCUSDT")
    assert out["error"]["code"] == "UPSTREAM_ERROR"
    assert out["error"]["retryable"] is True


def test_no_unavailable_key_when_everything_works(monkeypatch):
    _feeds(monkeypatch)
    assert "unavailable_feeds" not in svc.analyze_positioning("BTCUSDT")


# ── Parameters ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("bars", [0, 1, 501, 9999])
def test_history_bars_is_bounded(bars):
    out = svc.analyze_positioning("BTCUSDT", history_bars=bars)
    assert out["error"]["code"] == "INVALID_PARAMETER"


def test_unmappable_period_is_rejected():
    out = svc.analyze_positioning("BTCUSDT", period="7s")
    assert out["error"]["code"] == "INVALID_TIMEFRAME"
    assert "4h" in out["error"]["supported"]


@pytest.mark.parametrize("ours,theirs", [
    ("4h", "4h"), ("1D", "1d"), ("1d", "1d"), ("15m", "15m"), ("1h", "1h"),
])
def test_period_translation(ours, theirs):
    assert bn.binance_period_for(ours) == theirs


def test_period_translation_rejects_unknown():
    assert bn.binance_period_for("3s") is None
    assert bn.binance_period_for("") is None


def test_weekly_and_monthly_fall_back_to_daily():
    # Binance's data endpoints stop at 1d; sampling must not silently change
    # to something finer than the caller asked for.
    assert bn.binance_period_for("1W") == "1d"
    assert bn.binance_period_for("1M") == "1d"


# ── Client cache ──────────────────────────────────────────────────────────────

def test_fetch_is_cached_within_the_ttl(monkeypatch):
    calls = []

    class _Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return b'{"ok": 1}'

    def fake_urlopen(req, timeout=None):
        calls.append(req.full_url)
        return _Resp()

    monkeypatch.setattr(bn.urllib.request, "urlopen", fake_urlopen)
    bn._cache.clear()
    first = bn._fetch("/fapi/v1/test", {"symbol": "BTCUSDT"})
    second = bn._fetch("/fapi/v1/test", {"symbol": "BTCUSDT"})
    assert first == second == {"ok": 1}
    assert len(calls) == 1, "second identical read must come from cache"


def test_stale_cache_serves_when_upstream_fails(monkeypatch):
    key = ("binance", "/fapi/v1/test", "symbol=BTCUSDT")
    bn._cache.clear()
    bn._cache_put(key, {"stale": True})
    # Age the entry past the fresh TTL but inside the stale window.
    ts, value = bn._cache[key]
    bn._cache[key] = (ts - 300, value)

    monkeypatch.setenv("TRADINGVIEW_MCP_BINANCE_RETRIES", "0")

    def boom(req, timeout=None):
        raise OSError("network down")

    monkeypatch.setattr(bn.urllib.request, "urlopen", boom)
    # Positioning from minutes ago beats no answer at all.
    assert bn._fetch("/fapi/v1/test", {"symbol": "BTCUSDT"}) == {"stale": True}


def test_fetch_raises_when_there_is_nothing_cached(monkeypatch):
    bn._cache.clear()
    monkeypatch.setenv("TRADINGVIEW_MCP_BINANCE_RETRIES", "0")

    def boom(req, timeout=None):
        raise OSError("network down")

    monkeypatch.setattr(bn.urllib.request, "urlopen", boom)
    with pytest.raises(bn.BinanceUnavailable):
        bn._fetch("/fapi/v1/test", {"symbol": "NOPE"})
