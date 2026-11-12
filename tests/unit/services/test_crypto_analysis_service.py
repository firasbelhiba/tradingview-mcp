"""Unit tests for the crypto analysis service.

Every test here is offline: upstream fetches are monkeypatched so the maths
and the shapes are pinned without touching Yahoo or TradingView.
"""
import pytest

from tradingview_mcp.core.services import crypto_analysis_service as svc


# ── Symbol conversion ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("pair,expected", [
    ("BTCUSDT", "BTC-USD"),
    ("ethusdt", "ETH-USD"),
    ("SOLUSDC", "SOL-USD"),
    ("ADABUSD", "ADA-USD"),
    ("XRPFDUSD", "XRP-USD"),
    ("DOGEUSD", "DOGE-USD"),
    ("SOLBTC", "SOL-BTC"),
    ("ETHEUR", "ETH-EUR"),
    ("KUCOIN:BTCUSDT", "BTC-USD"),
    # Already-Yahoo and index forms pass through untouched.
    ("BTC-USD", "BTC-USD"),
    ("^GSPC", "^GSPC"),
    ("GC=F", "GC=F"),
    ("AAPL", "AAPL"),
    ("", ""),
    # The Yahoo ticker BTC is an ETF. A coin named bare becomes its pair
    # first, so the fallback charts the coin.
    ("BTC", "BTC-USD"),
    ("sol", "SOL-USD"),
])
def test_yahoo_crypto_symbol(pair, expected):
    assert svc.yahoo_crypto_symbol(pair) == expected


def test_stablecoin_quotes_all_map_to_usd():
    # Yahoo has no USDT/USDC/BUSD markets — every dollar stablecoin pair has
    # to resolve to the plain USD pair or the candle fetch 404s.
    for quote in ("USDT", "USDC", "BUSD", "FDUSD", "TUSD"):
        assert svc.yahoo_crypto_symbol(f"LINK{quote}") == "LINK-USD"


# ── Resampling ────────────────────────────────────────────────────────────────

def _hourly(day: str, hours, base=100.0):
    return [
        {
            "date": f"{day} {h:02d}:00",
            "open": base + h,
            "high": base + h + 2,
            "low": base + h - 1,
            "close": base + h + 1,
            "volume": 10,
        }
        for h in hours
    ]


def test_resample_aggregates_ohlcv_correctly():
    candles = _hourly("2026-11-10", range(4))
    out = svc._resample(candles, 4)
    assert len(out) == 1
    bar = out[0]
    assert bar["open"] == candles[0]["open"]
    assert bar["close"] == candles[-1]["close"]
    assert bar["high"] == max(c["high"] for c in candles)
    assert bar["low"] == min(c["low"] for c in candles)
    assert bar["volume"] == 40
    assert bar["bars"] == 4


def test_resample_buckets_align_to_utc_not_to_array_position():
    # Start at 02:00: the first bucket must close at 04:00 (a 2-bar partial),
    # not run 02:00-06:00. Otherwise the same symbol yields different 4h
    # candles depending on when the request ran.
    candles = _hourly("2026-11-10", range(2, 10))
    out = svc._resample(candles, 4)
    assert [b["bars"] for b in out] == [2, 4, 2]
    assert [b["complete"] for b in out] == [False, True, False]


def test_resample_collapses_yahoos_live_partial_candle():
    # Yahoo appends an in-progress bar stamped with the request time, not the
    # top of the hour: a "19:26" candle sitting next to the finished "19:00"
    # one. Both are hour 19, so bucketing naively produced a FIVE-bar 4h
    # candle and summed that hour's volume twice.
    candles = _hourly("2026-11-10", range(16, 20))
    live = dict(candles[-1])
    live["date"] = "2026-11-10 19:26"
    live["close"] = 999.0
    live["high"] = 1000.0
    candles.append(live)

    out = svc._resample(candles, 4)
    assert len(out) == 1
    bar = out[0]
    assert bar["bars"] == 4, "an hour must count once however many ticks Yahoo sends"
    assert bar["volume"] == 40, "the duplicated hour's volume must not be counted twice"
    # The live tick is the freshest price, so it wins the hour.
    assert bar["close"] == 999.0
    assert bar["high"] == 1000.0


def test_resample_marks_incomplete_buckets():
    partial = svc._resample(_hourly("2026-11-10", range(16, 18)), 4)[0]
    assert partial["bars"] == 2 and partial["complete"] is False
    whole = svc._resample(_hourly("2026-11-10", range(16, 20)), 4)[0]
    assert whole["bars"] == 4 and whole["complete"] is True


def test_resample_stamps_bars_at_the_bucket_boundary():
    # The bar's own label must be the boundary, not whichever hour happened to
    # arrive first, or a 4h candle starting at 02:00 would claim to be the
    # 00:00 bar.
    out = svc._resample(_hourly("2026-11-10", range(2, 10)), 4)
    assert [b["date"] for b in out] == [
        "2026-11-10 00:00", "2026-11-10 04:00", "2026-11-10 08:00",
    ]


def test_resample_factor_one_is_identity():
    candles = _hourly("2026-11-10", range(3))
    assert svc._resample(candles, 1) == candles


def test_resample_splits_across_days():
    candles = _hourly("2026-11-10", range(20, 24)) + _hourly("2026-11-11", range(0, 4))
    out = svc._resample(candles, 4)
    assert len(out) == 2
    assert out[0]["date"].startswith("2026-11-10")
    assert out[1]["date"].startswith("2026-11-11")


# ── Pivot detection ───────────────────────────────────────────────────────────

def test_pivot_indices_finds_local_highs_and_ignores_edges():
    values = [1, 2, 9, 2, 1, 2, 3, 2, 1]
    # Window 2 means indices 0,1 and the last two can never be pivots.
    assert svc._pivot_indices(values, 2, "high") == [2, 6]


def test_pivot_indices_requires_clear_bars_on_the_right():
    # A new extreme on the final bar is not a confirmed pivot; counting it
    # would invent divergences on the forming candle.
    values = [1, 2, 3, 4, 99]
    assert svc._pivot_indices(values, 2, "high") == []


def test_pivot_indices_finds_lows():
    values = [5, 4, 1, 4, 5, 4, 2, 4, 5]
    assert svc._pivot_indices(values, 2, "low") == [2, 6]


def test_pivot_indices_skips_none_values():
    values = [1, 2, None, 9, None, 2, 1, 2, 1]
    assert svc._pivot_indices(values, 2, "high") == [3]


# ── RSI divergence ────────────────────────────────────────────────────────────

def _flat_candles(n, highs=None, lows=None):
    out = []
    for i in range(n):
        h = highs[i] if highs else 100.0
        lo = lows[i] if lows else 90.0
        out.append({
            "date": f"2026-10-{(i % 28) + 1:02d}",
            "open": 95.0, "high": h, "low": lo, "close": 95.0, "volume": 1,
        })
    return out


def test_bearish_divergence_detected(monkeypatch):
    n = 30
    highs = [100.0] * n
    highs[8] = 120.0    # first pivot high
    highs[20] = 125.0   # higher pivot high
    rsi = [50.0] * n
    rsi[8] = 75.0       # strong momentum at the first high
    rsi[20] = 65.0      # weaker at the higher high -> bearish divergence

    monkeypatch.setattr(svc, "fetch_crypto_candles", lambda *a, **k: (_flat_candles(n, highs), "test source"))
    monkeypatch.setattr(svc, "calc_rsi", lambda closes, period=14: list(rsi))

    out = svc.detect_rsi_divergence("BTCUSDT", timeframe="4h", pivot_window=3)
    assert out["divergence_found"] is True
    types = [d["type"] for d in out["divergences"]]
    assert "bearish" in types
    bear = next(d for d in out["divergences"] if d["type"] == "bearish")
    assert [p["price"] for p in bear["price_pivots"]] == [120.0, 125.0]
    assert bear["rsi_change"] == -10.0
    assert bear["strength"] == 10.0
    assert bear["bars_since"] == n - 1 - 20


def test_bullish_divergence_detected(monkeypatch):
    n = 30
    lows = [90.0] * n
    lows[8] = 70.0
    lows[20] = 65.0     # lower low in price
    rsi = [50.0] * n
    rsi[8] = 25.0
    rsi[20] = 35.0      # higher RSI low -> bullish divergence

    monkeypatch.setattr(svc, "fetch_crypto_candles",
                        lambda *a, **k: (_flat_candles(n, lows=lows), "test source"))
    monkeypatch.setattr(svc, "calc_rsi", lambda closes, period=14: list(rsi))

    out = svc.detect_rsi_divergence("ETHUSDT", timeframe="4h", pivot_window=3)
    assert out["divergence_found"] is True
    assert any(d["type"] == "bullish" for d in out["divergences"])


def test_no_divergence_when_rsi_confirms(monkeypatch):
    n = 30
    highs = [100.0] * n
    highs[8] = 120.0
    highs[20] = 125.0
    rsi = [50.0] * n
    rsi[8] = 65.0
    rsi[20] = 75.0      # RSI confirms the higher high — no divergence

    monkeypatch.setattr(svc, "fetch_crypto_candles", lambda *a, **k: (_flat_candles(n, highs), "test source"))
    monkeypatch.setattr(svc, "calc_rsi", lambda closes, period=14: list(rsi))

    out = svc.detect_rsi_divergence("BTCUSDT", timeframe="4h", pivot_window=3)
    assert out["divergence_found"] is False
    assert out["divergences"] == []


@pytest.mark.parametrize("kwargs", [
    {"lookback": 5}, {"lookback": 500},
    {"pivot_window": 0}, {"pivot_window": 50},
    {"rsi_period": 1}, {"rsi_period": 99},
])
def test_divergence_rejects_out_of_range_params(kwargs):
    out = svc.detect_rsi_divergence("BTCUSDT", **kwargs)
    assert out["error"]["code"] == "INVALID_PARAMETER"


def test_divergence_rejects_timeframe_without_candles(monkeypatch):
    # 15m IS served now that Binance klines back these tools, so the test
    # needs an interval no provider has.
    out = svc.detect_rsi_divergence("BTCUSDT", timeframe="7s")
    assert out["error"]["code"] == "INVALID_TIMEFRAME"
    assert "1h" in out["error"]["supported"]
    assert "15m" in out["error"]["supported"], "Binance serves 15m natively"


def test_divergence_reports_too_few_candles(monkeypatch):
    monkeypatch.setattr(svc, "fetch_crypto_candles", lambda *a, **k: (_flat_candles(5), "test source"))
    out = svc.detect_rsi_divergence("BTCUSDT", timeframe="4h")
    assert out["error"]["code"] == "NO_DATA"


# ── Donchian breakout ─────────────────────────────────────────────────────────

def _channel_candles(n, last_high, last_close, body_high=100.0, body_low=90.0):
    out = _flat_candles(n - 1, highs=[body_high] * (n - 1), lows=[body_low] * (n - 1))
    out.append({
        "date": "2026-11-10", "open": 95.0, "high": last_high,
        "low": 95.0, "close": last_close, "volume": 1,
    })
    return out


def test_donchian_breakout_up_detected(monkeypatch):
    candles = _channel_candles(30, last_high=120.0, last_close=115.0)
    monkeypatch.setattr(svc, "_gather_candles", lambda syms, tf: ({"BTCUSDT": candles}, [], {"test source": 1}))
    out = svc.scan_donchian_breakout(symbols=["BTCUSDT"], timeframe="1D", period=20)
    row = out["rows"][0]
    assert row["state"] == "breakout_up"
    assert row["breakout"] is True
    assert row["channel_upper"] == 100.0
    assert out["breakouts"] == [row]


def test_donchian_channel_excludes_the_tested_bar(monkeypatch):
    # The regression guard: calc_donchian includes the current bar in its
    # window, so measuring the channel on all candles would compare this
    # bar's close against a high this bar itself set. Here the final bar
    # matches the prior high but closes below it — inside the channel, not a
    # breakout.
    candles = _channel_candles(30, last_high=100.0, last_close=99.0)
    monkeypatch.setattr(svc, "_gather_candles", lambda syms, tf: ({"BTCUSDT": candles}, [], {"test source": 1}))
    out = svc.scan_donchian_breakout(symbols=["BTCUSDT"], timeframe="1D", period=20)
    row = out["rows"][0]
    assert row["state"] == "inside_channel"
    assert row["breakout"] is False
    assert row["channel_upper"] == 100.0


def test_donchian_breakout_down_detected(monkeypatch):
    candles = _channel_candles(30, last_high=95.0, last_close=85.0)
    monkeypatch.setattr(svc, "_gather_candles", lambda syms, tf: ({"BTCUSDT": candles}, [], {"test source": 1}))
    out = svc.scan_donchian_breakout(symbols=["BTCUSDT"], timeframe="1D", period=20)
    assert out["rows"][0]["state"] == "breakout_down"


def test_donchian_skips_symbols_with_too_few_candles(monkeypatch):
    monkeypatch.setattr(svc, "_gather_candles", lambda syms, tf: ({"BTCUSDT": _flat_candles(5)}, [], {}))
    out = svc.scan_donchian_breakout(symbols=["BTCUSDT"], timeframe="1D", period=20)
    assert out["rows"] == []
    assert out["unavailable"] == ["BTCUSDT"]


@pytest.mark.parametrize("kwargs", [
    {"period": 2}, {"period": 500}, {"max_symbols": 0}, {"max_symbols": 99},
])
def test_donchian_rejects_out_of_range_params(kwargs):
    out = svc.scan_donchian_breakout(symbols=["BTCUSDT"], **kwargs)
    assert out["error"]["code"] == "INVALID_PARAMETER"


def test_donchian_rejects_timeframe_without_candles():
    out = svc.scan_donchian_breakout(symbols=["BTCUSDT"], timeframe="7s")
    assert out["error"]["code"] == "INVALID_TIMEFRAME"


def test_candle_timeframes_now_include_binance_native_intervals():
    # The old Yahoo path served 1h/4h/1D only; Binance adds the rest natively.
    supported = svc._supported_timeframes()
    for tf in ("5m", "15m", "30m", "1h", "4h", "1D", "1W"):
        assert tf in supported, tf
    assert svc._timeframe_has_provider("15m") is True
    assert svc._timeframe_has_provider("7s") is False


# ── Supertrend scan ───────────────────────────────────────────────────────────

def _trending_candles(n, start=100.0, step=1.0):
    out = []
    px = start
    for i in range(n):
        out.append({
            "date": f"2026-10-{(i % 28) + 1:02d}",
            "open": px, "high": px + 1, "low": px - 1, "close": px, "volume": 1,
        })
        px += step
    return out


def test_supertrend_scan_reports_direction(monkeypatch):
    candles = _trending_candles(60, step=1.0)  # steady uptrend
    monkeypatch.setattr(svc, "_gather_candles", lambda syms, tf: ({"BTCUSDT": candles}, [], {"test source": 1}))
    out = svc.scan_supertrend(symbols=["BTCUSDT"], timeframe="4h")
    row = out["rows"][0]
    assert row["direction"] == "bullish"
    assert row["symbol"] == "BTCUSDT"
    assert out["scanned"] == 1
    assert "Supertrend" in out["indicator"]


def test_supertrend_scan_flags_a_flip(monkeypatch):
    # Up for 40 bars, then a crash steep enough to close through the lower
    # band (the bar range is 2, so the band sits ~6 below; a 15/bar drop
    # clears it). Direction must flip and land in the flips list.
    candles = _trending_candles(40, step=1.0) + _trending_candles(20, start=139.0, step=-15.0)
    monkeypatch.setattr(svc, "_gather_candles", lambda syms, tf: ({"BTCUSDT": candles}, [], {"test source": 1}))
    out = svc.scan_supertrend(symbols=["BTCUSDT"], timeframe="4h", flip_within=20)
    row = out["rows"][0]
    assert row["direction"] == "bearish"
    assert row["bars_since_flip"] is not None
    assert row["just_flipped"] is True


def test_supertrend_marks_direction_unconfirmed_before_any_flip(monkeypatch):
    # A series that never penetrates a band carries only the seed guess.
    # pandas-ta and Pine seed the opposite way, so this run is exactly where
    # implementations legitimately disagree — it must not read as a signal.
    candles = _trending_candles(80, step=0.2)
    monkeypatch.setattr(svc, "_gather_candles", lambda syms, tf: ({"BTCUSDT": candles}, [], {"test source": 1}))
    out = svc.scan_supertrend(symbols=["BTCUSDT"], timeframe="4h")
    row = out["rows"][0]
    assert row["bars_since_flip"] is None
    assert row["direction_confirmed"] is False
    assert row["just_flipped"] is False


def test_supertrend_marks_direction_confirmed_after_a_flip(monkeypatch):
    candles = _trending_candles(40, step=1.0) + _trending_candles(20, start=139.0, step=-15.0)
    monkeypatch.setattr(svc, "_gather_candles", lambda syms, tf: ({"BTCUSDT": candles}, [], {"test source": 1}))
    out = svc.scan_supertrend(symbols=["BTCUSDT"], timeframe="4h")
    assert out["rows"][0]["direction_confirmed"] is True


def test_supertrend_explicit_symbols_skip_the_universe_scan(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("universe scan must not run when symbols are given")
    monkeypatch.setattr(svc, "_scan_universe", _boom)
    monkeypatch.setattr(svc, "_gather_candles",
                        lambda syms, tf: ({"BTCUSDT": _trending_candles(60)}, [], {}))
    out = svc.scan_supertrend(symbols=["BTCUSDT"], timeframe="4h")
    assert out["universe_note"] == "Explicit symbol list"


@pytest.mark.parametrize("kwargs", [
    {"max_symbols": 0}, {"max_symbols": 99}, {"flip_within": 0},
    {"flip_within": 50}, {"atr_period": 1}, {"multiplier": 0.1}, {"multiplier": 50},
])
def test_supertrend_rejects_out_of_range_params(kwargs):
    out = svc.scan_supertrend(symbols=["BTCUSDT"], **kwargs)
    assert out["error"]["code"] == "INVALID_PARAMETER"


# ── Watchlist ─────────────────────────────────────────────────────────────────

def test_watchlist_builds_rows_in_one_request(monkeypatch):
    calls = []

    def fake_scan(screener, interval, symbols, columns):
        calls.append(symbols)
        return {
            "KUCOIN:BTCUSDT": {
                "close": 73500.0, "change": 1.5, "RSI": 72.0, "EMA50": 70000.0,
                "ATR": 1500.0, svc.RELATIVE_VOLUME: 1.8,
                svc.AVERAGE_VOLUME: 1000.0, "volume": 1800.0, "Recommend.All": 0.4,
            },
            "KUCOIN:ETHUSDT": {
                "close": 2500.0, "change": -2.0, "RSI": 28.0, "EMA50": 2700.0,
                "ATR": 90.0, svc.RELATIVE_VOLUME: 0.7,
                svc.AVERAGE_VOLUME: 500.0, "volume": 350.0, "Recommend.All": -0.2,
            },
        }

    monkeypatch.setattr(svc, "resilient_scan_columns", fake_scan)
    out = svc.build_crypto_watchlist(["BTCUSDT", "ETHUSDT"], "KUCOIN", "4h")

    assert len(calls) == 1, "watchlist must cost exactly one scanner request"
    assert out["returned"] == 2
    btc = next(r for r in out["rows"] if r["symbol"] == "BTCUSDT")
    eth = next(r for r in out["rows"] if r["symbol"] == "ETHUSDT")
    assert btc["rsi_zone"] == "overbought"
    assert eth["rsi_zone"] == "oversold"
    assert btc["trend"] == "above EMA50"
    assert eth["trend"] == "below EMA50"
    assert btc["volume_ratio"] == 1.8
    # Sorted by change descending.
    assert out["rows"][0]["symbol"] == "BTCUSDT"


def test_watchlist_reports_unknown_symbols(monkeypatch):
    monkeypatch.setattr(svc, "resilient_scan_columns",
                        lambda *a, **k: {"KUCOIN:BTCUSDT": {"close": 1.0, "RSI": 50.0}})
    out = svc.build_crypto_watchlist(["BTCUSDT", "NOTACOINUSDT"], "KUCOIN", "4h")
    assert out["not_found"] == ["NOTACOINUSDT"]
    assert out["returned"] == 1


def test_watchlist_rejects_empty_symbols():
    assert svc.build_crypto_watchlist([], "KUCOIN")["error"]["code"] == "INVALID_PARAMETER"
    assert svc.build_crypto_watchlist(["  "], "KUCOIN")["error"]["code"] == "INVALID_PARAMETER"


def test_watchlist_truncates_oversized_requests(monkeypatch):
    seen = {}

    def fake_scan(screener, interval, symbols, columns):
        seen["n"] = len(symbols)
        return {}

    monkeypatch.setattr(svc, "resilient_scan_columns", fake_scan)
    out = svc.build_crypto_watchlist([f"COIN{i}USDT" for i in range(50)], "KUCOIN")
    assert seen["n"] == svc._MAX_WATCHLIST
    assert out["truncated_to"] == svc._MAX_WATCHLIST


# ── Trade plan ────────────────────────────────────────────────────────────────

class _FakeAnalysis:
    def __init__(self, indicators):
        self.indicators = indicators


_PLAN_INDICATORS = {
    "close": 100.0, "high": 102.0, "low": 98.0, "ATR": 3.0,
    "EMA20": 99.0, "EMA50": 95.0, "EMA200": 90.0, "RSI": 55.0,
    "Pivot.M.Classic.S1": 96.0, "Pivot.M.Classic.R1": 108.0,
    "Pivot.M.Fibonacci.S1": 97.0, "Pivot.M.Fibonacci.R1": 106.0,
    svc.RELATIVE_VOLUME: 1.4, "Recommend.All": 0.3, "open": 99.0,
}


def test_trade_plan_sizes_the_position_from_the_primary_entry(monkeypatch):
    monkeypatch.setattr(
        svc, "resilient_get_multiple_analysis",
        lambda screener, interval, symbols: {symbols[0]: _FakeAnalysis(dict(_PLAN_INDICATORS))},
    )
    out = svc.generate_crypto_trade_plan("BTCUSDT", "KUCOIN", "4h",
                                        account_size=10_000.0, risk_pct=1.0)
    sizing = out["position_sizing"]
    # Sizing only appears if the entry was read from the key compute_trade_setup
    # actually publishes (risk_reward.measured_from_entry) — a wrong key makes
    # this silently None.
    assert sizing is not None
    entry = out["trade_setup"]["risk_reward"]["measured_from_entry"]
    stop = out["trade_setup"]["stop_loss"]
    assert sizing["risk_amount"] == 100.0
    assert sizing["stop_distance"] == pytest.approx(abs(entry - stop), abs=1e-6)
    assert sizing["units"] == pytest.approx(100.0 / abs(entry - stop), rel=1e-6)
    assert sizing["notional"] == pytest.approx(sizing["units"] * entry, rel=1e-6)


def test_trade_plan_omits_sizing_without_account_size(monkeypatch):
    monkeypatch.setattr(
        svc, "resilient_get_multiple_analysis",
        lambda screener, interval, symbols: {symbols[0]: _FakeAnalysis(dict(_PLAN_INDICATORS))},
    )
    out = svc.generate_crypto_trade_plan("BTCUSDT", "KUCOIN", "4h")
    assert out["position_sizing"] is None
    assert out["trade_setup"]["stop_loss"] is not None
    assert out["trend_summary"]


def test_trade_plan_reports_missing_atr(monkeypatch):
    indicators = dict(_PLAN_INDICATORS)
    indicators["ATR"] = None
    monkeypatch.setattr(
        svc, "resilient_get_multiple_analysis",
        lambda screener, interval, symbols: {symbols[0]: _FakeAnalysis(indicators)},
    )
    monkeypatch.setattr(svc, "fetch_atr_for_ticker", lambda *a, **k: None)
    out = svc.generate_crypto_trade_plan("BTCUSDT", "KUCOIN", "4h")
    assert out["error"]["code"] == "NO_DATA"
    assert "ATR" in out["error"]["message"]


@pytest.mark.parametrize("kwargs", [
    {"risk_pct": 0}, {"risk_pct": -1}, {"risk_pct": 101},
    {"account_size": 0}, {"account_size": -5},
])
def test_trade_plan_rejects_bad_sizing_params(kwargs):
    out = svc.generate_crypto_trade_plan("BTCUSDT", **kwargs)
    assert out["error"]["code"] == "INVALID_PARAMETER"


def test_trade_plan_reports_unknown_symbol(monkeypatch):
    monkeypatch.setattr(svc, "resilient_get_multiple_analysis",
                        lambda screener, interval, symbols: {})
    out = svc.generate_crypto_trade_plan("NOTACOINUSDT", "KUCOIN", "4h")
    assert out["error"]["code"] == "SYMBOL_NOT_FOUND"


# ── Fibonacci ─────────────────────────────────────────────────────────────────

def test_fibonacci_uses_screener_period_extremes(monkeypatch):
    monkeypatch.setattr(
        svc, "resilient_scan_columns",
        lambda screener, interval, symbols, columns: {
            symbols[0]: {"close": 100.0, "price_52_week_high": 150.0,
                         "price_52_week_low": 50.0},
        },
    )
    monkeypatch.setattr(
        svc, "resilient_get_multiple_analysis",
        lambda screener, interval, symbols: {
            symbols[0]: _FakeAnalysis({
                "close": 100.0, "open": 99.0, "EMA50": 95.0, "EMA200": 90.0,
                "RSI": 55.0, "ATR": 3.0, svc.RELATIVE_VOLUME: 1.2,
            }),
        },
    )
    out = svc.analyze_crypto_fibonacci("BTCUSDT", "KUCOIN", "52W", "1D")
    assert out["swing_high"] == 150.0
    assert out["swing_low"] == 50.0
    assert "52W" in out["swing_source"]
    assert out["retracement_levels"]
    assert out["price_position"]
    assert out["context"]["rsi"] == 55.0


def test_fibonacci_falls_back_to_pivots(monkeypatch):
    monkeypatch.setattr(svc, "resilient_scan_columns",
                        lambda *a, **k: {})  # no period extremes available
    monkeypatch.setattr(
        svc, "resilient_get_multiple_analysis",
        lambda screener, interval, symbols: {
            symbols[0]: _FakeAnalysis({
                "close": 100.0, "open": 99.0, "EMA50": 95.0, "EMA200": 90.0,
                "Pivot.M.Fibonacci.R3": 130.0, "Pivot.M.Fibonacci.S3": 70.0,
                "RSI": 55.0, "ATR": 3.0,
            }),
        },
    )
    out = svc.analyze_crypto_fibonacci("BTCUSDT", "KUCOIN", "52W", "1D")
    assert out["swing_source"] == "pivot points (R3/S3 fallback)"
    assert out["swing_high"] == 130.0


def test_fibonacci_rejects_bad_lookback():
    out = svc.analyze_crypto_fibonacci("BTCUSDT", lookback="7Y")
    assert out["error"]["code"] == "INVALID_PARAMETER"
    assert "52W" in out["error"]["valid_lookbacks"]


def test_fibonacci_rejects_narrow_swing_range(monkeypatch):
    monkeypatch.setattr(
        svc, "resilient_scan_columns",
        lambda screener, interval, symbols, columns: {
            symbols[0]: {"close": 100.0, "price_52_week_high": 100.5,
                         "price_52_week_low": 100.0},
        },
    )
    monkeypatch.setattr(
        svc, "resilient_get_multiple_analysis",
        lambda screener, interval, symbols: {
            symbols[0]: _FakeAnalysis({"close": 100.0, "EMA50": 99.0}),
        },
    )
    out = svc.analyze_crypto_fibonacci("BTCUSDT", "KUCOIN", "52W", "1D")
    assert out["error"]["code"] == "NO_DATA"
    assert "narrow" in out["error"]["message"]
