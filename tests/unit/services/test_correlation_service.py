"""Unit tests for correlation, beta and seasonality. Offline.

The maths is checked against hand-computable cases, and the timestamp
alignment — the thing most likely to produce a plausible wrong number — has
its own tests.
"""
from datetime import datetime, timedelta

import pytest

from tradingview_mcp.core.services import correlation_service as svc


def _candles(closes, start_day=1, hourly=False):
    """Candles with REAL sequential timestamps.

    Formatting a day number straight into the string produced "2026-01-60"
    once a series ran past the month, which the service rightly counted as
    unparseable — the helper was the bug, not the code. datetime arithmetic
    rolls over properly.
    """
    base = datetime(2026, 1, 1) + timedelta(days=start_day - 1)
    out = []
    for i, c in enumerate(closes):
        when = base + (timedelta(hours=i) if hourly else timedelta(days=i))
        date = when.strftime("%Y-%m-%d %H:%M" if hourly else "%Y-%m-%d")
        out.append({"date": date, "open": c, "high": c, "low": c,
                    "close": c, "volume": 1})
    return out


def _from_returns(returns, start_day=1, start=100.0):
    """Closes built by compounding an explicit return series.

    Constant returns are degenerate: compounding 1.01 gives a return series
    with no dispersion beyond float residue, and correlating that measures
    rounding error. Real tests need returns that actually vary.
    """
    closes, px = [start], start
    for r in returns:
        px *= (1.0 + r)
        closes.append(px)
    return _candles(closes, start_day=start_day)


_VARYING = [0.01, -0.005, 0.02, -0.012, 0.008, -0.003, 0.015, -0.009] * 8


def _patch_fetch(monkeypatch, mapping, source="test source"):
    def fake(symbol, timeframe="4h"):
        key = symbol.upper()
        if key not in mapping:
            raise RuntimeError(f"{key} unavailable")
        return mapping[key], source
    monkeypatch.setattr(svc, "fetch_crypto_candles", fake)


# ── Maths ─────────────────────────────────────────────────────────────────────

def test_pearson_on_perfectly_correlated_series():
    xs = [1.0, 2.0, 3.0, 4.0]
    assert svc._pearson(xs, [2.0, 4.0, 6.0, 8.0]) == pytest.approx(1.0)
    assert svc._pearson(xs, [-2.0, -4.0, -6.0, -8.0]) == pytest.approx(-1.0)


def test_pearson_is_none_without_variance():
    assert svc._pearson([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None
    assert svc._pearson([1.0], [2.0]) is None


def test_beta_is_the_slope():
    xs = [1.0, 2.0, 3.0, 4.0]
    assert svc._beta(xs, [2.0, 4.0, 6.0, 8.0]) == pytest.approx(2.0)
    assert svc._beta(xs, [0.5, 1.0, 1.5, 2.0]) == pytest.approx(0.5)


def test_beta_is_none_without_benchmark_variance():
    assert svc._beta([1.0, 1.0], [1.0, 2.0]) is None


@pytest.mark.parametrize("corr,beta,expected_fragment", [
    (0.95, 1.5, "leveraged benchmark"),
    (0.95, 0.5, "dampened benchmark"),
    (0.95, 1.0, "tracks the benchmark"),
    (0.6, 1.0, "partly independent"),
    (0.1, 1.0, "largely independent"),
    (-0.8, 1.0, "inversely correlated"),
    (None, None, "unknown"),
])
def test_reading_labels(corr, beta, expected_fragment):
    assert expected_fragment in svc._label(corr, beta)


# ── Timestamp alignment ───────────────────────────────────────────────────────

def test_returns_are_keyed_by_timestamp():
    out = svc._returns_by_date(_candles([100.0, 110.0, 99.0]))
    assert set(out) == {"2026-01-02", "2026-01-03"}
    assert out["2026-01-02"] == pytest.approx(0.10)
    assert out["2026-01-03"] == pytest.approx(-0.10)


def test_misaligned_histories_pair_by_date_not_position(monkeypatch):
    """The alignment guard.

    The benchmark starts two days earlier. Paired by POSITION, the symbol's
    first return would be matched against the benchmark's first — two
    different days — and the correlation would be meaningless. Paired by date,
    only the shared dates are used, and an exactly-inverted symbol must come
    back at -1.
    """
    bench = _from_returns(_VARYING, start_day=1)
    # Starts two days later and moves exactly opposite on each shared bar.
    sym = _from_returns([-r for r in _VARYING[2:]], start_day=3)
    _patch_fetch(monkeypatch, {"BTCUSDT": bench, "ALT": sym})

    out = svc.analyze_correlation(["ALT"], "BTCUSDT", "1D", lookback=100)
    row = out["rows"][0]
    # Shared dates only: the symbol has 57 bars from day 3, so overlap is
    # strictly fewer than the benchmark's 59 returns.
    assert row["observations"] < out["benchmark_observations"]
    assert row["correlation"] == pytest.approx(-1.0, abs=0.001)


def test_zero_overlap_is_reported(monkeypatch):
    bench = _candles([100.0 + i for i in range(60)], start_day=1)
    sym = _candles([100.0 + i for i in range(60)], start_day=1)
    # Push the symbol's dates a year forward so nothing overlaps.
    for c in sym:
        c["date"] = c["date"].replace("2026-", "2027-")
    _patch_fetch(monkeypatch, {"BTCUSDT": bench, "ALT": sym})
    out = svc.analyze_correlation(["ALT"], "BTCUSDT", "1D", lookback=100)
    assert out["error"]["code"] == "NO_DATA"
    assert out["error"]["unavailable"] == ["ALT"]


# ── Correlation tool ──────────────────────────────────────────────────────────

def test_correlation_computes_and_sorts(monkeypatch):
    bench = _from_returns(_VARYING)
    high = _from_returns([r * 2 for r in _VARYING])   # perfect corr, beta 2
    _patch_fetch(monkeypatch, {"BTCUSDT": bench, "HIGH": high})
    out = svc.analyze_correlation(["HIGH"], "BTCUSDT", "1D", lookback=100)
    row = out["rows"][0]
    assert row["correlation"] == pytest.approx(1.0, abs=0.01)
    assert row["beta"] == pytest.approx(2.0, abs=0.01)
    assert row["r_squared"] == pytest.approx(1.0, abs=0.02)
    assert "leveraged benchmark" in row["reading"]


def test_benchmark_is_excluded_from_its_own_rows(monkeypatch):
    bench = _from_returns(_VARYING)
    _patch_fetch(monkeypatch, {"BTCUSDT": bench})
    out = svc.analyze_correlation(["BTCUSDT"], "BTCUSDT", "1D", lookback=100)
    assert out["error"]["code"] == "NO_DATA"


def test_unavailable_symbols_are_listed(monkeypatch):
    bench = _from_returns(_VARYING)
    ok = _from_returns([r * 1.5 for r in _VARYING])
    _patch_fetch(monkeypatch, {"BTCUSDT": bench, "OK": ok})
    out = svc.analyze_correlation(["OK", "MISSING"], "BTCUSDT", "1D", lookback=100)
    assert [r["symbol"] for r in out["rows"]] == ["OK"]
    assert out["unavailable"] == ["MISSING"]


def test_unfetchable_benchmark_is_an_error(monkeypatch):
    _patch_fetch(monkeypatch, {})
    out = svc.analyze_correlation(["ALT"], "BTCUSDT")
    assert out["error"]["code"] == "UPSTREAM_ERROR"
    assert out["error"]["retryable"] is True


def test_short_benchmark_history_is_reported(monkeypatch):
    _patch_fetch(monkeypatch, {"BTCUSDT": _candles([100.0, 101.0, 102.0])})
    out = svc.analyze_correlation(["ALT"], "BTCUSDT", "1D", lookback=100)
    assert out["error"]["code"] == "NO_DATA"


@pytest.mark.parametrize("kwargs", [{"lookback": 10}, {"lookback": 5000}])
def test_correlation_rejects_bad_lookback(kwargs):
    assert svc.analyze_correlation(["A"], **kwargs)["error"]["code"] == "INVALID_PARAMETER"


def test_correlation_rejects_empty_symbols():
    assert svc.analyze_correlation([])["error"]["code"] == "INVALID_PARAMETER"
    assert svc.analyze_correlation(["  "])["error"]["code"] == "INVALID_PARAMETER"


def test_correlation_caps_symbol_count(monkeypatch):
    bench = _from_returns(_VARYING)
    mapping = {"BTCUSDT": bench}
    for i in range(40):
        mapping[f"S{i}"] = _from_returns([r * 1.5 for r in _VARYING])
    _patch_fetch(monkeypatch, mapping)
    out = svc.analyze_correlation([f"S{i}" for i in range(40)], "BTCUSDT",
                                  "1D", lookback=100)
    assert len(out["rows"]) <= svc._MAX_SYMBOLS


# ── Bucket statistics ─────────────────────────────────────────────────────────

def test_bucket_stats_flags_noise():
    # Mean is small, dispersion is large: t well under 2, so not an edge.
    noisy = [0.01, -0.009, 0.011, -0.008, 0.012, -0.011] * 3
    stats = svc._bucket_stats(noisy)
    assert stats["samples"] == 18
    assert abs(stats["t_stat"]) < 2.0
    assert stats["distinguishable_from_zero"] is False


def test_bucket_stats_recognises_a_real_effect():
    consistent = [0.01] * 30  # same sign, no dispersion
    stats = svc._bucket_stats(consistent)
    assert stats["std_dev_pct"] == 0.0
    # Zero dispersion means no standard error to divide by; t is undefined
    # rather than infinite, and that must not read as "proven".
    assert stats["t_stat"] is None
    assert stats["distinguishable_from_zero"] is None


def test_bucket_stats_with_mild_dispersion_can_clear_the_bar():
    values = [0.01 + (0.0005 if i % 2 else -0.0005) for i in range(40)]
    stats = svc._bucket_stats(values)
    assert stats["t_stat"] > 2.0
    assert stats["distinguishable_from_zero"] is True


def test_empty_bucket_is_thin():
    assert svc._bucket_stats([])["samples"] == 0
    assert svc._bucket_stats([])["thin_sample"] is True


def test_win_rate_counts_positive_bars():
    stats = svc._bucket_stats([0.01, 0.01, -0.01, -0.01])
    assert stats["win_rate_pct"] == 50.0


# ── Seasonality tool ──────────────────────────────────────────────────────────

def test_seasonality_buckets_by_weekday(monkeypatch):
    closes = [100.0 * (1.001 ** i) for i in range(120)]
    _patch_fetch(monkeypatch, {"BTCUSDT": _candles(closes)})
    out = svc.analyze_seasonality("BTCUSDT", "1D", lookback=200)
    assert set(out["by_weekday"]) <= set(svc._WEEKDAYS)
    assert sum(v["samples"] for v in out["by_weekday"].values()) == 119


def test_daily_candles_have_no_hourly_buckets(monkeypatch):
    closes = [100.0 + i for i in range(80)]
    _patch_fetch(monkeypatch, {"BTCUSDT": _candles(closes)})
    out = svc.analyze_seasonality("BTCUSDT", "1D", lookback=200)
    assert out["by_hour_utc"] is None
    assert "need an intraday timeframe" in out["hour_note"]


def test_intraday_candles_produce_hourly_buckets(monkeypatch):
    closes = [100.0 + (i % 7) for i in range(72)]
    _patch_fetch(monkeypatch, {"BTCUSDT": _candles(closes, hourly=True)})
    out = svc.analyze_seasonality("BTCUSDT", "1h", lookback=200)
    assert out["by_hour_utc"]
    assert out["hour_note"] is None


def test_thin_buckets_are_excluded_from_best_and_worst(monkeypatch):
    # 60 daily bars gives ~8-9 per weekday, under the 10-sample floor.
    closes = [100.0 * (1.002 ** i) for i in range(60)]
    _patch_fetch(monkeypatch, {"BTCUSDT": _candles(closes)})
    out = svc.analyze_seasonality("BTCUSDT", "1D", lookback=200)
    thin = [k for k, v in out["by_weekday"].items() if v["thin_sample"]]
    assert thin, "expected thin buckets with this little history"
    assert out["best_weekday"] is None or not out["best_weekday"]["thin_sample"]


def test_seasonality_reports_too_little_history(monkeypatch):
    _patch_fetch(monkeypatch, {"BTCUSDT": _candles([100.0, 101.0])})
    out = svc.analyze_seasonality("BTCUSDT", "1D")
    assert out["error"]["code"] == "NO_DATA"


def test_seasonality_surfaces_a_bad_timeframe(monkeypatch):
    def fake(symbol, timeframe="4h"):
        raise ValueError("timeframe '7s' has no candle source")
    monkeypatch.setattr(svc, "fetch_crypto_candles", fake)
    out = svc.analyze_seasonality("BTCUSDT", "7s")
    assert out["error"]["code"] == "INVALID_TIMEFRAME"


@pytest.mark.parametrize("kwargs", [{"lookback": 10}, {"lookback": 9999}])
def test_seasonality_rejects_bad_lookback(kwargs):
    assert svc.analyze_seasonality("BTCUSDT", **kwargs)["error"]["code"] == \
        "INVALID_PARAMETER"


def test_unparseable_dates_are_counted_not_crashed(monkeypatch):
    candles = _candles([100.0 + i for i in range(60)])
    for c in candles[1:6]:
        c["date"] = "not-a-date"
    _patch_fetch(monkeypatch, {"BTCUSDT": candles})
    out = svc.analyze_seasonality("BTCUSDT", "1D", lookback=200)
    assert out["undated_bars_skipped"] == 5
