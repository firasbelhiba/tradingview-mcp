"""Unit tests for the market structure service.

Offline: the scanner call is monkeypatched so the statistics, the
denominators and the throttle handling are pinned without upstream.
"""
import pytest

from tradingview_mcp.core.services import market_structure_service as svc
from tradingview_mcp.core.services.indicators import RELATIVE_VOLUME


def _row(close=100.0, ema50=90.0, sma200=80.0, change=1.0, rsi=50.0, relvol=1.0):
    return {
        "close": close, "EMA50": ema50, "SMA200": sma200,
        "change": change, "RSI": rsi, RELATIVE_VOLUME: relvol,
    }


def _patch_scan(monkeypatch, rows, capture=None):
    def fake(screener, interval, symbols, columns):
        if capture is not None:
            capture.append({"screener": screener, "interval": interval,
                            "symbols": symbols, "columns": columns})
        return rows
    monkeypatch.setattr(svc, "resilient_scan_columns", fake)


def _patch_universe(monkeypatch, symbols):
    monkeypatch.setattr(svc, "load_symbols", lambda ex: list(symbols))


# ── Breadth statistics ────────────────────────────────────────────────────────

def test_breadth_counts_participation(monkeypatch):
    rows = {
        "KUCOIN:A": _row(close=100, ema50=90, sma200=80, change=2.0, rsi=75),
        "KUCOIN:B": _row(close=100, ema50=110, sma200=80, change=-1.0, rsi=25),
        "KUCOIN:C": _row(close=100, ema50=95, sma200=120, change=0.0, rsi=50),
        "KUCOIN:D": _row(close=100, ema50=99, sma200=99, change=3.0, rsi=60),
    }
    _patch_universe(monkeypatch, ["A", "B", "C", "D"])
    _patch_scan(monkeypatch, rows)

    out = svc.analyze_market_breadth("KUCOIN", "1D")
    tp = out["trend_participation"]
    assert tp["above_ema50"] == 3 and tp["above_ema50_pct"] == 75.0
    assert tp["above_sma200"] == 3 and tp["above_sma200_pct"] == 75.0

    ad = out["advance_decline"]
    assert (ad["advancing"], ad["declining"], ad["unchanged"]) == (2, 1, 1)
    assert ad["ratio"] == 2.0
    assert ad["median_change_pct"] == 1.0

    md = out["momentum_distribution"]
    assert md["overbought_pct"] == 25.0 and md["oversold_pct"] == 25.0
    assert md["median_rsi"] == 55.0


def test_missing_inputs_are_excluded_not_counted_as_below(monkeypatch):
    # The headline rule: a symbol with no EMA50 must not count toward
    # "below EMA50". Treating missing as a value is what made the old volume
    # tools report "Normal" for everything.
    rows = {
        "KUCOIN:A": _row(close=100, ema50=90),           # above
        "KUCOIN:B": {"close": 100, "EMA50": None,        # no EMA50 at all
                     "SMA200": None, "change": 1.0, "RSI": None},
        "KUCOIN:C": _row(close=100, ema50=95),           # above
    }
    _patch_universe(monkeypatch, ["A", "B", "C"])
    _patch_scan(monkeypatch, rows)

    out = svc.analyze_market_breadth("KUCOIN", "1D")
    # 2 of 2 symbols that HAVE an EMA50 are above it -> 100%, not 67%.
    assert out["trend_participation"]["above_ema50_pct"] == 100.0
    assert out["coverage"]["ema50"] == 2
    assert out["symbols_with_data"] == 3


def test_coverage_reports_each_denominator(monkeypatch):
    rows = {
        "KUCOIN:A": _row(),
        "KUCOIN:B": {"close": 1.0, "EMA50": 1.0, "SMA200": None,
                     "change": None, "RSI": 40.0},
    }
    _patch_universe(monkeypatch, ["A", "B"])
    _patch_scan(monkeypatch, rows)
    cov = svc.analyze_market_breadth("KUCOIN", "1D")["coverage"]
    assert cov["ema50"] == 2
    assert cov["sma200"] == 1
    assert cov["change"] == 1
    assert cov["rsi"] == 2


@pytest.mark.parametrize("above_pct_rows,adv,dec,expected", [
    # 75% above EMA50 with 3:1 advancers -> STRONG
    (0.75, 3, 1, "STRONG"),
    # 20% above with heavy decline -> WEAK
    (0.20, 1, 4, "WEAK"),
    # few above trend but broad advance today -> IMPROVING
    (0.20, 4, 1, "IMPROVING"),
    # most above trend but rolling over -> DETERIORATING
    (0.80, 1, 4, "DETERIORATING"),
])
def test_breadth_labels_follow_the_written_thresholds(
    monkeypatch, above_pct_rows, adv, dec, expected
):
    n = 20
    above = int(n * above_pct_rows)
    rows = {}
    for i in range(n):
        is_above = i < above
        # changes assigned to hit the requested advance/decline split
        if i < adv:
            change = 1.0
        elif i < adv + dec:
            change = -1.0
        else:
            change = 0.0
        rows[f"KUCOIN:S{i}"] = _row(
            close=100, ema50=90 if is_above else 110, change=change
        )
    _patch_universe(monkeypatch, [f"S{i}" for i in range(n)])
    _patch_scan(monkeypatch, rows)
    out = svc.analyze_market_breadth("KUCOIN", "1D")
    assert out["assessment"]["label"] == expected


def test_ratio_is_null_when_nothing_declined(monkeypatch):
    rows = {f"KUCOIN:S{i}": _row(close=100, ema50=90, change=1.0) for i in range(5)}
    _patch_universe(monkeypatch, [f"S{i}" for i in range(5)])
    _patch_scan(monkeypatch, rows)
    out = svc.analyze_market_breadth("KUCOIN", "1D")
    ad = out["advance_decline"]
    # A ratio with a zero denominator is undefined, not equal to the
    # advancing count — but the state is still unambiguously broad.
    assert ad["ratio"] is None
    assert ad["ratio_note"] == "undefined — no symbol declined"
    assert out["assessment"]["label"] == "STRONG"


def test_breadth_applies_relative_volume_filter(monkeypatch):
    rows = {
        "KUCOIN:A": _row(relvol=2.0, change=1.0),
        "KUCOIN:B": _row(relvol=0.1, change=-5.0),
    }
    _patch_universe(monkeypatch, ["A", "B"])
    _patch_scan(monkeypatch, rows)
    out = svc.analyze_market_breadth("KUCOIN", "1D", min_rel_volume=1.0)
    assert out["symbols_with_data"] == 1
    assert out["advance_decline"]["declining"] == 0
    assert out["filters"]["min_rel_volume"] == 1.0


def test_breadth_requests_every_symbol_in_one_call(monkeypatch):
    calls = []
    _patch_universe(monkeypatch, [f"S{i}" for i in range(300)])
    _patch_scan(monkeypatch, {f"KUCOIN:S{i}": _row() for i in range(300)}, calls)
    svc.analyze_market_breadth("KUCOIN", "1D")
    assert len(calls) == 1, "breadth must cost exactly one scanner request"
    assert len(calls[0]["symbols"]) == 300


def test_breadth_reports_throttling_rather_than_an_empty_market(monkeypatch):
    # TradingView answers a burst with empty bodies, not an error status.
    # Reporting "0% above EMA50" there would be a false market reading.
    _patch_universe(monkeypatch, ["A", "B"])
    _patch_scan(monkeypatch, {})
    out = svc.analyze_market_breadth("KUCOIN", "1D")
    assert out["error"]["code"] == "UPSTREAM_ERROR"
    assert "throttl" in out["error"]["message"].lower()
    assert out["error"]["retryable"] is True


def test_breadth_rejects_unknown_exchange(monkeypatch):
    _patch_universe(monkeypatch, [])
    out = svc.analyze_market_breadth("NOSUCHVENUE", "1D")
    assert out["error"]["code"] == "INVALID_EXCHANGE"


# ── Relative strength ─────────────────────────────────────────────────────────

def _rs_row(perf_1m=None, close=100.0, change=1.0, rsi=50.0, relvol=1.0, perf_w=None):
    return {
        "close": close, "change": change, "RSI": rsi, RELATIVE_VOLUME: relvol,
        "Perf.W": perf_w, "Perf.1M": perf_1m, "Perf.3M": None, "Perf.Y": None,
    }


def test_relative_strength_subtracts_the_benchmark(monkeypatch):
    rows = {
        "KUCOIN:BTCUSDT": _rs_row(perf_1m=9.0),
        "KUCOIN:STRONG": _rs_row(perf_1m=25.0),
        "KUCOIN:WEAK": _rs_row(perf_1m=4.0),
    }
    _patch_universe(monkeypatch, ["BTCUSDT", "STRONG", "WEAK"])
    _patch_scan(monkeypatch, rows)

    out = svc.rank_relative_strength("KUCOIN", "1M", "BTCUSDT", limit=5)
    assert out["benchmark"]["perf_1m_pct"] == 9.0
    by = {e["symbol"]: e for e in out["leaders"]}
    # Up 4% while the benchmark is up 9% is underperformance, not a gain.
    assert by["WEAK"]["relative_pct"] == -5.0
    assert by["WEAK"]["outperforming"] is False
    assert by["STRONG"]["relative_pct"] == 16.0
    assert by["STRONG"]["outperforming"] is True
    assert out["outperforming_benchmark"] == 1


def test_relative_strength_sorts_leaders_and_laggards(monkeypatch):
    rows = {f"KUCOIN:S{i}": _rs_row(perf_1m=float(i)) for i in range(10)}
    rows["KUCOIN:BTCUSDT"] = _rs_row(perf_1m=5.0)
    _patch_universe(monkeypatch, [f"S{i}" for i in range(10)] + ["BTCUSDT"])
    _patch_scan(monkeypatch, rows)
    out = svc.rank_relative_strength("KUCOIN", "1M", "BTCUSDT", limit=3)
    assert [e["symbol"] for e in out["leaders"]] == ["S9", "S8", "S7"]
    assert [e["symbol"] for e in out["laggards"]] == ["S0", "S1", "S2"]


def test_relative_strength_always_queries_the_daily_interval(monkeypatch):
    # Perf.* columns are daily-only; requesting them intraday returns nulls
    # for every symbol, so the interval must not follow a caller's timeframe.
    calls = []
    _patch_universe(monkeypatch, ["BTCUSDT", "A"])
    _patch_scan(monkeypatch, {"KUCOIN:BTCUSDT": _rs_row(perf_1m=1.0),
                              "KUCOIN:A": _rs_row(perf_1m=2.0)}, calls)
    svc.rank_relative_strength("KUCOIN", "1M", "BTCUSDT")
    assert calls[0]["interval"] == "1D"
    assert len(calls) == 1


def test_relative_strength_adds_a_benchmark_outside_the_universe(monkeypatch):
    calls = []
    _patch_universe(monkeypatch, ["A"])
    _patch_scan(monkeypatch, {"KUCOIN:A": _rs_row(perf_1m=2.0),
                              "KUCOIN:BTCUSDT": _rs_row(perf_1m=1.0)}, calls)
    svc.rank_relative_strength("KUCOIN", "1M", "BTCUSDT")
    assert "KUCOIN:BTCUSDT" in calls[0]["symbols"]


def test_relative_strength_warns_when_the_benchmark_has_no_data(monkeypatch):
    rows = {"KUCOIN:A": _rs_row(perf_1m=2.0), "KUCOIN:BTCUSDT": _rs_row(perf_1m=None)}
    _patch_universe(monkeypatch, ["A", "BTCUSDT"])
    _patch_scan(monkeypatch, rows)
    out = svc.rank_relative_strength("KUCOIN", "1M", "BTCUSDT")
    # Still ranked, but it must not be presented as relative strength.
    assert out["leaders"][0]["relative_pct"] is None
    assert out["leaders"][0]["outperforming"] is None
    assert out["outperforming_benchmark"] is None
    assert "not relative strength" in out["warning"]


def test_relative_strength_skips_symbols_without_performance(monkeypatch):
    rows = {
        "KUCOIN:BTCUSDT": _rs_row(perf_1m=5.0),
        "KUCOIN:HASDATA": _rs_row(perf_1m=7.0),
        "KUCOIN:NODATA": _rs_row(perf_1m=None),
    }
    _patch_universe(monkeypatch, ["BTCUSDT", "HASDATA", "NODATA"])
    _patch_scan(monkeypatch, rows)
    out = svc.rank_relative_strength("KUCOIN", "1M", "BTCUSDT")
    assert "NODATA" not in [e["symbol"] for e in out["leaders"]]
    assert out["ranked_symbols"] == 2


def test_relative_strength_applies_relative_volume_filter(monkeypatch):
    rows = {
        "KUCOIN:BTCUSDT": _rs_row(perf_1m=5.0, relvol=2.0),
        "KUCOIN:THIN": _rs_row(perf_1m=99.0, relvol=0.05),
    }
    _patch_universe(monkeypatch, ["BTCUSDT", "THIN"])
    _patch_scan(monkeypatch, rows)
    out = svc.rank_relative_strength("KUCOIN", "1M", "BTCUSDT", min_rel_volume=1.0)
    assert "THIN" not in [e["symbol"] for e in out["leaders"]]


def test_relative_strength_uses_the_window_column(monkeypatch):
    calls = []
    _patch_universe(monkeypatch, ["BTCUSDT", "A"])
    _patch_scan(monkeypatch, {"KUCOIN:BTCUSDT": _rs_row(perf_w=1.0),
                              "KUCOIN:A": _rs_row(perf_w=3.0)}, calls)
    out = svc.rank_relative_strength("KUCOIN", "1W", "BTCUSDT")
    assert out["performance_column"] == "Perf.W"
    assert out["leaders"][0]["perf_1w_pct"] == 3.0
    assert out["leaders"][0]["relative_pct"] == 2.0


@pytest.mark.parametrize("kwargs", [
    {"window": "5Y"}, {"window": "1h"}, {"limit": 0}, {"limit": 99},
])
def test_relative_strength_rejects_bad_params(kwargs):
    out = svc.rank_relative_strength("KUCOIN", **kwargs)
    assert out["error"]["code"] == "INVALID_PARAMETER"


def test_relative_strength_reports_throttling(monkeypatch):
    _patch_universe(monkeypatch, ["A"])
    _patch_scan(monkeypatch, {})
    out = svc.rank_relative_strength("KUCOIN", "1M", "BTCUSDT")
    assert out["error"]["code"] == "UPSTREAM_ERROR"
    assert "throttl" in out["error"]["message"].lower()


def test_relative_strength_reports_when_no_symbol_has_performance(monkeypatch):
    _patch_universe(monkeypatch, ["A"])
    _patch_scan(monkeypatch, {"KUCOIN:A": _rs_row(perf_1m=None)})
    out = svc.rank_relative_strength("KUCOIN", "1M", "BTCUSDT")
    assert out["error"]["code"] == "NO_DATA"


# ── The rendering pointer ─────────────────────────────────────────────────────

def test_breadth_and_strength_carry_their_rendering_pointers(monkeypatch):
    """Both are named in the server's PRESENTATION block.

    Breadth is a set of percentages over different denominators, so the hint
    has to point at coverage.note; a panel showing "18% above EMA50" without
    saying 18% of what is a worse artefact than the table it replaced.
    """
    rows = {f"KUCOIN:{s}": _row(change=float(i))
            for i, s in enumerate(("A", "B", "C"))}
    _patch_universe(monkeypatch, ["A", "B", "C"])
    _patch_scan(monkeypatch, rows)

    breadth = svc.analyze_market_breadth("KUCOIN", "1D")
    assert "coverage.note" in breadth["display_hint"]
    assert "denominator" in breadth["display_hint"]

    rs_rows = {
        "KUCOIN:BTCUSDT": _rs_row(perf_1m=5.0),
        "KUCOIN:STRONG": _rs_row(perf_1m=20.0),
        "KUCOIN:WEAK": _rs_row(perf_1m=1.0),
    }
    _patch_universe(monkeypatch, ["BTCUSDT", "STRONG", "WEAK"])
    _patch_scan(monkeypatch, rs_rows)
    strength = svc.rank_relative_strength("KUCOIN", "1M", "BTCUSDT", limit=2)
    assert "leaders and laggards" in strength["display_hint"]
    assert "how many symbols were measured" in strength["display_hint"]
