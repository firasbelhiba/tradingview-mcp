"""Unit tests for the trend-strength and volatility scans. Offline."""
import pytest

from tradingview_mcp.core.services import trend_scan_service as svc
from tradingview_mcp.core.services.indicators import RELATIVE_VOLUME


def _row(adx=30.0, plus=25.0, minus=10.0, close=100.0, sar=90.0,
         ema50=95.0, rsi=55.0, atr=2.0, relvol=1.0, vol_d=5.0,
         vol_w=10.0, vol_m=20.0):
    return {
        "close": close, "ADX": adx, "ADX+DI": plus, "ADX-DI": minus,
        "EMA50": ema50, "P.SAR": sar, "RSI": rsi, "ATR": atr,
        RELATIVE_VOLUME: relvol, "Volatility.D": vol_d,
        "Volatility.W": vol_w, "Volatility.M": vol_m,
    }


def _patch(monkeypatch, rows, symbols=None, capture=None):
    monkeypatch.setattr(svc, "_venue_tickers",
                        lambda ex: symbols or [f"KUCOIN:{s.split(':')[-1]}" for s in rows])

    def fake(screener, interval, tickers, columns):
        if capture is not None:
            capture.append({"interval": interval, "columns": list(columns)})
        return rows
    monkeypatch.setattr(svc, "resilient_scan_columns", fake)


# ── Trend strength ────────────────────────────────────────────────────────────

def test_min_adx_filters_chop(monkeypatch):
    _patch(monkeypatch, {
        "KUCOIN:TRENDING": _row(adx=40.0),
        "KUCOIN:CHOPPY": _row(adx=12.0),
    })
    out = svc.scan_trend_strength(min_adx=25.0)
    assert [r["symbol"] for r in out["rows"]] == ["TRENDING"]
    assert out["matched"] == 1


def test_direction_uses_di_alignment(monkeypatch):
    _patch(monkeypatch, {
        "KUCOIN:UP": _row(plus=30.0, minus=10.0),
        "KUCOIN:DOWN": _row(plus=10.0, minus=30.0),
    })
    up = svc.scan_trend_strength(direction="bullish")
    assert [r["symbol"] for r in up["rows"]] == ["UP"]
    assert up["rows"][0]["direction"] == "bullish"
    down = svc.scan_trend_strength(direction="bearish")
    assert [r["symbol"] for r in down["rows"]] == ["DOWN"]
    assert len(svc.scan_trend_strength(direction="both")["rows"]) == 2


def test_strength_labels_follow_thresholds(monkeypatch):
    _patch(monkeypatch, {
        "KUCOIN:VS": _row(adx=45.0),
        "KUCOIN:S": _row(adx=30.0),
    })
    rows = {r["symbol"]: r for r in svc.scan_trend_strength(min_adx=0)["rows"]}
    assert rows["VS"]["trend_strength"] == "very strong"
    assert rows["S"]["trend_strength"] == "strong"


def test_weak_label_is_reachable_when_min_adx_is_lowered(monkeypatch):
    _patch(monkeypatch, {"KUCOIN:W": _row(adx=10.0)})
    out = svc.scan_trend_strength(min_adx=0)
    assert out["rows"][0]["trend_strength"] == "weak"


def test_sar_side_is_reported(monkeypatch):
    _patch(monkeypatch, {
        "KUCOIN:SUP": _row(close=100.0, sar=90.0),
        "KUCOIN:RES": _row(close=100.0, sar=110.0),
    })
    rows = {r["symbol"]: r for r in svc.scan_trend_strength(min_adx=0)["rows"]}
    assert rows["SUP"]["sar_side"].startswith("below price")
    assert rows["RES"]["sar_side"].startswith("above price")


def test_rows_sort_by_adx_then_spread(monkeypatch):
    _patch(monkeypatch, {
        "KUCOIN:A": _row(adx=50.0, plus=30.0, minus=20.0),   # spread 10
        "KUCOIN:B": _row(adx=50.0, plus=40.0, minus=10.0),   # spread 30
        "KUCOIN:C": _row(adx=60.0),
    })
    assert [r["symbol"] for r in svc.scan_trend_strength(min_adx=0)["rows"]] == \
        ["C", "B", "A"]


def test_symbols_without_adx_are_counted_not_included(monkeypatch):
    rows = {"KUCOIN:OK": _row(), "KUCOIN:NOADX": _row()}
    rows["KUCOIN:NOADX"]["ADX"] = None
    _patch(monkeypatch, rows)
    out = svc.scan_trend_strength(min_adx=0)
    assert out["symbols_without_adx"] == 1
    assert [r["symbol"] for r in out["rows"]] == ["OK"]


def test_missing_di_excludes_the_symbol(monkeypatch):
    rows = {"KUCOIN:NODI": _row()}
    rows["KUCOIN:NODI"]["ADX+DI"] = None
    _patch(monkeypatch, rows)
    assert svc.scan_trend_strength(min_adx=0)["rows"] == []


def test_trend_scan_volume_filter(monkeypatch):
    _patch(monkeypatch, {
        "KUCOIN:THICK": _row(relvol=2.0),
        "KUCOIN:THIN": _row(relvol=0.1),
    })
    out = svc.scan_trend_strength(min_adx=0, min_rel_volume=1.0)
    assert [r["symbol"] for r in out["rows"]] == ["THICK"]


def test_trend_scan_costs_one_request(monkeypatch):
    calls = []
    _patch(monkeypatch, {f"KUCOIN:S{i}": _row() for i in range(50)}, capture=calls)
    svc.scan_trend_strength(min_adx=0)
    assert len(calls) == 1


@pytest.mark.parametrize("kwargs", [
    {"direction": "sideways"}, {"min_adx": -1}, {"min_adx": 101},
    {"limit": 0}, {"limit": 500},
])
def test_trend_scan_rejects_bad_params(kwargs):
    assert svc.scan_trend_strength(**kwargs)["error"]["code"] == "INVALID_PARAMETER"


def test_trend_scan_reports_throttling(monkeypatch):
    _patch(monkeypatch, {}, symbols=["KUCOIN:A"])
    out = svc.scan_trend_strength()
    assert out["error"]["code"] == "UPSTREAM_ERROR"
    assert "throttl" in out["error"]["message"].lower()


def test_trend_scan_rejects_unknown_exchange(monkeypatch):
    monkeypatch.setattr(svc, "_venue_tickers", lambda ex: [])
    assert svc.scan_trend_strength("NOPE")["error"]["code"] == "INVALID_EXCHANGE"


# ── Volatility ────────────────────────────────────────────────────────────────

def test_volatility_orders_both_ways(monkeypatch):
    _patch(monkeypatch, {
        "KUCOIN:CALM": _row(vol_d=2.0),
        "KUCOIN:WILD": _row(vol_d=50.0),
    })
    hi = svc.rank_volatility(order="highest")
    lo = svc.rank_volatility(order="lowest")
    assert [r["symbol"] for r in hi["rows"]] == ["WILD", "CALM"]
    assert [r["symbol"] for r in lo["rows"]] == ["CALM", "WILD"]


def test_volatility_always_queries_the_daily_interval(monkeypatch):
    calls = []
    _patch(monkeypatch, {"KUCOIN:A": _row()}, capture=calls)
    svc.rank_volatility(window="1W")
    assert calls[0]["interval"] == "1D", "volatility columns are daily-only"


def test_volatility_window_selects_the_column(monkeypatch):
    _patch(monkeypatch, {"KUCOIN:A": _row(vol_d=1.0, vol_w=2.0, vol_m=3.0)})
    assert svc.rank_volatility(window="1W")["rows"][0]["volatility_1w_pct"] == 2.0
    assert svc.rank_volatility(window="1M")["rows"][0]["volatility_1m_pct"] == 3.0


def test_atr_is_expressed_against_price(monkeypatch):
    _patch(monkeypatch, {"KUCOIN:A": _row(close=200.0, atr=10.0)})
    assert svc.rank_volatility()["rows"][0]["atr_pct_of_price"] == 5.0


def test_volatility_median_is_reported(monkeypatch):
    _patch(monkeypatch, {
        f"KUCOIN:S{i}": _row(vol_d=float(i)) for i in (1, 5, 9)
    })
    assert svc.rank_volatility()["median_volatility_pct"] == 5.0


def test_volatility_skips_symbols_without_data(monkeypatch):
    rows = {"KUCOIN:A": _row(), "KUCOIN:B": _row()}
    rows["KUCOIN:B"]["Volatility.D"] = None
    _patch(monkeypatch, rows)
    out = svc.rank_volatility()
    assert out["ranked_symbols"] == 1


def test_volatility_reports_when_nothing_has_data(monkeypatch):
    rows = {"KUCOIN:A": _row()}
    rows["KUCOIN:A"]["Volatility.D"] = None
    _patch(monkeypatch, rows)
    assert svc.rank_volatility()["error"]["code"] == "NO_DATA"


@pytest.mark.parametrize("kwargs", [
    {"window": "5Y"}, {"order": "sideways"}, {"limit": 0}, {"limit": 500},
])
def test_volatility_rejects_bad_params(kwargs):
    assert svc.rank_volatility(**kwargs)["error"]["code"] == "INVALID_PARAMETER"


def test_volatility_reports_throttling(monkeypatch):
    _patch(monkeypatch, {}, symbols=["KUCOIN:A"])
    out = svc.rank_volatility()
    assert out["error"]["code"] == "UPSTREAM_ERROR"
