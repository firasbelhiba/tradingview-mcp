"""Unit tests for the alert store and service.

Each test gets its own SQLite file via tmp_path, so nothing touches a real
user's alerts and the tests cannot interfere with each other. Upstream reads
are monkeypatched.
"""
import pytest

from tradingview_mcp.core.services import alert_service as svc
from tradingview_mcp.core.services import alert_store as store
from tradingview_mcp.core.services.indicators import RELATIVE_VOLUME


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADINGVIEW_MCP_ALERTS_DB", str(tmp_path / "alerts.sqlite3"))
    store._initialised.clear()
    yield
    store._initialised.clear()


def _scan(monkeypatch, rows, capture=None):
    def fake(screener, interval, symbols, columns):
        if capture is not None:
            capture.append({"interval": interval, "symbols": list(symbols),
                            "columns": list(columns)})
        return rows
    monkeypatch.setattr(svc, "resilient_scan_columns", fake)


# ── Store ─────────────────────────────────────────────────────────────────────

def test_alert_round_trips_through_the_store():
    created = store.add_alert("BTCUSDT", "KUCOIN", "price", "above", 80000.0, "4h",
                              note="breakout watch")
    assert created["id"] >= 1
    assert created["state"] == store.ARMED
    assert created["note"] == "breakout watch"

    fetched = store.get_alert(created["id"])
    assert fetched["symbol"] == "BTCUSDT"
    assert fetched["threshold"] == 80000.0


def test_marking_triggered_changes_state():
    a = store.add_alert("BTCUSDT", "KUCOIN", "price", "above", 1.0, "4h")
    store.mark_triggered(a["id"], 2.0)
    after = store.get_alert(a["id"])
    assert after["state"] == store.TRIGGERED
    assert after["triggered_value"] == 2.0
    assert after["triggered_at"]


def test_rearm_returns_an_alert_to_armed():
    a = store.add_alert("BTCUSDT", "KUCOIN", "price", "above", 1.0, "4h")
    store.mark_triggered(a["id"], 2.0)
    assert store.rearm(a["id"]) is True
    assert store.get_alert(a["id"])["state"] == store.ARMED


def test_list_filters_by_state():
    armed = store.add_alert("A", "KUCOIN", "price", "above", 1.0, "4h")
    fired = store.add_alert("B", "KUCOIN", "price", "above", 1.0, "4h")
    store.mark_triggered(fired["id"], 5.0)
    assert [r["symbol"] for r in store.list_alerts(store.ARMED)] == ["A"]
    assert [r["symbol"] for r in store.list_alerts(store.TRIGGERED)] == ["B"]
    assert len(store.list_alerts()) == 2


def test_delete_removes_the_row():
    a = store.add_alert("A", "KUCOIN", "price", "above", 1.0, "4h")
    assert store.delete_alert(a["id"]) is True
    assert store.get_alert(a["id"]) is None
    assert store.delete_alert(a["id"]) is False


def test_record_check_distinguishes_checked_from_never_checked():
    a = store.add_alert("A", "KUCOIN", "price", "above", 99.0, "4h")
    assert store.get_alert(a["id"])["last_checked_at"] is None
    store.record_check(a["id"], 50.0)
    after = store.get_alert(a["id"])
    assert after["last_checked_at"] is not None
    assert after["last_value"] == 50.0
    assert after["state"] == store.ARMED


# ── Create validation ─────────────────────────────────────────────────────────

def test_create_rejects_unknown_metric():
    out = svc.create_alert("BTCUSDT", "moon_phase", "above", 1.0)
    assert out["error"]["code"] == "INVALID_PARAMETER"
    assert "price" in out["error"]["valid_metrics"]


def test_create_rejects_unknown_operator():
    out = svc.create_alert("BTCUSDT", "price", "equals", 1.0)
    assert out["error"]["code"] == "INVALID_PARAMETER"
    assert out["error"]["valid_operators"] == ["above", "below"]


def test_create_rejects_non_numeric_threshold():
    out = svc.create_alert("BTCUSDT", "price", "above", "soon")
    assert out["error"]["code"] == "INVALID_PARAMETER"


def test_create_requires_a_symbol_for_per_symbol_metrics():
    out = svc.create_alert("", "price", "above", 1.0)
    assert out["error"]["code"] == "INVALID_PARAMETER"


def test_market_wide_metric_does_not_need_a_symbol():
    out = svc.create_alert("", "breadth_above_ema50_pct", "below", 10.0,
                           exchange="KUCOIN")
    assert "error" not in out
    assert out["symbol"] == "*"
    assert "KUCOIN" in out["condition"]


def test_create_normalises_and_describes():
    out = svc.create_alert("btcusdt", "RSI", "Below", 25, exchange="kucoin")
    assert out["symbol"] == "BTCUSDT"
    assert out["metric"] == "rsi"
    assert out["operator"] == "below"
    assert out["exchange"] == "KUCOIN"
    assert out["condition"] == "BTCUSDT rsi below 25.0 0-100"


def test_metric_catalogue_covers_every_metric():
    cat = svc.metric_catalogue()
    assert {c["metric"] for c in cat} == set(svc.METRICS)
    assert all(c["unit"] and c["description"] for c in cat)


# ── Checking ──────────────────────────────────────────────────────────────────

def test_check_with_no_alerts_says_so():
    out = svc.check_alerts()
    assert out["checked"] == 0
    assert out["fired"] == []
    assert "no alerts are armed" in out["note"]


def test_alert_fires_and_disarms(monkeypatch):
    svc.create_alert("BTCUSDT", "price", "above", 70000.0)
    _scan(monkeypatch, {"KUCOIN:BTCUSDT": {"close": 74000.0}})
    out = svc.check_alerts()
    assert out["fired_count"] == 1
    fired = out["fired"][0]
    assert fired["value"] == 74000.0
    assert fired["disarmed"] is True
    # A disarmed alert is not re-checked.
    assert svc.check_alerts()["checked"] == 0


def test_repeat_alert_stays_armed_after_firing(monkeypatch):
    svc.create_alert("BTCUSDT", "price", "above", 70000.0, repeat_after_fire=True)
    _scan(monkeypatch, {"KUCOIN:BTCUSDT": {"close": 74000.0}})
    first = svc.check_alerts()
    assert first["fired_count"] == 1
    assert "disarmed" not in first["fired"][0]
    assert svc.check_alerts()["fired_count"] == 1, "repeat alert must fire again"


def test_waiting_alert_reports_distance_to_trigger(monkeypatch):
    svc.create_alert("BTCUSDT", "price", "above", 80000.0)
    _scan(monkeypatch, {"KUCOIN:BTCUSDT": {"close": 74000.0}})
    out = svc.check_alerts()
    assert out["fired"] == []
    waiting = out["still_waiting"][0]
    assert waiting["distance_to_trigger"] == 6000.0


def test_below_operator(monkeypatch):
    svc.create_alert("BTCUSDT", "rsi", "below", 30.0)
    _scan(monkeypatch, {"KUCOIN:BTCUSDT": {"RSI": 24.0}})
    assert svc.check_alerts()["fired_count"] == 1


def test_below_operator_distance(monkeypatch):
    svc.create_alert("BTCUSDT", "rsi", "below", 30.0)
    _scan(monkeypatch, {"KUCOIN:BTCUSDT": {"RSI": 42.0}})
    assert svc.check_alerts()["still_waiting"][0]["distance_to_trigger"] == 12.0


def test_ema50_distance_is_computed(monkeypatch):
    svc.create_alert("BTCUSDT", "ema50_distance_pct", "below", -5.0)
    _scan(monkeypatch, {"KUCOIN:BTCUSDT": {"close": 90.0, "EMA50": 100.0}})
    out = svc.check_alerts()
    assert out["fired_count"] == 1
    assert out["fired"][0]["value"] == -10.0


def test_scanner_alerts_are_batched_into_one_request(monkeypatch):
    calls = []
    for sym in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
        svc.create_alert(sym, "price", "above", 1_000_000.0)
    svc.create_alert("BTCUSDT", "rsi", "below", 1.0)
    _scan(monkeypatch, {
        "KUCOIN:BTCUSDT": {"close": 1.0, "RSI": 50.0},
        "KUCOIN:ETHUSDT": {"close": 1.0, "RSI": 50.0},
        "KUCOIN:SOLUSDT": {"close": 1.0, "RSI": 50.0},
    }, calls)
    svc.check_alerts()
    assert len(calls) == 1, "four alerts on one exchange/timeframe = one request"
    assert set(calls[0]["columns"]) == {"close", "RSI"}
    assert len(calls[0]["symbols"]) == 3


def test_different_timeframes_are_separate_requests(monkeypatch):
    calls = []
    svc.create_alert("BTCUSDT", "price", "above", 1e9, timeframe="4h")
    svc.create_alert("BTCUSDT", "price", "above", 1e9, timeframe="1D")
    _scan(monkeypatch, {"KUCOIN:BTCUSDT": {"close": 1.0}}, calls)
    svc.check_alerts()
    assert len(calls) == 2
    assert {c["interval"] for c in calls} == {"4h", "1D"}


# ── Upstream problems must not look like a quiet market ───────────────────────

def test_throttled_scanner_is_reported_not_counted_as_unmet(monkeypatch):
    svc.create_alert("BTCUSDT", "price", "above", 70000.0)
    _scan(monkeypatch, {})  # TradingView answers a burst with empty bodies
    out = svc.check_alerts()
    assert out["fired"] == []
    assert out["still_waiting"] == []
    assert len(out["not_evaluated"]) == 1
    assert "throttling" in out["not_evaluated"][0]["reason"]
    assert out["upstream_problem"] is True


def test_scanner_exception_is_reported(monkeypatch):
    svc.create_alert("BTCUSDT", "price", "above", 1.0)

    def boom(*a, **k):
        raise RuntimeError("connection reset")

    monkeypatch.setattr(svc, "resilient_scan_columns", boom)
    out = svc.check_alerts()
    assert out["not_evaluated"][0]["reason"].startswith("scanner request failed")
    assert out["fired"] == []


def test_unknown_symbol_is_reported(monkeypatch):
    svc.create_alert("NOTACOIN", "price", "above", 1.0)
    _scan(monkeypatch, {"KUCOIN:OTHER": {"close": 5.0}})
    out = svc.check_alerts()
    assert "unknown to this exchange" in out["not_evaluated"][0]["reason"]


def test_missing_metric_value_is_reported(monkeypatch):
    svc.create_alert("BTCUSDT", "rsi", "below", 30.0)
    _scan(monkeypatch, {"KUCOIN:BTCUSDT": {"RSI": None}})
    out = svc.check_alerts()
    assert "no rsi data" in out["not_evaluated"][0]["reason"]


def test_a_failed_check_leaves_the_alert_armed(monkeypatch):
    created = svc.create_alert("BTCUSDT", "price", "above", 1.0)
    _scan(monkeypatch, {})
    svc.check_alerts()
    after = store.get_alert(created["id"])
    assert after["state"] == store.ARMED, "an outage must not disarm an alert"
    assert after["check_error"]


# ── Positioning and breadth sources ───────────────────────────────────────────

def test_funding_alert_reads_the_positioning_service(monkeypatch):
    svc.create_alert("BTCUSDT", "funding_annualised_pct", "below", 0.0)
    import tradingview_mcp.core.services.positioning_service as ps
    monkeypatch.setattr(ps, "analyze_positioning", lambda *a, **k: {
        "funding": {"annualised_pct": -5.3, "pct_per_interval": -0.0012},
        "open_interest": {"change_pct": 1.0},
        "crowd": {"long_pct": 70.0},
    })
    out = svc.check_alerts()
    assert out["fired_count"] == 1
    assert out["fired"][0]["value"] == -5.3


def test_positioning_error_is_surfaced(monkeypatch):
    svc.create_alert("SOLBTC", "funding_pct", "above", 0.0)
    import tradingview_mcp.core.services.positioning_service as ps
    monkeypatch.setattr(ps, "analyze_positioning", lambda *a, **k: {
        "error": {"code": "SYMBOL_NOT_FOUND"}
    })
    out = svc.check_alerts()
    assert "SYMBOL_NOT_FOUND" in out["not_evaluated"][0]["reason"]


def test_breadth_alert_reads_the_market_structure_service(monkeypatch):
    svc.create_alert("", "breadth_above_ema50_pct", "below", 10.0)
    import tradingview_mcp.core.services.market_structure_service as ms
    monkeypatch.setattr(ms, "analyze_market_breadth", lambda *a, **k: {
        "trend_participation": {"above_ema50_pct": 7.0}
    })
    out = svc.check_alerts()
    assert out["fired_count"] == 1
    assert out["fired"][0]["value"] == 7.0


# ── Service-level management ──────────────────────────────────────────────────

def test_list_reports_counts_and_conditions():
    svc.create_alert("BTCUSDT", "price", "above", 1.0)
    fired = svc.create_alert("ETHUSDT", "price", "above", 1.0)
    store.mark_triggered(fired["id"], 2.0)
    out = svc.list_alerts()
    assert out["count"] == 2 and out["armed"] == 1 and out["triggered"] == 1
    assert all(a["condition"] for a in out["alerts"])


def test_list_rejects_a_bad_state_filter():
    assert svc.list_alerts("sideways")["error"]["code"] == "INVALID_PARAMETER"


def test_delete_through_the_service():
    a = svc.create_alert("BTCUSDT", "price", "above", 1.0)
    out = svc.delete_alert(a["id"])
    assert out["deleted"] is True
    assert "BTCUSDT" in out["was"]


def test_delete_missing_alert_is_an_error():
    assert svc.delete_alert(4242)["error"]["code"] == "SYMBOL_NOT_FOUND"


def test_rearm_through_the_service():
    a = svc.create_alert("BTCUSDT", "price", "above", 1.0)
    store.mark_triggered(a["id"], 2.0)
    assert svc.rearm_alert(a["id"])["rearmed"] is True
    assert store.get_alert(a["id"])["state"] == store.ARMED


def test_rearm_missing_alert_is_an_error():
    assert svc.rearm_alert(4242)["error"]["code"] == "SYMBOL_NOT_FOUND"
