"""Unit tests for the lead-trader trade log. Offline.

Pagination and the performance maths get the attention: a paging bug silently
returns a partial history as if it were complete, and a wrong profit factor
or drawdown would misdescribe someone's record.
"""
import pytest

from tradingview_mcp.core.services import copytrading_trades_service as svc


def _raw(trade_id, pnl=100.0, ratio=0.1, lever="2E+1", inst="BTC-USDT-SWAP",
         side="long", open_ms=1_700_000_000_000, close_ms=1_700_003_600_000,
         margin=1000.0):
    return {
        "subPosId": str(trade_id), "pnl": str(pnl), "pnlRatio": str(ratio),
        "lever": lever, "instId": inst, "posSide": side,
        "openTime": str(open_ms), "closeTime": str(close_ms),
        "margin": str(margin), "subPos": "400", "openAvgPx": "70000",
        "closeAvgPx": "70700", "ccy": "USDT", "mgnMode": "cross",
    }


def _patch_pages(monkeypatch, pages, capture=None):
    """Serve successive pages, keyed by the `after` cursor."""
    calls = []

    def fake_get(path):
        calls.append(path)
        if capture is not None:
            capture.append(path)
        after = None
        if "after=" in path:
            after = path.split("after=")[1].split("&")[0]
        page = pages.get(after, [])
        if isinstance(page, Exception):
            raise page
        return {"code": "0", "data": page}

    monkeypatch.setattr(svc, "_get", fake_get)
    return calls


# ── Row mapping ───────────────────────────────────────────────────────────────

def test_trade_row_is_mapped_and_hold_time_computed():
    t = svc._trade(_raw(1, open_ms=1_700_000_000_000, close_ms=1_700_010_800_000))
    assert t["trade_id"] == "1"
    assert t["leverage"] == 20.0          # "2E+1"
    assert t["held_hours"] == 3.0         # 10,800,000 ms
    assert t["opened_at"].endswith("+00:00")
    assert t["pnl_pct_of_margin"] == 10.0


def test_hold_time_is_none_when_close_precedes_open():
    t = svc._trade(_raw(1, open_ms=2_000_000_000_000, close_ms=1_000_000_000_000))
    assert t["held_hours"] is None


def test_unparseable_timestamps_do_not_crash():
    raw = _raw(1)
    raw["closeTime"] = "not-a-time"
    t = svc._trade(raw)
    assert t["closed_at"] is None
    assert t["held_hours"] is None


# ── Pagination ────────────────────────────────────────────────────────────────

def test_pages_are_followed_until_a_short_page(monkeypatch):
    first = [_raw(i) for i in range(svc._PAGE_SIZE)]
    second = [_raw(1000 + i) for i in range(10)]   # short -> end of history
    _patch_pages(monkeypatch, {None: first, str(svc._PAGE_SIZE - 1): second})
    out = svc.get_lead_trader_trades("C", max_trades=500)
    assert out["pages_fetched"] == 2
    assert out["trades_fetched"] == svc._PAGE_SIZE + 10
    assert out["complete_history"] is True


def test_cursor_is_the_last_id_of_the_previous_page(monkeypatch):
    first = [_raw(i) for i in range(svc._PAGE_SIZE)]
    calls = _patch_pages(monkeypatch, {None: first,
                                       str(svc._PAGE_SIZE - 1): [_raw(9999)]})
    svc.get_lead_trader_trades("C", max_trades=500)
    assert f"after={svc._PAGE_SIZE - 1}" in calls[1]


def test_repeated_page_stops_paging(monkeypatch):
    """A cursor the API ignores would otherwise loop forever."""
    same = [_raw(i) for i in range(svc._PAGE_SIZE)]

    def fake_get(path):
        return {"code": "0", "data": same}

    monkeypatch.setattr(svc, "_get", fake_get)
    out = svc.get_lead_trader_trades("C", max_trades=2000)
    assert out["trades_fetched"] == svc._PAGE_SIZE
    assert out["pages_fetched"] == 2  # second page was all duplicates


def test_max_trades_is_respected(monkeypatch):
    first = [_raw(i) for i in range(svc._PAGE_SIZE)]
    _patch_pages(monkeypatch, {None: first,
                               str(svc._PAGE_SIZE - 1): [_raw(500 + i) for i in range(100)]})
    out = svc.get_lead_trader_trades("C", max_trades=120)
    assert out["trades_returned"] == 120


def test_truncation_is_declared(monkeypatch):
    page = [_raw(i) for i in range(svc._PAGE_SIZE)]

    def fake_get(path):
        after = path.split("after=")[1].split("&")[0] if "after=" in path else "0"
        base = int(after) + 1
        return {"code": "0", "data": [_raw(base + i) for i in range(svc._PAGE_SIZE)]}

    monkeypatch.setattr(svc, "_get", fake_get)
    out = svc.get_lead_trader_trades("C", max_trades=2000)
    assert out["complete_history"] is False
    assert "older trades exist" in out["truncation_note"]


def test_paging_error_after_some_rows_is_reported_not_fatal(monkeypatch):
    first = [_raw(i) for i in range(svc._PAGE_SIZE)]
    _patch_pages(monkeypatch, {None: first,
                               str(svc._PAGE_SIZE - 1): RuntimeError("rate limited")})
    out = svc.get_lead_trader_trades("C", max_trades=500)
    assert out["trades_returned"] == svc._PAGE_SIZE
    assert "paging stopped early" in out["incomplete"]
    assert out["complete_history"] is False


def test_error_on_the_first_page_is_an_error(monkeypatch):
    _patch_pages(monkeypatch, {None: RuntimeError("HTTP 503")})
    out = svc.get_lead_trader_trades("C")
    assert out["error"]["code"] == "UPSTREAM_ERROR"
    assert out["error"]["retryable"] is True


def test_no_trades_explains_the_likely_cause(monkeypatch):
    _patch_pages(monkeypatch, {None: []})
    out = svc.get_lead_trader_trades("C", instrument_type="SWAP")
    assert out["error"]["code"] == "NO_DATA"
    assert "lead on SPOT" in out["error"]["message"]


def test_spot_is_refused_with_the_measured_reason():
    """OKX rejects instType=SPOT here (400/51000) and returns nothing for
    spot traders with it omitted, so the tool refuses up front rather than
    forwarding a 400 or implying a spot log exists."""
    out = svc.get_lead_trader_trades("C", instrument_type="SPOT")
    assert out["error"]["code"] == "INVALID_PARAMETER"
    assert out["error"]["supported"] == ["SWAP"]
    assert "lead_trader_profile" in out["error"]["message"]


# ── Statistics ────────────────────────────────────────────────────────────────

def _trades(*specs):
    return [svc._trade(_raw(i, **spec)) for i, spec in enumerate(specs)]


def test_profit_factor_and_expectancy():
    trades = _trades({"pnl": 300.0}, {"pnl": 100.0}, {"pnl": -200.0})
    s = svc._statistics(trades)
    assert s["gross_profit"] == 400.0
    assert s["gross_loss"] == 200.0
    assert s["profit_factor"] == 2.0
    assert s["net_pnl"] == 200.0
    assert s["expectancy_per_trade"] == pytest.approx(66.67, abs=0.01)
    assert s["win_rate_pct"] == pytest.approx(66.7, abs=0.1)


def test_profit_factor_is_null_without_a_loss():
    """Infinity would read as a flawless record; null says 'undefined'."""
    s = svc._statistics(_trades({"pnl": 100.0}, {"pnl": 50.0}))
    assert s["profit_factor"] is None
    assert s["losses"] == 0


def test_win_loss_size_ratio():
    trades = _trades({"pnl": 200.0}, {"pnl": 200.0}, {"pnl": -100.0})
    s = svc._statistics(trades)
    assert s["avg_win"] == 200.0
    assert s["avg_loss"] == 100.0
    assert s["win_loss_size_ratio"] == 2.0


def test_breakeven_trades_are_counted_separately():
    s = svc._statistics(_trades({"pnl": 100.0}, {"pnl": 0.0}, {"pnl": -50.0}))
    assert (s["wins"], s["losses"], s["breakeven"]) == (1, 1, 1)


def test_leverage_and_hold_statistics():
    trades = _trades(
        {"lever": "1E+1", "open_ms": 0, "close_ms": 3_600_000},
        {"lever": "3E+1", "open_ms": 0, "close_ms": 10_800_000},
    )
    s = svc._statistics(trades)
    assert s["avg_leverage"] == 20.0
    assert s["max_leverage"] == 30
    assert s["median_hold_hours"] == 2.0
    assert s["longest_hold_hours"] == 3.0


def test_margin_return_extremes_are_surfaced():
    s = svc._statistics(_trades({"ratio": 10.84}, {"ratio": -4.0484}))
    assert s["best_trade_pct_of_margin"] == 1084.0
    assert s["worst_trade_pct_of_margin"] == -404.84


def test_median_of_even_and_odd_samples():
    assert svc._median([1.0, 3.0]) == 2.0
    assert svc._median([5.0, 1.0, 3.0]) == 3.0
    assert svc._median([]) is None


# ── Equity curve and drawdown ─────────────────────────────────────────────────

def test_equity_curve_is_ordered_by_close_time_not_input_order():
    # Fed newest-first, as the API returns them.
    trades = _trades(
        {"pnl": -60.0, "close_ms": 3_000_000_000_000},
        {"pnl": 100.0, "close_ms": 1_000_000_000_000},
    )
    curve, drawdown, pct = svc._equity_curve(trades)
    assert [c["cumulative_pnl"] for c in curve] == [100.0, 40.0]
    assert drawdown == -60.0
    assert pct == -60.0


def test_drawdown_is_zero_for_a_monotonic_winner():
    trades = _trades(
        {"pnl": 50.0, "close_ms": 1_000_000_000_000},
        {"pnl": 50.0, "close_ms": 2_000_000_000_000},
    )
    _, drawdown, pct = svc._equity_curve(trades)
    assert drawdown == 0.0 and pct == 0.0


def test_drawdown_percentage_is_relative_to_the_peak():
    trades = _trades(
        {"pnl": 1000.0, "close_ms": 1_000_000_000_000},
        {"pnl": -250.0, "close_ms": 2_000_000_000_000},
    )
    _, drawdown, pct = svc._equity_curve(trades)
    assert drawdown == -250.0
    assert pct == -25.0


def test_equity_curve_of_nothing():
    assert svc._equity_curve([]) == ([], None, None)


# ── Breakdowns and shaping ────────────────────────────────────────────────────

def test_breakdown_by_instrument_and_side(monkeypatch):
    _patch_pages(monkeypatch, {None: [
        _raw(1, pnl=500.0, inst="CL-USDT-SWAP", side="long"),
        _raw(2, pnl=-100.0, inst="CL-USDT-SWAP", side="short"),
        _raw(3, pnl=200.0, inst="ZEC-USDT-SWAP", side="long"),
    ]})
    out = svc.get_lead_trader_trades("C")
    by_inst = {b["instrument"]: b for b in out["by_instrument"]}
    assert by_inst["CL-USDT-SWAP"]["trades"] == 2
    assert by_inst["CL-USDT-SWAP"]["net_pnl"] == 400.0
    by_side = {b["side"]: b for b in out["by_side"]}
    assert by_side["long"]["net_pnl"] == 700.0
    assert by_side["short"]["net_pnl"] == -100.0


def test_instrument_filter_narrows_the_set(monkeypatch):
    _patch_pages(monkeypatch, {None: [
        _raw(1, inst="BTC-USDT-SWAP"), _raw(2, inst="ETH-USDT-SWAP"),
    ]})
    out = svc.get_lead_trader_trades("C", instrument="eth-usdt-swap")
    assert out["trades_returned"] == 1
    assert out["trades"][0]["instrument"] == "ETH-USDT-SWAP"


def test_instrument_filter_with_no_match(monkeypatch):
    _patch_pages(monkeypatch, {None: [_raw(1, inst="BTC-USDT-SWAP")]})
    out = svc.get_lead_trader_trades("C", instrument="DOGE-USDT-SWAP")
    assert out["error"]["code"] == "NO_DATA"


def test_rows_can_be_omitted(monkeypatch):
    _patch_pages(monkeypatch, {None: [_raw(1)]})
    out = svc.get_lead_trader_trades("C", include_trades=False)
    assert "trades" not in out
    assert out["statistics"]["trades"] == 1


def test_equity_curve_is_opt_in(monkeypatch):
    _patch_pages(monkeypatch, {None: [_raw(1)]})
    assert "equity_curve" not in svc.get_lead_trader_trades("C")
    assert "equity_curve" in svc.get_lead_trader_trades("C", include_equity_curve=True)


def test_trades_are_returned_newest_first(monkeypatch):
    _patch_pages(monkeypatch, {None: [
        _raw(1, close_ms=1_000_000_000_000), _raw(2, close_ms=3_000_000_000_000),
    ]})
    out = svc.get_lead_trader_trades("C")
    assert [t["trade_id"] for t in out["trades"]] == ["2", "1"]


def test_field_notes_explain_margin_return_and_drawdown(monkeypatch):
    _patch_pages(monkeypatch, {None: [_raw(1)]})
    notes = svc.get_lead_trader_trades("C")["field_notes"]
    assert "MARGIN" in notes["pnl_pct_of_margin"]
    assert "REALISED" in notes["realised_drawdown"]


@pytest.mark.parametrize("kwargs", [
    {"max_trades": 1}, {"max_trades": 99999}, {"instrument_type": "FUTURES"},
    {"instrument_type": "SPOT"},
])
def test_rejects_bad_params(kwargs):
    assert svc.get_lead_trader_trades("C", **kwargs)["error"]["code"] == \
        "INVALID_PARAMETER"


def test_requires_a_code():
    assert svc.get_lead_trader_trades("")["error"]["code"] == "INVALID_PARAMETER"
    assert svc.get_lead_trader_trades("  ")["error"]["code"] == "INVALID_PARAMETER"
