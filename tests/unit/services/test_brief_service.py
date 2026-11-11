"""Unit tests for the one-call market brief. Offline.

What matters here is not the numbers — each section has its own tests — but
that six feeds cannot take the answer down between them, and that the things
most easily lost in a tidy summary are surfaced.
"""
import pytest

from tradingview_mcp.core.services import brief_service as svc


def _patch_all(monkeypatch, **overrides):
    """Patch every section with a working default, overridable per test."""
    defaults = {
        "_breadth": lambda ex: {"exchange": "KUCOIN", "above_ema50_pct": 7.0,
                                "above_sma200_pct": 21.0, "advancing": 100,
                                "declining": 300, "median_change_pct": -1.0,
                                "verdict": "WEAK", "reasoning": "broad decline",
                                "measured_over": 498},
        "_bitcoin": lambda: {"price": 73000, "change_24h_pct": -1.5,
                             "btc_dominance_pct": 59.7,
                             "total_market_change_24h_pct": -3.0,
                             "verdict": "ALT_RISK", "reasoning": "x"},
        "_positioning": lambda sym: {"symbol": "BTCUSDT", "crowd_long_pct": 75.0,
                                     "top_traders_long_pct": 68.0,
                                     "divergence_pts": 7.0,
                                     "funding_annualised_pct": 8.1,
                                     "open_interest_change_pct": 6.5,
                                     "squeeze_risk": "ELEVATED_LONG_SQUEEZE",
                                     "reasoning": "y"},
        "_sentiment": lambda: {"fear_greed": 32, "classification": "Fear",
                               "change_over_14d": -19, "direction": "toward fear"},
        "_alerts": lambda: {"checked": 1, "fired": [], "still_waiting": 1,
                            "not_evaluated": 0},
        "_wallet": lambda: {"equity": 10000.0, "currency": "USDT",
                            "total_return_pct": 0.0, "unrealised_pnl": 0.0,
                            "open_positions": 1, "settled_this_check": 0,
                            "unsettled_positions": 0},
    }
    defaults.update(overrides)
    for name, fn in defaults.items():
        monkeypatch.setattr(svc, name, fn)


# ── Assembly ──────────────────────────────────────────────────────────────────

def test_brief_gathers_every_section(monkeypatch):
    _patch_all(monkeypatch)
    out = svc.market_brief()
    assert set(out["sections"]) == {
        "regime", "bitcoin", "positioning", "sentiment", "alerts", "wallet"
    }
    assert out["unavailable"] is None


def test_headline_restates_what_the_sections_decided(monkeypatch):
    _patch_all(monkeypatch)
    headline = svc.market_brief()["headline"]
    for fragment in ("WEAK", "BTC -1.5%", "ELEVATED_LONG_SQUEEZE", "Fear (32)"):
        assert fragment in headline, fragment


def test_sections_can_be_switched_off(monkeypatch):
    _patch_all(monkeypatch)
    out = svc.market_brief(include_wallet=False, include_alerts=False)
    assert "wallet" not in out["sections"]
    assert "alerts" not in out["sections"]


# ── One failure must not take the answer down ─────────────────────────────────

def test_a_failed_section_is_reported_and_the_rest_survive(monkeypatch):
    def boom():
        raise RuntimeError("alternative.me timed out")

    _patch_all(monkeypatch, _sentiment=boom)
    out = svc.market_brief()
    assert "sentiment" not in out["sections"]
    assert "timed out" in out["unavailable"]["sentiment"]
    assert len(out["sections"]) == 5, "the other five must still be there"
    assert any("sentiment unavailable" in a for a in out["attention"])


def test_an_error_envelope_counts_as_a_failure(monkeypatch):
    """A section returning {"error": ...} is not a section."""
    _patch_all(monkeypatch,
               _wallet=lambda: {"error": {"code": "NO_DATA",
                                          "message": "No paper wallet yet"}})
    out = svc.market_brief()
    assert "wallet" not in out["sections"]
    assert "NO_DATA" in out["unavailable"]["wallet"]


def test_total_failure_is_an_error(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("everything is down")

    _patch_all(monkeypatch, _breadth=boom, _bitcoin=boom, _positioning=boom,
               _sentiment=boom, _alerts=boom, _wallet=boom)
    out = svc.market_brief()
    assert out["error"]["code"] == "UPSTREAM_ERROR"
    assert out["error"]["retryable"] is True
    assert len(out["error"]["unavailable"]) == 6


# ── Attention: what a tidy summary would bury ─────────────────────────────────

def test_fired_alerts_reach_attention(monkeypatch):
    _patch_all(monkeypatch, _alerts=lambda: {
        "checked": 2,
        "fired": [{"alert_id": 1, "condition": "BTC funding below 0",
                   "value": -2.1}],
        "still_waiting": 1, "not_evaluated": 0,
    })
    out = svc.market_brief()
    assert any("alert fired" in a for a in out["attention"])
    assert out["fired_alerts"][0]["alert_id"] == 1


def test_unevaluated_alerts_are_distinguished_from_unmet_ones(monkeypatch):
    """The distinction the alert service exists to preserve."""
    _patch_all(monkeypatch, _alerts=lambda: {
        "checked": 3, "fired": [], "still_waiting": 1, "not_evaluated": 2,
    })
    attention = " ".join(svc.market_brief()["attention"])
    assert "could NOT be evaluated" in attention
    assert "not the same as" in attention


def test_unsettled_wallet_positions_reach_attention(monkeypatch):
    _patch_all(monkeypatch, _wallet=lambda: {
        "equity": 1.0, "currency": "USDT", "total_return_pct": 0.0,
        "unrealised_pnl": 0.0, "open_positions": 1,
        "settled_this_check": 0, "unsettled_positions": 2,
    })
    assert any("could not be checked" in a
               for a in svc.market_brief()["attention"])


def test_positions_that_hit_a_level_reach_attention(monkeypatch):
    _patch_all(monkeypatch, _wallet=lambda: {
        "equity": 1.0, "currency": "USDT", "total_return_pct": 0.0,
        "unrealised_pnl": 0.0, "open_positions": 0,
        "settled_this_check": 1, "unsettled_positions": 0,
    })
    assert any("hit a level" in a for a in svc.market_brief()["attention"])


def test_a_healthy_brief_has_nothing_to_flag(monkeypatch):
    _patch_all(monkeypatch)
    assert svc.market_brief()["attention"] == []


# ── Scope ─────────────────────────────────────────────────────────────────────

def test_the_brief_reports_rather_than_advises(monkeypatch):
    """It may assemble what sections concluded, never add a view of its own.

    A server that tells an assistant what a market means is doing something
    different from one that gathers readings.
    """
    _patch_all(monkeypatch)
    out = svc.market_brief()
    blob = " ".join([out["headline"] or "", " ".join(out["attention"]),
                     out["scope_note"]]).lower()
    for forbidden in ("you should", "recommend", "buy ", "sell ", "good entry",
                      "opportunity"):
        assert forbidden not in blob, f"brief drifted into advice: {forbidden}"
