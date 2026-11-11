"""Unit tests for the market-cap bubbles dataset. Offline.

The upstream call is mocked. What matters here is the shaping: a view that
drops stablecoins must still return the number of coins that was asked for,
a window that does not exist must be refused rather than silently becoming
24h, and a rate limit has to come back as a retryable error instead of an
empty board that looks like a market where nothing moved.
"""
import json
import urllib.error

import pytest

from tradingview_mcp.core.services import bubbles_service as svc


@pytest.fixture(autouse=True)
def _clear_cache():
    svc._cache.clear()
    yield
    svc._cache.clear()


def _coin(rank, symbol, mcap, **changes):
    row = {
        "market_cap_rank": rank,
        "symbol": symbol.lower(),
        "name": symbol,
        "current_price": 100.0,
        "market_cap": mcap,
        "total_volume": mcap / 20,
    }
    for window, field in svc.WINDOWS.items():
        row[field] = changes.get(window, 0.0)
    return row


def _patch(monkeypatch, coins):
    monkeypatch.setattr(svc, "_fetch", lambda limit: coins[:limit])


# ── Shaping ───────────────────────────────────────────────────────────────────

def test_it_returns_the_coins_with_every_window(monkeypatch):
    _patch(monkeypatch, [_coin(1, "BTC", 1e12, **{"24h": -2.0, "7d": -8.0})])
    out = svc.crypto_bubbles(limit=1)
    coin = out["coins"][0]
    assert coin["symbol"] == "BTC"
    assert set(coin["change_pct"]) == set(svc.WINDOWS)
    assert coin["change_pct"]["24h"] == -2.0
    assert coin["change_pct"]["7d"] == -8.0


def test_dropping_stablecoins_still_returns_the_number_asked_for(monkeypatch):
    """Returning 28 when 30 were asked for would be a quiet short-change."""
    coins = [_coin(1, "BTC", 1e12), _coin(2, "USDT", 9e11),
             _coin(3, "ETH", 5e11), _coin(4, "USDC", 4e11),
             _coin(5, "BNB", 3e11)]
    _patch(monkeypatch, coins)
    out = svc.crypto_bubbles(limit=3)
    assert [c["symbol"] for c in out["coins"]] == ["BTC", "ETH", "BNB"]
    assert out["count"] == 3
    assert out["stablecoins_removed"] == ["USDT", "USDC"]


def test_stablecoins_can_be_kept(monkeypatch):
    _patch(monkeypatch, [_coin(1, "BTC", 1e12), _coin(2, "USDT", 9e11)])
    out = svc.crypto_bubbles(limit=2, include_stablecoins=True)
    assert [c["symbol"] for c in out["coins"]] == ["BTC", "USDT"]
    assert out["stablecoins_removed"] is None


def test_a_stablecoin_is_judged_by_what_it_is_not_by_its_price(monkeypatch):
    """A depegged stablecoin is still a stablecoin, and would otherwise show
    up only on the day its bubble is most misleading."""
    _patch(monkeypatch, [_coin(1, "BTC", 1e12),
                         _coin(2, "USDT", 9e11, **{"24h": -12.0})])
    out = svc.crypto_bubbles(limit=2)
    assert [c["symbol"] for c in out["coins"]] == ["BTC"]


# ── Breadth ───────────────────────────────────────────────────────────────────

def test_breadth_counts_the_window_that_was_asked_for(monkeypatch):
    _patch(monkeypatch, [
        _coin(1, "BTC", 1e12, **{"24h": 1.0, "7d": -5.0}),
        _coin(2, "ETH", 5e11, **{"24h": 2.0, "7d": -9.0}),
        _coin(3, "BNB", 3e11, **{"24h": -1.0, "7d": 3.0}),
    ])
    day = svc.crypto_bubbles(limit=3, window="24h")["breadth"]
    assert (day["up"], day["down"]) == (2, 1)
    week = svc.crypto_bubbles(limit=3, window="7d")["breadth"]
    assert (week["up"], week["down"]) == (1, 2)


def test_breadth_names_the_extremes(monkeypatch):
    _patch(monkeypatch, [
        _coin(1, "BTC", 1e12, **{"24h": -1.0}),
        _coin(2, "ETH", 5e11, **{"24h": -19.0}),
        _coin(3, "BNB", 3e11, **{"24h": 4.0}),
    ])
    breadth = svc.crypto_bubbles(limit=3)["breadth"]
    assert breadth["worst"] == {"symbol": "ETH", "change_pct": -19.0}
    assert breadth["best"] == {"symbol": "BNB", "change_pct": 4.0}


def test_a_coin_with_no_figure_is_not_counted_as_flat(monkeypatch):
    """A missing change is missing, not zero — the same rule as everywhere."""
    coin = _coin(2, "NEW", 1e9)
    coin[svc.WINDOWS["1y"]] = None
    _patch(monkeypatch, [_coin(1, "BTC", 1e12, **{"1y": 10.0}), coin])
    out = svc.crypto_bubbles(limit=2, window="1y")
    assert out["coins"][1]["change_pct"]["1y"] is None
    assert out["breadth"]["measured"] == 1
    assert out["breadth"]["unchanged"] == 0


# ── Refusals ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("limit", [0, -1, 101, 500])
def test_an_impossible_limit_is_refused(monkeypatch, limit):
    _patch(monkeypatch, [_coin(1, "BTC", 1e12)])
    assert svc.crypto_bubbles(limit=limit)["error"]["code"] == "INVALID_PARAMETER"


def test_an_unknown_window_is_refused_rather_than_defaulted(monkeypatch):
    """Silently serving 24h for "5d" would answer a question nobody asked."""
    _patch(monkeypatch, [_coin(1, "BTC", 1e12)])
    out = svc.crypto_bubbles(window="5d")
    assert out["error"]["code"] == "INVALID_PARAMETER"
    assert "7d" in out["error"]["valid_windows"]


def test_a_rate_limit_is_retryable_and_says_why(monkeypatch):
    """An empty board would look like a market where nothing moved."""
    def boom(limit):
        raise urllib.error.HTTPError(svc._URL, 429, "Too Many Requests", {}, None)

    monkeypatch.setattr(svc, "_fetch", boom)
    out = svc.crypto_bubbles()
    assert out["error"]["code"] == "UPSTREAM_ERROR"
    assert out["error"]["retryable"] is True
    assert "rate limited" in out["error"]["message"]


def test_an_unreachable_upstream_is_an_error_not_an_empty_board(monkeypatch):
    def boom(limit):
        raise RuntimeError("dns failure")

    monkeypatch.setattr(svc, "_fetch", boom)
    out = svc.crypto_bubbles()
    assert out["error"]["code"] == "UPSTREAM_ERROR"
    assert out["error"]["retryable"] is True


def test_a_bad_request_is_not_retried_forever(monkeypatch):
    def boom(limit):
        raise urllib.error.HTTPError(svc._URL, 400, "Bad Request", {}, None)

    monkeypatch.setattr(svc, "_fetch", boom)
    assert svc.crypto_bubbles()["error"]["retryable"] is False


# ── What it says about itself ─────────────────────────────────────────────────

def test_it_says_the_change_is_price_not_market_cap(monkeypatch):
    _patch(monkeypatch, [_coin(1, "BTC", 1e12)])
    out = svc.crypto_bubbles(limit=1)
    assert "PRICE change, not market-cap change" in out["measures"]
    assert "CoinGecko" in out["source"]


def test_the_hint_warns_that_bitcoin_swamps_a_linear_radius(monkeypatch):
    """A thousand-to-one range drawn linearly is one circle and some dust."""
    _patch(monkeypatch, [_coin(1, "BTC", 1e12)])
    hint = svc.crypto_bubbles(limit=1)["display_hint"]
    assert "compress the radius" in hint
